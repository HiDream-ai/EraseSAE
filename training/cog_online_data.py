import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from training.hunyuan_online_data import (
    CELEBRITY_PREFIXES,
    EpochSampler,
    NON_CELEBRITY_PREFIXES,
    PromptMetadataDataset,
    find_token_indices,
    normalize_target_concepts,
    pad_distributed_indices,
    validate_training_records,
)
from training.streaming_stats import update_channel_moments


COG_NUDITY_PREFIXES = ["naked", "nude", "shirtless", "topless"]
COG_NO_NUDITY_PREFIXES = [
    "fully clothed",
    "dressed",
    "clothed",
    "wearing a top",
    "wearing-a-top",
]


def validate_cog_tokenizer_runtime():
    """Fail before model loading when SentencePiece/protobuf is incompatible."""
    try:
        from sentencepiece import sentencepiece_model_pb2  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "CogVideoX tokenizer dependencies are incompatible. Install the "
            "tested protobuf/SentencePiece combination with: "
            "python -m pip install 'protobuf==3.20.3' "
            "'sentencepiece==0.2.1'."
        ) from exc


def _first_jsonl(directory):
    files = sorted(Path(directory).glob("*.jsonl"))
    if not files:
        return None
    return files[0]


def _resolve_video_path(video_path, metadata_dir):
    path = Path(video_path)
    if path.is_absolute():
        return path
    return Path(metadata_dir) / path


class CogVideoMetadataDataset:
    """
    Reads raw CogVideoX metadata/videos and leaves activation extraction to the
    training iterator. This mirrors the old collect_data contract without
    writing per-timestep activation files.
    """

    def __init__(
        self,
        root_dir,
        task,
        target_concepts=None,
        num_frames=16,
        height=480,
        width=720,
        max_samples_per_subject=None,
    ):
        self.root_dir = Path(root_dir)
        self.task = task
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.max_samples_per_subject = max_samples_per_subject

        primary_group = "celebrity" if task == "celebrity" else "nsfw"
        target_concepts = normalize_target_concepts(target_concepts)
        self.concepts = target_concepts or self._discover_subjects(primary_group)
        self.concept_to_id = {name: i for i, name in enumerate(self.concepts)}
        self.records = []
        self._load_records()

        if not self.records:
            raise ValueError(
                f"No metadata records found under {self.root_dir}. "
                "Expected group/subject/*.jsonl with prompt and video_path fields."
            )
        validate_training_records(
            self.records,
            self.concepts,
            self.task,
            self.root_dir,
        )

    def _group_aliases(self, group_name):
        aliases = {
            "no_celebrity": ["no_celebrity", "non_celebrity"],
            "non_celebrity": ["non_celebrity", "no_celebrity"],
        }
        return aliases.get(group_name, [group_name])

    def _group_dir(self, group_name):
        for alias in self._group_aliases(group_name):
            candidate = self.root_dir / alias
            if candidate.exists():
                return candidate
            if self.root_dir.name == alias:
                return self.root_dir
        return None

    def _discover_subjects(self, group_name):
        group_dir = self._group_dir(group_name)
        if group_dir is None:
            return []
        return sorted(d.name for d in group_dir.iterdir() if d.is_dir())

    def _groups(self):
        if self.task == "celebrity":
            return ["celebrity", "no_celebrity"]
        if self.task == "nudity":
            return ["nsfw", "no_nsfw"]
        raise ValueError(f"Unsupported CogVideoX online task: {self.task}")

    def _load_records(self):
        for group in self._groups():
            group_dir = self._group_dir(group)
            if group_dir is None:
                continue

            for subject_dir in sorted(d for d in group_dir.iterdir() if d.is_dir()):
                subject = subject_dir.name
                label = self.concept_to_id.get(subject, -1)
                jsonl_path = _first_jsonl(subject_dir)
                if jsonl_path is None:
                    continue

                loaded = 0
                with jsonl_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if self.max_samples_per_subject and loaded >= self.max_samples_per_subject:
                            break
                        item = json.loads(line)
                        prompt = item.get("prompt") or item.get("original_prompt")
                        video_path = item.get("video_path") or item.get("path")
                        if not prompt or not video_path:
                            continue
                        self.records.append(
                            {
                                "prompt": prompt,
                                "video_path": str(_resolve_video_path(video_path, jsonl_path.parent)),
                                "subject": subject,
                                "label": label,
                                "case_number": item.get("case_number", loaded),
                                "seed": item.get("seed", 0),
                            }
                        )
                        loaded += 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = dict(self.records[index])
        record["original_video"] = self._load_video(record["video_path"])
        return record

    def _load_video(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        import cv2

        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError(f"Video contains no readable frames: {path}")

        frames_np = np.asarray(frames, dtype=np.float32) / 255.0
        frames_np = (frames_np - 0.5) * 2.0
        n_frames = frames_np.shape[0]
        if n_frames >= self.num_frames:
            idxs = np.linspace(0, n_frames - 1, num=self.num_frames).astype(np.int32)
            arr = frames_np[idxs]
        else:
            pad = np.repeat(frames_np[-1][None], self.num_frames - n_frames, axis=0)
            arr = np.concatenate([frames_np, pad], axis=0)

        return torch.tensor(arr, dtype=torch.float32).permute(3, 0, 1, 2).contiguous()


class OnlineCogActivationLoader:
    """
    Iterable dataloader that creates CogVideoX activations on demand.

    Each yielded batch matches the old offline Dataset contract:
    orig_act: [B, 1, L, D], mask: [B, 1, T, H, W], label: [B].
    """

    def __init__(
        self,
        model_path,
        raw_data_root,
        task,
        layer_name,
        target_concepts,
        batch_size,
        rank,
        world_size,
        device,
        dtype,
        num_frames=16,
        height=480,
        width=720,
        activation_input="video",
        inference_steps=30,
        guidance_scale=6.0,
        timesteps_per_video=1,
        timestep_min_index=700,
        sampling_mode="trajectory",
        trajectory_steps=4,
        seed=42,
        shuffle=True,
        max_samples_per_subject=None,
        empty_cache_every_step=True,
        attention_mask_quantile=0.8,
        attention_mask_dilation=0,
    ):
        self.model_path = model_path
        self.task = task
        self.layer_name = layer_name
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.activation_input = activation_input
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.timesteps_per_video = timesteps_per_video
        self.timestep_min_index = timestep_min_index
        self.sampling_mode = sampling_mode
        self.trajectory_steps = trajectory_steps
        self.seed = seed
        self.shuffle = shuffle
        self.empty_cache_every_step = empty_cache_every_step
        self.attention_mask_quantile = attention_mask_quantile
        self.attention_mask_dilation = attention_mask_dilation
        self.mean = None
        self.std = None
        self.sampler = EpochSampler()
        self.pipe = None
        self.eraser = None

        if activation_input == "prompt":
            self.dataset = PromptMetadataDataset(
                root_dir=raw_data_root,
                task=task,
                target_concepts=target_concepts,
                max_samples_per_subject=max_samples_per_subject,
            )
        elif activation_input == "video":
            self.dataset = CogVideoMetadataDataset(
                root_dir=raw_data_root,
                task=task,
                target_concepts=target_concepts,
                num_frames=num_frames,
                height=height,
                width=width,
                max_samples_per_subject=max_samples_per_subject,
            )
        else:
            raise ValueError(f"Unsupported activation_input={activation_input}. Use 'prompt' or 'video'.")
        self.concepts = self.dataset.concepts

        if self.sampling_mode == "trajectory":
            if self.trajectory_steps < 2:
                raise ValueError("activation_trajectory_steps must be at least 2.")
            if self.timesteps_per_video > self.trajectory_steps:
                raise ValueError(
                    "activation_timesteps_per_video cannot exceed "
                    "activation_trajectory_steps in trajectory mode."
                )

    def __len__(self):
        return self.batch_count(pad=True)

    def batch_count(self, pad=True):
        n = len(self._local_indices(pad=pad))
        return math.ceil((n * self.timesteps_per_video) / self.batch_size)

    def set_normalization(self, mean, std):
        self.mean = mean.cpu() if mean is not None else None
        self.std = std.cpu() if std is not None else None

    def unload(self):
        if self.eraser is not None:
            self.eraser.remove_hooks(verbose=self.rank == 0)
        self.eraser = None
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _local_indices(self, pad=True):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed + self.sampler.epoch)
            rng.shuffle(indices)

        if self.world_size <= 1:
            return indices

        if not pad:
            return indices[self.rank::self.world_size]

        num_samples = int(math.ceil(len(indices) / self.world_size))
        total_size = num_samples * self.world_size
        indices = pad_distributed_indices(indices, total_size)
        return indices[self.rank: total_size: self.world_size]

    def _build_pipeline(self):
        if self.pipe is not None:
            return

        validate_cog_tokenizer_runtime()

        from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel
        from training.activation_hooks import create_cog_activation_collector

        transformer = CogVideoXTransformer3DModel.from_pretrained(
            self.model_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
        ).to(self.device)
        transformer.requires_grad_(False)
        transformer.eval()

        self.eraser = create_cog_activation_collector(
            transformer,
            [self.layer_name],
        )
        self.eraser.register_hooks()

        self.pipe = CogVideoXPipeline.from_pretrained(
            self.model_path,
            transformer=transformer,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.pipe.transformer.eval()

        try:
            self.pipe.vae.enable_tiling()
        except Exception:
            pass

        if self.activation_input == "video" or self.sampling_mode == "noisy_latent":
            self._ensure_timesteps()

    def _ensure_timesteps(self):
        timesteps = getattr(self.pipe.scheduler, "timesteps", None)
        if timesteps is not None and timesteps.numel() > 0:
            return

        config_steps = getattr(getattr(self.pipe.scheduler, "config", None), "num_train_timesteps", None)
        if config_steps and self.timestep_min_index >= (self.inference_steps or 0):
            target_steps = config_steps
        else:
            target_steps = max(self.inference_steps or 0, self.timestep_min_index + 6)
            if config_steps:
                target_steps = min(config_steps, target_steps)
        target_steps = max(target_steps, 1)

        try:
            self.pipe.scheduler.set_timesteps(target_steps, device=self.device)
        except TypeError:
            self.pipe.scheduler.set_timesteps(target_steps)

    def _prefixes_for_record(self, subject):
        if self.task == "nudity":
            if subject in {"no_nudity", "clothed"}:
                return COG_NO_NUDITY_PREFIXES
            return COG_NUDITY_PREFIXES

        if subject in {"no_celebrity", "non_celebrity"}:
            return NON_CELEBRITY_PREFIXES
        return CELEBRITY_PREFIXES.get(subject, [subject.replace("_", " ")])

    def _pop_activation(self, timestep, conditional_index=None):
        acts_list = self.eraser.current_batch_activations.get(self.layer_name, [])
        if len(acts_list) < 2:
            raise RuntimeError(
                f"Layer {self.layer_name} did not produce both activation and attention mask."
            )

        orig_act = acts_list[0].detach().cpu().to(torch.float32)
        mask = acts_list[1].detach().cpu().to(torch.float32)
        if conditional_index is not None:
            orig_act = orig_act[conditional_index: conditional_index + 1]
            mask = mask[conditional_index: conditional_index + 1]
        self.eraser.current_batch_activations[self.layer_name] = []
        self.eraser.temp_attn_buffer = {}
        return {
            "orig_act": orig_act,
            "mask": mask,
            "timestep": float(timestep.detach().cpu().item()),
        }

    def _clear_hook_buffers(self):
        if self.eraser is None:
            return
        for name in self.eraser.current_batch_activations:
            self.eraser.current_batch_activations[name] = []
        self.eraser.temp_attn_buffer = {}

    def _set_hook_capture(self, capture_activations, capture_attention):
        self.eraser.capture_activations = capture_activations
        self.eraser.capture_attention = capture_attention

    def _append_attention_mask(self, token_idx, height, width, head_num, required):
        acts_list = self.eraser.current_batch_activations.get(self.layer_name, [])
        if not acts_list:
            raise RuntimeError(f"Layer {self.layer_name} did not produce an activation.")

        if token_idx.numel() == 0:
            visual_act = acts_list[0]
            spatial_tokens = height * width
            if visual_act.shape[1] % spatial_tokens != 0:
                raise RuntimeError(
                    f"Cannot reshape {visual_act.shape[1]} visual tokens into "
                    f"height={height}, width={width}."
                )
            frames = visual_act.shape[1] // spatial_tokens
            acts_list.append(
                torch.zeros(
                    (visual_act.shape[0], frames, height, width),
                    dtype=torch.float32,
                )
            )
            return

        before = len(acts_list)
        self.eraser.compute_step_mask(
            word_indices=token_idx,
            height=height,
            width=width,
            head_num=head_num,
            quantile=self.attention_mask_quantile,
            dilation=self.attention_mask_dilation,
        )
        if len(acts_list) == before and required:
            raise RuntimeError(
                f"Attention mask generation failed for target layer {self.layer_name}."
            )
        # A valid quantile operation can produce zero spatial support for an
        # isolated CFG branch/timestep. Keep the sample so attribution can
        # audit and filter the extracted conditional branch globally.

    def _sample_timestep(self, timesteps, generator):
        n_steps = timesteps.numel()
        if n_steps < 2:
            raise ValueError("Scheduler has fewer than two timesteps.")
        low = min(self.timestep_min_index, max(0, n_steps - 2))
        high = n_steps - 5 if n_steps - 5 > low else n_steps
        idx = torch.randint(
            low,
            high,
            (1,),
            device=self.device,
            generator=generator,
        )
        return timesteps[idx][0].to(timesteps.dtype)

    def _prepare_rotary_embeddings(self, latent_frames):
        try:
            return self.pipe._prepare_rotary_positional_embeddings(
                height=self.height,
                width=self.width,
                num_frames=latent_frames,
                device=self.device,
            )
        except TypeError:
            return self.pipe._prepare_rotary_positional_embeddings(
                height=self.height,
                width=self.width,
                num_frames=latent_frames,
                device=self.device,
                dtype=self.dtype,
            )

    def _make_zero_latent(self):
        channels = self.pipe.transformer.config.in_channels
        latent_t = (self.num_frames - 1) // self.pipe.vae_scale_factor_temporal + 1
        latent_h = self.height // self.pipe.vae_scale_factor_spatial
        latent_w = self.width // self.pipe.vae_scale_factor_spatial
        return torch.zeros(
            (1, latent_t, channels, latent_h, latent_w),
            dtype=self.dtype,
            device=self.device,
        )

    def _add_noise(self, latent, noise, timestep):
        try:
            return self.pipe.scheduler.add_noise(latent, noise, timestep.unsqueeze(0)).to(self.dtype)
        except AttributeError:
            return self.pipe.scheduler.scale_noise(latent, noise, timestep.unsqueeze(0)).to(self.dtype)

    def _scale_model_input(self, latent, timestep):
        if hasattr(self.pipe.scheduler, "scale_model_input"):
            return self.pipe.scheduler.scale_model_input(latent, timestep)
        return latent

    def _record_generator(self, record):
        base_seed = int(record.get("seed", 0) or 0)
        if base_seed == 0:
            base_seed = self.seed + int(record.get("case_number", 0))
        epoch_seed = base_seed + self.sampler.epoch * 1_000_003
        return torch.Generator(device=self.device).manual_seed(epoch_seed)

    def _capture_step_indices(self):
        count = min(self.timesteps_per_video, self.trajectory_steps)
        # Match Hunyuan's policy so a small capture budget spans the trajectory.
        return {
            math.ceil((index + 1) * self.trajectory_steps / count) - 1
            for index in range(count)
        }

    def _append_sample_metadata(self, sample, record, trajectory_step=None):
        sample["label"] = record["label"]
        sample["subject"] = record["subject"]
        sample["prompt"] = record.get("prompt")
        sample["seed"] = record.get("seed")
        sample["case_number"] = record.get("case_number")
        if trajectory_step is not None:
            sample["trajectory_step"] = trajectory_step
        return sample

    def _generate_trajectory_samples(self, record, token_idx):
        from diffusers import CogVideoXDPMScheduler

        generator = self._record_generator(record)
        self.pipe.scheduler.set_timesteps(self.trajectory_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        capture_steps = self._capture_step_indices()

        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=[record["prompt"]],
            negative_prompt=[""],
            do_classifier_free_guidance=True,
            device=self.device,
            num_videos_per_prompt=1,
        )
        prompt_embeds = torch.cat(
            [negative_prompt_embeds, prompt_embeds],
            dim=0,
        ).to(self.dtype)

        channels = self.pipe.transformer.config.in_channels
        latents = self.pipe.prepare_latents(
            1,
            channels,
            self.num_frames,
            self.height,
            self.width,
            self.dtype,
            self.device,
            generator,
        )
        image_rotary_emb = (
            self._prepare_rotary_embeddings(latents.shape[1])
            if self.pipe.transformer.config.use_rotary_positional_embeddings
            else None
        )
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta=0.0)
        old_pred_original_sample = None
        record_samples = []

        for step_index, timestep in enumerate(timesteps):
            should_capture = step_index in capture_steps
            self._set_hook_capture(
                capture_activations=should_capture,
                capture_attention=should_capture and token_idx.numel() > 0,
            )
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = self._scale_model_input(
                latent_model_input,
                timestep,
            )
            timestep_batch = timestep.expand(latent_model_input.shape[0])
            noise_pred = self.pipe.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep_batch,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0].float()
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            if should_capture:
                patch_size = getattr(self.pipe.transformer.config, "patch_size", 2)
                if isinstance(patch_size, (tuple, list)):
                    patch_h, patch_w = patch_size[-2], patch_size[-1]
                else:
                    patch_h = patch_w = patch_size
                latent_h = latent_model_input.shape[-2] // patch_h
                latent_w = latent_model_input.shape[-1] // patch_w
                self._append_attention_mask(
                    token_idx=token_idx,
                    height=latent_h,
                    width=latent_w,
                    head_num=self.pipe.transformer.config.num_attention_heads,
                    required=record["label"] >= 0,
                )
                sample = self._pop_activation(timestep, conditional_index=1)
                record_samples.append(
                    self._append_sample_metadata(sample, record, step_index)
                )
            else:
                self._clear_hook_buffers()

            if isinstance(self.pipe.scheduler, CogVideoXDPMScheduler):
                previous_timestep = timesteps[step_index - 1] if step_index > 0 else None
                latents, old_pred_original_sample = self.pipe.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    timestep,
                    previous_timestep,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            else:
                latents = self.pipe.scheduler.step(
                    noise_pred,
                    timestep,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
            latents = latents.to(self.dtype)
            del latent_model_input, timestep_batch, noise_pred
            del noise_pred_uncond, noise_pred_text

        self._set_hook_capture(True, True)
        del latents, prompt_embeds, image_rotary_emb
        return record_samples

    def _generate_noisy_latent_samples(self, record, token_idx):
        record_samples = []
        if "original_video" in record:
            video = record["original_video"].unsqueeze(0).to(self.device)
            encoded = self.pipe.vae.encode(video.to(self.dtype))
            latent = encoded.latent_dist.sample() * self.pipe.vae.config.scaling_factor
            latent = latent.permute(0, 2, 1, 3, 4).detach().to(self.dtype)
        else:
            video = None
            latent = self._make_zero_latent()
        latent_frames = latent.shape[1]

        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=[record["prompt"]],
            device=self.device,
            num_videos_per_prompt=1,
        )
        prompt_embeds = prompt_embeds.to(self.dtype)

        timesteps = self.pipe.scheduler.timesteps.to(self.device)
        generator = self._record_generator(record)
        for _ in range(self.timesteps_per_video):
            self._set_hook_capture(
                capture_activations=True,
                capture_attention=token_idx.numel() > 0,
            )
            timestep = self._sample_timestep(timesteps, generator)
            noise = torch.randn(
                latent.shape,
                generator=generator,
                device=self.device,
                dtype=latent.dtype,
            )
            z_t = self._add_noise(latent, noise, timestep)
            latent_model_input = self._scale_model_input(z_t, timestep)
            image_rotary_emb = self._prepare_rotary_embeddings(latent_frames)

            _ = self.pipe.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep.unsqueeze(0),
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]

            patch_size = getattr(self.pipe.transformer.config, "patch_size", 2)
            if isinstance(patch_size, (tuple, list)):
                patch_h, patch_w = patch_size[-2], patch_size[-1]
            else:
                patch_h = patch_w = patch_size
            latent_h = latent_model_input.shape[-2] // patch_h
            latent_w = latent_model_input.shape[-1] // patch_w

            self._append_attention_mask(
                token_idx=token_idx,
                height=latent_h,
                width=latent_w,
                head_num=self.pipe.transformer.config.num_attention_heads,
                required=record["label"] >= 0,
            )

            sample = self._pop_activation(timestep)
            record_samples.append(self._append_sample_metadata(sample, record))

            del z_t, noise, latent_model_input, image_rotary_emb
            self._clear_hook_buffers()

        if video is not None:
            del video
        self._set_hook_capture(True, True)
        del latent, prompt_embeds
        return record_samples

    def _generate_record_samples(self, record):
        self._build_pipeline()

        prefixes = self._prefixes_for_record(record["subject"]) if record["label"] >= 0 else []
        token_indices = find_token_indices(
            record["prompt"],
            prefixes,
            self.pipe.tokenizer,
            max_sequence_length=self.eraser.TEXT_LEN,
        )
        if record["label"] >= 0 and not token_indices:
            raise ValueError(
                f"No target tokens found for subject={record['subject']!r} "
                f"in prompt={record['prompt']!r}."
            )
        token_idx = torch.tensor(token_indices, dtype=torch.long, device=self.device)

        with torch.inference_mode():
            if self.activation_input == "prompt" and self.sampling_mode == "trajectory":
                record_samples = self._generate_trajectory_samples(record, token_idx)
            else:
                record_samples = self._generate_noisy_latent_samples(record, token_idx)

            self._clear_hook_buffers()
            if self.empty_cache_every_step and torch.cuda.is_available():
                torch.cuda.empty_cache()

        for sample in record_samples:
            sample["orig_act"] = sample["orig_act"].clone()
            sample["mask"] = sample["mask"].clone()
            yield sample

    def _collate(self, samples):
        orig_act = torch.stack([s["orig_act"] for s in samples], dim=0)
        if self.mean is not None and self.std is not None:
            orig_act = (orig_act - self.mean) / (self.std + 1e-8)

        return {
            "orig_act": orig_act,
            "mask": torch.stack([s["mask"] for s in samples], dim=0),
            "label": torch.tensor([s["label"] for s in samples], dtype=torch.long),
            "subject": [s["subject"] for s in samples],
            "prompt": [s.get("prompt") for s in samples],
            "seed": [s.get("seed") for s in samples],
            "case_number": [s.get("case_number") for s in samples],
            "trajectory_step": [s.get("trajectory_step") for s in samples],
            "timestep": torch.tensor([s["timestep"] for s in samples], dtype=torch.float32),
        }

    def _iter_samples(self, pad=True, progress=None):
        for idx in self._local_indices(pad=pad):
            record = self.dataset[idx]
            for sample in self._generate_record_samples(record):
                if progress is not None:
                    progress.update(1)
                yield sample
            del record
            gc.collect()

    def _iter_batches(self, pad=True, progress=None):
        batch = []
        for sample in self._iter_samples(pad=pad, progress=progress):
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []

        if batch:
            yield self._collate(batch)

    def iter_batches(self, pad=True):
        yield from self._iter_batches(pad=pad)

    def __iter__(self):
        yield from self.iter_batches(pad=True)

    def collect_global_stats(self, data_dim, desc=None):
        prev_mean, prev_std = self.mean, self.std
        self.mean, self.std = None, None

        try:
            sums = torch.zeros(data_dim, dtype=torch.float64)
            sq_sums = torch.zeros(data_dim, dtype=torch.float64)
            count = torch.zeros(1, dtype=torch.float64)

            total_samples = len(self._local_indices(pad=False)) * self.timesteps_per_video
            progress = tqdm(
                total=total_samples,
                desc=f"Online stats {desc or self.layer_name}",
                unit="sample",
                dynamic_ncols=True,
                disable=(self.rank != 0),
            )

            for sample in self._iter_samples(pad=False, progress=progress):
                update_channel_moments(
                    sample["orig_act"],
                    data_dim,
                    sums,
                    sq_sums,
                    count,
                )
                del sample
            progress.close()

            if dist.is_available() and dist.is_initialized():
                sums_d = sums.to(self.device)
                sq_sums_d = sq_sums.to(self.device)
                count_d = count.to(self.device)
                dist.all_reduce(sums_d, op=dist.ReduceOp.SUM)
                dist.all_reduce(sq_sums_d, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_d, op=dist.ReduceOp.SUM)
                sums = sums_d.cpu()
                sq_sums = sq_sums_d.cpu()
                count = count_d.cpu()

            if count.item() < 2:
                raise ValueError(f"Not enough online activations to compute stats for {desc or self.layer_name}.")

            mean = sums / count
            variance = (sq_sums - (sums * sums / count)).clamp_min(0.0) / (count - 1)
            std = torch.sqrt(variance)
            dead_mask = std < 1e-6
            std[dead_mask] = 1.0

            return mean.to(torch.float32), std.to(torch.float32), dead_mask
        finally:
            self.mean, self.std = prev_mean, prev_std


def create_online_cog_loader(args, layer_name, task, device, rank, world_size):
    return OnlineCogActivationLoader(
        model_path=args.base_model_path,
        raw_data_root=args.raw_data_root,
        task=task,
        layer_name=layer_name,
        target_concepts=args.target_celebs_list,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        device=device,
        dtype=args.dtype,
        num_frames=args.activation_num_frames,
        height=args.activation_height,
        width=args.activation_width,
        activation_input=getattr(args, "activation_input", "video"),
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        timesteps_per_video=args.activation_timesteps_per_video,
        timestep_min_index=args.activation_timestep_min_index,
        sampling_mode=args.activation_sampling_mode,
        trajectory_steps=args.activation_trajectory_steps,
        seed=args.global_seed,
        shuffle=True,
        max_samples_per_subject=args.max_samples_per_subject,
        empty_cache_every_step=args.empty_cache_every_step,
        attention_mask_quantile=args.attention_mask_quantile,
        attention_mask_dilation=args.attention_mask_dilation,
    )
