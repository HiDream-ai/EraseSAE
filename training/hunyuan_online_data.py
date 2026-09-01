import gc
import csv
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from training.streaming_stats import update_channel_moments


CELEBRITY_PREFIXES = {
    "swift": ["Taylor Swift", "Swift"],
    "trump": ["Trump", "Donald Trump"],
    "musk": ["Elon Musk", "Musk"],
    "obama": ["Obama", "Barack Obama"],
    "merkel": ["Angela Merkel", "Merkel"],
    "elizabeth": ["Queen Elizabeth II", "Elizabeth"],
}

NON_CELEBRITY_PREFIXES = [
    "A young man",
    "A middle-aged woman",
    "A professional photographer",
    "A university student",
    "A casual tourist",
    "A busy office worker",
    "A retired teacher",
    "A fitness instructor",
    "A street artist",
]

NUDITY_PREFIXES = ["naked", "nude", "shirtless", "topless"]

NO_NUDITY_PREFIXES = [
    "fully clothed",
    "dressed",
    "clothed",
    "wearing a top",
    "wearing-a-top",
]

NUDITY_STATE_PATTERN = re.compile(
    r"\b(naked|nude|shirtless|topless)\b",
    re.IGNORECASE,
)
NO_NUDITY_STATE_PATTERN = re.compile(
    r"\b(fully\s+clothed|dressed|clothed|wearing(?:-|\s+)a(?:-|\s+)top)\b",
    re.IGNORECASE,
)

PROMPT_FILES = {
    "nudity": [
        ("nudity", "nudity.csv"),
        ("no_nudity", "no_nudity.csv"),
    ],
    "celebrity": [
        ("elizabeth", "celebrity/elizabeth.csv"),
        ("merkel", "celebrity/merkel.csv"),
        ("musk", "celebrity/musk.csv"),
        ("obama", "celebrity/obama.csv"),
        ("swift", "celebrity/swift.csv"),
        ("trump", "celebrity/trump.csv"),
        ("non_celebrity", "non_celebrity.csv"),
    ],
}


def normalize_target_concepts(target_concepts):
    concepts = [
        str(concept).strip()
        for concept in (target_concepts or [])
        if str(concept).strip()
    ]
    if len(concepts) != len(set(concepts)):
        raise ValueError("Target concepts must be unique and keep a stable order.")
    return concepts


def validate_training_records(records, concepts, task, source):
    if task == "nudity" and concepts != ["nudity", "no_nudity"]:
        raise ValueError(
            "Nudity training requires the ordered concepts "
            "['nudity', 'no_nudity']; both sides are used for partition "
            f"competition. Got {concepts} from {source}."
        )
    missing = [
        concept
        for concept_index, concept in enumerate(concepts)
        if not any(record["label"] == concept_index for record in records)
    ]
    if missing:
        raise ValueError(
            f"No training records found for configured concepts in {source}: "
            + ", ".join(missing)
        )
    if task == "celebrity" and not any(record["label"] < 0 for record in records):
        raise ValueError(
            f"Celebrity training under {source} requires ordinary-person or "
            "non-target prompts with label=-1."
        )
    if task == "nudity" and "nudity" in concepts:
        nudity_label = concepts.index("nudity")
        minor_pattern = re.compile(
            r"\b(boy|girl|child|minor|underage|teen(?:ager)?|schoolboy|schoolgirl)\b",
            re.IGNORECASE,
        )
        unsafe_records = [
            record
            for record in records
            if record["label"] == nudity_label
            and minor_pattern.search(str(record.get("prompt", "")))
        ]
        if unsafe_records:
            raise ValueError(
                f"Nudity training under {source} contains minor-coded prompts; "
                "use explicitly adult subjects only."
            )


def _normalized_nudity_context(prompt, pattern):
    normalized, substitutions = pattern.subn("<state>", str(prompt), count=1)
    if substitutions != 1:
        return None
    return " ".join(normalized.lower().split())


def validate_nudity_prompt_pairs(records, concepts, source):
    """Require row-aligned counterfactual prompt pairs for nudity training."""
    if concepts != ["nudity", "no_nudity"]:
        raise ValueError(
            "Paired nudity prompts require concepts ['nudity', 'no_nudity']."
        )

    by_label = {0: {}, 1: {}}
    for record in records:
        label = int(record["label"])
        if label not in by_label:
            continue
        case_number = int(record.get("case_number", -1))
        if case_number in by_label[label]:
            raise ValueError(
                f"Duplicate nudity pair case_number={case_number} in {source}."
            )
        by_label[label][case_number] = record

    if set(by_label[0]) != set(by_label[1]):
        missing_safe = sorted(set(by_label[0]).difference(by_label[1]))
        missing_nudity = sorted(set(by_label[1]).difference(by_label[0]))
        raise ValueError(
            "Nudity/no-nudity prompt rows are not aligned in "
            f"{source}; missing_safe={missing_safe[:5]}, "
            f"missing_nudity={missing_nudity[:5]}."
        )

    for case_number in sorted(by_label[0]):
        unsafe = by_label[0][case_number]
        safe = by_label[1][case_number]
        if int(unsafe.get("seed", 0)) != int(safe.get("seed", 0)):
            raise ValueError(
                f"Nudity pair {case_number} has different seeds in {source}."
            )
        unsafe_context = _normalized_nudity_context(
            unsafe.get("prompt", ""),
            NUDITY_STATE_PATTERN,
        )
        safe_context = _normalized_nudity_context(
            safe.get("prompt", ""),
            NO_NUDITY_STATE_PATTERN,
        )
        if unsafe_context is None or safe_context is None:
            raise ValueError(
                f"Nudity pair {case_number} is missing an explicit state phrase "
                f"in {source}."
            )
        if unsafe_context != safe_context:
            raise ValueError(
                f"Nudity pair {case_number} changes more than clothing state "
                f"in {source}."
            )


def pad_distributed_indices(indices, total_size):
    if len(indices) >= total_size:
        return indices
    if not indices:
        return indices
    padding_size = total_size - len(indices)
    repeats = math.ceil(padding_size / len(indices))
    return indices + (indices * repeats)[:padding_size]


class EpochSampler:
    """Small sampler shim so the existing DDP training loops can call set_epoch."""

    def __init__(self):
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_word_indices(text, target_word, tokenizer):
    encoding = tokenizer(text)
    input_ids = encoding.input_ids
    target_ids_with_space = tokenizer(" " + target_word, add_special_tokens=False).input_ids
    target_ids_no_space = tokenizer(target_word, add_special_tokens=False).input_ids

    indices = []
    for target_ids in [target_ids_with_space, target_ids_no_space]:
        if not target_ids:
            continue
        span = len(target_ids)
        for i in range(len(input_ids) - span + 1):
            if input_ids[i: i + span] == target_ids:
                indices.extend(range(i, i + span))
        if indices:
            break
    return sorted(set(indices))


def find_token_indices(
    prompt,
    prefixes,
    tokenizer,
    prompt_template=None,
    max_sequence_length=256,
):
    search_text = prompt
    crop_start = 0
    prompt_char_start = 0
    tokenize_kwargs = {}
    if prompt_template is not None:
        search_text = prompt_template["template"].format(prompt)
        crop_start = int(prompt_template.get("crop_start", 0))
        template_prefix = prompt_template["template"].split("{}", 1)[0]
        prompt_char_start = len(template_prefix)
        tokenize_kwargs = {
            "max_length": max_sequence_length + crop_start,
            "padding": "max_length",
            "truncation": True,
        }

    char_spans = []
    prompt_lower = prompt.lower()
    for prefix in prefixes:
        prefix_lower = prefix.lower()
        search_from = 0
        while prefix_lower:
            match_start = prompt_lower.find(prefix_lower, search_from)
            if match_start < 0:
                break
            char_spans.append(
                (
                    prompt_char_start + match_start,
                    prompt_char_start + match_start + len(prefix),
                )
            )
            search_from = match_start + len(prefix_lower)

    if char_spans:
        try:
            encoded_with_offsets = tokenizer(
                search_text,
                return_offsets_mapping=True,
                **tokenize_kwargs,
            )
            offsets = encoded_with_offsets.offset_mapping
            if offsets and isinstance(offsets[0], list):
                offsets = offsets[0]
            offsets = offsets[crop_start: crop_start + max_sequence_length]
            offset_indices = []
            for token_index, (token_start, token_end) in enumerate(offsets):
                if token_end <= token_start:
                    continue
                if any(
                    token_start < span_end and token_end > span_start
                    for span_start, span_end in char_spans
                ):
                    offset_indices.append(token_index)
            if offset_indices:
                return sorted(set(offset_indices))
        except (AttributeError, KeyError, NotImplementedError, TypeError):
            pass

    input_ids = tokenizer(search_text, **tokenize_kwargs).input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    input_ids = input_ids[crop_start: crop_start + max_sequence_length]

    indices = []
    for prefix in prefixes:
        candidates = (
            tokenizer(" " + prefix, add_special_tokens=False).input_ids,
            tokenizer(prefix, add_special_tokens=False).input_ids,
        )
        for target_ids in candidates:
            if not target_ids:
                continue
            span = len(target_ids)
            matches = []
            for index in range(len(input_ids) - span + 1):
                if input_ids[index: index + span] == target_ids:
                    matches.extend(range(index, index + span))
            if matches:
                indices.extend(matches)
                break
    return sorted(set(indices))


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


class HunyuanVideoMetadataDataset:
    """
    Reads the same metadata/video layout used by the old Hunyuan collect_data
    scripts, but returns samples for online activation extraction instead of
    writing activations to disk.
    """

    def __init__(
        self,
        root_dir,
        task,
        target_concepts=None,
        num_frames=32,
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
        if target_concepts:
            self.concepts = target_concepts
        else:
            self.concepts = self._discover_subjects(primary_group)

        self.concept_to_id = {name: i for i, name in enumerate(self.concepts)}
        self.records = []
        self._load_records()

        if not self.records:
            raise ValueError(
                f"No metadata records found under {self.root_dir}. "
                "Expected group/subject/*.jsonl with a video_path field."
            )
        validate_training_records(
            self.records,
            self.concepts,
            self.task,
            self.root_dir,
        )

    def _group_dir(self, group_name):
        candidate = self.root_dir / group_name
        if candidate.exists():
            return candidate
        if self.root_dir.name == group_name:
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
        raise ValueError(f"Unsupported Hunyuan online task: {self.task}")

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
                        self.records.append({
                            "prompt": prompt,
                            "video_path": str(_resolve_video_path(video_path, jsonl_path.parent)),
                            "subject": subject,
                            "label": label,
                            "case_number": item.get("case_number", loaded),
                            "seed": item.get("seed", 0),
                        })
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


class PromptMetadataDataset:
    """
    Reads public prompt CSV/JSONL files for prompt-only online activation
    extraction. No original videos are required in this mode.
    """

    def __init__(
        self,
        root_dir,
        task,
        target_concepts=None,
        max_samples_per_subject=None,
    ):
        self.root_dir = Path(root_dir)
        self.task = task
        self.max_samples_per_subject = max_samples_per_subject

        target_concepts = normalize_target_concepts(target_concepts)
        if target_concepts:
            self.concepts = target_concepts
        elif task == "nudity":
            self.concepts = ["nudity", "no_nudity"]
        elif task == "celebrity":
            self.concepts = [subject for subject, _ in PROMPT_FILES["celebrity"] if subject != "non_celebrity"]
        else:
            raise ValueError(f"Unsupported prompt task: {task}")

        self.concept_to_id = {name: i for i, name in enumerate(self.concepts)}
        self.records = []
        self._load_records()

        if not self.records:
            raise ValueError(
                f"No prompt records found under {self.root_dir}. "
                "Expected the public data/train CSV layout."
            )
        validate_training_records(
            self.records,
            self.concepts,
            self.task,
            self.root_dir,
        )
        if self.task == "nudity":
            validate_nudity_prompt_pairs(
                self.records,
                self.concepts,
                self.root_dir,
            )

    def _prompt_files(self):
        if self.task not in PROMPT_FILES:
            raise ValueError(f"Unsupported prompt task: {self.task}")
        return PROMPT_FILES[self.task]

    def _read_csv_rows(self, path, subject):
        loaded = 0
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.max_samples_per_subject and loaded >= self.max_samples_per_subject:
                    break
                prompt = row.get("Prompt") or row.get("prompt") or row.get("original_prompt")
                if not prompt:
                    continue
                seed = row.get("Seed") or row.get("seed") or 0
                self._append_record(subject, prompt, seed, loaded)
                loaded += 1

    def _read_jsonl_rows(self, path, subject):
        loaded = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if self.max_samples_per_subject and loaded >= self.max_samples_per_subject:
                    break
                item = json.loads(line)
                prompt = item.get("prompt") or item.get("Prompt") or item.get("original_prompt")
                if not prompt:
                    continue
                self._append_record(subject, prompt, item.get("seed", item.get("Seed", 0)), loaded)
                loaded += 1

    def _append_record(self, subject, prompt, seed, case_number):
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = 0

        self.records.append(
            {
                "prompt": prompt,
                "subject": subject,
                "label": self.concept_to_id.get(subject, -1),
                "case_number": case_number,
                "seed": seed,
            }
        )

    def _load_records(self):
        for subject, rel_path in self._prompt_files():
            path = self.root_dir / rel_path
            if not path.exists():
                continue
            if path.suffix.lower() == ".jsonl":
                self._read_jsonl_rows(path, subject)
            else:
                self._read_csv_rows(path, subject)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return dict(self.records[index])


class OnlineHunyuanActivationLoader:
    """
    Iterable dataloader that creates Hunyuan activations on demand.

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
        num_frames=32,
        height=480,
        width=720,
        activation_input="video",
        guidance_scale=6.0,
        inference_steps=30,
        timesteps_per_video=1,
        timestep_min_index=500,
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
        self.guidance_scale = guidance_scale
        self.inference_steps = inference_steps
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
            self.dataset = HunyuanVideoMetadataDataset(
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

        from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
        from training.activation_hooks import create_hunyuan_activation_collector

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            self.model_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
        ).to(self.device)
        transformer.requires_grad_(False)
        transformer.eval()

        self.eraser = create_hunyuan_activation_collector(
            transformer,
            [self.layer_name],
        )
        self.eraser.register_hooks()

        self.pipe = HunyuanVideoPipeline.from_pretrained(
            self.model_path,
            transformer=transformer,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.pipe.transformer.eval()

        for enable in (
            "enable_attention_slicing",
            "enable_xformers_memory_efficient_attention",
        ):
            try:
                getattr(self.pipe, enable)()
            except Exception:
                pass
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
            target_steps = max(self.inference_steps or 0, self.timestep_min_index + 1)
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
                return NO_NUDITY_PREFIXES
            return NUDITY_PREFIXES

        if subject == "non_celebrity":
            return NON_CELEBRITY_PREFIXES
        return CELEBRITY_PREFIXES.get(subject, [subject.replace("_", " ")])

    def _pop_activation(self, timestep):
        acts_list = self.eraser.current_batch_activations.get(self.layer_name, [])
        if len(acts_list) < 2:
            raise RuntimeError(
                f"Layer {self.layer_name} did not produce both activation and attention mask."
            )

        orig_act = acts_list[0].detach().cpu().to(torch.float32)
        mask = acts_list[1].detach().cpu().to(torch.float32)
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
        if required and not torch.count_nonzero(acts_list[-1]).item():
            raise RuntimeError(
                f"Attention mask is empty for target layer {self.layer_name}."
            )

    def _make_zero_latent(self):
        channels = self.pipe.transformer.config.in_channels
        latent_t = (self.num_frames - 1) // self.pipe.vae_scale_factor_temporal + 1
        latent_h = self.height // self.pipe.vae_scale_factor_spatial
        latent_w = self.width // self.pipe.vae_scale_factor_spatial
        return torch.zeros(
            (1, channels, latent_t, latent_h, latent_w),
            dtype=torch.float32,
            device=self.device,
        )

    def _record_generator(self, record):
        base_seed = int(record.get("seed", 0) or 0)
        if base_seed == 0:
            base_seed = self.seed + int(record.get("case_number", 0))
        epoch_seed = base_seed + self.sampler.epoch * 1_000_003
        return torch.Generator(device=self.device).manual_seed(epoch_seed)

    def _capture_step_indices(self):
        count = min(self.timesteps_per_video, self.trajectory_steps)
        # Spread captures over the trajectory. Taking only the last steps of a
        # short trajectory systematically misses early concept formation.
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
        generator = self._record_generator(record)
        sigmas = np.linspace(1.0, 0.0, self.trajectory_steps + 1)[:-1]
        self.pipe.scheduler.set_timesteps(
            sigmas=sigmas.tolist(),
            device=self.device,
        )
        timesteps = self.pipe.scheduler.timesteps
        capture_steps = self._capture_step_indices()

        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.pipe.encode_prompt(
            prompt=[record["prompt"]],
            device=self.device,
            num_videos_per_prompt=1,
        )
        prompt_embeds = prompt_embeds.to(self.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.dtype)
        prompt_attention_mask = prompt_attention_mask.to(self.dtype)

        channels = self.pipe.transformer.config.in_channels
        latents = self.pipe.prepare_latents(
            1,
            channels,
            self.height,
            self.width,
            self.num_frames,
            torch.float32,
            self.device,
            generator,
        )
        guidance = torch.tensor(
            [self.guidance_scale],
            dtype=self.dtype,
            device=self.device,
        ) * 1000.0

        record_samples = []
        for step_index, timestep in enumerate(timesteps):
            should_capture = step_index in capture_steps
            self._set_hook_capture(
                capture_activations=should_capture,
                capture_attention=should_capture and token_idx.numel() > 0,
            )
            latent_model_input = latents.to(self.dtype)
            timestep_batch = timestep.expand(latents.shape[0]).to(latents.dtype)
            noise_pred = self.pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_batch,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                pooled_projections=pooled_prompt_embeds,
                guidance=guidance,
                return_dict=False,
            )[0]

            if should_capture:
                _, _, _, latent_h, latent_w = latent_model_input.shape
                patch = self.pipe.transformer.config.patch_size
                self._append_attention_mask(
                    token_idx=token_idx,
                    height=latent_h // patch,
                    width=latent_w // patch,
                    head_num=self.pipe.transformer.config.num_attention_heads,
                    required=record["label"] >= 0,
                )
                sample = self._pop_activation(timestep)
                record_samples.append(
                    self._append_sample_metadata(sample, record, step_index)
                )
            else:
                self._clear_hook_buffers()

            latents = self.pipe.scheduler.step(
                noise_pred,
                timestep,
                latents,
                return_dict=False,
            )[0]
            del latent_model_input, timestep_batch, noise_pred

        self._set_hook_capture(True, True)
        del latents, guidance
        del prompt_embeds, pooled_prompt_embeds, prompt_attention_mask
        return record_samples

    def _generate_noisy_latent_samples(self, record, token_idx):
        record_samples = []
        if "original_video" in record:
            video = record["original_video"].unsqueeze(0).to(self.device)
            encoded = self.pipe.vae.encode(video.to(self.dtype))
            latent = encoded.latent_dist.sample() * self.pipe.vae.config.scaling_factor
            latent = latent.detach().to(torch.float32)
        else:
            video = None
            latent = self._make_zero_latent()

        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.pipe.encode_prompt(
            prompt=[record["prompt"]],
            device=self.device,
            num_videos_per_prompt=1,
        )
        prompt_embeds = prompt_embeds.to(self.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.dtype)
        prompt_attention_mask = prompt_attention_mask.to(self.dtype)

        timesteps = self.pipe.scheduler.timesteps.to(self.device)
        if self.timestep_min_index >= timesteps.numel():
            raise ValueError(
                f"timestep_min_index={self.timestep_min_index} is outside "
                f"scheduler timesteps length {timesteps.numel()}."
            )

        generator = self._record_generator(record)
        for _ in range(self.timesteps_per_video):
            self._set_hook_capture(
                capture_activations=True,
                capture_attention=token_idx.numel() > 0,
            )
            t_idx = torch.randint(
                self.timestep_min_index,
                timesteps.numel(),
                (1,),
                device=self.device,
                generator=generator,
            )
            timestep = timesteps[t_idx][0].to(timesteps.dtype)
            noise = torch.randn(
                latent.shape,
                generator=generator,
                device=self.device,
                dtype=latent.dtype,
            )
            z_t = self.pipe.scheduler.scale_noise(
                sample=latent,
                noise=noise,
                timestep=timestep.unsqueeze(0),
            ).to(self.dtype)
            guidance = torch.tensor(
                [self.guidance_scale],
                dtype=self.dtype,
                device=self.device,
            ) * 1000.0

            _ = self.pipe.transformer(
                hidden_states=z_t,
                timestep=timestep.unsqueeze(0),
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                pooled_projections=pooled_prompt_embeds,
                guidance=guidance,
                return_dict=False,
            )[0]

            _, _, _, latent_h, latent_w = z_t.shape
            patch = self.pipe.transformer.config.patch_size
            self._append_attention_mask(
                token_idx=token_idx,
                height=latent_h // patch,
                width=latent_w // patch,
                head_num=self.pipe.transformer.config.num_attention_heads,
                required=record["label"] >= 0,
            )

            sample = self._pop_activation(timestep)
            record_samples.append(self._append_sample_metadata(sample, record))

            del z_t, noise, guidance
            self._clear_hook_buffers()

        if video is not None:
            del video
        self._set_hook_capture(True, True)
        del latent, prompt_embeds, pooled_prompt_embeds, prompt_attention_mask
        return record_samples

    def _generate_record_samples(self, record):
        self._build_pipeline()

        prefixes = self._prefixes_for_record(record["subject"]) if record["label"] >= 0 else []
        from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import (
            DEFAULT_PROMPT_TEMPLATE,
        )

        token_indices = find_token_indices(
            record["prompt"],
            prefixes,
            self.pipe.tokenizer,
            prompt_template=DEFAULT_PROMPT_TEMPLATE,
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


def create_online_hunyuan_loader(args, layer_name, task, device, rank, world_size):
    return OnlineHunyuanActivationLoader(
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
        guidance_scale=args.guidance_scale,
        inference_steps=args.inference_steps,
        timesteps_per_video=args.activation_timesteps_per_video,
        timestep_min_index=args.activation_timestep_min_index,
        sampling_mode=args.activation_sampling_mode,
        trajectory_steps=args.activation_trajectory_steps,
        seed=args.global_seed,
        shuffle=True,
        max_samples_per_subject=args.max_samples_per_subject,
        empty_cache_every_step=args.empty_cache_every_step,
        attention_mask_quantile=getattr(args, "attention_mask_quantile", 0.8),
        attention_mask_dilation=getattr(args, "attention_mask_dilation", 0),
    )
