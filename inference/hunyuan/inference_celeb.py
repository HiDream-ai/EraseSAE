import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from diffusers.utils import export_to_video


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs.config_loader import get_model_path, load_experiment_config
from inference.cli_utils import (
    load_prompt_records,
    ordered_concepts_from_map,
    output_video_path,
    prompt_mentions_identity,
    resolve_checkpoint,
    validated_target_kernels,
)
from inference.mask_artifacts import (
    capture_effective_mask,
    save_step_mask_artifacts,
)
from inference.hunyuan.celebrity_mask import build_dynamic_celebrity_mask
from inference.hunyuan.hunyuan_pipline import HunyuanVideoPipeline
from model.celebrity_partitioned_sae import TopKConvPSAE
from model.checkpoint_io import load_checkpoint_model_config


DEVICE = "cuda:0"
DEFAULT_MODEL_PATH = get_model_path("hunyuan")
INFERENCE_DEFAULTS = load_experiment_config("hunyuan_celebrity").get(
    "inference",
    {},
)
DEFAULT_CHECKPOINT_DIR = "checkpoints/hunyuan/celebrity"
LAYER_NAME = "single_transformer_blocks.14.proj_out"
IDENTITIES = ["trump", "obama", "musk", "merkel", "elizabeth", "swift"]
IDENTITY_ALIASES = {
    "trump": ("donald trump", "trump"),
    "obama": ("barack obama", "obama"),
    "musk": ("elon musk", "musk"),
    "merkel": ("angela merkel", "merkel"),
    "elizabeth": ("queen elizabeth ii", "queen elizabeth", "elizabeth ii"),
    "swift": ("taylor swift",),
}
DEFAULT_REFERENCE_PROMPT = INFERENCE_DEFAULTS.get(
    "mask_cfg_reference_prompt",
    "nudity",
)
DEFAULT_PROMPT = (
    "Queen Elizabeth II is adjusting a hat on her head on the palace balcony "
    "during a parade."
)


def _training_config(checkpoint_dir):
    candidates = sorted((Path(checkpoint_dir) / "logs").glob("*_config.json"))
    if not candidates:
        return {}
    with candidates[-1].open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _spatial_grid(pipe, height, width):
    patch_size = pipe.transformer.config.patch_size
    if isinstance(patch_size, (tuple, list)):
        patch_height, patch_width = patch_size[-2:]
    else:
        patch_height = patch_width = int(patch_size)
    scale = int(pipe.vae_scale_factor_spatial)
    denominator_h = scale * int(patch_height)
    denominator_w = scale * int(patch_width)
    if height % denominator_h or width % denominator_w:
        raise ValueError(
            "Inference resolution is incompatible with the VAE and transformer "
            f"patch sizes: height={height}, width={width}."
        )
    return height // denominator_h, width // denominator_w


def _diagnostic_summary(records):
    metric_names = (
        "target_activation_mean",
        "target_activation_peak",
        "target_activation_nonzero_fraction",
        "foreground_mask_fraction",
        "identity_mask_fraction",
        "target_gate_fraction",
        "intersection_mask_fraction",
        "final_mask_fraction",
        "target_identity_score",
        "strongest_competitor_score",
    )
    summary = {
        "recorded_steps": len(records),
        "active_mask_steps": sum(record["mask_active"] for record in records),
    }
    for metric_name in metric_names:
        values = [float(record[metric_name]) for record in records]
        if values:
            summary[metric_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

    winner_counts = Counter()
    for record in records:
        winner_counts.update(record.get("identity_winners", []))
    summary["identity_winner_counts"] = dict(winner_counts)
    return summary


class CelebrityMaskEraser:
    def __init__(
        self,
        pipe,
        sae,
        target_identity,
        identity_names,
        identity_kernels,
        identity_activation_references,
        data_mean,
        data_std,
        grid_height,
        grid_width,
        args,
    ):
        self.pipe = pipe
        self.sae = sae
        self.target_identity = target_identity
        self.identity_names = identity_names
        self.identity_kernels = identity_kernels
        self.identity_activation_references = identity_activation_references
        self.data_mean = data_mean
        self.data_std = data_std
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.args = args
        self.save_step_masks = bool(getattr(args, "save_step_masks", False))
        self.prompt_target_match = False
        self.records = []
        self.mask_snapshots = []
        self.handles = []

        device = next(sae.parameters()).device
        dtype = next(sae.parameters()).dtype
        self.mean = data_mean.to(device=device, dtype=dtype).view(1, -1, 1, 1)
        self.std = data_std.to(device=device, dtype=dtype).view(1, -1, 1, 1)

    def _hook(self, module, inputs, output):
        del module, inputs
        step_index = getattr(self.pipe, "_current_step_index", None)
        if step_index is None or not (
            self.args.mask_start_step <= step_index <= self.args.mask_end_step
        ):
            return output

        # Preserve the conditional-branch mask while the reference branch runs.
        if getattr(self.pipe, "_current_sae_mask", None) is not None:
            return output

        text_length = 256
        if output.shape[1] <= text_length:
            return output
        batch_size, _, hidden_size = output.shape
        visual = output[:, :-text_length, :]
        visual_length = visual.shape[1]
        spatial_size = self.grid_height * self.grid_width
        if visual_length % spatial_size:
            raise RuntimeError(
                f"Cannot reshape {visual_length} visual tokens into a "
                f"{self.grid_height}x{self.grid_width} latent grid."
            )
        frames = visual_length // spatial_size
        x_folded = (
            visual.transpose(1, 2)
            .reshape(
                batch_size,
                hidden_size,
                frames,
                self.grid_height,
                self.grid_width,
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(
                batch_size * frames,
                hidden_size,
                self.grid_height,
                self.grid_width,
            )
        )

        with torch.no_grad():
            normalized = (
                x_folded - self.mean.to(x_folded.dtype)
            ) / (self.std.to(x_folded.dtype) + 1e-8)
            sae_output = self.sae(
                normalized,
                route_all=True,
                return_global_topk=False,
            )
            mask, diagnostics = build_dynamic_celebrity_mask(
                f_gen=sae_output["f_gen"],
                f_id=sae_output["f_id"],
                identity_names=self.identity_names,
                identity_kernels=self.identity_kernels,
                identity_activation_references=self.identity_activation_references,
                target_identity=self.target_identity,
                batch_size=batch_size,
                frames=frames,
                background_threshold=self.args.mask_background_threshold,
                identity_threshold=self.args.mask_identity_threshold,
                competition_ratio=self.args.identity_competition_ratio,
                min_relative_score=self.args.identity_min_relative_score,
                score_top_fraction=self.args.identity_score_top_fraction,
                dilation=self.args.mask_dilation,
                identity_gate_enabled=not self.args.disable_identity_gate,
            )

            prompt_gate_passed = (
                not self.args.enable_prompt_gate or self.prompt_target_match
            )
            if not prompt_gate_passed:
                mask = torch.zeros_like(mask)
            mask_active = bool(mask.any().item())
            self.pipe._current_sae_mask = mask.detach() if mask_active else None
            self.records.append(
                {
                    "step_index": int(step_index),
                    "mask_active": mask_active,
                    "prompt_gate_passed": prompt_gate_passed,
                    **diagnostics,
                }
            )
        return output

    def register(self):
        self.clear()
        for name, module in self.pipe.transformer.named_modules():
            if name == LAYER_NAME:
                self.handles.append(module.register_forward_hook(self._hook))
                break
        if not self.handles:
            raise RuntimeError(f"Target layer not found: {LAYER_NAME}")
        self.pipe._capture_effective_sae_mask = self.save_step_masks
        print(f"SAE mask hook registered on: {LAYER_NAME}")

    def clear(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.pipe._current_sae_mask = None
        self.pipe._current_effective_sae_mask = None
        self.pipe._capture_effective_sae_mask = False

    def prepare_prompt(self, prompt):
        self.records.clear()
        self.mask_snapshots.clear()
        self.pipe._current_sae_mask = None
        self.pipe._current_effective_sae_mask = None
        self.pipe._capture_effective_sae_mask = self.save_step_masks
        self.prompt_target_match = prompt_mentions_identity(
            prompt,
            self.target_identity,
            IDENTITY_ALIASES,
        )

    def clear_step_mask(self, pipe, step_index, timestep, callback_kwargs):
        del timestep
        if (
            self.save_step_masks
            and self.args.mask_start_step
            <= step_index
            <= self.args.mask_end_step
        ):
            self.mask_snapshots.append(capture_effective_mask(pipe, step_index))
        pipe._current_sae_mask = None
        pipe._current_effective_sae_mask = None
        return callback_kwargs


def _generation_args(prompt, seed, args):
    return {
        "prompt": prompt,
        "num_inference_steps": args.steps,
        "num_frames": args.frames,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "generator": torch.Generator(device=DEVICE).manual_seed(seed),
    }


def generate_original(pipe, prompt, seed, args):
    pipe._current_sae_mask = None
    pipe._current_effective_sae_mask = None
    call_args = _generation_args(prompt, seed, args)
    call_args["mask_cfg_scale"] = 0.0
    return pipe(**call_args).frames[0]


def generate_erased(pipe, eraser, prompt, reference_prompt, seed, args):
    eraser.prepare_prompt(prompt)
    call_args = _generation_args(prompt, seed, args)
    call_args.update(
        {
            "negative_prompt": reference_prompt,
            "true_cfg_scale": 1.0,
            "mask_cfg_scale": args.mask_cfg_scale,
            "mask_cfg_start_step": args.mask_start_step,
            "mask_cfg_end_step": args.mask_end_step,
            "callback_on_step_end": eraser.clear_step_mask,
        }
    )
    return pipe(**call_args).frames[0]


def _reference_prompt(args):
    if args.reference_prompt is not None:
        return args.reference_prompt
    return DEFAULT_REFERENCE_PROMPT


def _build_validation_prompt_plan(
    prompt_records,
    target_identities,
    validation_mode,
):
    indexed_records = list(enumerate(prompt_records))
    if validation_mode == "cross":
        return indexed_records, {
            target_identity: indexed_records
            for target_identity in target_identities
        }

    missing_identity = [
        index for index, record in indexed_records if record.identity is None
    ]
    if missing_identity:
        rows = ", ".join(str(index + 1) for index in missing_identity[:5])
        raise ValueError(
            "Self validation requires an Identity column for every prompt; "
            f"missing at loaded prompt rows: {rows}."
        )

    unknown_identities = sorted(
        {
            record.identity
            for _, record in indexed_records
            if record.identity not in IDENTITIES
        }
    )
    if unknown_identities:
        raise ValueError(
            "Unsupported prompt identities in self validation: "
            + ", ".join(unknown_identities)
        )

    records_by_target = {
        target_identity: [
            (index, record)
            for index, record in indexed_records
            if record.identity == target_identity
        ]
        for target_identity in target_identities
    }
    empty_targets = [
        target_identity
        for target_identity, records in records_by_target.items()
        if not records
    ]
    if empty_targets:
        raise ValueError(
            "Self validation has no loaded prompts for: "
            + ", ".join(empty_targets)
            + ". Increase --max-prompts or update the prompt CSV."
        )

    selected_indices = {
        index
        for records in records_by_target.values()
        for index, _ in records
    }
    original_records = [
        item for item in indexed_records if item[0] in selected_indices
    ]
    return original_records, records_by_target


def load_sae_assets(args):
    checkpoint = resolve_checkpoint(
        args.checkpoint_dir,
        LAYER_NAME,
        checkpoint=args.checkpoint,
    )
    print(f"Using SAE checkpoint: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    asset_root = checkpoint.parent
    training_config = _training_config(asset_root)
    model_config = load_checkpoint_model_config(checkpoint, LAYER_NAME)

    layer_file_name = LAYER_NAME.replace(".", "_")
    map_path = asset_root / f"kernel_identity_map_{layer_file_name}.pt"
    if not map_path.is_file():
        raise FileNotFoundError(f"Attribution map not found: {map_path}")
    identity_map = torch.load(map_path, map_location="cpu", weights_only=False)
    identity_names = ordered_concepts_from_map(identity_map)
    unsupported = sorted(set(identity_names) - set(IDENTITIES))
    if unsupported:
        raise RuntimeError(
            "The checkpoint contains identities unsupported by this CLI: "
            + ", ".join(unsupported)
        )

    d_model = state_dict["decoder_gen.weight"].shape[0]
    n_gen = state_dict["decoder_gen.weight"].shape[1]
    total_id = state_dict["decoder_id.weight"].shape[1]
    if total_id % len(identity_names):
        raise RuntimeError(
            f"ID dictionary size {total_id} is incompatible with "
            f"{len(identity_names)} identities."
        )
    n_id_per_identity = total_id // len(identity_names)
    k_gen = int(model_config.get("k_gen", training_config.get("k_gen", args.k_gen)))
    k_id = int(model_config.get("k_id", training_config.get("k_id", args.k_id)))
    configured_identities = model_config.get("concepts")
    if configured_identities is not None and configured_identities != identity_names:
        raise RuntimeError(
            "Checkpoint model_config concepts do not match the attribution map."
        )
    sae = TopKConvPSAE(
        d_model=d_model,
        celebs=identity_names,
        n_gen=n_gen,
        n_id_per_celeb=n_id_per_identity,
        k_gen=k_gen,
        k_id=k_id,
    ).to(DEVICE, dtype=torch.bfloat16)
    sae.load_state_dict(state_dict)
    sae.eval()

    stats_path = asset_root / "online_stats" / f"stats_{layer_file_name}.pt"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Normalization statistics not found: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)

    identity_kernels = {}
    identity_activation_references = {}
    for identity_index, identity_name in enumerate(identity_names):
        start, end = sae.get_celeb_fields(identity_index)
        kernels = validated_target_kernels(
            identity_map,
            identity_name,
            start,
            end,
        )
        info = identity_map[identity_name]
        activation_by_kernel = dict(
            zip(info["top_kernels"], info["avg_activation"])
        )
        identity_kernels[identity_name] = kernels
        identity_activation_references[identity_name] = torch.tensor(
            [activation_by_kernel[kernel] for kernel in kernels],
            dtype=torch.float32,
        )
        purity = float(info.get("purity", 0.0))
        print(
            f"Attribution {identity_name:9s}: kernels={len(kernels):2d}, "
            f"purity={purity:6.2f}%"
        )
        if purity < args.warn_below_purity:
            print(
                f"[WARNING] {identity_name} purity is below "
                f"{args.warn_below_purity:.2f}%."
            )

    return (
        sae,
        identity_names,
        identity_kernels,
        identity_activation_references,
        stats["mean"],
        stats["std"],
    )


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("HunyuanVideo inference requires a CUDA device.")

    target_identities = list(
        dict.fromkeys(args.target_identities or [args.target_identity])
    )
    print("--- HunyuanVideo SAE celebrity erasure ---")
    print(f"Base model: {args.base_model_path}")
    print(f"Targets: {target_identities}")
    print(
        "Masked CFG toward reference: "
        f"scale={args.mask_cfg_scale}, "
        f"steps=[{args.mask_start_step}, {args.mask_end_step}]"
    )
    print(
        "Gates: identity="
        f"{'disabled' if args.disable_identity_gate else 'enabled'}, "
        f"prompt={'enabled' if args.enable_prompt_gate else 'disabled'}"
    )
    print("Global true CFG: disabled (true_cfg_scale=1.0)")
    if args.save_step_masks:
        print("Applied-mask PNG capture: enabled.")

    pipe = HunyuanVideoPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
    )
    if args.sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(device=DEVICE)
        print("Sequential CPU offload enabled.")
    elif args.cpu_offload:
        pipe.enable_model_cpu_offload(device=DEVICE)
        print("Model CPU offload enabled.")
    else:
        pipe.to(DEVICE)

    (
        sae,
        identity_names,
        identity_kernels,
        identity_activation_references,
        data_mean,
        data_std,
    ) = load_sae_assets(args)
    missing_targets = sorted(set(target_identities) - set(identity_names))
    if missing_targets:
        raise RuntimeError(
            "Requested identities are absent from the checkpoint: "
            + ", ".join(missing_targets)
        )
    grid_height, grid_width = _spatial_grid(pipe, args.height, args.width)
    prompt_records = load_prompt_records(
        prompt=args.prompt,
        prompts_file=args.prompts_file,
        default_prompt=DEFAULT_PROMPT,
        max_prompts=args.max_prompts,
    )
    original_records, records_by_target = _build_validation_prompt_plan(
        prompt_records,
        target_identities,
        args.validation_mode,
    )
    print(f"Validation mode: {args.validation_mode}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    multiple_targets = len(target_identities) > 1
    generate_originals = not args.skip_original and (
        args.prompts_file is None or args.generate_originals
    )
    if generate_originals:
        original_dir = output_root / "originals" if multiple_targets else output_root
        original_dir.mkdir(parents=True, exist_ok=True)
        for progress, (index, record) in enumerate(original_records, start=1):
            seed = record.seed if record.seed is not None else args.seed + index * args.seed_step
            print(
                f"\n[Original {progress}/{len(original_records)}] "
                f"seed={seed} prompt={record.prompt}"
            )
            video = generate_original(pipe, record.prompt, seed, args)
            path = output_video_path(
                str(original_dir),
                index,
                record.prompt,
                "reference" if multiple_targets else target_identities[0],
                seed,
                variant="original",
            )
            export_to_video(video, str(path), fps=args.fps)
            print(f"Original video saved to: {path}")
            del video
            torch.cuda.empty_cache()

    for target_identity in target_identities:
        target_dir = (
            output_root / f"target_{target_identity}"
            if multiple_targets
            else output_root
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        reference_prompt = _reference_prompt(args)
        print(
            f"\n=== Target: {target_identity}; "
            f"CFG reference: {reference_prompt!r} ==="
        )
        eraser = CelebrityMaskEraser(
            pipe=pipe,
            sae=sae,
            target_identity=target_identity,
            identity_names=identity_names,
            identity_kernels=identity_kernels,
            identity_activation_references=identity_activation_references,
            data_mean=data_mean,
            data_std=data_std,
            grid_height=grid_height,
            grid_width=grid_width,
            args=args,
        )
        try:
            eraser.register()
            target_records = records_by_target[target_identity]
            for progress, (index, record) in enumerate(target_records, start=1):
                seed = record.seed if record.seed is not None else args.seed + index * args.seed_step
                print(
                    f"\n[{progress}/{len(target_records)}] "
                    f"seed={seed} prompt={record.prompt}"
                )
                video = generate_erased(
                    pipe,
                    eraser,
                    record.prompt,
                    reference_prompt,
                    seed,
                    args,
                )
                path = output_video_path(
                    str(target_dir),
                    index,
                    record.prompt,
                    target_identity,
                    seed,
                    variant="erased",
                )
                export_to_video(video, str(path), fps=args.fps)
                print(f"Erased video saved to: {path}")

                if args.save_step_masks:
                    mask_dir = save_step_mask_artifacts(
                        eraser.mask_snapshots,
                        path,
                    )
                    print(f"Step masks saved to: {mask_dir}")

                if args.save_diagnostics:
                    report = {
                        "prompt": record.prompt,
                        "prompt_identity": record.identity,
                        "seed": seed,
                        "target_identity": target_identity,
                        "reference_prompt": reference_prompt,
                        "mask_cfg_scale": args.mask_cfg_scale,
                        "mask_start_step": args.mask_start_step,
                        "mask_end_step": args.mask_end_step,
                        "identity_gate_enabled": not args.disable_identity_gate,
                        "prompt_gate_enabled": args.enable_prompt_gate,
                        "prompt_target_match": eraser.prompt_target_match,
                        "summary": _diagnostic_summary(eraser.records),
                        "steps": list(eraser.records),
                    }
                    diagnostics_path = path.with_suffix(".diagnostics.json")
                    diagnostics_path.write_text(
                        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
                        encoding="utf-8",
                    )
                    print(f"Diagnostics saved to: {diagnostics_path}")

                del video
                torch.cuda.empty_cache()
        finally:
            eraser.clear()
            print("SAE mask hook removed.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run HunyuanVideo partitioned-SAE celebrity erasure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--prompt",
        help=(
            "Generate one erased video from this prompt. An original using the "
            "same seed is also generated unless --skip-original is set."
        ),
    )
    input_group.add_argument(
        "--prompts-file",
        help=(
            "Prompt input in TXT (one prompt per line), CSV, or JSONL format. "
            "CSV/JSONL records accept Prompt, Seed, and Identity fields; Identity "
            "is required by --validation-mode self."
        ),
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Process only the first N loaded prompt records; unset means all records.",
    )
    parser.add_argument(
        "--target-identity",
        choices=IDENTITIES,
        default="elizabeth",
        help="Celebrity partition to erase when --target-identities is not supplied.",
    )
    parser.add_argument(
        "--target-identities",
        nargs="+",
        choices=IDENTITIES,
        default=None,
        help=(
            "Erase multiple celebrity partitions sequentially while reusing one "
            "loaded HunyuanVideo pipeline. Overrides --target-identity."
        ),
    )
    parser.add_argument(
        "--validation-mode",
        choices=("self", "cross"),
        default="cross",
        help=(
            "Celebrity prompt pairing for --prompts-file: 'self' applies each "
            "target only to records with the same Identity field; 'cross' applies "
            "every selected target to every loaded prompt."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help=(
            "Celebrity checkpoint run directory, or task root containing timestamped "
            "run directories. The latest compatible final checkpoint is selected."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Explicit SAE .pt checkpoint. Its parent directory must also contain "
            "the matching attribution map and online_stats directory."
        ),
    )
    parser.add_argument(
        "--base-model-path",
        default=DEFAULT_MODEL_PATH,
        help="Repository-relative model directory or Hugging Face HunyuanVideo ID.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(DEFAULT_CHECKPOINT_DIR) / "output_masked_cfg"),
        help="Directory in which original, erased, mask, and diagnostic files are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5210645,
        help="Fallback seed for records without a Seed field.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help=(
            "Increment applied per zero-based prompt index: effective seed is "
            "--seed + index * --seed-step when the record has no Seed field."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=INFERENCE_DEFAULTS.get("inference_steps", 30),
        help="Number of scheduler denoising steps for both original and erased videos.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=INFERENCE_DEFAULTS.get("num_frames", 32),
        help="Number of decoded output frames per video.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=INFERENCE_DEFAULTS.get("height", 480),
        help="Output height in pixels; it must match the VAE/transformer patch grid.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=INFERENCE_DEFAULTS.get("width", 720),
        help="Output width in pixels; it must match the VAE/transformer patch grid.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=INFERENCE_DEFAULTS.get("fps", 8),
        help="Playback frame rate written to each MP4; it does not change generation.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=INFERENCE_DEFAULTS.get("guidance_scale", 6.0),
        help="HunyuanVideo text classifier-free guidance scale.",
    )
    parser.add_argument(
        "--reference-prompt",
        default=None,
        help=(
            "Reference concept used inside the SAE mask. Masked predictions move "
            "toward this prompt; when unset, use mask_cfg_reference_prompt from "
            "configs/hunyuan/celebrity.json."
        ),
    )
    parser.add_argument(
        "--mask-cfg-scale",
        type=float,
        default=INFERENCE_DEFAULTS.get("mask_cfg_scale", 10.0),
        help=(
            "Strength of the masked replacement: conditional + scale * "
            "(reference - conditional). Outside the mask, normal CFG is preserved."
        ),
    )
    parser.add_argument(
        "--mask-start-step",
        type=int,
        default=INFERENCE_DEFAULTS.get("mask_start_step", 5),
        help="First zero-based denoising step at which masked CFG may be applied.",
    )
    parser.add_argument(
        "--mask-end-step",
        type=int,
        default=INFERENCE_DEFAULTS.get("mask_end_step"),
        help="Last zero-based denoising step, inclusive; unset means --steps - 1.",
    )
    parser.add_argument(
        "--mask-background-threshold",
        type=float,
        default=INFERENCE_DEFAULTS.get("mask_background_threshold", 0.4),
        help=(
            "Keep locations whose framewise min-max-normalized generic SAE response "
            "is below this value. Raising it generally expands the eligible mask."
        ),
    )
    parser.add_argument(
        "--mask-identity-threshold",
        type=float,
        default=INFERENCE_DEFAULTS.get("mask_identity_threshold", 0.2),
        help=(
            "Keep locations whose framewise min-max-normalized target-identity "
            "response is above this value. Raising it generally shrinks the mask."
        ),
    )
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=INFERENCE_DEFAULTS.get("mask_dilation", 0),
        help="Binary-mask dilation radius measured in SAE activation-grid cells.",
    )
    parser.add_argument(
        "--identity-competition-ratio",
        type=float,
        default=INFERENCE_DEFAULTS.get("identity_competition_ratio", 1.05),
        help=(
            "Require the calibrated target video score to be at least this multiple "
            "of the strongest non-target celebrity score."
        ),
    )
    parser.add_argument(
        "--identity-min-relative-score",
        type=float,
        default=INFERENCE_DEFAULTS.get("identity_min_relative_score", 0.05),
        help="Minimum calibrated target video score required to open the identity gate.",
    )
    parser.add_argument(
        "--identity-score-top-fraction",
        type=float,
        default=INFERENCE_DEFAULTS.get("identity_score_top_fraction", 0.2),
        help=(
            "Top fraction of spatial cells averaged per frame when computing each "
            "calibrated celebrity score; must be in (0, 1]."
        ),
    )
    parser.add_argument(
        "--disable-identity-gate",
        action="store_true",
        help=(
            "Disable the video-level target-versus-competitor score gate. Spatial "
            "generic and identity thresholds still determine the applied mask."
        ),
    )
    prompt_gate = parser.add_mutually_exclusive_group()
    prompt_gate.add_argument(
        "--enable-prompt-gate",
        action="store_true",
        help=(
            "Apply masks only when the prompt explicitly contains a configured "
            "alias for the selected celebrity. Disabled by default."
        ),
    )
    prompt_gate.add_argument(
        "--disable-prompt-gate",
        action="store_false",
        dest="enable_prompt_gate",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(
        enable_prompt_gate=INFERENCE_DEFAULTS.get("prompt_gate_enabled", False)
    )
    parser.add_argument(
        "--k-gen",
        type=int,
        default=64,
        help=(
            "Legacy fallback generic Top-K used only when checkpoint metadata and "
            "the saved training config do not provide k_gen."
        ),
    )
    parser.add_argument(
        "--k-id",
        type=int,
        default=12,
        help=(
            "Legacy fallback identity Top-K used only when checkpoint metadata and "
            "the saved training config do not provide k_id."
        ),
    )
    parser.add_argument(
        "--warn-below-purity",
        type=float,
        default=INFERENCE_DEFAULTS.get("warn_below_purity", 70.0),
        help=(
            "Print a warning when a loaded attribution partition has lower held-out "
            "purity (percent). This warning does not block inference."
        ),
    )
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Write per-step gate, score, and mask statistics beside each erased MP4.",
    )
    parser.add_argument(
        "--save-step-masks",
        action="store_true",
        help=(
            "Save one grayscale PNG per active denoising step showing the "
            "effective mask actually applied by masked CFG."
        ),
    )
    parser.add_argument(
        "--skip-original",
        action="store_true",
        help="Never generate unmodified comparison videos.",
    )
    parser.add_argument(
        "--generate-originals",
        action="store_true",
        help=(
            "Also generate originals for --prompts-file input. Single-prompt runs "
            "already generate an original unless --skip-original is set."
        ),
    )
    offload_group = parser.add_mutually_exclusive_group()
    offload_group.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable Diffusers model-level CPU offload to reduce resident GPU memory.",
    )
    offload_group.add_argument(
        "--sequential-cpu-offload",
        action="store_true",
        help=(
            "Offload individual submodules between calls for the lowest GPU memory "
            "use; this is slower than model-level CPU offload."
        ),
    )
    args = parser.parse_args(argv)

    if args.mask_end_step is None:
        args.mask_end_step = args.steps - 1
    if args.steps < 1:
        parser.error("--steps must be positive.")
    if not 0 <= args.mask_start_step <= args.mask_end_step < args.steps:
        parser.error("Mask steps must satisfy 0 <= start <= end < steps.")
    if args.mask_cfg_scale <= 0:
        parser.error("--mask-cfg-scale must be positive.")
    if not 0.0 <= args.mask_background_threshold <= 1.0:
        parser.error("--mask-background-threshold must be in [0, 1].")
    if not 0.0 <= args.mask_identity_threshold <= 1.0:
        parser.error("--mask-identity-threshold must be in [0, 1].")
    if args.mask_dilation < 0:
        parser.error("--mask-dilation cannot be negative.")
    if args.identity_competition_ratio < 1.0:
        parser.error("--identity-competition-ratio must be at least 1.")
    if args.identity_min_relative_score < 0:
        parser.error("--identity-min-relative-score cannot be negative.")
    if not 0.0 < args.identity_score_top_fraction <= 1.0:
        parser.error("--identity-score-top-fraction must be in (0, 1].")
    if not 0.0 <= args.warn_below_purity <= 100.0:
        parser.error("--warn-below-purity must be in [0, 100].")
    return args


if __name__ == "__main__":
    run(parse_args())
