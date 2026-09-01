import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers.utils import export_to_video


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs.config_loader import get_model_path, load_experiment_config
from inference.cli_utils import (
    activation_references_for_kernels,
    load_prompt_records,
    ordered_concepts_from_map,
    output_video_path,
    resolve_checkpoint,
    validated_target_kernels,
)
from inference.mask_artifacts import (
    capture_effective_mask,
    save_step_mask_artifacts,
)
from inference.hunyuan.hunyuan_pipline import HunyuanVideoPipeline
from inference.hunyuan.nudity_mask import build_dynamic_nudity_mask
from model.checkpoint_io import load_checkpoint_model_config
from model.partitioned_sae import TopKConvPSAE


DEVICE = "cuda:0"
DEFAULT_MODEL_PATH = get_model_path("hunyuan")
INFERENCE_DEFAULTS = load_experiment_config("hunyuan_nudity").get(
    "inference",
    {},
)
DEFAULT_CHECKPOINT_DIR = "checkpoints/hunyuan/nudity"
LAYER_NAME = "single_transformer_blocks.15.proj_out"
TARGET_CONCEPT = "nudity"
COMPETITOR_CONCEPT = "no_nudity"
EXPECTED_CONCEPTS = [TARGET_CONCEPT, COMPETITOR_CONCEPT]
DEFAULT_REFERENCE_PROMPT = INFERENCE_DEFAULTS.get(
    "mask_cfg_reference_prompt",
    INFERENCE_DEFAULTS.get("mask_cfg_prompt", TARGET_CONCEPT),
)
DEFAULT_PROMPT = (
    "A fully nude adult model is standing in a brightly lit figure-drawing "
    "studio, with their entire body clearly visible from head to toe."
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
        "active_target_kernels",
        "foreground_mask_fraction",
        "raw_concept_mask_fraction",
        "concept_mask_fraction",
        "intersection_mask_fraction",
        "final_mask_fraction",
        "target_gate_fraction",
        "target_relative_score",
        "competitor_relative_score",
    )
    summary = {
        "recorded_steps": len(records),
        "candidate_mask_steps": sum(
            bool(record.get("candidate_mask_active", False))
            for record in records
        ),
        "active_mask_steps": sum(
            bool(record.get("mask_active", False)) for record in records
        ),
    }
    for metric_name in metric_names:
        values = [
            float(record[metric_name])
            for record in records
            if metric_name in record
        ]
        if values:
            summary[metric_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
    return summary


class NudityMaskEraser:
    """Use the nudity SAE partition only to locate masked-CFG regions."""

    def __init__(
        self,
        pipe,
        sae,
        concept_kernels,
        activation_references,
        data_mean,
        data_std,
        grid_height,
        grid_width,
        args,
    ):
        self.pipe = pipe
        self.sae = sae
        self.concept_kernels = concept_kernels
        self.activation_references = activation_references
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.args = args
        self.save_step_masks = bool(getattr(args, "save_step_masks", False))
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
        if step_index is None:
            return output
        within_mask_window = (
            self.args.mask_start_step <= step_index <= self.args.mask_end_step
        )
        if not within_mask_window:
            return output

        # The reference branch traverses this layer after the conditional branch.
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
        folded = (
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
                folded - self.mean.to(folded.dtype)
            ) / (self.std.to(folded.dtype) + 1e-8)
            sae_output = self.sae(
                normalized,
                route_all=True,
                return_global_topk=False,
            )
            f_id = sae_output["f_id"]
            target_activations = f_id[:, self.concept_kernels[TARGET_CONCEPT]]
            competitor_activations = (
                None
                if self.args.disable_concept_gate
                else f_id[:, self.concept_kernels[COMPETITOR_CONCEPT]]
            )
            mask_result = build_dynamic_nudity_mask(
                f_gen=sae_output["f_gen"],
                target_activations=target_activations,
                competitor_activations=competitor_activations,
                target_activation_references=self.activation_references[
                    TARGET_CONCEPT
                ],
                competitor_activation_references=(
                    None
                    if self.args.disable_concept_gate
                    else self.activation_references[COMPETITOR_CONCEPT]
                ),
                batch_size=batch_size,
                frames=frames,
                background_threshold=self.args.mask_background_threshold,
                concept_threshold=self.args.mask_concept_threshold,
                competition_ratio=self.args.competition_ratio,
                min_relative_score=self.args.min_relative_score,
                score_top_fraction=self.args.score_top_fraction,
                dilation=self.args.mask_dilation,
                combine_mode=self.args.mask_mode,
                return_components=False,
            )
            mask, raw_diagnostics = mask_result

            target_float = target_activations.float()
            active_per_frame = (
                target_float.flatten(2).amax(dim=-1) > 0
            ).float().sum(dim=1)
            candidate_mask_active = bool(mask.any().item())
            mask_active = within_mask_window and candidate_mask_active
            self.pipe._current_sae_mask = mask.detach() if mask_active else None
            record = {
                "step_index": int(step_index),
                "within_mask_window": within_mask_window,
                "candidate_mask_active": candidate_mask_active,
                "mask_active": mask_active,
                "target_activation_mean": float(target_float.mean().item()),
                "target_activation_peak": float(target_float.amax().item()),
                "target_activation_nonzero_fraction": float(
                    (target_float > 0).float().mean().item()
                ),
                "active_target_kernels": float(active_per_frame.mean().item()),
                "foreground_mask_fraction": raw_diagnostics[
                    "foreground_fraction"
                ],
                "raw_concept_mask_fraction": raw_diagnostics[
                    "raw_concept_fraction"
                ],
                "concept_mask_fraction": raw_diagnostics["concept_fraction"],
                "intersection_mask_fraction": raw_diagnostics[
                    "intersection_fraction"
                ],
                "final_mask_fraction": raw_diagnostics["final_fraction"],
                "target_gate_fraction": raw_diagnostics[
                    "target_gate_fraction"
                ],
                "target_relative_score": raw_diagnostics.get(
                    "target_relative_score",
                    0.0,
                ),
                "competitor_relative_score": raw_diagnostics.get(
                    "no_nudity_relative_score",
                    0.0,
                ),
                "valid_generic_frames": raw_diagnostics[
                    "valid_generic_frames"
                ],
                "valid_concept_frames": raw_diagnostics[
                    "valid_concept_frames"
                ],
                "frame_count": raw_diagnostics["frame_count"],
            }
            self.records.append(record)
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

    def prepare_prompt(self):
        self.records.clear()
        self.mask_snapshots.clear()
        self.pipe._current_sae_mask = None
        self.pipe._current_effective_sae_mask = None
        self.pipe._capture_effective_sae_mask = self.save_step_masks

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
    eraser.prepare_prompt()
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
    concepts = ordered_concepts_from_map(identity_map)
    if concepts != EXPECTED_CONCEPTS:
        raise RuntimeError(
            "Checkpoint concept order does not match Hunyuan nudity inference: "
            f"expected={EXPECTED_CONCEPTS}, found={concepts}."
        )

    d_model = state_dict["decoder_gen.weight"].shape[0]
    n_gen = state_dict["decoder_gen.weight"].shape[1]
    total_id = state_dict["decoder_id.weight"].shape[1]
    if total_id % len(concepts):
        raise RuntimeError(
            f"ID dictionary size {total_id} is incompatible with "
            f"{len(concepts)} concepts."
        )
    n_id_per_concept = total_id // len(concepts)
    k_gen = int(model_config.get("k_gen", training_config.get("k_gen", args.k_gen)))
    k_id = int(model_config.get("k_id", training_config.get("k_id", args.k_id)))
    configured_concepts = model_config.get("concepts")
    if configured_concepts is not None and configured_concepts != concepts:
        raise RuntimeError(
            "Checkpoint model_config concepts do not match the attribution map."
        )

    sae = TopKConvPSAE(
        d_model=d_model,
        celebs=concepts,
        n_gen=n_gen,
        n_id_per_celeb=n_id_per_concept,
        k_gen=k_gen,
        k_id=k_id,
    ).to(DEVICE, dtype=torch.bfloat16)
    sae.load_state_dict(state_dict)
    sae.eval()

    stats_path = asset_root / "online_stats" / f"stats_{layer_file_name}.pt"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Normalization statistics not found: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    if "mean" not in stats or "std" not in stats:
        raise RuntimeError(f"Normalization statistics are incomplete: {stats_path}")

    concept_kernels = {}
    activation_references = {}
    for concept_index, concept in enumerate(concepts):
        start, end = sae.get_celeb_fields(concept_index)
        selected = validated_target_kernels(identity_map, concept, start, end)
        concept_kernels[concept] = selected
        activation_references[concept] = torch.tensor(
            activation_references_for_kernels(identity_map, concept, selected),
            dtype=torch.float32,
        )
        purity = float(identity_map[concept].get("purity", 0.0))
        print(
            f"Attribution {concept:12s}: kernels={len(selected):2d}, "
            f"purity={purity:6.2f}%"
        )
        if purity < args.warn_below_purity:
            print(
                f"[WARNING] {concept} purity is below "
                f"{args.warn_below_purity:.2f}%."
            )

    return (
        sae,
        concepts,
        concept_kernels,
        activation_references,
        stats["mean"],
        stats["std"],
    )


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("HunyuanVideo inference requires a CUDA device.")

    reference_prompt = (
        args.reference_prompt
        if args.reference_prompt is not None
        else DEFAULT_REFERENCE_PROMPT
    )
    print("--- HunyuanVideo SAE nudity erasure ---")
    print(f"Base model: {args.base_model_path}")
    print(f"Target: {TARGET_CONCEPT}; CFG reference: {reference_prompt!r}")
    print(
        "Masked CFG toward reference: "
        f"scale={args.mask_cfg_scale}, "
        f"steps=[{args.mask_start_step}, {args.mask_end_step}]"
    )
    competition = "disabled" if args.disable_concept_gate else "enabled"
    print(f"Prompt gate: disabled; SAE partition competition: {competition}")
    print("Global true CFG: disabled (true_cfg_scale=1.0)")
    if args.save_step_masks:
        print("Step-mask capture: enabled for every denoising step.")

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
        concepts,
        concept_kernels,
        activation_references,
        data_mean,
        data_std,
    ) = load_sae_assets(args)
    if concepts != EXPECTED_CONCEPTS:
        raise RuntimeError("Loaded SAE concepts changed during asset validation.")
    grid_height, grid_width = _spatial_grid(pipe, args.height, args.width)
    prompt_records = load_prompt_records(
        prompt=args.prompt,
        prompts_file=args.prompts_file,
        default_prompt=DEFAULT_PROMPT,
        max_prompts=args.max_prompts,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_originals = not args.skip_original and (
        args.prompts_file is None or args.generate_originals
    )
    if generate_originals:
        for index, record in enumerate(prompt_records):
            seed = (
                record.seed
                if record.seed is not None
                else args.seed + index * args.seed_step
            )
            print(
                f"\n[Original {index + 1}/{len(prompt_records)}] "
                f"seed={seed} prompt={record.prompt}"
            )
            video = generate_original(pipe, record.prompt, seed, args)
            path = output_video_path(
                str(output_dir),
                index,
                record.prompt,
                TARGET_CONCEPT,
                seed,
                variant="original",
            )
            export_to_video(video, str(path), fps=args.fps)
            print(f"Original video saved to: {path}")
            del video
            torch.cuda.empty_cache()

    eraser = NudityMaskEraser(
        pipe=pipe,
        sae=sae,
        concept_kernels=concept_kernels,
        activation_references=activation_references,
        data_mean=data_mean,
        data_std=data_std,
        grid_height=grid_height,
        grid_width=grid_width,
        args=args,
    )
    try:
        eraser.register()
        for index, record in enumerate(prompt_records):
            seed = (
                record.seed
                if record.seed is not None
                else args.seed + index * args.seed_step
            )
            print(
                f"\n[{index + 1}/{len(prompt_records)}] "
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
                str(output_dir),
                index,
                record.prompt,
                TARGET_CONCEPT,
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
                    "task": "nudity",
                    "prompt": record.prompt,
                    "seed": seed,
                    "target_concept": TARGET_CONCEPT,
                    "reference_prompt": reference_prompt,
                    "mask_mode": args.mask_mode,
                    "mask_background_threshold": args.mask_background_threshold,
                    "mask_concept_threshold": args.mask_concept_threshold,
                    "competition_ratio": args.competition_ratio,
                    "min_relative_score": args.min_relative_score,
                    "score_top_fraction": args.score_top_fraction,
                    "mask_dilation": args.mask_dilation,
                    "mask_cfg_scale": args.mask_cfg_scale,
                    "mask_start_step": args.mask_start_step,
                    "mask_end_step": args.mask_end_step,
                    "concept_gate_enabled": not args.disable_concept_gate,
                    "prompt_gate_enabled": False,
                    "prompt_target_match": None,
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
        description="Run HunyuanVideo partitioned-SAE nudity erasure.",
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
            "CSV/JSONL records may provide Prompt and Seed fields."
        ),
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Process only the first N loaded prompt records; unset means all records.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help=(
            "Nudity checkpoint run directory, or task root containing timestamped "
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
        default=4191321,
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
            "configs/hunyuan/nudity.json."
        ),
    )
    parser.add_argument(
        "--mask-cfg-prompt",
        dest="reference_prompt",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mask-cfg-scale",
        type=float,
        default=INFERENCE_DEFAULTS.get("mask_cfg_scale", 3.0),
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
        "--mask-mode",
        choices=("intersection", "concept"),
        default=INFERENCE_DEFAULTS.get("mask_mode", "intersection"),
        help=(
            "Spatial mask composition: 'intersection' requires both target-concept "
            "activity and low generic activity; 'concept' uses the gated target "
            "concept mask without the generic foreground filter."
        ),
    )
    parser.add_argument(
        "--mask-background-threshold",
        type=float,
        default=INFERENCE_DEFAULTS.get("mask_background_threshold", 0.5),
        help=(
            "For intersection mode, keep locations whose framewise min-max-normalized "
            "generic SAE response is below this value. Raising it generally expands "
            "the eligible foreground mask."
        ),
    )
    parser.add_argument(
        "--mask-concept-threshold",
        type=float,
        default=INFERENCE_DEFAULTS.get("mask_concept_threshold", 0.05),
        help=(
            "Keep locations whose framewise min-max-normalized nudity response is "
            "above this value. Raising it generally shrinks the concept mask."
        ),
    )
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=INFERENCE_DEFAULTS.get("mask_dilation", 2),
        help="Binary-mask dilation radius measured in SAE activation-grid cells.",
    )
    parser.add_argument(
        "--competition-ratio",
        type=float,
        default=INFERENCE_DEFAULTS.get(
            "competition_ratio",
            INFERENCE_DEFAULTS.get("concept_competition_ratio", 1.05),
        ),
        help=(
            "Require the calibrated nudity video score to be at least this multiple "
            "of the calibrated no_nudity score."
        ),
    )
    parser.add_argument(
        "--concept-competition-ratio",
        dest="competition_ratio",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--min-relative-score",
        type=float,
        default=INFERENCE_DEFAULTS.get(
            "min_relative_score",
            INFERENCE_DEFAULTS.get("concept_min_relative_score", 0.05),
        ),
        help="Minimum calibrated nudity video score required to open the concept gate.",
    )
    parser.add_argument(
        "--concept-min-relative-score",
        dest="min_relative_score",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--score-top-fraction",
        type=float,
        default=INFERENCE_DEFAULTS.get(
            "score_top_fraction",
            INFERENCE_DEFAULTS.get("concept_score_top_fraction", 0.2),
        ),
        help=(
            "Top fraction of spatial cells averaged per frame when computing "
            "calibrated nudity and no_nudity scores; must be in (0, 1]."
        ),
    )
    parser.add_argument(
        "--concept-score-top-fraction",
        dest="score_top_fraction",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable-concept-gate",
        action="store_true",
        help=(
            "Disable the video-level nudity-versus-no_nudity score gate. Spatial "
            "generic/concept thresholds and --mask-mode still determine the mask."
        ),
    )
    parser.add_argument(
        "--k-gen",
        type=int,
        default=128,
        help=(
            "Legacy fallback generic Top-K used only when checkpoint metadata and "
            "the saved training config do not provide k_gen."
        ),
    )
    parser.add_argument(
        "--k-id",
        type=int,
        default=128,
        help=(
            "Legacy fallback concept Top-K used only when checkpoint metadata and "
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
    originals = parser.add_mutually_exclusive_group()
    originals.add_argument(
        "--skip-original",
        action="store_true",
        help="Never generate unmodified comparison videos.",
    )
    originals.add_argument(
        "--generate-originals",
        action="store_true",
        help=(
            "Also generate originals for --prompts-file input. Single-prompt runs "
            "already generate an original unless --skip-original is set."
        ),
    )
    offload = parser.add_mutually_exclusive_group()
    offload.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable Diffusers model-level CPU offload to reduce resident GPU memory.",
    )
    offload.add_argument(
        "--sequential-cpu-offload",
        action="store_true",
        help=(
            "Offload individual submodules between calls for the lowest GPU memory "
            "use; this is slower than model-level CPU offload."
        ),
    )
    parser.add_argument(
        "--erasure-mode",
        choices=("mask-cfg",),
        default="mask-cfg",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    if args.mask_end_step is None:
        args.mask_end_step = args.steps - 1
    if args.steps < 1 or args.frames < 1 or args.height < 1 or args.width < 1:
        parser.error("--steps, --frames, --height, and --width must be positive.")
    if args.fps < 1:
        parser.error("--fps must be positive.")
    if args.guidance_scale <= 0:
        parser.error("--guidance-scale must be positive.")
    if not 0 <= args.mask_start_step <= args.mask_end_step < args.steps:
        parser.error("Mask steps must satisfy 0 <= start <= end < steps.")
    if args.mask_cfg_scale <= 0:
        parser.error("--mask-cfg-scale must be positive.")
    if not 0.0 <= args.mask_background_threshold <= 1.0:
        parser.error("--mask-background-threshold must be in [0, 1].")
    if not 0.0 <= args.mask_concept_threshold <= 1.0:
        parser.error("--mask-concept-threshold must be in [0, 1].")
    if args.mask_dilation < 0:
        parser.error("--mask-dilation cannot be negative.")
    if args.competition_ratio < 1.0:
        parser.error("--competition-ratio must be at least 1.")
    if args.min_relative_score < 0:
        parser.error("--min-relative-score cannot be negative.")
    if not 0.0 < args.score_top_fraction <= 1.0:
        parser.error("--score-top-fraction must be in (0, 1].")
    if args.reference_prompt is not None and not args.reference_prompt.strip():
        parser.error("--reference-prompt cannot be empty.")
    if not 0.0 <= args.warn_below_purity <= 100.0:
        parser.error("--warn-below-purity must be in [0, 100].")
    return args


if __name__ == "__main__":
    run(parse_args())
