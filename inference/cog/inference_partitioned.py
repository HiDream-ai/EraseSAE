import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from diffusers.utils import export_to_video


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs.config_loader import get_model_path, load_experiment_config
from inference.cli_utils import (
    activation_references_for_kernels,
    load_prompt_records,
    ordered_concepts_from_map,
    output_video_path,
    resolve_attributed_bundle,
    validated_target_kernels,
)
from inference.cog.cog_pipline import CogVideoXPipeline
from inference.hunyuan.celebrity_mask import build_dynamic_celebrity_mask
from inference.hunyuan.nudity_mask import build_dynamic_nudity_mask
from inference.mask_artifacts import (
    capture_effective_mask,
    save_step_mask_artifacts,
)
from model.checkpoint_io import load_checkpoint_model_config
from model.partitioned_sae import TopKConvPSAE


DEVICE = "cuda:0"
CELEBRITIES = ["trump", "obama", "musk", "merkel", "elizabeth", "swift"]
TASKS = {
    "celebrity": {
        "config": "cogvideo_celebrity",
        "checkpoint_dir": "checkpoints/cog/celebrity",
        "layer": "transformer_blocks.14.ff.net.2",
        "default_prompt": (
            "Elon Musk is adjusting a microphone in a podcast studio."
        ),
    },
    "nudity": {
        "config": "cogvideo_nudity",
        "checkpoint_dir": "checkpoints/cog/nudity",
        "layer": "transformer_blocks.3.ff.net.2",
        "default_prompt": "A nude adult person is standing beside a swimming pool.",
    },
}


def _training_config(checkpoint_dir):
    candidates = sorted((Path(checkpoint_dir) / "logs").glob("*_config.json"))
    if not candidates:
        return {}
    with candidates[-1].open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _spatial_grid(pipe, height, width):
    patch_size = getattr(pipe.transformer.config, "patch_size", 2)
    if isinstance(patch_size, (tuple, list)):
        patch_height, patch_width = patch_size[-2:]
    else:
        patch_height = patch_width = int(patch_size)
    scale = int(pipe.vae_scale_factor_spatial)
    denominator_height = scale * int(patch_height)
    denominator_width = scale * int(patch_width)
    if height % denominator_height or width % denominator_width:
        raise ValueError(
            "Inference resolution is incompatible with the CogVideoX VAE and "
            f"patch sizes: height={height}, width={width}."
        )
    return height // denominator_height, width // denominator_width


def _diagnostic_summary(records):
    metric_names = (
        "target_activation_mean",
        "target_activation_peak",
        "target_activation_nonzero_fraction",
        "foreground_mask_fraction",
        "raw_concept_mask_fraction",
        "identity_mask_fraction",
        "target_gate_fraction",
        "instantaneous_target_gate_fraction",
        "intersection_mask_fraction",
        "final_mask_fraction",
        "generic_dynamic_range_mean",
        "concept_dynamic_range_mean",
        "target_identity_score",
        "strongest_competitor_score",
        "target_relative_score",
        "competitor_relative_score",
    )
    summary = {
        "recorded_steps": len(records),
        "active_mask_steps": sum(record["mask_active"] for record in records),
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

    winners = Counter()
    for record in records:
        winners.update(record.get("identity_winners", []))
    summary["identity_winner_counts"] = dict(winners)
    return summary


def _print_mask_summary(summary, gate_mode):
    def mean(name):
        metric = summary.get(name)
        return float(metric["mean"]) if metric else 0.0

    target_score = mean("target_relative_score")
    if "target_relative_score" not in summary:
        target_score = mean("target_identity_score")
    competitor_score = mean("competitor_relative_score")
    if "competitor_relative_score" not in summary:
        competitor_score = mean("strongest_competitor_score")
    raw_concept = mean("raw_concept_mask_fraction")
    if "raw_concept_mask_fraction" not in summary:
        raw_concept = mean("identity_mask_fraction")
    gate_fraction = mean("target_gate_fraction")
    instantaneous_gate_fraction = mean("instantaneous_target_gate_fraction")
    if "instantaneous_target_gate_fraction" not in summary:
        instantaneous_gate_fraction = gate_fraction
    foreground = mean("foreground_mask_fraction")
    final = mean("final_mask_fraction")
    winner_counts = summary.get("identity_winner_counts", {})
    winners = ",".join(
        f"{name}:{count}" for name, count in sorted(winner_counts.items())
    ) or "n/a"
    print(
        "[Mask] "
        f"gate={gate_mode} "
        f"active_steps={summary['active_mask_steps']}/{summary['recorded_steps']} "
        f"target_score={target_score:.4f} "
        f"competitor_score={competitor_score:.4f} "
        f"gate_fraction={gate_fraction:.4f} "
        f"instant_gate={instantaneous_gate_fraction:.4f} "
        f"raw_concept={raw_concept:.4f} "
        f"foreground={foreground:.4f} "
        f"final={final:.4f} "
        f"winners={winners}"
    )
    if summary["recorded_steps"] and not summary["active_mask_steps"]:
        if gate_fraction == 0.0:
            reason = (
                "the video-level concept gate rejected the target; changing "
                "the spatial threshold cannot restore this mask"
            )
        elif raw_concept == 0.0:
            reason = "the target heatmap did not pass the spatial threshold"
        elif foreground == 0.0 or final == 0.0:
            reason = "the foreground intersection removed the concept mask"
        else:
            reason = "no effective mask reached masked CFG"
        print(f"[Mask][WARNING] Every applied mask is empty because {reason}.")
    elif final > 0.8:
        print(
            "[Mask][WARNING] The effective mask covers over 80% of the latent "
            "grid; increase the spatial threshold or reduce dilation."
        )


def load_sae_assets(args, task_settings):
    layer_name = task_settings["layer"]
    assets = resolve_attributed_bundle(
        args.checkpoint_dir,
        layer_name,
        checkpoint=args.checkpoint,
    )
    checkpoint = assets["checkpoint"]
    print(f"Using SAE checkpoint: {checkpoint}")
    print(f"Using validated attribution bundle: {assets['bundle']}")
    asset_root = checkpoint.parent
    map_path = assets["attribution_map"]
    stats_path = assets["normalization_stats"]

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity_map = torch.load(map_path, map_location="cpu", weights_only=False)
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    concepts = ordered_concepts_from_map(identity_map)
    expected = CELEBRITIES if args.task == "celebrity" else ["nudity", "no_nudity"]
    if concepts != expected:
        raise RuntimeError(
            f"Checkpoint concept order does not match task {args.task}: "
            f"expected={expected}, found={concepts}."
        )
    if assets["payload"].get("concepts") != expected:
        raise RuntimeError(
            "Attribution bundle concept order does not match the requested task."
        )

    training_config = _training_config(asset_root)
    model_config = load_checkpoint_model_config(checkpoint, layer_name)
    d_model = state_dict["decoder_gen.weight"].shape[0]
    n_gen = state_dict["decoder_gen.weight"].shape[1]
    total_id = state_dict["decoder_id.weight"].shape[1]
    if total_id % len(concepts):
        raise RuntimeError(
            f"ID dictionary size {total_id} is incompatible with {len(concepts)} concepts."
        )
    n_id_per_concept = total_id // len(concepts)
    k_gen = int(model_config.get("k_gen", training_config.get("k_gen", 128)))
    k_id = int(model_config.get("k_id", training_config.get("k_id", 128)))
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

    kernels = {}
    references = {}
    for concept_index, concept in enumerate(concepts):
        start, end = sae.get_celeb_fields(concept_index)
        selected = validated_target_kernels(identity_map, concept, start, end)
        kernels[concept] = selected
        references[concept] = torch.tensor(
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

    return sae, concepts, kernels, references, stats["mean"], stats["std"]


class CogMaskEraser:
    def __init__(
        self,
        pipe,
        sae,
        task,
        target_concept,
        concepts,
        kernels,
        references,
        data_mean,
        data_std,
        grid_height,
        grid_width,
        layer_name,
        args,
    ):
        self.pipe = pipe
        self.sae = sae
        self.task = task
        self.target_concept = target_concept
        self.concepts = concepts
        self.kernels = kernels
        self.references = references
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.layer_name = layer_name
        self.args = args
        self.save_step_masks = bool(getattr(args, "save_step_masks", False))
        self.records = []
        self.mask_snapshots = []
        self.handles = []
        self.latched_competition_gate = None

        device = next(sae.parameters()).device
        dtype = next(sae.parameters()).dtype
        self.mean = data_mean.to(device=device, dtype=dtype).view(1, -1, 1, 1)
        self.std = data_std.to(device=device, dtype=dtype).view(1, -1, 1, 1)

    def _fold_conditional_visual(self, output):
        text_length = int(
            getattr(self.pipe.transformer.config, "max_text_seq_length", 226)
        )
        if output.shape[1] <= text_length:
            raise RuntimeError(
                f"Target layer output has no visual tokens: shape={tuple(output.shape)}."
            )
        # The pipeline concatenates [reference, conditional]. Inference is
        # deliberately one prompt at a time, so only the final row is scored.
        visual = output[-1:, text_length:, :]
        batch_size, visual_length, hidden_size = visual.shape
        spatial_size = self.grid_height * self.grid_width
        if visual_length % spatial_size:
            raise RuntimeError(
                f"Cannot reshape {visual_length} visual tokens into a "
                f"{self.grid_height}x{self.grid_width} grid."
            )
        frames = visual_length // spatial_size
        folded = (
            visual.reshape(
                batch_size,
                frames,
                self.grid_height,
                self.grid_width,
                hidden_size,
            )
            .permute(0, 1, 4, 2, 3)
            .reshape(
                batch_size * frames,
                hidden_size,
                self.grid_height,
                self.grid_width,
            )
        )
        return folded, batch_size, frames

    def _build_mask(self, sae_output, batch_size, frames):
        f_gen = sae_output["f_gen"]
        f_id = sae_output["f_id"]
        if self.task == "celebrity":
            use_latched_gate = (
                self.args.concept_gate_policy == "first_step"
                and self.args.concept_gate_mode == "competition"
                and not self.args.disable_concept_gate
            )
            gate_override = (
                self.latched_competition_gate
                if use_latched_gate
                else None
            )
            mask, diagnostics = build_dynamic_celebrity_mask(
                f_gen=f_gen,
                f_id=f_id,
                identity_names=self.concepts,
                identity_kernels=self.kernels,
                identity_activation_references=self.references,
                target_identity=self.target_concept,
                batch_size=batch_size,
                frames=frames,
                background_threshold=self.args.mask_background_threshold,
                identity_threshold=self.args.mask_concept_threshold,
                competition_ratio=self.args.competition_ratio,
                min_relative_score=self.args.min_relative_score,
                score_top_fraction=self.args.score_top_fraction,
                dilation=self.args.mask_dilation,
                identity_gate_enabled=not self.args.disable_concept_gate,
                gate_mode=self.args.concept_gate_mode,
                competition_gate_override=gate_override,
            )
            if use_latched_gate and self.latched_competition_gate is None:
                self.latched_competition_gate = bool(
                    diagnostics["target_gate_fraction"] > 0.0
                )
            diagnostics["concept_gate_policy"] = self.args.concept_gate_policy
            diagnostics["latched_competition_gate"] = (
                self.latched_competition_gate
            )
            return mask, diagnostics

        target = f_id[:, self.kernels["nudity"]]
        gate_mode = self.args.concept_gate_mode
        competitor = (
            f_id[:, self.kernels["no_nudity"]]
            if gate_mode == "competition"
            else None
        )
        mask, diagnostics = build_dynamic_nudity_mask(
            f_gen=f_gen,
            target_activations=target,
            competitor_activations=competitor,
            target_activation_references=self.references["nudity"],
            competitor_activation_references=(
                self.references["no_nudity"]
                if gate_mode == "competition"
                else None
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
            gate_mode=gate_mode,
        )
        diagnostics = {
            "gate_mode": diagnostics["gate_mode"],
            "target_activation_mean": diagnostics["target_activation_mean"],
            "target_activation_peak": diagnostics["target_activation_peak"],
            "target_activation_nonzero_fraction": diagnostics[
                "target_activation_nonzero_fraction"
            ],
            "foreground_mask_fraction": diagnostics["foreground_fraction"],
            "raw_concept_mask_fraction": diagnostics["raw_concept_fraction"],
            "identity_mask_fraction": diagnostics["concept_fraction"],
            "intersection_mask_fraction": diagnostics["intersection_fraction"],
            "final_mask_fraction": diagnostics["final_fraction"],
            "target_gate_fraction": diagnostics["target_gate_fraction"],
            "target_relative_score": diagnostics.get("target_relative_score", 0.0),
            "competitor_relative_score": diagnostics.get(
                "no_nudity_relative_score", 0.0
            ),
            "valid_generic_frames": diagnostics["valid_generic_frames"],
            "valid_identity_frames": diagnostics["valid_concept_frames"],
            "generic_dynamic_range_mean": diagnostics[
                "generic_dynamic_range_mean"
            ],
            "concept_dynamic_range_mean": diagnostics[
                "concept_dynamic_range_mean"
            ],
            "frame_count": diagnostics["frame_count"],
        }
        return mask, diagnostics

    def _hook(self, module, inputs, output):
        del module, inputs
        if getattr(self.pipe, "_suspend_sae_mask_hook", False):
            return output
        step_index = getattr(self.pipe, "_current_step_index", None)
        if step_index is None or not (
            self.args.mask_start_step <= step_index <= self.args.mask_end_step
        ):
            return output
        if getattr(self.pipe, "_current_sae_mask", None) is not None:
            return output

        folded, batch_size, frames = self._fold_conditional_visual(output)
        with torch.no_grad():
            normalized = (
                folded - self.mean.to(folded.dtype)
            ) / (self.std.to(folded.dtype) + 1e-8)
            sae_output = self.sae(
                normalized,
                route_all=True,
                return_global_topk=False,
            )
            mask, diagnostics = self._build_mask(sae_output, batch_size, frames)
            mask_active = bool(mask.any().item())
            self.pipe._current_sae_mask = mask.detach() if mask_active else None
            self.records.append(
                {
                    "step_index": int(step_index),
                    "mask_active": mask_active,
                    **diagnostics,
                }
            )
        return output

    def register(self):
        self.clear()
        for name, module in self.pipe.transformer.named_modules():
            if name == self.layer_name:
                self.handles.append(module.register_forward_hook(self._hook))
                break
        if not self.handles:
            raise RuntimeError(f"Target layer not found: {self.layer_name}")
        self.pipe._capture_effective_sae_mask = self.save_step_masks
        print(f"SAE mask hook registered on: {self.layer_name}")

    def clear(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.pipe._current_sae_mask = None
        self.pipe._current_effective_sae_mask = None
        self.pipe._capture_effective_sae_mask = False

    def prepare(self):
        self.records.clear()
        self.mask_snapshots.clear()
        self.latched_competition_gate = None
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
        "negative_prompt": "",
        "num_inference_steps": args.steps,
        "num_frames": args.frames,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "use_dynamic_cfg": args.use_dynamic_cfg,
        "generator": torch.Generator(device=DEVICE).manual_seed(seed),
        "mask_cfg_scale": args.mask_cfg_scale,
        "mask_cfg_start_step": args.mask_start_step,
        "mask_cfg_end_step": args.mask_end_step,
    }


def generate_original(pipe, prompt, seed, args):
    pipe._current_sae_mask = None
    pipe._current_effective_sae_mask = None
    call_args = _generation_args(prompt, seed, args)
    call_args["mask_cfg_scale"] = 0.0
    return pipe(**call_args).frames[0]


def generate_erased(pipe, eraser, prompt, reference_prompt, seed, args):
    eraser.prepare()
    call_args = _generation_args(prompt, seed, args)
    call_args["mask_reference_prompt"] = reference_prompt
    call_args["callback_on_step_end"] = eraser.clear_step_mask
    return pipe(**call_args).frames[0]


def _reference_prompt(args):
    if args.reference_prompt is not None:
        return args.reference_prompt
    return "nudity"


def _build_celebrity_prompt_plan(prompt_records, targets, validation_mode):
    indexed_records = list(enumerate(prompt_records))
    if validation_mode == "cross":
        return indexed_records, {target: indexed_records for target in targets}

    missing_identity = [
        index for index, record in indexed_records if record.identity is None
    ]
    if missing_identity:
        rows = ", ".join(str(index + 1) for index in missing_identity[:5])
        raise ValueError(
            "Self inference requires an Identity column for every prompt; "
            f"missing at loaded prompt rows: {rows}."
        )

    unknown_identities = sorted(
        {
            record.identity
            for _, record in indexed_records
            if record.identity not in CELEBRITIES
        }
    )
    if unknown_identities:
        raise ValueError(
            "Unsupported prompt identities in self inference: "
            + ", ".join(unknown_identities)
        )

    records_by_target = {
        target: [
            (index, record)
            for index, record in indexed_records
            if record.identity == target
        ]
        for target in targets
    }
    empty_targets = [
        target for target, records in records_by_target.items() if not records
    ]
    if empty_targets:
        raise ValueError(
            "Self inference has no loaded prompts for: "
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


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CogVideoX inference requires a CUDA device.")
    settings = TASKS[args.task]
    targets = (
        list(dict.fromkeys(args.target_identities or [args.target_identity]))
        if args.task == "celebrity"
        else ["nudity"]
    )
    print("--- CogVideoX partitioned-SAE inference ---")
    print(f"Task: {args.task}; targets: {targets}")
    print(f"Base model: {args.base_model_path}")
    print(f"Dynamic CFG: {'enabled' if args.use_dynamic_cfg else 'disabled'}")
    print(
        "Masked CFG toward reference: "
        f"scale={args.mask_cfg_scale}, "
        f"steps=[{args.mask_start_step}, {args.mask_end_step}]"
    )
    print(
        "Prompt gate: disabled; "
        f"SAE concept gate: {args.concept_gate_mode} "
        f"({args.concept_gate_policy})"
    )
    if args.save_step_masks:
        print("Applied-mask PNG capture: enabled.")

    sae, concepts, kernels, references, mean, std = load_sae_assets(args, settings)
    missing = sorted(set(targets) - set(concepts))
    if missing:
        raise RuntimeError("Targets absent from checkpoint: " + ", ".join(missing))

    pipe = CogVideoXPipeline.from_pretrained(
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

    grid_height, grid_width = _spatial_grid(pipe, args.height, args.width)
    prompt_records = load_prompt_records(
        prompt=args.prompt,
        prompts_file=args.prompts_file,
        default_prompt=settings["default_prompt"],
        max_prompts=args.max_prompts,
    )
    if args.task == "celebrity":
        original_records, records_by_target = _build_celebrity_prompt_plan(
            prompt_records,
            targets,
            args.validation_mode,
        )
        print(f"Celebrity prompt mode: {args.validation_mode}")
    else:
        original_records = list(enumerate(prompt_records))
        records_by_target = {"nudity": original_records}

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    multiple_targets = len(targets) > 1
    generate_originals = not args.skip_original and (
        args.prompts_file is None or args.generate_originals
    )
    if generate_originals:
        original_dir = output_root / "originals" if multiple_targets else output_root
        original_dir.mkdir(parents=True, exist_ok=True)
        original_target = "reference" if multiple_targets else targets[0]
        for position, (index, record) in enumerate(original_records, start=1):
            seed = record.seed if record.seed is not None else args.seed + index * args.seed_step
            print(
                f"\n[Original {position}/{len(original_records)}] "
                f"seed={seed} prompt={record.prompt}"
            )
            video = generate_original(pipe, record.prompt, seed, args)
            path = output_video_path(
                str(original_dir),
                index,
                record.prompt,
                original_target,
                seed,
                variant="original",
            )
            export_to_video(video, str(path), fps=args.fps)
            print(f"Original video saved to: {path}")
            del video
            torch.cuda.empty_cache()

    for target in targets:
        target_dir = output_root / f"target_{target}" if multiple_targets else output_root
        target_dir.mkdir(parents=True, exist_ok=True)
        reference_prompt = _reference_prompt(args)
        print(f"\n=== Target: {target}; reference: {reference_prompt!r} ===")
        eraser = CogMaskEraser(
            pipe=pipe,
            sae=sae,
            task=args.task,
            target_concept=target,
            concepts=concepts,
            kernels=kernels,
            references=references,
            data_mean=mean,
            data_std=std,
            grid_height=grid_height,
            grid_width=grid_width,
            layer_name=settings["layer"],
            args=args,
        )
        try:
            eraser.register()
            target_records = records_by_target[target]
            for position, (index, record) in enumerate(target_records, start=1):
                seed = (
                    record.seed
                    if record.seed is not None
                    else args.seed + index * args.seed_step
                )
                print(
                    f"\n[{position}/{len(target_records)}] "
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
                    target,
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

                mask_summary = _diagnostic_summary(eraser.records)
                _print_mask_summary(mask_summary, args.concept_gate_mode)

                if args.save_diagnostics:
                    report = {
                        "task": args.task,
                        "prompt": record.prompt,
                        "seed": seed,
                        "target_identity": target,
                        "validation_mode": args.validation_mode,
                        "reference_prompt": reference_prompt,
                        "prompt_gate_enabled": False,
                        "prompt_target_match": None,
                        "mask_cfg_scale": args.mask_cfg_scale,
                        "use_dynamic_cfg": args.use_dynamic_cfg,
                        "mask_start_step": args.mask_start_step,
                        "mask_end_step": args.mask_end_step,
                        "concept_gate_mode": args.concept_gate_mode,
                        "concept_gate_policy": args.concept_gate_policy,
                        "concept_gate_enabled": args.concept_gate_mode != "off",
                        "summary": mask_summary,
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
        description=(
            "Run CogVideoX partitioned-SAE celebrity or nudity erasure. "
            "Task-specific defaults are loaded after parsing from the matching JSON config."
        )
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        required=True,
        help=(
            "Select the checkpoint family, target transformer layer, concept list, "
            "default prompt, and inference config."
        ),
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
            "is required by celebrity --validation-mode self."
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
        choices=CELEBRITIES,
        default="musk",
        help=(
            "Celebrity partition to erase when --task celebrity and "
            "--target-identities is not supplied; ignored for nudity."
        ),
    )
    parser.add_argument(
        "--target-identities",
        nargs="+",
        choices=CELEBRITIES,
        default=None,
        help=(
            "Erase multiple celebrity partitions sequentially while reusing one "
            "loaded CogVideoX pipeline. Celebrity-only; overrides --target-identity."
        ),
    )
    parser.add_argument(
        "--validation-mode",
        choices=("self", "cross"),
        default="cross",
        help=(
            "Celebrity prompt pairing for --prompts-file: 'self' applies each "
            "target only to records with the same Identity field; 'cross' applies "
            "every selected target to every loaded prompt. Nudity uses 'cross'."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Checkpoint run directory, or task root containing timestamped runs. "
            "The newest run with a complete validated attribution bundle is selected; "
            "when omitted, use the selected task's checkpoint directory."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Explicit SAE .pt checkpoint. It must exactly match the checkpoint in "
            "the validated attribution bundle stored in the same directory."
        ),
    )
    parser.add_argument(
        "--base-model-path",
        default=get_model_path("cogvideox_5b"),
        help="Repository-relative model directory or Hugging Face CogVideoX-5b ID.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for original, erased, mask, and diagnostic files. When "
            "omitted, use <checkpoint-dir>/output_partitioned."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        default=None,
        help="Denoising steps; omitted loads inference_steps from the task config (fallback 30).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Decoded output frames; omitted loads num_frames from the task config (fallback 16).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            "Output height in pixels; omitted loads the task config (fallback 480), "
            "and the value must match the VAE/transformer patch grid."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Output width in pixels; omitted loads the task config (fallback 720), "
            "and the value must match the VAE/transformer patch grid."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help=(
            "MP4 playback rate; omitted loads fps from the task config (fallback 8). "
            "It does not change video generation."
        ),
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help=(
            "CogVideoX text classifier-free guidance scale; omitted loads the task "
            "config (fallback 6.0). Must exceed 1 so the reference branch is encoded."
        ),
    )
    dynamic_cfg = parser.add_mutually_exclusive_group()
    dynamic_cfg.add_argument(
        "--use-dynamic-cfg",
        dest="use_dynamic_cfg",
        action="store_true",
        help=(
            "Force CogVideoX's timestep-dependent text CFG schedule on. When neither "
            "dynamic-CFG flag is set, use use_dynamic_cfg from the task config."
        ),
    )
    dynamic_cfg.add_argument(
        "--no-dynamic-cfg",
        dest="use_dynamic_cfg",
        action="store_false",
        help="Force a constant --guidance-scale at every denoising step.",
    )
    parser.set_defaults(use_dynamic_cfg=None)
    parser.add_argument(
        "--reference-prompt",
        default=None,
        help=(
            "Reference concept used inside the SAE mask. Masked predictions move "
            "toward this prompt; omitted loads mask_cfg_reference_prompt from the "
            "selected task config."
        ),
    )
    parser.add_argument(
        "--mask-cfg-scale",
        type=float,
        default=None,
        help=(
            "Strength of masked replacement: conditional + scale * (reference - "
            "conditional). Outside the mask, normal text CFG is preserved. Omitted "
            "loads the task config (fallback 3.0)."
        ),
    )
    parser.add_argument(
        "--mask-start-step",
        type=int,
        default=None,
        help=(
            "First zero-based denoising step at which masked CFG may be applied; "
            "omitted loads the task config (fallback 5)."
        ),
    )
    parser.add_argument(
        "--mask-end-step",
        type=int,
        default=None,
        help=(
            "Last zero-based masked-CFG step, inclusive; omitted loads the task "
            "config, or uses --steps - 1 when the config value is null."
        ),
    )
    parser.add_argument(
        "--mask-background-threshold",
        type=float,
        default=None,
        help=(
            "Keep locations whose framewise min-max-normalized generic SAE response "
            "is below this value. Raising it generally expands the eligible mask; "
            "omitted loads the task config (fallback 0.4)."
        ),
    )
    parser.add_argument(
        "--mask-concept-threshold",
        type=float,
        default=None,
        help=(
            "Keep locations whose framewise min-max-normalized target response is "
            "above this value. Raising it generally shrinks the mask; omitted loads "
            "the task config (fallback 0.2)."
        ),
    )
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=None,
        help=(
            "Binary-mask dilation radius in SAE activation-grid cells; omitted "
            "loads the task config (fallback 0)."
        ),
    )
    parser.add_argument(
        "--mask-mode",
        choices=("intersection", "concept"),
        default=None,
        help=(
            "Nudity spatial composition: 'intersection' requires target activity "
            "and low generic activity; 'concept' omits the generic filter. Celebrity "
            "masks always use intersection. Omitted loads the task config."
        ),
    )
    parser.add_argument(
        "--competition-ratio",
        type=float,
        default=None,
        help=(
            "For competition gating, require the calibrated target score to be at "
            "least this multiple of the strongest competitor (or no_nudity). Omitted "
            "loads the task config (fallback 1.05). Values below 1 allow bounded "
            "near-ties; values above 1 require a target lead."
        ),
    )
    parser.add_argument(
        "--min-relative-score",
        type=float,
        default=None,
        help=(
            "Minimum calibrated target video score required by 'competition' and "
            "'target' gate modes; omitted loads the task config (fallback 0.05)."
        ),
    )
    parser.add_argument(
        "--score-top-fraction",
        type=float,
        default=None,
        help=(
            "Top fraction of spatial cells averaged per frame to form calibrated "
            "video scores; must be in (0, 1]. Omitted loads the task config "
            "(fallback 0.2)."
        ),
    )
    parser.add_argument(
        "--concept-gate-mode",
        choices=("competition", "target", "off"),
        default=None,
        help=(
            "Video-level gate: 'competition' compares the target against other "
            "celebrity partitions or no_nudity using --competition-ratio; 'target' "
            "requires only the absolute target score; 'off' keeps only spatial "
            "masking. Omitted loads the task config."
        ),
    )
    parser.add_argument(
        "--concept-gate-policy",
        choices=("per_step", "first_step"),
        default=None,
        help=(
            "Celebrity competition routing policy: 'per_step' re-evaluates every "
            "step; 'first_step' latches the first competition decision, while later "
            "steps still require absolute target evidence. Ignored for nudity and "
            "non-competition modes."
        ),
    )
    parser.add_argument(
        "--disable-concept-gate",
        action="store_true",
        help=(
            "Compatibility alias that forces --concept-gate-mode off. Spatial "
            "generic/concept thresholds and mask composition remain active."
        ),
    )
    originals = parser.add_mutually_exclusive_group()
    originals.add_argument(
        "--generate-originals",
        action="store_true",
        help=(
            "Also generate originals for --prompts-file input. Single-prompt runs "
            "already generate an original unless --skip-original is set."
        ),
    )
    originals.add_argument(
        "--skip-original",
        action="store_true",
        help="Never generate unmodified comparison videos.",
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
        "--warn-below-purity",
        type=float,
        default=70.0,
        help=(
            "Print a warning when a loaded attribution partition has lower held-out "
            "purity (percent). This warning does not block inference."
        ),
    )
    args = parser.parse_args(argv)

    settings = TASKS[args.task]
    inference_defaults = load_experiment_config(settings["config"]).get(
        "inference", {}
    )
    if args.reference_prompt is None:
        configured_reference = inference_defaults.get(
            "mask_cfg_reference_prompt"
        )
        if configured_reference is not None:
            args.reference_prompt = str(configured_reference)
    args.checkpoint_dir = args.checkpoint_dir or settings["checkpoint_dir"]
    args.output_dir = args.output_dir or str(
        Path(args.checkpoint_dir) / "output_partitioned"
    )
    defaults = {
        "steps": ("inference_steps", 30, int),
        "frames": ("num_frames", 16, int),
        "height": ("height", 480, int),
        "width": ("width", 720, int),
        "fps": ("fps", 8, int),
        "guidance_scale": ("guidance_scale", 6.0, float),
        "use_dynamic_cfg": ("use_dynamic_cfg", True, bool),
        "mask_cfg_scale": ("mask_cfg_scale", 3.0, float),
        "mask_start_step": ("mask_start_step", 5, int),
        "mask_background_threshold": (
            "mask_background_threshold",
            0.4,
            float,
        ),
        "mask_concept_threshold": ("mask_concept_threshold", 0.2, float),
        "mask_dilation": ("mask_dilation", 0, int),
        "mask_mode": ("mask_mode", "intersection", str),
        "competition_ratio": ("competition_ratio", 1.05, float),
        "min_relative_score": ("min_relative_score", 0.05, float),
        "score_top_fraction": ("score_top_fraction", 0.2, float),
        "concept_gate_mode": ("concept_gate_mode", "competition", str),
        "concept_gate_policy": ("concept_gate_policy", "per_step", str),
    }
    for attribute, (config_key, fallback, converter) in defaults.items():
        if getattr(args, attribute) is None:
            setattr(
                args,
                attribute,
                converter(inference_defaults.get(config_key, fallback)),
            )
    if args.mask_end_step is None:
        configured_end = inference_defaults.get("mask_end_step")
        args.mask_end_step = (
            args.steps - 1 if configured_end is None else int(configured_end)
        )

    if args.task == "nudity" and args.target_identities is not None:
        parser.error("--target-identities is only valid for celebrity inference.")
    if args.task == "nudity" and args.validation_mode != "cross":
        parser.error("--validation-mode is only valid for celebrity inference.")
    if args.steps < 1 or args.frames < 1 or args.height < 1 or args.width < 1:
        parser.error("--steps, --frames, --height, and --width must be positive.")
    if args.guidance_scale <= 1.0:
        parser.error("--guidance-scale must be greater than 1 to encode a reference.")
    if args.mask_cfg_scale < 0:
        parser.error("--mask-cfg-scale cannot be negative.")
    if not 0 <= args.mask_start_step <= args.mask_end_step < args.steps:
        parser.error("Mask step bounds must satisfy 0 <= start <= end < steps.")
    if args.mask_dilation < 0:
        parser.error("--mask-dilation cannot be negative.")
    if not 0.0 <= args.mask_background_threshold <= 1.0:
        parser.error("--mask-background-threshold must be in [0, 1].")
    if not 0.0 <= args.mask_concept_threshold <= 1.0:
        parser.error("--mask-concept-threshold must be in [0, 1].")
    if args.competition_ratio <= 0.0:
        parser.error("--competition-ratio must be positive.")
    if args.min_relative_score < 0.0:
        parser.error("--min-relative-score cannot be negative.")
    if not 0.0 < args.score_top_fraction <= 1.0:
        parser.error("--score-top-fraction must be in (0, 1].")
    if args.reference_prompt is not None and not args.reference_prompt.strip():
        parser.error("--reference-prompt cannot be empty.")
    if args.disable_concept_gate:
        args.concept_gate_mode = "off"
    return args


if __name__ == "__main__":
    run(parse_args())
