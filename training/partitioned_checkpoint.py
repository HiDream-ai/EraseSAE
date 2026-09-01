import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

from model.checkpoint_io import checkpoint_pointer_path, load_checkpoint_model_config


def _configured_model_kwargs(args, classes, d_model):
    return {
        "d_model": int(d_model),
        "celebs": list(classes),
        "n_gen": int(args.n_gen_kernels),
        "n_id_per_celeb": int(args.n_id_per_celeb),
        "k_gen": int(args.k_gen),
        "k_id": int(args.k_id),
    }


def _checkpoint_model_kwargs(
    args,
    classes,
    d_model,
    checkpoint,
    layer_name,
    verbose,
):
    metadata = load_checkpoint_model_config(checkpoint, layer_name)
    if not metadata:
        if verbose:
            print(
                "[Checkpoint] Final pointer has no model_config; falling back "
                "to the current CLI/config values for this legacy checkpoint."
            )
        return _configured_model_kwargs(args, classes, d_model)

    required = {
        "d_model",
        "n_gen",
        "n_id_per_concept",
        "k_gen",
        "k_id",
        "concepts",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise RuntimeError(
            f"Checkpoint model_config is incomplete for {checkpoint}: "
            + ", ".join(missing)
        )

    saved_classes = list(metadata["concepts"])
    if saved_classes != list(classes):
        raise RuntimeError(
            "Checkpoint concept order does not match the attribution dataset: "
            f"checkpoint={saved_classes}, dataset={list(classes)}."
        )
    if int(metadata["d_model"]) != int(d_model):
        raise RuntimeError(
            f"Checkpoint d_model={metadata['d_model']} does not match "
            f"the target layer dimension {d_model}."
        )

    return {
        "d_model": int(metadata["d_model"]),
        "celebs": saved_classes,
        "n_gen": int(metadata["n_gen"]),
        "n_id_per_celeb": int(metadata["n_id_per_concept"]),
        "k_gen": int(metadata["k_gen"]),
        "k_id": int(metadata["k_id"]),
    }


def build_partitioned_sae(
    model_class,
    args,
    classes,
    d_model,
    *,
    checkpoint=None,
    layer_name=None,
    verbose=True,
):
    """Construct a partitioned SAE and optionally restore an exact checkpoint."""
    if checkpoint is None:
        kwargs = _configured_model_kwargs(args, classes, d_model)
    else:
        if layer_name is None:
            raise ValueError("layer_name is required when loading a checkpoint.")
        checkpoint = Path(checkpoint)
        kwargs = _checkpoint_model_kwargs(
            args,
            classes,
            d_model,
            checkpoint,
            layer_name,
            verbose,
        )

    model = model_class(**kwargs)
    if checkpoint is None:
        return model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid SAE checkpoint payload: {checkpoint}")
    state_dict = {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in payload.items()
    }
    model.load_state_dict(state_dict, strict=True)
    if verbose:
        print(f"[Checkpoint] Restored SAE weights and TopK config from: {checkpoint}")
    return model


def validate_partitioned_bundle(
    args,
    layer_name,
    classes,
    rank,
    device,
    raw_model=None,
):
    """Publish a bundle manifest only when all online inference assets exist."""
    if getattr(args, "activation_source", "offline") != "online":
        return None

    status = torch.ones(1, dtype=torch.int32, device=device)
    error = None
    manifest_path = None
    if rank == 0:
        try:
            root = Path(args.sae_save_dir)
            layer_slug = layer_name.replace(".", "_")
            pointer_path = checkpoint_pointer_path(root, layer_name)
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            checkpoint_name = pointer["checkpoint"]
            if (
                not isinstance(checkpoint_name, str)
                or Path(checkpoint_name).name != checkpoint_name
            ):
                raise RuntimeError(
                    f"Checkpoint pointer contains an invalid filename: {pointer_path}"
                )
            checkpoint_path = root / checkpoint_name
            model_config = pointer.get("model_config")
            if not isinstance(model_config, dict):
                raise RuntimeError(
                    f"Checkpoint pointer has no model_config: {pointer_path}"
                )
            if list(model_config.get("concepts", [])) != list(classes):
                raise RuntimeError(
                    "Checkpoint pointer concept order does not match training: "
                    f"pointer={model_config.get('concepts')}, "
                    f"training={list(classes)}."
                )
            d_model = int(model_config.get("d_model", 0))
            partition_size = int(model_config.get("n_id_per_concept", 0))
            if d_model < 1 or partition_size < 1:
                raise RuntimeError(
                    f"Checkpoint pointer has invalid model dimensions: {pointer_path}"
                )
            if raw_model is not None:
                if d_model != int(raw_model.d_model):
                    raise RuntimeError("Checkpoint pointer d_model does not match the SAE.")
                if partition_size != int(raw_model.n_id_per_celeb):
                    raise RuntimeError(
                        "Checkpoint pointer partition size does not match the SAE."
                    )
            map_path = root / f"kernel_identity_map_{layer_slug}.pt"
            stats_path = root / "online_stats" / f"stats_{layer_slug}.pt"
            for artifact in (checkpoint_path, map_path, stats_path):
                if not artifact.is_file():
                    raise FileNotFoundError(
                        f"Required checkpoint bundle artifact is missing: {artifact}"
                    )

            identity_map = torch.load(
                map_path,
                map_location="cpu",
                weights_only=False,
            )
            if list(identity_map) != list(classes):
                raise RuntimeError(
                    "Attribution map concept order does not match training: "
                    f"map={list(identity_map)}, training={list(classes)}."
                )
            for concept_index, concept in enumerate(classes):
                entry = identity_map[concept]
                selected = [int(value) for value in entry.get("top_kernels", [])]
                scores = entry.get("scores")
                references = entry.get("avg_activation")
                survival_rates = entry.get("survival_rates")
                assigned_range = [
                    concept_index * partition_size,
                    (concept_index + 1) * partition_size,
                ]
                if list(entry.get("assigned_range", [])) != assigned_range:
                    raise RuntimeError(
                        f"Attribution range is invalid for {concept!r}."
                    )
                if len(selected) < int(args.min_mining_kernels):
                    raise RuntimeError(
                        f"Attribution map contains too few kernels for {concept!r}."
                    )
                if len(selected) != len(set(selected)):
                    raise RuntimeError(
                        f"Attribution map contains duplicate kernels for {concept!r}."
                    )
                if not all(assigned_range[0] <= value < assigned_range[1] for value in selected):
                    raise RuntimeError(
                        f"Attribution map contains out-of-partition kernels for {concept!r}."
                    )
                if not all(
                    isinstance(values, (list, tuple))
                    and len(values) == len(selected)
                    for values in (scores, references, survival_rates)
                ):
                    raise RuntimeError(
                        f"Attribution vectors are misaligned for {concept!r}."
                    )
                score_tensor = torch.as_tensor(scores, dtype=torch.float64)
                reference_tensor = torch.as_tensor(references, dtype=torch.float64)
                survival_tensor = torch.as_tensor(survival_rates, dtype=torch.float64)
                if not torch.isfinite(score_tensor).all() or (
                    score_tensor <= float(args.min_mining_score)
                ).any():
                    raise RuntimeError(
                        f"Attribution scores are invalid for {concept!r}."
                    )
                if not torch.isfinite(reference_tensor).all() or (
                    reference_tensor <= 0
                ).any():
                    raise RuntimeError(
                        f"Activation references are invalid for {concept!r}."
                    )
                if not torch.isfinite(survival_tensor).all() or (
                    survival_tensor <= float(args.min_mining_survival)
                ).any():
                    raise RuntimeError(
                        f"Kernel survival rates are invalid for {concept!r}."
                    )
                purity = float(entry.get("purity", -1.0))
                if not math.isfinite(purity) or purity < float(args.min_mining_purity):
                    raise RuntimeError(
                        f"Attribution purity is invalid for {concept!r}."
                    )
                if int(entry.get("sample_count", 0)) < 1:
                    raise RuntimeError(
                        f"Attribution sample count is invalid for {concept!r}."
                    )

            stats = torch.load(
                stats_path,
                map_location="cpu",
                weights_only=False,
            )
            if not all(torch.is_tensor(stats.get(key)) for key in ("mean", "std")):
                raise RuntimeError(
                    f"Normalization statistics are incomplete: {stats_path}"
                )
            mean = stats["mean"].reshape(-1)
            std = stats["std"].reshape(-1)
            if mean.numel() != d_model or std.numel() != d_model:
                raise RuntimeError(
                    f"Normalization statistics have the wrong dimension: {stats_path}"
                )
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise RuntimeError(
                    f"Normalization statistics contain non-finite values: {stats_path}"
                )
            if (std <= 0).any():
                raise RuntimeError(
                    f"Normalization statistics contain non-positive std: {stats_path}"
                )

            manifest_path = root / f"bundle_{layer_slug}.json"
            temporary_path = manifest_path.with_suffix(
                manifest_path.suffix + ".tmp"
            )
            temporary_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "layer": layer_name,
                        "concepts": list(classes),
                        "checkpoint": checkpoint_path.name,
                        "checkpoint_pointer": pointer_path.name,
                        "attribution_map": map_path.name,
                        "normalization_stats": str(stats_path.relative_to(root)),
                        "model_config": model_config,
                        "attribution_thresholds": {
                            "min_purity": float(args.min_mining_purity),
                            "min_kernels": int(args.min_mining_kernels),
                            "min_survival": float(args.min_mining_survival),
                            "min_score": float(args.min_mining_score),
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(manifest_path)
            print(f"[Checkpoint] Validated bundle manifest: {manifest_path}")
        except Exception as exc:
            status.zero_()
            error = f"{type(exc).__name__}: {exc}"
            print(f"[Checkpoint] Bundle validation failed: {error}")

    if dist.is_available() and dist.is_initialized():
        dist.broadcast(status, src=0)
    if not status.item():
        detail = error if rank == 0 else "see the rank-0 bundle validation error"
        raise RuntimeError(f"Checkpoint bundle is incomplete: {detail}")
    return manifest_path
