import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist

from training.logging_utils import append_jsonl


STATS_SCHEMA_VERSION = 4


def _dataset_digest(dataset):
    records = getattr(dataset, "records", None)
    if records is None:
        return None

    digest = hashlib.sha256()
    for record in records:
        stable_record = {
            key: record.get(key)
            for key in (
                "prompt",
                "video_path",
                "subject",
                "label",
                "case_number",
                "seed",
            )
            if key in record
        }
        digest.update(
            json.dumps(
                stable_record,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _stats_signature(layer_dataloader, layer_name, data_dim):
    capture_step_indices = None
    capture_steps_fn = getattr(layer_dataloader, "_capture_step_indices", None)
    if callable(capture_steps_fn):
        capture_step_indices = sorted(int(index) for index in capture_steps_fn())

    return {
        "schema_version": STATS_SCHEMA_VERSION,
        "loader": (
            f"{type(layer_dataloader).__module__}."
            f"{type(layer_dataloader).__qualname__}"
        ),
        "model_path": str(getattr(layer_dataloader, "model_path", "")),
        "layer": layer_name,
        "task": getattr(layer_dataloader, "task", None),
        "activation_input": getattr(layer_dataloader, "activation_input", None),
        "data_dim": int(data_dim),
        "records": len(getattr(layer_dataloader, "dataset", [])),
        "dataset_sha256": _dataset_digest(
            getattr(layer_dataloader, "dataset", None)
        ),
        "concepts": list(getattr(layer_dataloader, "concepts", [])),
        "num_frames": getattr(layer_dataloader, "num_frames", None),
        "height": getattr(layer_dataloader, "height", None),
        "width": getattr(layer_dataloader, "width", None),
        "guidance_scale": getattr(layer_dataloader, "guidance_scale", None),
        "inference_steps": getattr(layer_dataloader, "inference_steps", None),
        "timesteps_per_video": getattr(
            layer_dataloader, "timesteps_per_video", None
        ),
        "timestep_min_index": getattr(
            layer_dataloader, "timestep_min_index", None
        ),
        "sampling_mode": getattr(layer_dataloader, "sampling_mode", None),
        "trajectory_steps": getattr(
            layer_dataloader, "trajectory_steps", None
        ),
        "capture_step_indices": capture_step_indices,
        "seed": getattr(layer_dataloader, "seed", None),
        "dtype": str(getattr(layer_dataloader, "dtype", None)),
    }


def _load_cached_stats(stats_path, signature, data_dim):
    if not stats_path.exists():
        return None, "cache file does not exist"

    try:
        payload = torch.load(
            stats_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        return None, f"cache could not be read: {exc}"

    if not isinstance(payload, dict):
        return None, "cache payload is not a dictionary"
    if payload.get("signature") != signature:
        return None, "cache signature does not match this run"

    tensors = tuple(payload.get(name) for name in ("mean", "std", "dead_mask"))
    if not all(torch.is_tensor(tensor) for tensor in tensors):
        return None, "cache is missing normalization tensors"

    mean, std, dead_mask = tensors
    if any(tensor.numel() != data_dim for tensor in tensors):
        return None, "cache tensor shape does not match the target layer"
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        return None, "cache contains non-finite values"
    if (std <= 0).any():
        return None, "cache contains a non-positive standard deviation"

    return (
        mean.reshape(data_dim).float(),
        std.reshape(data_dim).float(),
        dead_mask.reshape(data_dim).bool(),
    ), None


def _load_checkpoint_stats(stats_path, data_dim):
    """Load the immutable normalization basis paired with trained SAE weights."""
    if not stats_path.exists():
        return None, "checkpoint statistics do not exist"
    try:
        payload = torch.load(
            stats_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        return None, f"checkpoint statistics could not be read: {exc}"
    if not isinstance(payload, dict):
        return None, "checkpoint statistics payload is not a dictionary"

    mean = payload.get("mean")
    std = payload.get("std")
    if not torch.is_tensor(mean) or not torch.is_tensor(std):
        return None, "checkpoint statistics are missing mean/std tensors"
    if mean.numel() != data_dim or std.numel() != data_dim:
        return None, "checkpoint statistics do not match the target layer"
    mean = mean.reshape(data_dim).float()
    std = std.reshape(data_dim).float()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        return None, "checkpoint statistics contain non-finite values"
    if (std <= 0).any():
        return None, "checkpoint statistics contain a non-positive std"

    dead_mask = payload.get("dead_mask")
    if not torch.is_tensor(dead_mask) or dead_mask.numel() != data_dim:
        dead_mask = torch.zeros(data_dim, dtype=torch.bool)
    else:
        dead_mask = dead_mask.reshape(data_dim).bool()
    return (mean, std, dead_mask), None


def _broadcast_cached_stats(cached, data_dim, layer_dataloader, rank):
    if not (dist.is_available() and dist.is_initialized()):
        return cached

    device = layer_dataloader.device
    use_cache = torch.tensor(
        [int(cached is not None) if rank == 0 else 0],
        dtype=torch.int32,
        device=device,
    )
    dist.broadcast(use_cache, src=0)
    if not use_cache.item():
        return None

    if rank == 0:
        mean, std, dead_mask = cached
        mean = mean.to(device)
        std = std.to(device)
        dead_mask_u8 = dead_mask.to(device=device, dtype=torch.uint8)
    else:
        mean = torch.empty(data_dim, dtype=torch.float32, device=device)
        std = torch.empty(data_dim, dtype=torch.float32, device=device)
        dead_mask_u8 = torch.empty(data_dim, dtype=torch.uint8, device=device)

    dist.broadcast(mean, src=0)
    dist.broadcast(std, src=0)
    dist.broadcast(dead_mask_u8, src=0)
    return mean.cpu(), std.cpu(), dead_mask_u8.bool().cpu()


def prepare_online_normalization(args, layer_dataloader, layer_name, layer_fn, data_dim, rank):
    mode = getattr(args, "normalization_mode", "global")
    dataset_size = len(getattr(layer_dataloader, "dataset", []))
    local_records = len(layer_dataloader._local_indices(pad=False)) if hasattr(layer_dataloader, "_local_indices") else dataset_size
    timesteps_per_record = getattr(layer_dataloader, "timesteps_per_video", 1)
    total_activation_samples = dataset_size * timesteps_per_record
    local_activation_samples = local_records * timesteps_per_record
    concepts = getattr(layer_dataloader, "concepts", None)
    stats_root = Path(args.sae_save_dir) / "online_stats"
    stats_path = stats_root / f"stats_{layer_fn}.pt"
    signature = _stats_signature(layer_dataloader, layer_name, data_dim)
    signature["normalization_mode"] = mode
    cache_status = "not_applicable"

    if getattr(args, "skip_train", False):
        if getattr(args, "recompute_online_stats", False):
            raise ValueError(
                "--recompute-online-stats cannot be used with --skip-train: "
                "the saved SAE must keep its original normalization basis."
            )
        checkpoint_stats = None
        checkpoint_reason = None
        if rank == 0:
            checkpoint_stats, checkpoint_reason = _load_checkpoint_stats(
                stats_path,
                data_dim,
            )
        checkpoint_stats = _broadcast_cached_stats(
            checkpoint_stats,
            data_dim,
            layer_dataloader,
            rank,
        )
        if checkpoint_stats is None:
            detail = checkpoint_reason or "see the rank-0 error"
            raise RuntimeError(
                "--skip-train requires the normalization statistics paired with "
                f"the checkpoint at {stats_path}: {detail}."
            )
        mean, std, dead_mask = checkpoint_stats
        layer_dataloader.set_normalization(mean, std)
        if rank == 0:
            print(
                "[Online] Skip-train mode locked to checkpoint normalization: "
                f"{stats_path}"
            )
            append_jsonl(
                args,
                "stage_events",
                {
                    "phase": "normalization",
                    "layer": layer_name,
                    "mode": "checkpoint_locked",
                    "cache": "hit",
                    "records": dataset_size,
                    "timesteps_per_record": timesteps_per_record,
                    "activation_samples": total_activation_samples,
                    "max_samples_per_subject": getattr(
                        args, "max_samples_per_subject", None
                    ),
                    "concepts": concepts,
                    "stats_path": str(stats_path),
                },
            )
        return mean, std, dead_mask

    if mode == "none":
        if rank == 0:
            print(f"[Online] Skipping normalization stats for {layer_name}; using mean=0, std=1.")
        mean = torch.zeros(data_dim, dtype=torch.float32)
        std = torch.ones(data_dim, dtype=torch.float32)
        dead_mask = torch.zeros(data_dim, dtype=torch.bool)
    elif mode == "global":
        cached = None
        cache_reason = None
        if rank == 0 and not getattr(args, "recompute_online_stats", False):
            cached, cache_reason = _load_cached_stats(
                stats_path,
                signature,
                data_dim,
            )
        cached = _broadcast_cached_stats(
            cached,
            data_dim,
            layer_dataloader,
            rank,
        )
        if cached is not None:
            mean, std, dead_mask = cached
            cache_status = "hit"
            if rank == 0:
                print(f"[Online] Reusing compatible normalization stats: {stats_path}")
            layer_dataloader.set_normalization(mean, std)
            if rank == 0:
                append_jsonl(
                    args,
                    "stage_events",
                    {
                        "phase": "normalization",
                        "layer": layer_name,
                        "mode": mode,
                        "cache": cache_status,
                        "records": dataset_size,
                        "timesteps_per_record": timesteps_per_record,
                        "activation_samples": total_activation_samples,
                        "max_samples_per_subject": getattr(
                            args, "max_samples_per_subject", None
                        ),
                        "concepts": concepts,
                        "stats_path": str(stats_path),
                    },
                )
            return mean, std, dead_mask

        cache_status = "miss"
        if rank == 0:
            if getattr(args, "recompute_online_stats", False):
                cache_reason = "forced by --recompute-online-stats"
            print(f"[Online] Normalization cache miss: {cache_reason}")
            print(f"[Online] Collecting normalization stats for {layer_name} without saving activations.")
            print(
                "[Online] Stats dataset: "
                f"records={dataset_size}, timesteps_per_record={timesteps_per_record}, "
                f"activation_samples={total_activation_samples}, "
                f"max_samples_per_subject={getattr(args, 'max_samples_per_subject', None)}"
            )
            if concepts is not None:
                print(f"[Online] Concepts: {concepts}")
        if rank != 0:
            print(
                f"[Online][rank{rank}] local records={local_records}, "
                f"activation_samples={local_activation_samples}"
            )
        mean, std, dead_mask = layer_dataloader.collect_global_stats(
            data_dim=data_dim,
            desc=layer_name,
        )
    else:
        raise ValueError(f"Unsupported normalization_mode: {mode}")

    layer_dataloader.set_normalization(mean, std)

    if rank == 0:
        stats_root.mkdir(exist_ok=True, parents=True)
        payload = {
            "mean": mean,
            "std": std,
            "dead_mask": dead_mask,
            "source": "online",
            "normalization_mode": mode,
            "signature": signature,
        }
        temporary_path = stats_path.with_suffix(stats_path.suffix + ".tmp")
        torch.save(
            payload,
            temporary_path,
        )
        temporary_path.replace(stats_path)
        print(f"[Online] Normalization stats saved to: {stats_path}")
        append_jsonl(
            args,
            "stage_events",
            {
                "phase": "normalization",
                "layer": layer_name,
                "mode": mode,
                "cache": cache_status,
                "records": dataset_size,
                "timesteps_per_record": timesteps_per_record,
                "activation_samples": total_activation_samples,
                "max_samples_per_subject": getattr(args, "max_samples_per_subject", None),
                "concepts": concepts,
                "stats_path": str(stats_path),
            },
        )

    return mean, std, dead_mask
