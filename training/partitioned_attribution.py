import torch
import torch.distributed as dist
from tqdm import tqdm

from training.logging_utils import append_jsonl, log_loader_summary
from training.mining_utils import finalize_distributed_partitioned_mining
from training.partitioned_checkpoint import validate_partitioned_bundle
from training.replay_utils import prepare_attribution_batches


def fold_spatiotemporal_activations(orig_act, mask):
    """Convert flattened video tokens to frame-major Conv2d batches."""
    if orig_act.ndim == 4:
        if orig_act.shape[1] != 1:
            raise ValueError(
                "orig_act must have shape [B, 1, L, D] or [B, L, D]."
            )
        orig_act = orig_act[:, 0]
    if orig_act.ndim != 3:
        raise ValueError(
            "orig_act must have shape [B, 1, L, D] or [B, L, D]."
        )
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B, 1, T, H, W].")

    batch_size, token_count, hidden_size = orig_act.shape
    if mask.shape[0] != batch_size:
        raise ValueError("orig_act and mask batch sizes do not match.")
    frames, height, width = mask.shape[2:]
    expected_tokens = frames * height * width
    if token_count != expected_tokens:
        raise ValueError(
            f"Activation token count {token_count} does not match mask grid "
            f"T*H*W={frames}*{height}*{width}={expected_tokens}."
        )

    # Input tokens are frame-major: [B, T, H, W, D]. Keep T adjacent to B
    # before folding. Reshaping [B, D, T, H, W] directly would mix channels
    # and frames when B or T is greater than one.
    folded = (
        orig_act.reshape(batch_size, frames, height, width, hidden_size)
        .permute(0, 1, 4, 2, 3)
        .contiguous()
        .reshape(batch_size * frames, hidden_size, height, width)
    )
    return folded, (batch_size, frames, height, width, hidden_size)


def partition_target_rows_by_mask(labels, frame_major_mask):
    """Split target rows into usable and empty-mask subsets."""
    if labels.ndim != 1:
        raise ValueError("labels must have shape [B].")
    if frame_major_mask.ndim != 5:
        raise ValueError("frame_major_mask must have shape [B, T, 1, H, W].")
    if frame_major_mask.shape[0] != labels.shape[0]:
        raise ValueError("labels and frame_major_mask batch sizes do not match.")

    target_rows = torch.nonzero(labels >= 0, as_tuple=True)[0]
    if target_rows.numel() == 0:
        return target_rows, target_rows

    target_areas = frame_major_mask[target_rows].flatten(start_dim=1).sum(dim=1)
    has_support = target_areas > 0
    return target_rows[has_support], target_rows[~has_support]


def _batch_row_metadata(batch, row):
    metadata = {}
    for key in (
        "subject",
        "case_number",
        "seed",
        "timestep",
        "trajectory_step",
        "prompt",
    ):
        values = batch.get(key)
        if values is None:
            continue
        value = values[row]
        if torch.is_tensor(value):
            value = value.item()
        metadata[key] = value
    return metadata


def _all_reduce(tensors):
    if not (dist.is_available() and dist.is_initialized()):
        return
    for tensor in tensors:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def run_partitioned_attribution(
    *,
    args,
    data_loader,
    raw_model,
    classes,
    layer_name,
    rank,
    device,
):
    """Run one exact post-training pass and save a validated kernel map."""
    classes = list(classes)
    model_classes = list(raw_model.celebs)
    if classes != model_classes:
        raise ValueError(
            "Attribution concept order must match the SAE partitions: "
            f"loader={classes}, model={model_classes}."
        )
    if getattr(args, "activation_source", "offline") == "online":
        log_loader_summary(args, "attribution", layer_name, data_loader, rank)

    total_id_channels = int(raw_model.total_id_kernels)
    num_classes = len(classes)
    sum_targets = torch.zeros(
        (num_classes, total_id_channels),
        dtype=torch.float64,
        device=device,
    )
    count_targets = torch.zeros(num_classes, dtype=torch.float64, device=device)
    count_nonzero_targets = torch.zeros_like(sum_targets)
    skipped_empty_targets = torch.zeros(
        num_classes,
        dtype=torch.float64,
        device=device,
    )
    sum_common = torch.zeros(total_id_channels, dtype=torch.float64, device=device)
    count_common = torch.zeros(1, dtype=torch.float64, device=device)

    batches, batch_count = prepare_attribution_batches(args, data_loader, rank)
    with torch.inference_mode():
        for batch in tqdm(
            batches,
            total=batch_count,
            desc=f"Attributing {layer_name}",
            disable=(rank != 0),
        ):
            orig_act = batch["orig_act"].to(device)
            mask = batch["mask"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.long)
            folded, shape = fold_spatiotemporal_activations(orig_act, mask)
            batch_size, frames, height, width, _ = shape

            if labels.ndim != 1 or labels.shape[0] != batch_size:
                raise ValueError("label must have shape [B].")
            if (labels < -1).any() or (labels >= num_classes).any():
                raise ValueError(
                    "Attribution labels must be -1 or valid concept indices."
                )

            output = raw_model(
                folded,
                route_all=True,
                return_global_topk=False,
            )
            features = output["f_id"].reshape(
                batch_size,
                frames,
                total_id_channels,
                height,
                width,
            ).float()
            frame_major_mask = mask.permute(0, 2, 1, 3, 4).clamp(0.0, 1.0)

            target_rows, empty_target_rows = partition_target_rows_by_mask(
                labels,
                frame_major_mask,
            )
            if empty_target_rows.numel() > 0:
                empty_labels = labels[empty_target_rows]
                skipped_empty_targets.index_add_(
                    0,
                    empty_labels,
                    torch.ones(
                        empty_labels.shape[0],
                        dtype=torch.float64,
                        device=device,
                    ),
                )
                bad_samples = [
                    _batch_row_metadata(batch, row)
                    for row in empty_target_rows.tolist()
                ]
                print(
                    f"[Attribution][rank{rank}] Skipping target samples with "
                    f"empty conditional masks: {bad_samples}"
                )

            if target_rows.numel() > 0:
                target_mask = frame_major_mask[target_rows]
                target_area = target_mask.sum(dim=(1, 3, 4))
                target_features = features[target_rows]
                activation_per_sample = (
                    (target_features * target_mask).sum(
                        dim=(1, 3, 4),
                        dtype=torch.float64,
                    )
                    / target_area.clamp_min(1.0).to(torch.float64)
                )
                target_labels = labels[target_rows]
                sum_targets.index_add_(0, target_labels, activation_per_sample)
                count_targets.index_add_(
                    0,
                    target_labels,
                    torch.ones(
                        target_labels.shape[0],
                        dtype=torch.float64,
                        device=device,
                    ),
                )
                count_nonzero_targets.index_add_(
                    0,
                    target_labels,
                    (activation_per_sample > 1e-6).to(torch.float64),
                )

                # Kernels used throughout the non-target part of a target video
                # are not concept-local. Include those regions in the common
                # baseline instead of comparing only against ordinary prompts.
                outside_mask = 1.0 - target_mask
                outside_area = outside_mask.sum()
                sum_common += (
                    target_features * outside_mask
                ).sum(dim=(0, 1, 3, 4), dtype=torch.float64)
                count_common += outside_area.to(torch.float64)

            common_rows = torch.nonzero(labels == -1, as_tuple=True)[0]
            if common_rows.numel() > 0:
                common_features = features[common_rows]
                sum_common += common_features.sum(
                    dim=(0, 1, 3, 4),
                    dtype=torch.float64,
                )
                count_common += float(
                    common_rows.numel() * frames * height * width
                )

            del output, features, folded, orig_act, mask

    _all_reduce(
        (
            sum_targets,
            count_targets,
            count_nonzero_targets,
            skipped_empty_targets,
            sum_common,
            count_common,
        )
    )

    total_target_samples = count_targets + skipped_empty_targets
    empty_fractions = skipped_empty_targets / total_target_samples.clamp_min(1.0)
    max_empty_fraction = float(
        getattr(args, "max_empty_target_mask_fraction", 0.05)
    )
    quality_rows = {
        name: {
            "valid_samples": int(count_targets[index].item()),
            "skipped_empty_masks": int(skipped_empty_targets[index].item()),
            "empty_fraction": float(empty_fractions[index].item()),
        }
        for index, name in enumerate(classes)
    }
    if rank == 0:
        print(
            "[Attribution] Conditional-mask quality: "
            + ", ".join(
                f"{name}={values['skipped_empty_masks']}/"
                f"{values['valid_samples'] + values['skipped_empty_masks']} "
                f"empty ({100.0 * values['empty_fraction']:.3f}%)"
                for name, values in quality_rows.items()
            )
        )
        append_jsonl(
            args,
            "stage_events",
            {
                "phase": "attribution_mask_quality",
                "layer": layer_name,
                "max_empty_target_mask_fraction": max_empty_fraction,
                "classes": quality_rows,
            },
        )

    if (count_targets == 0).any():
        missing = [
            classes[index]
            for index in torch.nonzero(count_targets == 0, as_tuple=True)[0].tolist()
        ]
        raise RuntimeError(
            "No valid post-training attribution masks were observed for: "
            + ", ".join(missing)
        )
    excessive_empty = torch.nonzero(
        empty_fractions > max_empty_fraction,
        as_tuple=True,
    )[0]
    if excessive_empty.numel() > 0:
        detail = ", ".join(
            f"{classes[index]}={100.0 * empty_fractions[index].item():.3f}%"
            for index in excessive_empty.tolist()
        )
        raise RuntimeError(
            "Empty conditional attribution masks exceed the configured "
            f"{100.0 * max_empty_fraction:.3f}% limit: {detail}."
        )

    finalize_distributed_partitioned_mining(
        classes=classes,
        raw_model=raw_model,
        sum_targets=sum_targets,
        count_targets=count_targets,
        count_nonzero_targets=count_nonzero_targets,
        sum_common=sum_common,
        count_common=count_common,
        top_k=args.top_k,
        save_dir=args.sae_save_dir,
        layer_fn=layer_name.replace(".", "_"),
        rank=rank,
        device=device,
        min_purity=args.min_mining_purity,
        min_kernels=args.min_mining_kernels,
        survival_threshold=args.min_mining_survival,
        score_threshold=args.min_mining_score,
        skipped_target_counts=skipped_empty_targets,
        max_empty_target_mask_fraction=max_empty_fraction,
    )
    validate_partitioned_bundle(
        args=args,
        layer_name=layer_name,
        classes=classes,
        rank=rank,
        device=device,
        raw_model=raw_model,
    )
