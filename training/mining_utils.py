import numpy as np
import torch
import torch.distributed as dist

from pathlib import Path


def quarantine_existing_map(map_path):
    """Move a stale inference map aside without deleting prior results."""
    map_path = Path(map_path)
    if not map_path.exists():
        return None

    counter = 0
    while True:
        invalid_suffix = ".invalid" if counter == 0 else f".invalid.{counter}"
        destination = map_path.with_name(
            map_path.stem + invalid_suffix + map_path.suffix
        )
        if not destination.exists():
            map_path.rename(destination)
            return destination
        counter += 1


def quarantine_stale_attribution(
    save_dir,
    layer_fn,
    rank,
    artifact_prefix="kernel_identity_map",
):
    """Prevent a new training run from being paired with an older map."""
    if rank != 0:
        return None
    artifact_path = Path(save_dir) / f"{artifact_prefix}_{layer_fn}.pt"
    quarantined = quarantine_existing_map(artifact_path)
    if quarantined is not None:
        print(f"[Mining] Stale attribution quarantined at: {quarantined}")
    if artifact_prefix == "kernel_identity_map":
        bundle_path = Path(save_dir) / f"bundle_{layer_fn}.json"
        quarantined_bundle = quarantine_existing_map(bundle_path)
        if quarantined_bundle is not None:
            print(
                "[Mining] Stale bundle manifest quarantined at: "
                f"{quarantined_bundle}"
            )
    return quarantined


def analyze_partitioned_kernels(
    classes,
    raw_model,
    target_means,
    survival_rates,
    common_mean,
    sample_counts,
    top_k,
    survival_threshold=0.1,
    score_threshold=1e-8,
    min_purity=0.0,
    min_kernels=1,
):
    """Mine only positive, stable kernels from each concept's assigned block."""

    target_means = np.asarray(target_means)
    survival_rates = np.asarray(survival_rates)
    common_mean = np.asarray(common_mean)
    sample_counts = np.asarray(sample_counts)

    identity_kernel_map = {}
    diagnostics = {
        "valid": True,
        "survival_threshold": survival_threshold,
        "score_threshold": score_threshold,
        "min_purity": min_purity,
        "min_kernels": min_kernels,
        "classes": {},
    }
    failures = []
    eps = 1e-6

    for index, name in enumerate(classes):
        vec_target = target_means[index]
        other_indices = [other for other in range(len(classes)) if other != index]
        if other_indices:
            vec_other_max = np.max(target_means[other_indices], axis=0)
        else:
            vec_other_max = np.zeros_like(common_mean)
        negative_baseline = np.maximum(common_mean, vec_other_max)

        contrast_ratio = (vec_target + eps) / (negative_baseline + eps)
        base_score = vec_target * np.maximum(0.0, np.log(contrast_ratio))
        stable = survival_rates[index] > survival_threshold
        final_score = np.where(np.isfinite(base_score), base_score, 0.0) * stable

        start, end = raw_model.get_celeb_fields(index)
        assigned_scores = final_score[start:end]
        assigned_targets = vec_target[start:end]
        assigned_negatives = negative_baseline[start:end]
        assigned_survival = survival_rates[index, start:end]
        positive_local = np.flatnonzero(assigned_scores > score_threshold)
        if positive_local.size:
            order = np.argsort(assigned_scores[positive_local])[::-1]
            selected = (positive_local[order[:top_k]] + start).tolist()
        else:
            selected = []

        total_positive_energy = float(final_score[final_score > score_threshold].sum())
        assigned_positive_energy = float(
            assigned_scores[assigned_scores > score_threshold].sum()
        )
        purity = (
            100.0 * assigned_positive_energy / (total_positive_energy + eps)
        )
        sample_count = int(sample_counts[index])

        reason = None
        if sample_count == 0:
            reason = "no attribution samples"
        elif len(selected) < min_kernels:
            reason = (
                f"only {len(selected)} positive stable kernels; "
                f"at least {min_kernels} required"
            )
        elif purity < min_purity:
            reason = (
                f"assigned-partition purity {purity:.2f}% is below "
                f"the required {min_purity:.2f}%"
            )
        if reason is not None:
            failures.append(f"{name}: {reason}")

        diagnostics["classes"][name] = {
            "valid": reason is None,
            "reason": reason,
            "sample_count": sample_count,
            "assigned_range": [start, end],
            "positive_kernel_count": int(positive_local.size),
            "stable_kernel_count": int(
                np.count_nonzero(assigned_survival > survival_threshold)
            ),
            "selected_kernels": selected,
            "purity": purity,
            "assigned_target_mean": float(assigned_targets.mean()),
            "assigned_target_max": float(assigned_targets.max()),
            "assigned_negative_mean": float(assigned_negatives.mean()),
            "assigned_negative_max": float(assigned_negatives.max()),
            "assigned_survival_mean": float(assigned_survival.mean()),
            "assigned_survival_max": float(assigned_survival.max()),
            "max_assigned_score": (
                float(assigned_scores.max()) if assigned_scores.size else 0.0
            ),
        }

        print(
            f"[Mining] {name:15} samples={sample_count:4d} "
            f"range=[{start},{end}) positive={positive_local.size:4d} "
            f"purity={purity:6.2f}%"
        )
        for kernel_index in selected[:3]:
            print(
                f"  Kernel {kernel_index:4d}: "
                f"Target={vec_target[kernel_index]:.4f} "
                f"Common={common_mean[kernel_index]:.4f} "
                f"OtherMax={vec_other_max[kernel_index]:.4f} "
                f"Score={final_score[kernel_index]:.4f} "
                f"Survival={survival_rates[index, kernel_index]:.2%}"
            )

        identity_kernel_map[name] = {
            "top_kernels": selected,
            "assigned_range": [start, end],
            "purity": purity,
            "avg_activation": vec_target[selected].tolist(),
            "survival_rates": survival_rates[index, selected].tolist(),
            "scores": final_score[selected].tolist(),
            "sample_count": sample_count,
        }

    diagnostics["valid"] = not failures
    diagnostics["failures"] = failures
    return identity_kernel_map, diagnostics


def finalize_distributed_partitioned_mining(
    classes,
    raw_model,
    sum_targets,
    count_targets,
    count_nonzero_targets,
    sum_common,
    count_common,
    top_k,
    save_dir,
    layer_fn,
    rank,
    device,
    min_purity=0.0,
    min_kernels=1,
    survival_threshold=0.1,
    score_threshold=1e-8,
    skipped_target_counts=None,
    max_empty_target_mask_fraction=None,
):
    """Analyze on rank 0, broadcast validity, and never save an invalid map."""

    status = torch.ones(1, dtype=torch.int32, device=device)
    failure_summary = None

    if rank == 0:
        try:
            counts = count_targets.detach().cpu().numpy()
            target_means = (
                sum_targets.detach().cpu().numpy()
                / (counts.reshape(-1, 1) + 1e-9)
            )
            survival_rates = (
                count_nonzero_targets.detach().cpu().numpy()
                / (counts.reshape(-1, 1) + 1e-9)
            )
            if count_common.item() > 0:
                common_mean = (sum_common / count_common).detach().cpu().numpy()
            else:
                common_mean = np.zeros(sum_common.shape[0], dtype=np.float32)

            kernel_map, diagnostics = analyze_partitioned_kernels(
                classes=classes,
                raw_model=raw_model,
                target_means=target_means,
                survival_rates=survival_rates,
                common_mean=common_mean,
                sample_counts=counts,
                top_k=top_k,
                min_purity=min_purity,
                min_kernels=min_kernels,
                survival_threshold=survival_threshold,
                score_threshold=score_threshold,
            )
            diagnostics["common_position_count"] = float(count_common.item())
            if skipped_target_counts is not None:
                skipped_counts = skipped_target_counts.detach().cpu().numpy()
                diagnostics["attribution_mask_quality"] = {
                    "max_empty_target_mask_fraction": (
                        float(max_empty_target_mask_fraction)
                        if max_empty_target_mask_fraction is not None
                        else None
                    ),
                    "classes": {
                        name: {
                            "valid_samples": int(counts[index]),
                            "skipped_empty_masks": int(skipped_counts[index]),
                            "empty_fraction": float(
                                skipped_counts[index]
                                / max(counts[index] + skipped_counts[index], 1.0)
                            ),
                        }
                        for index, name in enumerate(classes)
                    },
                }

            save_root = Path(save_dir)
            diagnostic_path = save_root / f"mining_diagnostics_{layer_fn}.pt"
            diagnostic_tmp = diagnostic_path.with_suffix(
                diagnostic_path.suffix + ".tmp"
            )
            torch.save(diagnostics, diagnostic_tmp)
            diagnostic_tmp.replace(diagnostic_path)
            print(f"[Mining] Diagnostics saved to: {diagnostic_path}")

            map_path = save_root / f"kernel_identity_map_{layer_fn}.pt"
            if diagnostics["valid"]:
                map_tmp = map_path.with_suffix(map_path.suffix + ".tmp")
                torch.save(kernel_map, map_tmp)
                map_tmp.replace(map_path)
                print(f"[Mining] Valid kernel map saved to: {map_path}")
            else:
                status.zero_()
                failure_summary = "; ".join(diagnostics["failures"])
                quarantined_path = quarantine_existing_map(map_path)
                if quarantined_path is not None:
                    print(
                        "[Mining] Previous map quarantined at: "
                        f"{quarantined_path}"
                    )
                print(
                    "[Mining] Refusing to save an invalid kernel map: "
                    f"{failure_summary}"
                )
        except Exception as exc:
            status.zero_()
            failure_summary = f"{type(exc).__name__}: {exc}"
            print(f"[Mining] Analysis failed: {failure_summary}")

    if dist.is_available() and dist.is_initialized():
        dist.broadcast(status, src=0)

    if status.item() == 0:
        detail = failure_summary if rank == 0 else "see the rank-0 mining diagnostics"
        raise RuntimeError(f"Partitioned feature mining failed: {detail}")
