import torch
import torch.nn.functional as F


def _framewise_minmax(heatmap, eps=1e-6):
    if heatmap.ndim != 4 or heatmap.shape[1] != 1:
        raise ValueError("heatmap must have shape [B*F, 1, H, W].")

    flat = heatmap.float().flatten(1)
    minimum = flat.amin(dim=1).view(-1, 1, 1, 1)
    maximum = flat.amax(dim=1).view(-1, 1, 1, 1)
    dynamic_range = maximum - minimum
    normalized = (heatmap.float() - minimum) / dynamic_range.clamp_min(eps)
    valid = dynamic_range > eps
    return normalized, dynamic_range, valid


def dilate_spatial_mask(mask, radius):
    """Dilate a folded video mask by `radius` latent-grid cells."""
    if radius < 0:
        raise ValueError("radius cannot be negative.")
    if radius == 0:
        return mask
    kernel_size = 2 * radius + 1
    return F.max_pool2d(mask.float(), kernel_size, stride=1, padding=radius)


def build_dynamic_nudity_mask(
    f_gen,
    target_activations,
    competitor_activations=None,
    target_activation_references=None,
    competitor_activation_references=None,
    batch_size=None,
    frames=None,
    background_threshold=0.5,
    concept_threshold=0.05,
    competition_ratio=1.05,
    min_relative_score=0.05,
    score_top_fraction=0.2,
    dilation=2,
    combine_mode="intersection",
    gate_mode=None,
    return_components=False,
):
    """Build a calibrated nudity mask with optional no-nudity competition."""
    if f_gen.ndim != 4 or target_activations.ndim != 4:
        raise ValueError("SAE activations must have shape [B*F, C, H, W].")
    if f_gen.shape[0] != target_activations.shape[0]:
        raise ValueError("Generic and target activations must share a batch size.")
    if f_gen.shape[-2:] != target_activations.shape[-2:]:
        raise ValueError("Generic and target activations must share a spatial shape.")
    if not 0.0 <= background_threshold <= 1.0:
        raise ValueError("background_threshold must be in [0, 1].")
    if not 0.0 <= concept_threshold <= 1.0:
        raise ValueError("concept_threshold must be in [0, 1].")
    if competition_ratio <= 0.0:
        raise ValueError("competition_ratio must be positive.")
    if min_relative_score < 0.0:
        raise ValueError("min_relative_score cannot be negative.")
    if not 0.0 < score_top_fraction <= 1.0:
        raise ValueError("score_top_fraction must be in (0, 1].")
    if combine_mode not in {"intersection", "concept"}:
        raise ValueError("combine_mode must be 'intersection' or 'concept'.")
    if gate_mode is None:
        gate_mode = "competition" if competitor_activations is not None else "off"
    if gate_mode not in {"competition", "target", "off"}:
        raise ValueError("gate_mode must be 'competition', 'target', or 'off'.")

    generic_heatmap = f_gen.float().mean(dim=1, keepdim=True)
    concept_heatmap = target_activations.float().mean(dim=1, keepdim=True)
    generic_norm, generic_range, generic_valid = _framewise_minmax(
        generic_heatmap
    )
    concept_norm, concept_range, concept_valid = _framewise_minmax(
        concept_heatmap
    )

    foreground_mask = (
        (generic_norm < background_threshold) & generic_valid
    ).float()
    raw_concept_mask = (
        (concept_norm > concept_threshold) & concept_valid
    ).float()

    target_gate = torch.ones(
        target_activations.shape[0],
        dtype=torch.bool,
        device=target_activations.device,
    )
    target_score = None
    competitor_score = None
    if gate_mode != "off":
        if batch_size is None or frames is None:
            raise ValueError(
                "batch_size and frames are required for calibrated gating."
            )
        if batch_size * frames != target_activations.shape[0]:
            raise ValueError("batch_size * frames does not match folded activations.")
        if target_activation_references is None:
            raise ValueError(
                "target_activation_references are required for calibrated gating."
            )

        spatial_count = target_activations.shape[-2] * target_activations.shape[-1]
        score_count = max(1, int(spatial_count * score_top_fraction))

        def calibrated_video_score(activations, references, description):
            references = torch.as_tensor(
                references,
                dtype=torch.float32,
                device=activations.device,
            )
            if references.numel() == 1:
                references = references.expand(activations.shape[1])
            if references.numel() != activations.shape[1]:
                raise ValueError(
                    f"{description} kernel/reference counts do not match."
                )
            calibrated = activations.float() / references.view(
                1, -1, 1, 1
            ).clamp_min(1e-6)
            frame_score = calibrated.mean(dim=1).flatten(1).topk(
                score_count,
                dim=1,
            ).values.mean(dim=1)
            return frame_score.view(batch_size, frames).mean(dim=1)

        target_score = calibrated_video_score(
            target_activations,
            target_activation_references,
            "Target",
        )
        video_gate = target_score >= min_relative_score

    if gate_mode == "competition":
        if competitor_activations is None:
            raise ValueError(
                "competitor_activations are required for competition gating."
            )
        if competitor_activations.ndim != 4:
            raise ValueError(
                "competitor_activations must have shape [B*F, C, H, W]."
            )
        if competitor_activations.shape[0] != target_activations.shape[0]:
            raise ValueError("Target and competitor batch sizes do not match.")
        if competitor_activations.shape[-2:] != target_activations.shape[-2:]:
            raise ValueError("Target and competitor spatial shapes do not match.")
        if competitor_activation_references is None:
            raise ValueError(
                "competitor_activation_references are required for partition competition."
            )
        competitor_score = calibrated_video_score(
            competitor_activations,
            competitor_activation_references,
            "Competitor",
        )
        video_gate = video_gate & (
            target_score >= competitor_score * competition_ratio
        )

    if gate_mode != "off":
        target_gate = video_gate[:, None].expand(batch_size, frames).reshape(-1)

    concept_mask = raw_concept_mask * target_gate.view(-1, 1, 1, 1).float()
    intersection = foreground_mask * concept_mask
    combined = concept_mask if combine_mode == "concept" else intersection
    final_mask = dilate_spatial_mask(combined, dilation).clamp_(0.0, 1.0)

    diagnostics = {
        "gate_mode": gate_mode,
        "target_activation_mean": float(target_activations.float().mean().item()),
        "target_activation_peak": float(target_activations.float().amax().item()),
        "target_activation_nonzero_fraction": float(
            (target_activations > 0).float().mean().item()
        ),
        "foreground_fraction": float(foreground_mask.mean().item()),
        "raw_concept_fraction": float(raw_concept_mask.mean().item()),
        "concept_fraction": float(concept_mask.mean().item()),
        "intersection_fraction": float(intersection.mean().item()),
        "final_fraction": float(final_mask.mean().item()),
        "generic_dynamic_range_mean": float(generic_range.mean().item()),
        "concept_dynamic_range_mean": float(concept_range.mean().item()),
        "valid_generic_frames": int(generic_valid.sum().item()),
        "valid_concept_frames": int(concept_valid.sum().item()),
        "target_gate_fraction": float(target_gate.float().mean().item()),
        "frame_count": int(f_gen.shape[0]),
    }
    if target_score is not None:
        diagnostics["target_relative_score"] = float(
            target_score.mean().item()
        )
    if competitor_score is not None:
        diagnostics["no_nudity_relative_score"] = float(
            competitor_score.mean().item()
        )
    if return_components:
        components = {
            "foreground": foreground_mask,
            "concept_raw": raw_concept_mask,
            "concept_gated": concept_mask,
            "intersection": intersection,
            "final": final_mask,
        }
        return final_mask, diagnostics, components
    return final_mask, diagnostics
