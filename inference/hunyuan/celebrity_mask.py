import torch

from inference.hunyuan.nudity_mask import dilate_spatial_mask


def _framewise_minmax(heatmap, eps=1e-6):
    if heatmap.ndim != 4 or heatmap.shape[1] != 1:
        raise ValueError("heatmap must have shape [B*F, 1, H, W].")

    flat = heatmap.float().flatten(1)
    minimum = flat.amin(dim=1).view(-1, 1, 1, 1)
    maximum = flat.amax(dim=1).view(-1, 1, 1, 1)
    dynamic_range = maximum - minimum
    normalized = (heatmap.float() - minimum) / dynamic_range.clamp_min(eps)
    return normalized, dynamic_range, dynamic_range > eps


def build_dynamic_celebrity_mask(
    f_gen,
    f_id,
    identity_names,
    identity_kernels,
    identity_activation_references,
    target_identity,
    batch_size,
    frames,
    background_threshold=0.4,
    identity_threshold=0.2,
    competition_ratio=1.05,
    min_relative_score=0.05,
    score_top_fraction=0.2,
    dilation=0,
    identity_gate_enabled=True,
    gate_mode="competition",
    competition_gate_override=None,
):
    """Build an SAE identity mask with calibrated cross-identity competition."""
    if f_gen.ndim != 4 or f_id.ndim != 4:
        raise ValueError("SAE activations must have shape [B*F, C, H, W].")
    if f_gen.shape[0] != f_id.shape[0]:
        raise ValueError("Generic and identity activations must share a batch size.")
    if f_gen.shape[-2:] != f_id.shape[-2:]:
        raise ValueError("Generic and identity activations must share a spatial shape.")
    if f_id.shape[0] != batch_size * frames:
        raise ValueError("batch_size * frames does not match folded activations.")
    if target_identity not in identity_names:
        raise ValueError(f"Unknown target identity: {target_identity}")
    if not 0.0 <= background_threshold <= 1.0:
        raise ValueError("background_threshold must be in [0, 1].")
    if not 0.0 <= identity_threshold <= 1.0:
        raise ValueError("identity_threshold must be in [0, 1].")
    if competition_ratio <= 0.0:
        raise ValueError("competition_ratio must be positive.")
    if min_relative_score < 0.0:
        raise ValueError("min_relative_score cannot be negative.")
    if not 0.0 < score_top_fraction <= 1.0:
        raise ValueError("score_top_fraction must be in (0, 1].")
    if gate_mode not in {"competition", "target", "off"}:
        raise ValueError("gate_mode must be 'competition', 'target', or 'off'.")
    if competition_gate_override is not None and gate_mode != "competition":
        raise ValueError(
            "competition_gate_override requires gate_mode='competition'."
        )

    target_kernels = identity_kernels[target_identity]
    if not target_kernels:
        raise ValueError(f"No kernels configured for {target_identity}.")

    generic_heatmap = f_gen.float().mean(dim=1, keepdim=True)
    target_activations = f_id[:, target_kernels].float()
    identity_heatmap = target_activations.mean(dim=1, keepdim=True)
    generic_norm, generic_range, generic_valid = _framewise_minmax(
        generic_heatmap
    )
    identity_norm, identity_range, identity_valid = _framewise_minmax(
        identity_heatmap
    )
    foreground_mask = (
        (generic_norm < background_threshold) & generic_valid
    ).float()
    concept_mask = ((identity_norm > identity_threshold) & identity_valid).float()

    height, width = f_id.shape[-2:]
    score_count = max(1, int(height * width * score_top_fraction))
    frame_score_tensors = []
    for identity_name in identity_names:
        kernels = identity_kernels[identity_name]
        references = torch.as_tensor(
            identity_activation_references[identity_name],
            device=f_id.device,
            dtype=torch.float32,
        )
        if len(kernels) != references.numel() or not kernels:
            raise ValueError(
                f"Kernel/reference mismatch for identity {identity_name}."
            )
        activations = f_id[:, kernels].float()
        calibrated = activations / references.view(1, -1, 1, 1).clamp_min(1e-6)
        frame_score = calibrated.mean(dim=1).flatten(1).topk(
            score_count,
            dim=1,
        ).values.mean(dim=1)
        frame_score_tensors.append(frame_score)

    frame_scores = torch.stack(frame_score_tensors, dim=1)
    video_scores = frame_scores.view(batch_size, frames, -1).mean(dim=1)
    target_index = identity_names.index(target_identity)
    target_score = video_scores[:, target_index]
    competitor_scores = video_scores.clone()
    competitor_scores[:, target_index] = -torch.inf
    strongest_competitor = competitor_scores.amax(dim=1)
    absolute_gate = target_score >= min_relative_score
    if gate_mode == "competition":
        identity_gate = absolute_gate & (
            target_score >= strongest_competitor * competition_ratio
        )
    elif gate_mode == "target":
        identity_gate = absolute_gate
    else:
        identity_gate = torch.ones_like(target_score, dtype=torch.bool)
    if not identity_gate_enabled:
        identity_gate = torch.ones_like(target_score, dtype=torch.bool)
    instantaneous_identity_gate = identity_gate.clone()
    if competition_gate_override is not None and identity_gate_enabled:
        if competition_gate_override:
            # Keep requiring absolute target evidence after routing is latched.
            identity_gate = absolute_gate
        else:
            identity_gate = torch.zeros_like(identity_gate)

    folded_gate = identity_gate[:, None].expand(batch_size, frames).reshape(
        batch_size * frames,
        1,
        1,
        1,
    )
    gated_identity_mask = concept_mask * folded_gate.float()
    intersection = foreground_mask * gated_identity_mask
    final_mask = dilate_spatial_mask(intersection, dilation).clamp_(0.0, 1.0)

    winner_indices = video_scores.argmax(dim=1).tolist()
    diagnostics = {
        "gate_mode": (
            "off"
            if not identity_gate_enabled
            else (
                f"{gate_mode}_latched"
                if competition_gate_override is not None
                else gate_mode
            )
        ),
        "target_activation_mean": float(target_activations.mean().item()),
        "target_activation_peak": float(target_activations.amax().item()),
        "target_activation_nonzero_fraction": float(
            (target_activations > 0).float().mean().item()
        ),
        "foreground_mask_fraction": float(foreground_mask.mean().item()),
        "identity_mask_fraction": float(concept_mask.mean().item()),
        "target_gate_fraction": float(identity_gate.float().mean().item()),
        "instantaneous_target_gate_fraction": float(
            instantaneous_identity_gate.float().mean().item()
        ),
        "intersection_mask_fraction": float(intersection.mean().item()),
        "final_mask_fraction": float(final_mask.mean().item()),
        "target_identity_score": float(target_score.mean().item()),
        "strongest_competitor_score": float(
            strongest_competitor.mean().item()
        ),
        "identity_scores": {
            identity_name: float(video_scores[:, index].mean().item())
            for index, identity_name in enumerate(identity_names)
        },
        "identity_winners": [identity_names[index] for index in winner_indices],
        "generic_dynamic_range_mean": float(generic_range.mean().item()),
        "identity_dynamic_range_mean": float(identity_range.mean().item()),
        "valid_generic_frames": int(generic_valid.sum().item()),
        "valid_identity_frames": int(identity_valid.sum().item()),
        "frame_count": int(f_id.shape[0]),
    }
    return final_mask, diagnostics
