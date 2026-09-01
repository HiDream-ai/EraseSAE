import torch


def apply_masked_cfg(
    conditional_prediction,
    reference_prediction,
    mask,
    scale,
    outside_prediction=None,
):
    """Move masked predictions toward a reference and preserve the outside baseline."""
    if conditional_prediction.shape != reference_prediction.shape:
        raise ValueError("Conditional and reference predictions must have the same shape.")
    if outside_prediction is not None and outside_prediction.shape != conditional_prediction.shape:
        raise ValueError("Outside and conditional predictions must have the same shape.")
    if mask.ndim != conditional_prediction.ndim:
        raise ValueError("Mask rank must match the noise prediction rank.")
    if scale < 0:
        raise ValueError("scale cannot be negative.")

    mask = mask.to(
        device=conditional_prediction.device,
        dtype=conditional_prediction.dtype,
    ).clamp(0.0, 1.0)
    inside_prediction = conditional_prediction + float(scale) * (
        reference_prediction - conditional_prediction
    )
    if outside_prediction is None:
        outside_prediction = conditional_prediction
    return outside_prediction + (inside_prediction - outside_prediction) * mask
