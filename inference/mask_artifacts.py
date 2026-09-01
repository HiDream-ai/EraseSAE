from pathlib import Path

import torch
from PIL import Image


def capture_effective_mask(pipe, step_index):
    """Copy the final spatial mask used by masked CFG into a PNG-ready tensor."""
    effective_mask = getattr(pipe, "_current_effective_sae_mask", None)
    if effective_mask is None:
        raise RuntimeError(
            "The pipeline did not expose the effective masked-CFG mask."
        )
    if effective_mask.ndim != 5 or effective_mask.shape[1] != 1:
        raise RuntimeError(
            "Expected effective masked-CFG mask shape [B, 1, F, H, W], "
            f"got {tuple(effective_mask.shape)}."
        )
    if effective_mask.shape[0] != 1:
        raise RuntimeError(
            "Step-mask PNG output currently requires inference batch size 1."
        )

    quantized_mask = (
        effective_mask.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(device="cpu", dtype=torch.uint8)
    )
    return {
        "step_index": int(step_index),
        "effective_mask": quantized_mask,
    }


def _render_effective_mask_grid(snapshot, output_path):
    mask = snapshot["effective_mask"]
    if mask.ndim != 5 or mask.shape[0] != 1 or mask.shape[1] != 1:
        raise ValueError(
            "Effective mask snapshots must have shape [1, 1, F, H, W]."
        )
    if mask.dtype != torch.uint8:
        raise ValueError("Effective mask snapshots must be uint8 tensors.")

    frames = mask[0, 0]
    frame_count, height, width = frames.shape
    if frame_count < 1:
        raise ValueError("An effective mask must contain at least one frame.")

    columns = min(8, frame_count)
    rows = (frame_count + columns - 1) // columns
    canvas = torch.zeros(
        (rows * height, columns * width),
        dtype=torch.uint8,
    )
    for frame_index, frame in enumerate(frames):
        row, column = divmod(frame_index, columns)
        canvas[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = frame

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.numpy()).save(output_path)


def save_step_mask_artifacts(snapshots, video_path):
    """Write only one final effective-mask PNG for each captured denoising step."""
    if not snapshots:
        raise RuntimeError(
            "--save-step-masks was requested, but no mask snapshots were captured."
        )

    video_path = Path(video_path)
    output_dir = video_path.parent / f"{video_path.stem}_applied_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("step_*.png"):
        stale_path.unlink()

    ordered_snapshots = sorted(snapshots, key=lambda item: item["step_index"])
    step_indices = [int(snapshot["step_index"]) for snapshot in ordered_snapshots]
    if len(step_indices) != len(set(step_indices)):
        raise ValueError("Effective mask snapshots contain duplicate steps.")

    for snapshot in ordered_snapshots:
        step_index = int(snapshot["step_index"])
        _render_effective_mask_grid(
            snapshot,
            output_dir / f"step_{step_index:03d}.png",
        )
    return output_dir
