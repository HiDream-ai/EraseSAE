import json
from pathlib import Path
from typing import Iterable, Optional


def _epoch_candidates(root: Path, file_prefix: str):
    candidates = []
    for path in root.glob(f"{file_prefix}*.pt"):
        epoch_text = path.stem.removeprefix(file_prefix)
        if epoch_text.isdigit():
            candidates.append((int(epoch_text), path))
    return candidates


def checkpoint_pointer_path(
    root: Path,
    layer_name: str,
    checkpoint_type: str = "conv_sae",
) -> Path:
    layer_file_name = layer_name.replace(".", "_")
    return Path(root) / f"latest_{checkpoint_type}_{layer_file_name}.json"


def load_checkpoint_model_config(
    checkpoint: Path,
    layer_name: str,
    checkpoint_type: str = "conv_sae",
):
    """Load portable SAE construction parameters from a final pointer."""
    checkpoint = Path(checkpoint)
    pointer_path = checkpoint_pointer_path(
        checkpoint.parent,
        layer_name,
        checkpoint_type,
    )
    if not pointer_path.is_file():
        return {}

    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid checkpoint pointer: {pointer_path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint pointer: {pointer_path}")
    if payload.get("checkpoint") != checkpoint.name:
        return {}
    model_config = payload.get("model_config", {})
    if not isinstance(model_config, dict):
        raise RuntimeError(
            f"model_config must be a dictionary in checkpoint pointer: {pointer_path}"
        )
    return model_config


def _checkpoint_from_pointer(
    root: Path,
    layer_name: str,
    file_prefix: str,
    checkpoint_type: str,
):
    pointer_path = checkpoint_pointer_path(root, layer_name, checkpoint_type)
    if not pointer_path.is_file():
        return None

    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
        filename = payload["checkpoint"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid checkpoint pointer: {pointer_path}") from exc

    if not isinstance(filename, str) or Path(filename).name != filename:
        raise RuntimeError(
            f"Checkpoint pointer must contain a local filename: {pointer_path}"
        )
    checkpoint = root / filename
    if not checkpoint.name.startswith(file_prefix):
        raise RuntimeError(
            f"Checkpoint pointer does not match layer/type: {pointer_path}"
        )
    epoch_text = checkpoint.stem.removeprefix(file_prefix)
    if not epoch_text.isdigit():
        raise RuntimeError(
            f"Checkpoint pointer has an invalid epoch suffix: {pointer_path}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint pointer target does not exist: {checkpoint}"
        )
    return checkpoint


def _checkpoint_in_directory(root, layer_name, file_prefix, checkpoint_type):
    pointed = _checkpoint_from_pointer(
        root,
        layer_name,
        file_prefix,
        checkpoint_type,
    )
    if pointed is not None:
        return pointed
    candidates = _epoch_candidates(root, file_prefix)
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return None


def resolve_layer_checkpoint(
    checkpoint_dir: str,
    layer_name: str,
    checkpoint: Optional[str] = None,
    checkpoint_type: str = "conv_sae",
) -> Path:
    """Resolve an explicit checkpoint or the latest checkpoint bundle.

    A checkpoint directory may either be a run directory itself or contain
    timestamp-named run directories. A final-checkpoint pointer takes priority
    within a directory; legacy bundles fall back to their largest epoch suffix.
    """
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")

    layer_file_name = layer_name.replace(".", "_")
    file_prefix = f"{checkpoint_type}_{layer_file_name}_"

    direct_checkpoint = _checkpoint_in_directory(
        root,
        layer_name,
        file_prefix,
        checkpoint_type,
    )
    if direct_checkpoint is not None:
        return direct_checkpoint

    for run_dir in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    ):
        run_checkpoint = _checkpoint_in_directory(
            run_dir,
            layer_name,
            file_prefix,
            checkpoint_type,
        )
        if run_checkpoint is not None:
            return run_checkpoint

    raise FileNotFoundError(
        f"No SAE checkpoints matching {file_prefix}<epoch>.pt were found "
        f"under {root} or its immediate run directories."
    )


def resolve_checkpoint_run(
    checkpoint_dir: str,
    layer_names: Iterable[str],
    checkpoint_type: str = "conv_sae",
) -> Path:
    """Select one run directory and require every requested layer in it."""
    layer_names = list(layer_names)
    if not layer_names:
        raise ValueError("At least one target layer is required.")

    first_checkpoint = resolve_layer_checkpoint(
        checkpoint_dir,
        layer_names[0],
        checkpoint_type=checkpoint_type,
    )
    run_dir = first_checkpoint.parent
    for layer_name in layer_names[1:]:
        resolve_layer_checkpoint(
            str(run_dir),
            layer_name,
            checkpoint_type=checkpoint_type,
        )
    return run_dir
