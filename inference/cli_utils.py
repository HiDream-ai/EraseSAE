import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from model.checkpoint_io import resolve_layer_checkpoint


PROMPT_COLUMNS = ("prompt", "text", "caption", "description")


@dataclass(frozen=True)
class PromptRecord:
    prompt: str
    seed: Optional[int] = None
    identity: Optional[str] = None


def validated_target_kernels(
    identity_map,
    target_identity,
    assigned_start,
    assigned_end,
    score_threshold=1e-8,
):
    """Load only positively scored kernels from the requested partition."""
    if target_identity not in identity_map:
        raise RuntimeError(
            f"Target {target_identity!r} is absent from the attribution map."
        )

    entry = identity_map[target_identity]
    kernels = entry.get("top_kernels", [])
    scores = entry.get("scores")
    if scores is None:
        raise RuntimeError(
            "This is a legacy attribution map without validated kernel scores. "
            "Rerun training/attribution with the current code."
        )
    if len(kernels) != len(scores):
        raise RuntimeError("Attribution map kernel and score lengths do not match.")

    selected = [
        int(kernel)
        for kernel, score in zip(kernels, scores)
        if assigned_start <= int(kernel) < assigned_end
        and math.isfinite(float(score))
        and float(score) > score_threshold
    ]
    if not selected:
        raise RuntimeError(
            f"No positive validated kernels found for {target_identity!r} "
            f"inside [{assigned_start}, {assigned_end})."
        )
    return selected


def activation_references_for_kernels(identity_map, target_identity, kernels):
    """Return positive held-out mean activations aligned to selected kernels."""
    entry = identity_map.get(target_identity)
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"Target {target_identity!r} is absent from the attribution map."
        )
    mapped_kernels = entry.get("top_kernels", [])
    references = entry.get("avg_activation")
    if references is None or len(mapped_kernels) != len(references):
        raise RuntimeError(
            f"Attribution activation references are invalid for "
            f"{target_identity!r}."
        )
    by_kernel = {
        int(kernel): float(reference)
        for kernel, reference in zip(mapped_kernels, references)
    }
    try:
        selected = [by_kernel[int(kernel)] for kernel in kernels]
    except KeyError as exc:
        raise RuntimeError(
            f"A selected kernel has no activation reference for "
            f"{target_identity!r}."
        ) from exc
    if not all(math.isfinite(value) and value > 0.0 for value in selected):
        raise RuntimeError(
            f"Activation references must be finite and positive for "
            f"{target_identity!r}."
        )
    return selected


def ordered_concepts_from_map(identity_map):
    """Recover the training partition order from an attribution map."""
    if not isinstance(identity_map, dict) or not identity_map:
        raise RuntimeError("Attribution map must be a non-empty dictionary.")

    partitions = []
    for concept, entry in identity_map.items():
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"Attribution entry for {concept!r} must be a dictionary."
            )
        assigned_range = entry.get("assigned_range")
        if (
            not isinstance(assigned_range, (list, tuple))
            or len(assigned_range) != 2
        ):
            raise RuntimeError(
                f"Attribution entry for {concept!r} has no valid assigned_range."
            )
        start, end = (int(value) for value in assigned_range)
        if start < 0 or end <= start:
            raise RuntimeError(
                f"Attribution entry for {concept!r} has invalid assigned_range "
                f"{assigned_range!r}."
            )
        partitions.append((start, end, str(concept)))

    partitions.sort()
    partition_size = partitions[0][1] - partitions[0][0]
    expected_start = 0
    for start, end, concept in partitions:
        if start != expected_start or end - start != partition_size:
            raise RuntimeError(
                "Attribution map partitions must be contiguous, equally sized, "
                f"and start at zero; {concept!r} uses [{start}, {end})."
            )
        expected_start = end
    return [concept for _, _, concept in partitions]


def prompt_mentions_identity(prompt, target_identity, identity_aliases):
    """Match an explicit identity alias without accepting partial words."""
    if target_identity not in identity_aliases:
        raise KeyError(f"No prompt aliases configured for {target_identity!r}.")
    normalized_prompt = re.sub(r"\s+", " ", prompt.lower()).strip()
    return any(
        re.search(rf"(?<!\w){re.escape(alias.lower())}(?!\w)", normalized_prompt)
        for alias in identity_aliases[target_identity]
    )


def _record_from_mapping(
    record: Dict[str, object],
    source: Path,
    line_number: int,
) -> PromptRecord:
    normalized = {str(key).strip().lower(): value for key, value in record.items()}
    prompt = None
    for column in PROMPT_COLUMNS:
        value = normalized.get(column)
        if value is not None and str(value).strip():
            prompt = str(value).strip()
            break
    if prompt is None:
        raise ValueError(
            f"No prompt column found in {source} at row {line_number}. "
            f"Expected one of: {', '.join(PROMPT_COLUMNS)}."
        )

    seed_value = normalized.get("seed")
    seed = None
    if seed_value is not None and str(seed_value).strip():
        try:
            seed = int(seed_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid seed in {source} at row {line_number}: {seed_value!r}"
            ) from error

    identity_value = normalized.get("identity")
    identity = None
    if identity_value is not None and str(identity_value).strip():
        identity = str(identity_value).strip().lower()
    return PromptRecord(prompt=prompt, seed=seed, identity=identity)


def load_prompt_records(
    prompt: Optional[str],
    prompts_file: Optional[str],
    default_prompt: str,
    max_prompts: Optional[int] = None,
) -> List[PromptRecord]:
    if prompt and prompts_file:
        raise ValueError("Use either --prompt or --prompts-file, not both.")

    if prompt:
        records = [PromptRecord(prompt=prompt.strip())]
    elif prompts_file:
        source = Path(prompts_file)
        if not source.is_file():
            raise FileNotFoundError(f"Prompt file not found: {source}")

        suffix = source.suffix.lower()
        if suffix == ".csv":
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ValueError(f"CSV file has no header: {source}")
                records = [
                    _record_from_mapping(record, source, row_number)
                    for row_number, record in enumerate(reader, start=2)
                ]
        elif suffix == ".jsonl":
            records = []
            with source.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"Expected a JSON object in {source} at line {line_number}."
                        )
                    records.append(_record_from_mapping(record, source, line_number))
        else:
            with source.open("r", encoding="utf-8") as handle:
                records = [
                    PromptRecord(prompt=line.strip())
                    for line in handle
                    if line.strip()
                ]
    else:
        records = [PromptRecord(prompt=default_prompt)]

    records = [record for record in records if record.prompt]
    if max_prompts is not None:
        if max_prompts <= 0:
            raise ValueError("--max-prompts must be greater than zero.")
        records = records[:max_prompts]
    if not records:
        raise ValueError("No prompts were loaded.")
    return records


def resolve_checkpoint(
    checkpoint_dir: str,
    layer_name: str,
    checkpoint: Optional[str] = None,
) -> Path:
    return resolve_layer_checkpoint(
        checkpoint_dir=checkpoint_dir,
        layer_name=layer_name,
        checkpoint=checkpoint,
    )


def _bundle_member(root: Path, value, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Attribution bundle has no valid {description} path.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(
            f"Attribution bundle {description} must be relative to its run directory."
        )
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Attribution bundle {description} not found: {path}")
    return path


def resolve_attributed_bundle(checkpoint_dir, layer_name, checkpoint=None):
    """Resolve the newest run that completed both training and attribution."""
    layer_slug = layer_name.replace(".", "_")
    bundle_name = f"bundle_{layer_slug}.json"
    explicit_checkpoint = None
    if checkpoint is not None:
        explicit_checkpoint = Path(checkpoint)
        if not explicit_checkpoint.is_file():
            raise FileNotFoundError(
                f"SAE checkpoint not found: {explicit_checkpoint}"
            )
        candidates = [explicit_checkpoint.parent]
    else:
        root = Path(checkpoint_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {root}")
        candidates = [root]
        candidates.extend(
            sorted(
                (path for path in root.iterdir() if path.is_dir()),
                key=lambda path: path.name,
                reverse=True,
            )
        )

    for run_dir in candidates:
        bundle_path = run_dir / bundle_name
        if not bundle_path.is_file():
            continue
        try:
            payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid attribution bundle: {bundle_path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid attribution bundle: {bundle_path}")
        if payload.get("layer") != layer_name:
            raise RuntimeError(
                f"Attribution bundle layer mismatch in {bundle_path}: "
                f"expected {layer_name!r}, found {payload.get('layer')!r}."
            )

        bundled_checkpoint = _bundle_member(
            run_dir,
            payload.get("checkpoint"),
            "checkpoint",
        )
        attribution_map = _bundle_member(
            run_dir,
            payload.get("attribution_map"),
            "attribution map",
        )
        normalization_stats = _bundle_member(
            run_dir,
            payload.get("normalization_stats"),
            "normalization statistics",
        )
        if (
            explicit_checkpoint is not None
            and bundled_checkpoint.resolve() != explicit_checkpoint.resolve()
        ):
            raise RuntimeError(
                "The explicit checkpoint does not match the checkpoint validated "
                f"by {bundle_path}."
            )
        return {
            "checkpoint": bundled_checkpoint,
            "attribution_map": attribution_map,
            "normalization_stats": normalization_stats,
            "bundle": bundle_path,
            "payload": payload,
        }

    location = explicit_checkpoint.parent if explicit_checkpoint else Path(checkpoint_dir)
    raise FileNotFoundError(
        f"No completed attribution bundle {bundle_name!r} was found under {location}."
    )


def output_video_path(
    output_dir: str,
    index: int,
    prompt: str,
    target: str,
    seed: int,
    variant: str = "erased",
) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:64]
    if not slug:
        slug = "prompt"
    target_slug = re.sub(r"[^a-z0-9]+", "_", target.lower()).strip("_") or "target"
    variant_slug = re.sub(r"[^a-z0-9]+", "_", variant.lower()).strip("_")
    return (
        Path(output_dir)
        / f"{index:04d}_{target_slug}_{variant_slug}_{slug}_seed{seed}.mp4"
    )
