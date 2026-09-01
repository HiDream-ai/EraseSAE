import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self):
        return self.streams[0].fileno()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _run_id():
    return (
        os.environ.get("ERASESAE_RUN_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or time.strftime("%Y%m%d_%H%M%S")
    )


def setup_run_logging(args, script_name, rank):
    log_dir = Path(getattr(args, "log_dir", None) or Path(args.sae_save_dir) / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = getattr(args, "run_id", None) or _run_id()
    args.run_id = run_id
    args.log_dir = str(log_dir)

    log_path = log_dir / f"{run_id}_{script_name}_rank{rank}.log"
    log_file = log_path.open("a", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    if rank == 0:
        config_path = log_dir / f"{run_id}_{script_name}_config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(
                {k: _jsonable(v) for k, v in vars(args).items() if not k.startswith("_")},
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"[Log] stdout/stderr: {log_path}")
        print(f"[Log] config: {config_path}")
    return log_path


def append_jsonl(args, filename, record):
    log_dir = Path(getattr(args, "log_dir", None) or Path(args.sae_save_dir) / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = getattr(args, "run_id", None) or _run_id()
    path = log_dir / f"{run_id}_{filename}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")


def log_train_metrics(args, layer_name, epoch, step, metrics):
    record = {
        "phase": "train",
        "layer": layer_name,
        "epoch": epoch,
        "step": step,
        **metrics,
    }
    append_jsonl(args, "train_metrics", record)


def distributed_mean_metrics(metrics, device):
    """Return detached scalar metrics averaged over all training ranks."""
    names = list(metrics)
    values = []
    for name in names:
        value = metrics[name]
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Metric {name!r} must be scalar.")
            value = value.detach().float().to(device)
        else:
            value = torch.tensor(float(value), dtype=torch.float32, device=device)
        values.append(value)

    packed = torch.stack(values)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed /= dist.get_world_size()
    return {
        name: float(value)
        for name, value in zip(names, packed.cpu().tolist())
    }


def require_finite_across_ranks(value, name, device):
    """Make every rank fail together when one rank observes a non-finite scalar."""
    if not torch.is_tensor(value) or value.numel() != 1:
        raise ValueError(f"{name} must be a scalar tensor.")
    finite = torch.tensor(
        [int(torch.isfinite(value.detach()).item())],
        dtype=torch.int32,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError(
            f"Non-finite {name} observed on at least one distributed rank."
        )


def log_loader_summary(args, phase, layer_name, loader, rank):
    if rank != 0:
        return
    dataset_size = len(getattr(loader, "dataset", []))
    timesteps_per_record = getattr(loader, "timesteps_per_video", 1)
    batches = len(loader) if hasattr(loader, "__len__") else None
    concepts = getattr(loader, "concepts", None)
    capture_steps_fn = getattr(loader, "_capture_step_indices", None)
    capture_step_indices = (
        sorted(int(index) for index in capture_steps_fn())
        if callable(capture_steps_fn)
        else None
    )
    record = {
        "phase": phase,
        "layer": layer_name,
        "records": dataset_size,
        "timesteps_per_record": timesteps_per_record,
        "activation_samples": dataset_size * timesteps_per_record,
        "batches_per_epoch": batches,
        "batch_size": getattr(args, "batch_size", None),
        "max_samples_per_subject": getattr(args, "max_samples_per_subject", None),
        "concepts": concepts,
        "sampling_mode": getattr(loader, "sampling_mode", None),
        "trajectory_steps": getattr(loader, "trajectory_steps", None),
        "capture_step_indices": capture_step_indices,
    }
    print(
        f"[Online] {phase}: records={record['records']}, "
        f"timesteps_per_record={record['timesteps_per_record']}, "
        f"activation_samples={record['activation_samples']}, "
        f"batches_per_epoch={record['batches_per_epoch']}, "
        f"batch_size={record['batch_size']}, "
        f"capture_steps={record['capture_step_indices']}"
    )
    if concepts is not None:
        print(f"[Online] {phase} concepts: {concepts}")
    append_jsonl(args, "stage_events", record)


def init_wandb(args, run_name):
    mode = getattr(args, "wandb_mode", "disabled")
    if mode == "disabled":
        if not getattr(args, "_wandb_disabled_printed", False):
            print("[W&B] disabled; using local logs only.")
            args._wandb_disabled_printed = True
        return None

    try:
        import wandb

        if wandb.run is None:
            run_id = getattr(args, "run_id", None)
            display_name = f"{run_name}_{run_id}" if run_id else run_name
            wandb.init(
                project=getattr(args, "wandb_name", "EraseSAE"),
                name=display_name,
                mode=mode,
            )
        return wandb
    except Exception as exc:
        print(f"[W&B] disabled after init failure: {exc}")
        args.wandb_mode = "disabled"
        return None


def log_wandb(args, metrics):
    if getattr(args, "wandb_mode", "disabled") == "disabled":
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics)
    except Exception as exc:
        print(f"[W&B] log skipped after failure: {exc}")
        args.wandb_mode = "disabled"
