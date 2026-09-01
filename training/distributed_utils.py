import os

import torch
import torch.distributed as dist


def _visible_cuda_count():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return torch.cuda.device_count()
    return len([item for item in visible.split(",") if item.strip()])


def launch_training(args, worker_fn):
    """Launch one worker process.

    Multi-GPU training is expected to be launched by torchrun, which provides
    RANK, LOCAL_RANK, and WORLD_SIZE. A plain python invocation remains valid
    for single-process runs.
    """
    if "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ["WORLD_SIZE"])
        if rank == 0:
            print(
                f"[INFO] torchrun detected: world_size={world_size}, "
                f"local_rank={local_rank}"
            )
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        if _visible_cuda_count() > 1:
            print(
                "[INFO] Multiple GPUs are visible, but torchrun was not used. "
                "Running a single process on local_rank=0."
            )
        else:
            print("[INFO] Running single-process training.")

    try:
        worker_fn(rank, world_size, args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def init_distributed_device(rank, world_size):
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NCCL distributed training.")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside available CUDA device count "
            f"({torch.cuda.device_count()})."
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return local_rank, torch.device("cuda", local_rank)
