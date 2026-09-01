import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from model.checkpoint_io import checkpoint_pointer_path


class ReplayedBatchLoader:
    """Repeat one streamed batch in memory before requesting the next batch."""

    def __init__(self, loader, reuse):
        self.loader = loader
        self.reuse = reuse

    @property
    def sampler(self):
        return self.loader.sampler

    def __len__(self):
        return len(self.loader) * self.reuse

    def __iter__(self):
        for batch in self.loader:
            for _ in range(self.reuse):
                yield batch


@dataclass(frozen=True)
class ReplayPlan:
    base_batches: int
    batch_reuse: int
    total_updates: int
    generation_passes: int

    def effective_epoch(self, completed_updates):
        return completed_updates / self.base_batches


def build_replay_loader(args, data_loader):
    reuse = 1
    if getattr(args, "activation_source", "offline") == "online":
        reuse = int(getattr(args, "online_batch_reuse", 1))
    if reuse < 1:
        raise ValueError("online_batch_reuse must be at least 1.")

    base_batches = len(data_loader)
    if base_batches < 1:
        raise ValueError("The activation loader contains no batches.")

    total_updates = int(args.sae_epochs) * base_batches
    plan = ReplayPlan(
        base_batches=base_batches,
        batch_reuse=reuse,
        total_updates=total_updates,
        generation_passes=math.ceil(total_updates / (base_batches * reuse)),
    )
    return ReplayedBatchLoader(data_loader, reuse), plan


def set_loader_epoch(loader, generation_pass):
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(generation_pass)


def prepare_attribution_batches(args, data_loader, rank):
    """Build an exact, deterministic attribution pass after online training."""
    if getattr(args, "activation_source", "offline") != "online":
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        if world_size <= 1:
            return iter(data_loader), len(data_loader)
        if not hasattr(data_loader, "dataset") or data_loader.batch_size is None:
            raise TypeError(
                "Offline attribution requires a DataLoader with a dataset and "
                "an explicit batch_size."
            )
        exact_indices = list(range(rank, len(data_loader.dataset), world_size))
        exact_loader = DataLoader(
            Subset(data_loader.dataset, exact_indices),
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            drop_last=False,
        )
        if rank == 0:
            print(
                "[Offline] Attribution disables DistributedSampler padding: "
                f"local_batches={len(exact_loader)}"
            )
        return iter(exact_loader), len(exact_loader)

    attribution_epoch = int(args.sae_epochs)
    set_loader_epoch(data_loader, attribution_epoch)
    if not hasattr(data_loader, "iter_batches") or not hasattr(
        data_loader, "batch_count"
    ):
        raise TypeError(
            "Online activation loaders must expose iter_batches() and "
            "batch_count()."
        )

    batch_count = data_loader.batch_count(pad=False)
    if rank == 0:
        print(
            "[Online] Attribution uses an unseen latent trajectory: "
            f"sampler_epoch={attribution_epoch}, local_batches={batch_count}, "
            "DDP_padding=disabled"
        )
    return data_loader.iter_batches(pad=False), batch_count


def save_final_checkpoint(
    args,
    model,
    layer_name,
    rank,
    checkpoint_type="conv_sae",
):
    """Persist the exact model state used by post-training attribution."""
    if rank != 0:
        return None

    final_epoch = int(args.sae_epochs) - 1
    if final_epoch < 0:
        raise ValueError("sae_epochs must be at least 1.")

    output_dir = Path(args.sae_save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_slug = layer_name.replace(".", "_")
    save_path = output_dir / f"{checkpoint_type}_{layer_slug}_{final_epoch}.pt"
    temporary_path = save_path.with_suffix(f"{save_path.suffix}.tmp")
    torch.save(model.state_dict(), temporary_path)
    temporary_path.replace(save_path)

    pointer_path = checkpoint_pointer_path(
        output_dir,
        layer_name,
        checkpoint_type,
    )
    pointer_temporary_path = pointer_path.with_suffix(
        f"{pointer_path.suffix}.tmp"
    )
    pointer_temporary_path.write_text(
        json.dumps(
            {
                "checkpoint": save_path.name,
                "epoch": final_epoch,
                "model_config": {
                    key: value
                    for key, value in {
                        "class_name": type(model).__name__,
                        "d_model": getattr(model, "d_model", None),
                        "n_gen": getattr(model, "n_gen", None),
                        "n_id": getattr(model, "n_id", None),
                        "n_id_per_concept": getattr(
                            model, "n_id_per_celeb", None
                        ),
                        "k_gen": getattr(
                            getattr(model, "act_gen", None), "k", None
                        ),
                        "k_id": getattr(model, "k_id", None),
                        "concepts": list(getattr(model, "celebs", [])),
                    }.items()
                    if value is not None
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pointer_temporary_path.replace(pointer_path)
    print(f"[Checkpoint] Final SAE saved to: {save_path}")
    return save_path
