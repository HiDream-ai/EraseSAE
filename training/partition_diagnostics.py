import torch
import torch.distributed as dist

from training.logging_utils import append_jsonl


class PartitionHealthTracker:
    SAMPLE_COUNT = 0
    GLOBAL_HIT_SUM = 1
    ACTIVE_KERNEL_SUM = 2
    MASK_COVERAGE_SUM = 3
    MASKED_ACTIVATION_SUM = 4

    def __init__(self, raw_model, device):
        self.raw_model = raw_model
        self.device = device
        self.values = torch.zeros(
            (len(raw_model.celebs), 5),
            dtype=torch.float64,
            device=device,
        )

    def reset(self):
        self.values.zero_()

    @torch.no_grad()
    def update(self, out, labels, mask_3d):
        batch_size = labels.shape[0]
        frames = mask_3d.shape[2]
        height, width = mask_3d.shape[-2:]
        total_channels = self.raw_model.total_id_kernels

        global_topk = out.get("id_global_topk_indices")
        if global_topk is None:
            raise ValueError(
                "Partition health tracking requires return_global_topk=True."
            )
        global_indices = global_topk.detach().view(
            batch_size,
            frames,
            -1,
        )
        f_id = out["f_id"].detach().view(
            batch_size,
            frames,
            total_channels,
            height,
            width,
        )
        mask = mask_3d.detach().permute(0, 2, 1, 3, 4).float()

        for class_index in range(len(self.raw_model.celebs)):
            rows = torch.nonzero(labels == class_index, as_tuple=True)[0]
            if rows.numel() == 0:
                continue

            start, end = self.raw_model.get_celeb_fields(class_index)
            class_indices = global_indices[rows]
            global_hit = (
                (class_indices >= start) & (class_indices < end)
            ).float().mean(dim=(1, 2))

            class_features = f_id[rows, :, start:end]
            active_kernels = (
                class_features.amax(dim=(1, 3, 4)) > 1e-6
            ).float().sum(dim=1)
            class_mask = mask[rows]
            mask_coverage = class_mask.mean(dim=(1, 2, 3, 4))
            masked_activation = (
                (class_features * class_mask).sum(dim=(1, 2, 3, 4))
                / (
                    class_mask.sum(dim=(1, 2, 3, 4))
                    * self.raw_model.n_id_per_celeb
                    + 1e-6
                )
            )

            self.values[class_index, self.SAMPLE_COUNT] += rows.numel()
            self.values[class_index, self.GLOBAL_HIT_SUM] += global_hit.sum()
            self.values[class_index, self.ACTIVE_KERNEL_SUM] += active_kernels.sum()
            self.values[class_index, self.MASK_COVERAGE_SUM] += mask_coverage.sum()
            self.values[class_index, self.MASKED_ACTIVATION_SUM] += masked_activation.sum()

    def summarize(self, args, layer_name, generation_pass, effective_epoch, rank):
        reduced = self.values.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        if rank != 0:
            return

        chance_hit = (
            self.raw_model.n_id_per_celeb / self.raw_model.total_id_kernels
        )
        for class_index, class_name in enumerate(self.raw_model.celebs):
            count = reduced[class_index, self.SAMPLE_COUNT].item()
            denominator = max(count, 1.0)
            global_hit = (
                reduced[class_index, self.GLOBAL_HIT_SUM].item() / denominator
            )
            active_kernels = (
                reduced[class_index, self.ACTIVE_KERNEL_SUM].item() / denominator
            )
            mask_coverage = (
                reduced[class_index, self.MASK_COVERAGE_SUM].item() / denominator
            )
            masked_activation = (
                reduced[class_index, self.MASKED_ACTIVATION_SUM].item() / denominator
            )
            warning = None
            if count == 0:
                warning = "no samples"
            elif effective_epoch >= 5:
                if mask_coverage <= 1e-8:
                    warning = "target attention mask is empty"
                elif active_kernels < 1.0:
                    warning = "assigned partition has no active kernels"
                elif masked_activation <= 1e-8:
                    warning = "assigned partition is inactive inside target mask"

            record = {
                "phase": "partition_health",
                "layer": layer_name,
                "generation_pass": generation_pass,
                "effective_epoch": effective_epoch,
                "class": class_name,
                "samples": int(count),
                "global_assigned_hit_rate": global_hit,
                "chance_hit_rate": chance_hit,
                "mean_active_assigned_kernels": active_kernels,
                "mean_mask_coverage": mask_coverage,
                "mean_masked_activation": masked_activation,
                "warning": warning,
            }
            print(
                f"[Health] {class_name:15} samples={int(count):4d} "
                f"global_hit={global_hit:6.2%} "
                f"active={active_kernels:6.2f} "
                f"mask={mask_coverage:6.2%} "
                f"activation={masked_activation:.5f}"
                + (f" WARNING={warning}" if warning else "")
            )
            append_jsonl(args, "partition_health", record)
