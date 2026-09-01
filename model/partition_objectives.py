import torch
import torch.nn.functional as F


def get_partition_schedule(epoch, total_epochs, leak_start_fraction=0.3):
    """Return stable reconstruction, separation, leakage, and temporal ramps."""
    if total_epochs < 1:
        raise ValueError("total_epochs must be at least 1.")
    if not 0.0 <= leak_start_fraction < 1.0:
        raise ValueError("leak_start_fraction must be in [0, 1).")

    partition_start = int(0.1 * total_epochs)
    temporal_start = int(0.2 * total_epochs)
    leak_start = int(leak_start_fraction * total_epochs)
    if epoch < partition_start:
        return {
            "partition": 0.0,
            "face_recon": 0.0,
            "leak": 0.0,
            "temp": 0.0,
        }

    partition_progress = (epoch - partition_start) / max(
        total_epochs - partition_start,
        1,
    )
    partition_progress = min(1.0, max(0.0, partition_progress))

    if epoch < temporal_start:
        temporal_progress = 0.0
    else:
        temporal_progress = (epoch - temporal_start) / max(
            total_epochs - temporal_start,
            1,
        )
        temporal_progress = min(0.5, max(0.0, temporal_progress))

    if epoch < leak_start:
        leak_progress = 0.0
    else:
        leak_progress = (epoch - leak_start) / max(total_epochs - leak_start, 1)
        leak_progress = min(1.0, max(0.0, leak_progress))

    return {
        "partition": partition_progress ** 0.5,
        "face_recon": partition_progress,
        "leak": leak_progress,
        "temp": temporal_progress,
    }


def partition_contrastive_loss(
    z_id_pre,
    labels,
    target_mask,
    n_id_per_partition,
    k_id,
    margin=0.1,
):
    """Make each target sample score its assigned partition above all others."""
    if z_id_pre.ndim != 4:
        raise ValueError("z_id_pre must have shape [B*T, C, H, W].")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [B].")
    if target_mask.ndim != 5 or target_mask.shape[1] != 1:
        raise ValueError("target_mask must have shape [B, 1, T, H, W].")

    batch_size = labels.shape[0]
    folded_batch, total_channels, height, width = z_id_pre.shape
    frames = target_mask.shape[2]
    if folded_batch != batch_size * frames:
        raise ValueError(
            f"Expected {batch_size * frames} folded rows, got {folded_batch}."
        )
    if target_mask.shape[-2:] != (height, width):
        raise ValueError("target_mask and z_id_pre spatial shapes do not match.")
    if total_channels % n_id_per_partition != 0:
        raise ValueError("ID channels are not divisible by the partition size.")

    num_partitions = total_channels // n_id_per_partition
    valid_rows = torch.nonzero(labels >= 0, as_tuple=True)[0]
    if valid_rows.numel() == 0 or num_partitions < 2:
        return z_id_pre.sum() * 0.0
    if (labels[valid_rows] >= num_partitions).any():
        raise ValueError("A label is outside the configured partition range.")

    selected_k = min(int(k_id), int(n_id_per_partition))
    if selected_k < 1:
        raise ValueError("k_id must be at least 1.")

    folded_mask = (
        target_mask.permute(0, 2, 1, 3, 4)
        .reshape(folded_batch, 1, height, width)
        .float()
    )
    mask_area = folded_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    partition_scores = []
    for partition_index in range(num_partitions):
        start = partition_index * n_id_per_partition
        partition_pre = z_id_pre[:, start:start + n_id_per_partition]
        strengths = partition_pre.flatten(2).amax(dim=-1)
        top_indices = strengths.topk(selected_k, dim=1, sorted=False).indices
        selected = torch.gather(
            partition_pre,
            1,
            top_indices[:, :, None, None].expand(-1, -1, height, width),
        )
        score = (
            F.relu(selected).float() * folded_mask
        ).sum(dim=(1, 2, 3)) / (mask_area * selected_k)
        partition_scores.append(score)

    scores = torch.stack(partition_scores, dim=1).view(
        batch_size,
        frames,
        num_partitions,
    ).mean(dim=1)
    valid_scores = scores[valid_rows]
    valid_labels = labels[valid_rows]
    owner_score = valid_scores.gather(1, valid_labels[:, None]).squeeze(1)
    competitors = valid_scores.clone()
    competitors.scatter_(1, valid_labels[:, None], -torch.inf)
    strongest_other = competitors.amax(dim=1)
    return F.relu(float(margin) + strongest_other - owner_score).mean()


def partition_leakage_loss(
    z_id_pre,
    labels,
    target_mask,
    n_id_per_partition,
    k_id,
    margin=0.0,
):
    """Penalize target-partition leakage without materializing all activations."""
    if z_id_pre.ndim != 4:
        raise ValueError("z_id_pre must have shape [B*T, C, H, W].")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [B].")
    if target_mask.ndim != 5 or target_mask.shape[1] != 1:
        raise ValueError("target_mask must have shape [B, 1, T, H, W].")

    batch_size = labels.shape[0]
    folded_batch, total_channels, height, width = z_id_pre.shape
    if target_mask.shape[0] != batch_size:
        raise ValueError("labels and target_mask batch sizes do not match.")
    if target_mask.shape[-2:] != (height, width):
        raise ValueError("target_mask and z_id_pre spatial shapes do not match.")
    if total_channels % n_id_per_partition != 0:
        raise ValueError("ID channels are not divisible by the partition size.")

    frames = target_mask.shape[2]
    if folded_batch != batch_size * frames:
        raise ValueError(
            f"Expected {batch_size * frames} folded rows, got {folded_batch}."
        )

    num_partitions = total_channels // n_id_per_partition
    if (labels >= num_partitions).any():
        raise ValueError("A label is outside the configured partition range.")

    folded_labels = labels.repeat_interleave(frames)
    folded_mask = (
        target_mask.permute(0, 2, 1, 3, 4)
        .reshape(folded_batch, 1, height, width)
        .bool()
    )
    selected_k = min(int(k_id), int(n_id_per_partition))
    if selected_k < 1:
        raise ValueError("k_id must be at least 1.")

    violation_sum = torch.zeros((), dtype=torch.float32, device=z_id_pre.device)
    penalty_count = torch.zeros((), dtype=torch.float32, device=z_id_pre.device)

    for partition_index in range(num_partitions):
        start = partition_index * n_id_per_partition
        end = start + n_id_per_partition
        partition_pre = z_id_pre[:, start:end]
        strengths = partition_pre.flatten(2).amax(dim=-1)
        top_indices = torch.topk(
            strengths,
            k=selected_k,
            dim=1,
            sorted=False,
        ).indices
        selected = torch.gather(
            partition_pre,
            1,
            top_indices[:, :, None, None].expand(-1, -1, height, width),
        )

        is_owner = (folded_labels == partition_index).view(-1, 1, 1, 1)
        is_other_target = (
            (folded_labels >= 0) & (folded_labels != partition_index)
        ).view(-1, 1, 1, 1)
        is_common = (folded_labels < 0).view(-1, 1, 1, 1)

        # A partition learns background suppression from its own examples and
        # all common prompts. Other celebrities only provide cross-identity
        # negatives inside their target masks; repeating their full background
        # as a negative for every partition overweights leakage supervision.
        invalid_region = (
            is_common
            | (is_owner & ~folded_mask)
            | (is_other_target & folded_mask)
        )
        violations = F.relu(F.relu(selected).float() - float(margin))
        violation_sum = violation_sum + (
            violations * invalid_region.to(violations.dtype)
        ).sum()
        penalty_count = penalty_count + (
            invalid_region.sum(dtype=torch.float32) * n_id_per_partition
        )

    return violation_sum / penalty_count.clamp_min(1.0)
