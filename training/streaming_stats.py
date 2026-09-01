import torch


def update_channel_moments(
    activation,
    data_dim,
    sums,
    squared_sums,
    count,
    chunk_rows=2048,
):
    """Accumulate float64 channel moments without a full-size float64 copy."""
    rows = activation.reshape(-1, data_dim)
    for start in range(0, rows.shape[0], chunk_rows):
        chunk = rows[start:start + chunk_rows].to(torch.float64)
        sums.add_(chunk.sum(dim=0))
        squared_sums.add_((chunk * chunk).sum(dim=0))
        count.add_(chunk.shape[0])

