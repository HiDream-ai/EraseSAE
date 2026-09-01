import math

import torch
import torch.nn.functional as F


def binarize_attention_mask(mask, quantile=0.8, dilation=0):
    """Threshold each frame independently and optionally expand its support."""
    if mask.ndim < 2:
        raise ValueError("attention mask must have at least two spatial dimensions.")
    if not 0.0 <= quantile < 1.0:
        raise ValueError("attention mask quantile must be in [0, 1).")
    if int(dilation) != dilation or dilation < 0:
        raise ValueError("attention mask dilation must be a non-negative integer.")

    flat = mask.float().flatten(start_dim=-2)
    threshold = torch.quantile(flat, quantile, dim=-1, keepdim=True)
    has_spatial_variation = (
        flat.amax(dim=-1, keepdim=True) - flat.amin(dim=-1, keepdim=True)
    ) > 1e-8
    binary = (flat > threshold) & has_spatial_variation
    binary = binary.reshape(mask.shape).to(torch.float32)

    dilation = int(dilation)
    if dilation:
        height, width = binary.shape[-2:]
        binary = F.max_pool2d(
            binary.reshape(-1, 1, height, width),
            kernel_size=2 * dilation + 1,
            stride=1,
            padding=dilation,
        ).reshape(binary.shape)
    return binary


def compute_cross_attention_mask(
    query,
    key,
    token_idx,
    head_num,
    text_len,
    height,
    width,
    text_first,
    query_chunk_size=64,
    key_chunk_size=4096,
    use_fp16=True,
):
    """Compute exact selected-token attention without materializing S x S."""

    batch_size, query_length, inner_dim = query.shape
    _, key_length, key_inner_dim = key.shape
    if query_length != key_length:
        raise ValueError("query and key sequence lengths do not match.")
    if inner_dim != key_inner_dim or inner_dim % head_num != 0:
        raise ValueError(
            f"inner_dim={inner_dim} must match and be divisible by heads={head_num}."
        )

    visual_len = query_length - text_len
    if visual_len <= 0:
        raise ValueError(f"visual_len must be positive, got {visual_len}.")
    spatial_size = height * width
    if visual_len % spatial_size != 0:
        raise ValueError(
            f"visual_len={visual_len} is not divisible by height*width={spatial_size}."
        )

    token_idx = torch.as_tensor(
        token_idx,
        device=query.device,
        dtype=torch.long,
    )
    if token_idx.numel() == 0:
        raise ValueError("token_idx must contain at least one text token.")
    if token_idx.min() < 0 or token_idx.max() >= text_len:
        raise IndexError("A target token index is outside the text sequence.")

    head_dim = inner_dim // head_num
    q = query.view(batch_size, query_length, head_num, head_dim).permute(0, 2, 1, 3)
    k = key.view(batch_size, key_length, head_num, head_dim).permute(0, 2, 1, 3)
    if use_fp16:
        q = q.half()
        k = k.half()

    if text_first:
        visual_start = text_len
        selected_key_indices = token_idx
    else:
        visual_start = 0
        selected_key_indices = visual_len + token_idx
    selected_keys = k.index_select(2, selected_key_indices)
    scale = 1.0 / math.sqrt(head_dim)

    attention_chunks = []
    for query_start in range(0, visual_len, query_chunk_size):
        query_end = min(query_start + query_chunk_size, visual_len)
        q_chunk = q[
            :,
            :,
            visual_start + query_start:visual_start + query_end,
            :,
        ]

        chunk_shape = q_chunk.shape[:-1]
        running_max = torch.full(
            chunk_shape,
            -torch.inf,
            dtype=q_chunk.dtype,
            device=q_chunk.device,
        )
        normalizer = torch.zeros_like(running_max)

        for key_start in range(0, key_length, key_chunk_size):
            key_end = min(key_start + key_chunk_size, key_length)
            scores = (
                torch.matmul(
                    q_chunk,
                    k[:, :, key_start:key_end, :].transpose(-1, -2),
                )
                * scale
            )
            chunk_max = scores.amax(dim=-1)
            new_max = torch.maximum(running_max, chunk_max)
            normalizer = (
                normalizer * torch.exp(running_max - new_max)
                + torch.exp(scores - new_max.unsqueeze(-1)).sum(dim=-1)
            )
            running_max = new_max
            del scores, chunk_max, new_max

        log_denom = running_max + normalizer.log()
        selected_scores = torch.matmul(
            q_chunk,
            selected_keys.transpose(-1, -2),
        ) * scale
        selected_probability = torch.exp(
            selected_scores - log_denom.unsqueeze(-1)
        ).sum(dim=-1)
        attention_chunks.append(selected_probability.mean(dim=1))

    attention = torch.cat(attention_chunks, dim=1)
    frames = visual_len // spatial_size
    attention = attention.reshape(batch_size, frames, height, width)
    minimum = attention.amin(dim=(2, 3), keepdim=True)
    maximum = attention.amax(dim=(2, 3), keepdim=True)
    return (attention - minimum) / (maximum - minimum).clamp_min(1e-6)
