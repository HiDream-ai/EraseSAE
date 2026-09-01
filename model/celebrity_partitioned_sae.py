import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from training.logging_utils import init_wandb, log_train_metrics, log_wandb
from training.partition_diagnostics import PartitionHealthTracker
from training.replay_utils import (
    build_replay_loader,
    save_final_checkpoint,
    set_loader_epoch,
)
from model.partition_objectives import partition_leakage_loss


# ==========================================
# 1. 核心组件与模型定义
# ==========================================

class TopKConvActivation(nn.Module):
    """
    卷积版 TopK 激活：在空间维度(H,W)上寻找最大响应，以此作为 TopK 的排序依据。
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def topk_indices(self, x, channel_mask=None, k=None):
        latent_strength = x.flatten(2).amax(dim=-1)
        if channel_mask is not None:
            if channel_mask.shape != latent_strength.shape:
                raise ValueError(
                    f"channel_mask shape {channel_mask.shape} does not match "
                    f"latent strength shape {latent_strength.shape}."
                )
            latent_strength = latent_strength.masked_fill(~channel_mask, -torch.inf)

        k_curr = min(self.k if k is None else k, x.shape[1])
        _, topk_indices = torch.topk(
            latent_strength,
            k=k_curr,
            dim=1,
            sorted=False,
        )
        return topk_indices

    def forward(self, x, channel_mask=None, k=None, return_indices=False):
        topk_indices = self.topk_indices(x, channel_mask=channel_mask, k=k)
        latent_strength = x.new_zeros((x.shape[0], x.shape[1]))
        mask = torch.zeros_like(latent_strength)
        mask.scatter_(1, topk_indices, 1.0)
        if channel_mask is not None:
            mask = mask * channel_mask.to(mask.dtype)

        activated = F.relu(x) * mask.view(mask.shape[0], mask.shape[1], 1, 1)
        if return_indices:
            return activated, topk_indices
        return activated

class TopKConvPSAE(nn.Module):
    """
    分区卷积 SAE (Partitioned Convolutional SAE) - TopK 版本
    包含：通用分支(Gen) + 身份分支(Id)
    """
    def __init__(self, d_model, celebs, n_gen=128, n_id_per_celeb=16, k_gen=32, k_id=8):
        super().__init__()
        self.d_model = d_model
        self.n_gen = n_gen
        self.celebs = celebs
        self.n_id_per_celeb = n_id_per_celeb
        self.k_id = k_id
        self.total_id_kernels = n_id_per_celeb * len(celebs)
        self.total_k_id_kernels = k_id * len(celebs)

        # --- Encoder ---
        self.encoder_gen = nn.Conv2d(d_model, n_gen, 3, padding=1)
        self.encoder_id = nn.Conv2d(d_model, self.total_id_kernels, 3, padding=1)
        self.b_enc_gen = nn.Parameter(torch.zeros(n_gen))
        self.b_enc_id = nn.Parameter(torch.zeros(self.total_id_kernels))

        # --- Decoder ---
        self.decoder_gen = nn.Conv2d(n_gen, d_model, 1, bias=False)
        self.decoder_id = nn.Conv2d(self.total_id_kernels, d_model, 1, bias=False)

        # Global Decoder Bias (Geometric Median 将被赋值于此)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # --- Activations ---
        self.act_gen = TopKConvActivation(k_gen)
        self.act_id = TopKConvActivation(k_id)

    def get_celeb_fields(self, label):
        """获取特定名人在 ID 通道中的索引范围"""
        start = label * self.n_id_per_celeb
        return start, start + self.n_id_per_celeb

    def _label_channel_mask(self, labels):
        if labels.ndim != 1 or labels.shape[0] == 0:
            raise ValueError("labels must be a non-empty 1D tensor.")
        if labels.shape[0] and (labels >= len(self.celebs)).any():
            raise ValueError("A training label is outside the configured concept range.")

        allowed = torch.zeros(
            (labels.shape[0], self.total_id_kernels),
            dtype=torch.bool,
            device=labels.device,
        )
        valid_rows = torch.nonzero(labels >= 0, as_tuple=True)[0]
        if valid_rows.numel() > 0:
            offsets = torch.arange(self.n_id_per_celeb, device=labels.device)
            channels = labels[valid_rows, None] * self.n_id_per_celeb + offsets[None]
            allowed[valid_rows[:, None], channels] = True
        return allowed

    def _activate_all_id_partitions(self, z_id_pre_scaled):
        """Apply TopK independently inside every configured identity partition."""
        batch_size, _, height, width = z_id_pre_scaled.shape
        num_concepts = len(self.celebs)
        blocks = z_id_pre_scaled.view(
            batch_size,
            num_concepts,
            self.n_id_per_celeb,
            height,
            width,
        ).reshape(
            batch_size * num_concepts,
            self.n_id_per_celeb,
            height,
            width,
        )
        block_activations, local_indices = self.act_id(
            blocks,
            k=self.k_id,
            return_indices=True,
        )
        block_activations = block_activations.view(
            batch_size,
            self.total_id_kernels,
            height,
            width,
        )
        offsets = (
            torch.arange(num_concepts, device=z_id_pre_scaled.device)
            * self.n_id_per_celeb
        ).view(1, num_concepts, 1)
        global_indices = (
            local_indices.view(batch_size, num_concepts, -1) + offsets
        ).reshape(batch_size, -1)
        return block_activations, global_indices

    def forward(
        self,
        x_folded,
        labels=None,
        route_all=False,
        return_global_topk=True,
    ):
        """
        Args:
            x_folded: [B*T, D, H, W] - 时间维度已折叠
            labels: target concept for routed training; -1 keeps the original
                global ID competition so negative examples remain trainable.
            route_all: independently activate every partition for attribution.
            return_global_topk: retain the legacy global-routing diagnostic.
        """
        if route_all and labels is not None:
            raise ValueError("route_all and labels cannot be used together.")

        x_centered = x_folded - self.b_dec.view(1, -1, 1, 1)

        z_gen_pre = self.encoder_gen(x_centered) + self.b_enc_gen.view(1, -1, 1, 1)
        z_id_pre = self.encoder_id(x_centered) + self.b_enc_id.view(1, -1, 1, 1)

        w_dec_gen_norm = self.decoder_gen.weight.squeeze().norm(dim=0).view(1, -1, 1, 1)
        w_dec_id_norm = self.decoder_id.weight.squeeze().norm(dim=0).view(1, -1, 1, 1)

        z_gen_pre_scaled = z_gen_pre * w_dec_gen_norm
        z_id_pre_scaled = z_id_pre * w_dec_id_norm

        f_gen = self.act_gen(z_gen_pre_scaled)
        global_id_topk = None
        if route_all:
            f_id, routed_id_topk = self._activate_all_id_partitions(
                z_id_pre_scaled
            )
            if return_global_topk:
                global_id_topk = self.act_id.topk_indices(z_id_pre_scaled)
        elif labels is None:
            f_id, selected_indices = self.act_id(
                z_id_pre_scaled,
                return_indices=True,
            )
            routed_id_topk = selected_indices
            if return_global_topk:
                global_id_topk = selected_indices
        else:
            if labels.shape[0] != x_folded.shape[0]:
                raise ValueError(
                    f"Expected {x_folded.shape[0]} folded labels, got {labels.shape[0]}."
                )
            self._label_channel_mask(labels)
            if return_global_topk:
                global_id_topk = self.act_id.topk_indices(z_id_pre_scaled)
            f_id = torch.zeros_like(z_id_pre_scaled)
            routed_id_topk = torch.full(
                (labels.shape[0], self.k_id),
                -1,
                dtype=torch.long,
                device=labels.device,
            )

            target_rows = torch.nonzero(labels >= 0, as_tuple=True)[0]
            if target_rows.numel() > 0:
                target_mask = self._label_channel_mask(labels[target_rows])
                target_activations, target_indices = self.act_id(
                    z_id_pre_scaled[target_rows],
                    channel_mask=target_mask,
                    k=self.k_id,
                    return_indices=True,
                )
                f_id = f_id.index_copy(0, target_rows, target_activations)
                routed_id_topk[target_rows] = target_indices

            common_rows = torch.nonzero(labels < 0, as_tuple=True)[0]
            if common_rows.numel() > 0:
                common_activations, common_indices = self.act_id(
                    z_id_pre_scaled[common_rows],
                    return_indices=True,
                )
                f_id = f_id.index_copy(0, common_rows, common_activations)
                routed_id_topk[common_rows] = common_indices[:, :self.k_id]

        f_gen_descaled = f_gen / (w_dec_gen_norm + 1e-8)
        f_id_descaled = f_id / (w_dec_id_norm + 1e-8)

        x_gen_recon = self.decoder_gen(f_gen_descaled)
        x_id_recon = self.decoder_id(f_id_descaled)

        x_recon = x_gen_recon + x_id_recon + self.b_dec.view(1, -1, 1, 1)

        return {
            "x_recon": x_recon,
            "x_gen_recon": x_gen_recon,          # 不含 b_dec 的纯 Gen 重建
            "x_id_recon": x_id_recon,            # 不含 b_dec 的纯 Id 重建
            "b_dec_expanded": self.b_dec.view(1, -1, 1, 1),
            "f_gen": f_gen,
            "f_id": f_id,
            "z_gen_pre": z_gen_pre_scaled,
            "z_id_pre": z_id_pre_scaled,
            "id_global_topk_indices": global_id_topk,
            "id_routed_topk_indices": routed_id_topk,
        }

# ==========================================
# 2. 初始化工具 (SOTA 必需)
# ==========================================

@torch.no_grad()
def initialize_decoder_bias_to_geometric_median(sae, data_loader, device, num_batches=20):
    """
    计算数据集的几何中位点并初始化 SAE 的 b_dec。
    """
    print(">>> Initializing Decoder Bias to Geometric Median...")
    sae.eval()
    accumulated = torch.zeros(sae.d_model, device=device)
    count = 0

    for i, batch in enumerate(data_loader):
        if i >= num_batches: break
        x = batch["orig_act"].to(device)
        x_flat = x.view(-1, sae.d_model)
        accumulated += x_flat.sum(dim=0)
        count += x_flat.shape[0]

    mean_act = accumulated / count
    sae.b_dec.data = mean_act
    print(f">>> Initialization complete. Bias mean: {mean_act.mean().item():.4f}")


# ==========================================
# 3. 辅助损失函数
# ==========================================

def calculate_aux_loss(raw_model, out, residual, branch, valid_mask=None):
    """
    Args:
        residual: [B*T, D, H, W]
        branch: "gen" or "id"
        valid_mask: [B*T, C, H, W] float, 1.0 = 允许激活. None = 全部允许.
    """
    if branch == "gen":
        f_act = out["f_gen"]
        z_pre = out["z_gen_pre"]
        decoder = raw_model.decoder_gen
        n_kernels = raw_model.n_gen
    elif branch == "id":
        f_act = out["f_id"]
        z_pre = out["z_id_pre"]
        decoder = raw_model.decoder_id
        n_kernels = raw_model.total_id_kernels
    else:
        return torch.tensor(0.0, device=residual.device)

    B_T, C, H, W = f_act.shape
    activation_max = f_act.permute(0, 2, 3, 1).reshape(-1, n_kernels).max(dim=0)[0]
    dead_mask = (activation_max == 0)
    num_dead = dead_mask.sum().item()

    if num_dead == 0:
        return torch.tensor(0.0, device=residual.device)

    valid_flat = None
    eligible_dead = dead_mask
    if valid_mask is not None:
        valid_flat = valid_mask.reshape(B_T, n_kernels, -1).bool()
        eligible_dead = dead_mask & valid_flat.any(dim=(0, 2))
        num_dead = eligible_dead.sum().item()
        if num_dead == 0:
            return torch.tensor(0.0, device=residual.device)

    z_dead = torch.where(
        eligible_dead.view(1, -1, 1, 1),
        z_pre,
        torch.tensor(-float('inf'), device=z_pre.device)
    )

    k_aux = n_kernels // 2
    if valid_flat is not None:
        max_valid_channels = int(
            valid_flat.any(dim=-1).sum(dim=1).max().item()
        )
        k_aux = min(k_aux, max_valid_channels)
    if k_aux == 0:
        return torch.tensor(0.0, device=residual.device)
    z_dead_max_spatial = z_dead.view(B_T, n_kernels, -1).max(dim=-1)[0]
    if valid_flat is not None:
        z_dead_max_spatial = z_dead_max_spatial.masked_fill(
            ~valid_flat.any(dim=-1),
            -torch.inf,
        )
    topk_val, topk_idx = torch.topk(
        z_dead_max_spatial,
        k=min(k_aux, n_kernels),
        dim=1,
    )

    aux_mask = torch.zeros_like(z_dead_max_spatial)
    aux_mask.scatter_(1, topk_idx, 1.0)
    finite_mask = torch.zeros_like(aux_mask)
    finite_mask.scatter_(1, topk_idx, torch.isfinite(topk_val).to(aux_mask.dtype))
    aux_mask = aux_mask * finite_mask

    z_safe = torch.where(torch.isfinite(z_dead), z_dead.clamp_min(-10.0), 0.0)
    f_aux = F.leaky_relu(z_safe, negative_slope=0.1) * aux_mask.view(B_T, n_kernels, 1, 1)

    if valid_mask is not None:
        f_aux = f_aux * valid_mask

    w_dec_norm = decoder.weight.squeeze().norm(dim=0).view(1, -1, 1, 1)
    f_aux_descaled = f_aux / (w_dec_norm + 1e-8)

    recon_aux = decoder(f_aux_descaled)
    loss_aux_raw = (recon_aux - residual).pow(2).mean()

    scale = min(num_dead / k_aux, 1.0) if k_aux > 0 else 1.0
    return loss_aux_raw * scale


# ==========================================
# 4. 统一 Warmup Schedule
# ==========================================

def get_warmup_schedule(epoch, total_epochs, leak_start_fraction=0.3):
    """
    统一的 warmup schedule, 返回各 loss 的权重系数 (0.0 ~ 1.0)

    Phase 1 (0% ~ 10%):  纯重建预热, 所有分区约束关闭
    Phase 2 (10% ~ 100%): 重建和分区稀疏约束逐步加压
    Leakage (默认 30% ~ 100%): 身份分区先学会正样本重建，再逐步抑制泄漏

    Returns:
        dict: 'partition' (gen_sparse), 'face_recon', 'leak', 'temp'
    """
    if total_epochs < 1:
        raise ValueError("total_epochs must be at least 1.")
    if not 0.0 <= leak_start_fraction < 1.0:
        raise ValueError("leak_start_fraction must be in [0, 1).")

    phase1_end = int(0.1 * total_epochs)
    phase2_temp_start = int(0.2 * total_epochs)
    leak_start = int(leak_start_fraction * total_epochs)

    if epoch < phase1_end:
        return {
            'partition': 0.0,
            'face_recon': 0.0,
            'leak': 0.0,
            'temp': 0.0,
        }

    # 分区约束: sqrt ramp (前期快速上升, 后期平缓)
    partition_progress = (epoch - phase1_end) / (total_epochs - phase1_end)
    partition_progress = min(1.0, max(0.0, partition_progress))

    # 时间一致性: 比分区更晚启动, 线性上升, 最高 0.5 防止过强
    if epoch < phase2_temp_start:
        temp_progress = 0.0
    else:
        temp_progress = (epoch - phase2_temp_start) / (total_epochs - phase2_temp_start)
        temp_progress = min(0.5, max(0.0, temp_progress))

    if epoch < leak_start:
        leak_progress = 0.0
    else:
        leak_progress = (epoch - leak_start) / (total_epochs - leak_start)
        leak_progress = min(1.0, max(0.0, leak_progress))

    return {
        'partition': partition_progress ** 0.5,
        'face_recon': partition_progress,
        'leak': leak_progress,
        'temp': temp_progress,
    }


# ==========================================
# 5. 构建 penalty_mask
# ==========================================

def build_penalty_mask(labels, mask, f_id, raw_model, is_5d=True):
    """
    penalty_mask[b, c, ...] = True: 该位置的 Id 激活是非法的, 需要被惩罚.

    规则:
    - 普通人 (label < 0): 所有 Id 通道全部非法
    - 名人 (label >= 0): 只有"对的人"在"对的区域(mask > 0)"是合法的
    """
    B = labels.shape[0]
    penalty_mask = torch.ones_like(f_id, dtype=torch.bool)

    for b in range(B):
        label = labels[b].item()
        if label >= 0:
            start, end = raw_model.get_celeb_fields(label)
            is_face_region = (mask[b] > 0)
            if is_5d:
                penalty_mask[b, start:end, :, :, :] = ~is_face_region
            else:
                penalty_mask[b, start:end, :, :] = ~is_face_region

    return penalty_mask


# ==========================================
# 6. HunyuanVideo 训练循环
# ==========================================

def train_conv_sae_final(args, model, data_loader, device, layer_name):
    is_ddp = dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0
    raw_model = model.module if is_ddp else model

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    W_BG = args.bg_exclusive_weight
    W_FACE = args.face_weight
    W_LEAK = args.partition_beta
    W_TEMP = args.temp_consistency
    W_AUX = args.aux_coeff
    replay_loader, replay_plan = build_replay_loader(args, data_loader)
    completed_updates = 0
    next_checkpoint_epoch = 5
    health_tracker = PartitionHealthTracker(raw_model, device)

    if rank == 0:
        init_wandb(args, f"Layer_{layer_name}_SOTA")
        print(
            "[Train] "
            f"base_batches={replay_plan.base_batches}, "
            f"batch_reuse={replay_plan.batch_reuse}, "
            f"optimizer_updates={replay_plan.total_updates}, "
            f"generation_passes={replay_plan.generation_passes}"
        )
        pbar = tqdm(
            range(replay_plan.generation_passes),
            desc=f"Training {layer_name}",
        )
    else:
        pbar = range(replay_plan.generation_passes)

    for generation_pass in pbar:
        set_loader_epoch(replay_loader, generation_pass)
        health_tracker.reset()
        batch_iter = tqdm(replay_loader, leave=False) if rank == 0 else replay_loader

        for step, batch_data in enumerate(batch_iter):
            if completed_updates >= replay_plan.total_updates:
                break
            epoch = replay_plan.effective_epoch(completed_updates)
            schedule_epoch = int(epoch)
            optimizer.zero_grad()

            # --- 1. 数据准备 ---
            x_pos = batch_data["orig_act"].to(device).squeeze(1)  # [B, L, D]
            mask_3d = batch_data["mask"].to(device)               # [B, 1, T, H, W]
            labels = batch_data["label"].to(device)               # [B]

            B, L, D = x_pos.shape
            T = mask_3d.shape[2]
            h, w = mask_3d.shape[3], mask_3d.shape[4]

            x_pos_3d = x_pos.view(B, T, h, w, D).permute(0, 4, 1, 2, 3)        # [B, D, T, H, W]
            x_folded = x_pos_3d.permute(0, 2, 1, 3, 4).reshape(B * T, D, h, w)  # [B*T, D, H, W]

            # --- 2. 前向传播 ---
            folded_labels = labels.repeat_interleave(T)
            collect_health = step % replay_plan.batch_reuse == 0
            out = model(
                x_folded,
                labels=folded_labels,
                return_global_topk=collect_health,
            )
            if collect_health:
                health_tracker.update(out, labels, mask_3d)

            # --- 3. 维度还原 ---
            x_gen_recon_3d = out["x_gen_recon"].view(B, T, D, h, w).permute(0, 2, 1, 3, 4)  # [B, D, T, H, W]
            x_id_recon_3d = out["x_id_recon"].view(B, T, D, h, w).permute(0, 2, 1, 3, 4)
            b_dec = out["b_dec_expanded"]  # [1, D, 1, 1]
            f_id_3d = out["f_id"].view(B, T, -1, h, w).permute(0, 2, 1, 3, 4)  # [B, C_id, T, H, W]
            f_gen_3d = out["f_gen"].view(B, T, -1, h, w).permute(0, 2, 1, 3, 4)

            # --- 4. Warmup Schedule ---
            schedule = get_warmup_schedule(
                schedule_epoch,
                args.sae_epochs,
                getattr(args, "leak_start_fraction", 0.3),
            )

            # --- 5. Masks ---
            is_celeb = (labels >= 0).float().view(B, 1, 1, 1, 1)
            is_common = 1.0 - is_celeb
            bg_mask = ((1.0 - mask_3d) + (mask_3d * is_common)).float()  # [B, 1, T, H, W]
            face_mask = (mask_3d * is_celeb).float()

            # ============================================================
            # Loss A: 背景重建 (Gen + b_dec 独占)
            # ============================================================
            x_gen_full = x_gen_recon_3d + b_dec.unsqueeze(2)
            loss_bg = (bg_mask * (x_gen_full - x_pos_3d)**2).sum() / (bg_mask.sum() * D + 1e-6)

            # ============================================================
            # Loss B: Gen latent 在人脸区稀疏化
            # 惩罚 latent activations (不影响 b_dec)
            # ============================================================
            loss_gen_sparse = (face_mask * f_gen_3d.abs()).sum() / (face_mask.sum() * raw_model.n_gen + 1e-6)

            # ============================================================
            # Loss C: 人脸联合重建 (Gen.detach + Id + b_dec)
            # detach Gen: Id 学补残差, Gen 不收矛盾梯度
            # ============================================================
            x_face_recon = x_gen_recon_3d.detach() + x_id_recon_3d + b_dec.unsqueeze(2)
            loss_face_recon = (face_mask * (x_face_recon - x_pos_3d)**2).sum() / (face_mask.sum() * D + 1e-6)

            # ============================================================
            # Loss D: 身份泄露 - margin + 固定分母
            # ============================================================
            penalty_mask = build_penalty_mask(labels, mask_3d, f_id_3d, raw_model, is_5d=True)
            margin = 0.01
            # Reconstruction uses the routed target partition. Leakage must also
            # inspect every other partition or cross-identity responses receive
            # no negative gradient.
            loss_leak = partition_leakage_loss(
                z_id_pre=out["z_id_pre"],
                labels=labels,
                target_mask=mask_3d,
                n_id_per_partition=raw_model.n_id_per_celeb,
                k_id=raw_model.k_id,
                margin=margin,
            )

            # ============================================================
            # Loss E: 时间一致性
            # Gen: decoded output 帧间一致 (背景不该闪烁)
            # Id:  latent f_id 帧间一致 (同一身份的激活模式应稳定,
            #       但 decoded output 必须允许变化以拟合逐帧残差)
            # ============================================================
            # Gen: latent 激活帧间一致性 (约束 f_gen 而非 decoded output)
            f_gen_diff = f_gen_3d[:, :, 1:] - f_gen_3d[:, :, :-1]
            bg_mask_t = bg_mask[:, :, 1:]
            # bg_mask_t: [B, 1, T-1, H, W] 广播到 [B, C_gen, T-1, H, W]
            loss_temp_gen = (bg_mask_t * f_gen_diff.pow(2)).sum() / (bg_mask_t.sum() * raw_model.n_gen + 1e-6)

            # Id: latent 激活帧间一致性 (约束 f_id 而非 x_id_recon)
            f_id_diff = f_id_3d[:, :, 1:] - f_id_3d[:, :, :-1]
            face_mask_t = face_mask[:, :, 1:]
            # face_mask_t: [B, 1, T-1, H, W] 广播到 [B, C_id, T-1, H, W]
            loss_temp_id = (face_mask_t * f_id_diff.pow(2)).sum() / (face_mask_t.sum() * raw_model.total_id_kernels + 1e-6)

            loss_temp = loss_temp_gen + loss_temp_id

            # ============================================================
            # Loss F: 辅助损失 (复活死核)
            # ============================================================
            residual_folded = (x_folded - out["x_recon"]).detach()
            loss_aux_gen = calculate_aux_loss(raw_model, out, residual_folded, branch="gen")

            valid_mask_id = (~penalty_mask).permute(0, 2, 1, 3, 4).reshape(B * T, -1, h, w).float()
            loss_aux_id = calculate_aux_loss(raw_model, out, residual_folded, branch="id", valid_mask=valid_mask_id)
            loss_aux = loss_aux_gen + loss_aux_id

            # ============================================================
            # Loss G: 全图重建保底
            # ============================================================
            x_recon_3d = out["x_recon"].view(B, T, D, h, w).permute(0, 2, 1, 3, 4)
            loss_global_recon = (x_recon_3d - x_pos_3d).pow(2).mean()

            # ============================================================
            # 总损失
            # ============================================================
            if schedule['partition'] == 0.0:
                # Phase 1: 纯重建预热
                total_loss = loss_global_recon + W_AUX * loss_aux
            else:
                # Phase 2: 分区训练
                p = schedule['partition']
                leak = schedule['leak']
                t = schedule['temp']
                f = schedule['face_recon']
                w_global = 0.1 * (1.0 - p)  # 全图保底随分区上升而衰减

                total_loss = (
                    W_BG * loss_bg +
                    p * W_BG * loss_gen_sparse +
                    f * W_FACE * loss_face_recon +
                    leak * W_LEAK * loss_leak +
                    t * W_TEMP * loss_temp +
                    W_AUX * loss_aux +
                    w_global * loss_global_recon
                )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # --- 日志 ---
            if rank == 0:
                metrics = {
                    "loss/total": total_loss.item(),
                    "loss/bg": loss_bg.item(),
                    "loss/gen_sparse": loss_gen_sparse.item(),
                    "loss/face_recon": loss_face_recon.item(),
                    "loss/leak": loss_leak.item(),
                    "loss/temp": loss_temp.item(),
                    "loss/aux": loss_aux.item(),
                    "loss/global_recon": loss_global_recon.item(),
                    "schedule/partition": schedule['partition'],
                    "schedule/leak": schedule['leak'],
                    "schedule/temp": schedule['temp'],
                    "schedule/face_recon": schedule['face_recon'],
                    "train/effective_epoch": epoch,
                    "train/generation_pass": generation_pass,
                    "train/replay_index": step % replay_plan.batch_reuse,
                    "train/update": completed_updates,
                }
                log_wandb(args, metrics)
                log_train_metrics(args, layer_name, epoch, step, metrics)

            completed_updates += 1
            completed_epoch = replay_plan.effective_epoch(completed_updates)
            if rank == 0 and completed_epoch + 1e-9 >= next_checkpoint_epoch:
                save_epoch = next_checkpoint_epoch - 1
                save_path = (
                    f"{args.sae_save_dir}/conv_sae_"
                    f"{layer_name.replace('.', '_')}_{save_epoch}.pt"
                )
                torch.save(raw_model.state_dict(), save_path)
                next_checkpoint_epoch += 5

        health_tracker.summarize(
            args=args,
            layer_name=layer_name,
            generation_pass=generation_pass,
            effective_epoch=replay_plan.effective_epoch(completed_updates),
            rank=rank,
        )

    save_final_checkpoint(args, raw_model, layer_name, rank)

