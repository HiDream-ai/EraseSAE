import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from training.logging_utils import (
    distributed_mean_metrics,
    init_wandb,
    log_train_metrics,
    log_wandb,
    require_finite_across_ranks,
)
from training.partition_diagnostics import PartitionHealthTracker
from training.replay_utils import (
    build_replay_loader,
    save_final_checkpoint,
    set_loader_epoch,
)
from model.partition_objectives import (
    get_partition_schedule,
    partition_contrastive_loss,
    partition_leakage_loss,
)

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
        # Encoder Bias
        self.b_enc_gen = nn.Parameter(torch.zeros(n_gen))
        self.b_enc_id = nn.Parameter(torch.zeros(self.total_id_kernels))

        # --- Decoder ---
        # SOTA: Decoder 权重建议 Unit Norm 初始化，这里通过 forward 中的 rescale 动态处理
        self.decoder_gen = nn.Conv2d(n_gen, d_model, 1, bias=False)
        self.decoder_id = nn.Conv2d(self.total_id_kernels, d_model, 1, bias=False)
        
        # Global Decoder Bias (Geometric Median 将被赋值于此)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # --- Activations ---
        self.act_gen = TopKConvActivation(k_gen)
        self.act_id = TopKConvActivation(k_id * len(celebs))

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
        """Apply TopK independently inside every configured concept partition."""
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

        # --- SOTA 技巧 1: Input Centering (去偏) ---
        # Encoder 应该学习相对于 Geometric Median 的残差
        # b_dec: [D] -> [1, D, 1, 1]
        x_centered = x_folded - self.b_dec.view(1, -1, 1, 1)

        # --- Encoding ---
        z_gen_pre = self.encoder_gen(x_centered) + self.b_enc_gen.view(1, -1, 1, 1)
        z_id_pre = self.encoder_id(x_centered) + self.b_enc_id.view(1, -1, 1, 1)

        # --- SOTA 技巧 2: Rescale Acts by Decoder Norm (范数缩放) ---
        # 防止通过把 Encoder 权重变大、Decoder 权重变小来绕过 TopK
        w_dec_gen_norm = self.decoder_gen.weight.squeeze().norm(dim=0).view(1, -1, 1, 1)
        w_dec_id_norm = self.decoder_id.weight.squeeze().norm(dim=0).view(1, -1, 1, 1)

        z_gen_pre_scaled = z_gen_pre * w_dec_gen_norm
        z_id_pre_scaled = z_id_pre * w_dec_id_norm

        # --- TopK Activation ---
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

        # --- Decoding ---
        # 必须除以范数，保持数学等价性
        f_gen_descaled = f_gen / (w_dec_gen_norm + 1e-8)
        f_id_descaled = f_id / (w_dec_id_norm + 1e-8)

        x_gen_recon = self.decoder_gen(f_gen_descaled)
        x_id_recon = self.decoder_id(f_id_descaled)
        
        # 加上 Bias 恢复原空间
        x_recon = x_gen_recon + x_id_recon + self.b_dec.view(1, -1, 1, 1)

        return {
            "x_recon": x_recon,
            "x_gen_only": x_gen_recon + self.b_dec.view(1, -1, 1, 1), # 用于背景损失
            "x_id_only": x_id_recon,
            "x_gen_recon": x_gen_recon,
            "x_id_recon": x_id_recon,
            "b_dec_expanded": self.b_dec.view(1, -1, 1, 1),
            "f_gen": f_gen,
            "f_id": f_id,
            "z_gen_pre": z_gen_pre_scaled, # 返回缩放后的 Pre-Act 用于 Aux Loss
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
    这是 SAE Lens 和 Anthropic 推荐的关键初始化步骤。
    """
    print(">>> Initializing Decoder Bias to Geometric Median...")
    sae.eval()
    accumulated = torch.zeros(sae.d_model, device=device)
    count = 0
    
    for i, batch in enumerate(data_loader):
        if i >= num_batches: break
        # 获取数据: [B, L, D]
        x = batch["orig_act"].to(device)
        # 展平所有维度 [N, D]
        x_flat = x.view(-1, sae.d_model)
        accumulated += x_flat.sum(dim=0)
        count += x_flat.shape[0]
    
    # 使用 Mean 近似 Geometric Median (在大规模数据下通常足够有效)
    mean_act = accumulated / count
    sae.b_dec.data = mean_act
    print(f">>> Initialization complete. Bias mean: {mean_act.mean().item():.4f}")


# ==========================================
# 3. 训练循环 (含 DDP, Aux Loss, 时空约束)
# ==========================================
def calculate_aux_loss(raw_model, out, residual, branch, valid_mask=None):
    """
    统一的辅助损失计算函数 (支持 Mask)。
    
    Args:
        raw_model: SAE 模型实例
        out: forward 返回的字典
        residual: [B*T, D, H, W] 重建残差
        branch: "gen", "common", 或 "spec" (用于选择参数)
        valid_mask: [B, T, 1, H, W] 或 None. 
                    True 表示该区域允许激活 (用于 Specific 分支的身份隔离)。
                    None 表示全图允许 (用于 Gen/Common 分支)。
    """
    # 1. 根据分支选择组件
    if branch == "gen":
        f_act = out["f_gen"]         # [B*T, C, H, W]
        z_pre = out["z_gen_pre"]     # [B*T, C, H, W]
        decoder = raw_model.decoder_gen
        n_kernels = raw_model.n_gen
    elif branch == "id":
        f_act = out["f_id"]
        z_pre = out["z_id_pre"]
        decoder = raw_model.decoder_id
        n_kernels = raw_model.total_id_kernels
    else:
        return torch.tensor(0.0, device=residual.device)

    # 1. 识别死核 (在本 Batch 中，所有位置的最大激活均为 0)
    # f_act: [B*T, C, H, W] -> max -> [C]
    # 注意：这里需要跨 Batch 和跨 Spatial 统计
    B_T, C, H, W = f_act.shape
    activation_max = f_act.permute(0, 2, 3, 1).reshape(-1, n_kernels).max(dim=0)[0]
    dead_mask = (activation_max == 0)
    num_dead = dead_mask.sum().item()

    if num_dead == 0:
        return torch.tensor(0.0, device=residual.device)

    # 2. 准备死核的预激活值
    # 只保留 Dead Kernels 的 z_pre，其他设为 -inf (TopK 时会被忽略)
    valid_flat = None
    eligible_dead = dead_mask
    if valid_mask is not None:
        if valid_mask.ndim == 5:
            valid_mask = valid_mask.permute(0, 2, 1, 3, 4).reshape(B_T, C, H, W)
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

    # 3. 在死核中通过 TopK 选择最相关的 (K_aux)
    # 启发式：K_aux 通常设为 C // 2
    k_aux = n_kernels // 2
    if valid_flat is not None:
        max_valid_channels = int(
            valid_flat.any(dim=-1).sum(dim=1).max().item()
        )
        k_aux = min(k_aux, max_valid_channels)
    if k_aux == 0:
        return torch.tensor(0.0, device=residual.device)
    
    # 获取 z_dead 的最大响应值用于排序
    z_dead_max_spatial = z_dead.view(z_dead.shape[0], n_kernels, -1).max(dim=-1)[0] # [B*T, C]
    
    # 选出 Top-K_aux 个死核
    # 这里的 TopK 是针对每个样本选出的最强死核
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
    
    # 构造 Aux Mask
    aux_mask = torch.zeros_like(z_dead_max_spatial)
    aux_mask.scatter_(1, topk_idx, 1.0)
    finite_mask = torch.zeros_like(aux_mask)
    finite_mask.scatter_(1, topk_idx, torch.isfinite(topk_val).to(aux_mask.dtype))
    aux_mask = aux_mask * finite_mask
    
    # 4. 激活与重构
    # ReLU
    # 1. 钳制极端负值，防止 Loss 爆炸
    z_safe = torch.where(torch.isfinite(z_dead), z_dead.clamp_min(-10.0), 0.0)
    
    # 2. 使用 LeakyReLU 允许负梯度回传 (复活死核的关键)
    f_aux = F.leaky_relu(z_safe, negative_slope=0.1) * aux_mask.view(*aux_mask.shape, 1, 1)
    #f_aux = F.relu(z_dead) * aux_mask.view(*aux_mask.shape, 1, 1)
    # --- 🔥 关键融合点: 应用 Valid Mask ---
    if valid_mask is not None:
        f_aux = f_aux * valid_flat.view(B_T, C, H, W).to(f_aux.dtype)
    # Descale (反向缩放)
    w_dec_norm = decoder.weight.squeeze().norm(dim=0).view(1, -1, 1, 1)
    f_aux_descaled = f_aux / (w_dec_norm + 1e-8)
    
    # 5. 计算损失 (死核拟合残差)
    recon_aux = decoder(f_aux_descaled)
    loss_aux_raw = (recon_aux - residual).pow(2).mean()
    
    # SOTA: 动态缩放因子 (死核越少，权重越小)
    scale = min(num_dead / k_aux, 1.0)
    
    return loss_aux_raw * scale

def train_conv_sae_final(args, model, data_loader, device, layer_name):
    # --- DDP Setup ---
    is_ddp = dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0
    raw_model = model.module if is_ddp else model # 访问自定义属性用 raw_model
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # --- 权重配置 ---
    # 根据你的任务经验值设定
    W_BG = args.bg_exclusive_weight       # 背景独占 (Background Exclusive)
    W_FACE = args.face_weight     # 人脸联合 (Face Joint Recon)
    W_TEMP = args.temp_consistency     # 时间一致性 (Temporal Consistency)
    W_AUX = args.aux_coeff       # 辅助损失 (Auxiliary Loss)
    W_CONTRAST = getattr(args, "partition_contrast_weight", 0.0)
    CONTRAST_MARGIN = getattr(args, "partition_contrast_margin", 0.1)
    replay_loader, replay_plan = build_replay_loader(args, data_loader)
    completed_updates = 0
    next_checkpoint_epoch = 5
    health_tracker = PartitionHealthTracker(raw_model, device)

    #h, w = 30, 45     # HunyuanVideo Latent Spatial Dim

    # --- 初始化 (Rank 0 负责) ---
    # 建议在训练开始前调用一次 initialize_decoder_bias_to_geometric_median
    # if rank == 0: initialize_decoder_bias_to_geometric_median(raw_model, data_loader, device)
    # dist.barrier() # 等待初始化完成

    if rank == 0:
        init_wandb(args, f"Layer_{layer_name}_SOTA")
        world_size = dist.get_world_size() if is_ddp else 1
        per_rank_batch = getattr(
            args,
            "batch_size",
            getattr(data_loader, "batch_size", None),
        )
        global_batch = (
            per_rank_batch * world_size
            if isinstance(per_rank_batch, int)
            else "unknown"
        )
        print(
            "[Train] "
            f"base_batches={replay_plan.base_batches}, "
            f"per_rank_batch={per_rank_batch or 'unknown'}, "
            f"global_batch={global_batch}, "
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
            # x_pos: [B, L, D] (Hunyuan Output)
            x_pos = batch_data["orig_act"].to(device).squeeze(1)
            # mask_3d: [B, 1, T, H, W]
            mask_3d = batch_data["mask"].to(device)
            labels = batch_data["label"].to(device)
            
            B, L, D = x_pos.shape
            T = mask_3d.shape[2]
            h,w = mask_3d.shape[3],mask_3d.shape[4]
            # 还原为 5D 时空张量: [B, D, T, H, W]
            x_pos_3d = x_pos.view(B, T, h, w, D).permute(0, 4, 1, 2, 3)
            # 折叠时间维度用于 Conv2d: [B*T, D, H, W]
            x_folded = x_pos_3d.permute(0, 2, 1, 3, 4).reshape(B * T, D, h,w)

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

            # --- 3. 维度还原 (用于 Loss 计算) ---
            # 重构结果: [B, D, T, H, W]
            #x_recon_3d = out["x_recon"].view(B, T, D, h, w).permute(0, 2, 1, 3, 4)
            x_gen_only_3d = out["x_gen_only"].view(B, T, D, h, w).permute(0, 2, 1, 3, 4)
            x_id_only_3d = out["x_id_only"].view(B, T, D, h, w).permute(0, 2, 1, 3, 4)
            # 身份特征: [B, C_id, T, H, W]
            f_id_3d = out["f_id"].view(B, T, -1, h, w).permute(0, 2, 1, 3, 4)
            f_gen_3d = out["f_gen"].view(B, T, -1, h, w).permute(0, 2, 1, 3, 4)

            

            # --- 4. Losses ---
            schedule = get_partition_schedule(
                schedule_epoch,
                args.sae_epochs,
                getattr(args, "leak_start_fraction", 0.3),
            )


            is_celeb = (labels >= 0).float().view(B, 1, 1, 1, 1) # [B, 1, 1, 1, 1]
            is_common = 1.0 - is_celeb

            # 定义 Mask
            bg_mask = ((1.0 - mask_3d)+((mask_3d * is_common))).float() # [B, 1, T, H, W]
            face_mask = (mask_3d * is_celeb).float()
            # A. 空间契约损失 (Spatial Contract)
            # 背景区 (Mask=0): 要求 Gen 分支能独立重构原图
            #bg_mask = torch.where(mask_3d > 0.05, 0.0, 1.0)
            loss_gen_bg = (bg_mask * (x_gen_only_3d - x_pos_3d)**2).sum() / (bg_mask.sum() * D + 1e-6)
            #loss_gen_face = (face_mask * (x_gen_recon - x_pos_3d)**2).sum() / (face_mask.sum() * D + 1e-6)
            # Suppress generic latent usage in the target region. Penalizing
            # x_gen_only would also force b_dec toward zero and conflicts with
            # background reconstruction.
            loss_gen_sparse = (face_mask * f_gen_3d.abs()).sum() / (
                face_mask.sum() * raw_model.n_gen + 1e-6
            )

            #x_reshaped = x_pos_3d.permute(0, 2, 1, 3, 4).reshape(B*T, D, h,w)

            #x_blurred_2d = TF.gaussian_blur(x_reshaped, kernel_size=15, sigma=5.0)
            
            # 还原维度
            #x_blurred_3d = x_blurred_2d.view(B, T, D, h,w).permute(0, 2, 1, 3, 4)

            #loss_gen_blur = (face_mask * (x_gen_only_3d - x_blurred_3d)**2).sum() / (face_mask.sum() * D + 1e-6)


            # 关键：人脸区域权重设低，防止 Gen 偷学身份细节
            loss_stage_1 = W_BG * loss_gen_bg + (
                schedule["partition"] * W_BG * loss_gen_sparse
            )
            #loss_stage_1 = W_BG * loss_gen_bg
            #loss_stage_1 = W_BG * loss_gen_bg + cur_silence_weight_1 * loss_gen_blur



            # 人脸区 (Mask=1): 要求 Gen + Id 联合重构原图
            # 空间加权 MSE: 人脸区域赋予更高权重
            # 关键：x_gen.detach() 和 x_common.detach()
            current_recon_2 = x_gen_only_3d.detach() + x_id_only_3d
            # 构造一个 Batch 级别的掩码，区分名人与普通人
            # labels: [B]
            
            
            # A. 名人的任务：完美重建 (Reconstruction)
            # 只计算 is_celeb 为 1 的样本
            loss_recon_celeb = (
                face_mask * (current_recon_2 - x_pos_3d) ** 2
            ).sum() / (face_mask.sum() * D + 1e-6)
            
            """# B. 普通人的任务：全图静默 (Total Silence)
            # 对于普通人，Specific 分支不应该输出任何东西（不管是脸还是背景）
            # 所以直接惩罚 x_spec_recon 的平方和
            mse_silence = x_id_only_3d ** 2
            # 注意：这里是全图约束 (不仅是 face_mask)，因为普通人压根就不该激活 Specific
            loss_silence_normal = (mse_silence * is_common).sum() / (is_common.sum() * x_id_only_3d.shape[2:].numel() * D + 1e-6)"""
            
            # C. Specific 分支的背景静默 (针对名人的背景)
            # 名人的背景也不能有 Specific 特征
            #loss_silence_spec_bg = (bg_mask * x_id_only_3d**2 * is_celeb).sum() / (bg_mask.sum() * D + 1e-6)

            # 合并 Stage 3 Loss
            # 这里的 5.0 是强约束，确保普通人绝对无法激活 Specific
            loss_stage_2 = (
                schedule["face_recon"] * W_FACE * loss_recon_celeb
            )
            #loss_face = (face_weight_map * (x_recon_3d - x_pos_3d)**2).mean()

            # B. 身份泄露损失 (Identity Leakage)
            # 目标：非目标名人的核在 Mask 区域不准激活；目标名人的核在背景区域不准激活
            
            #z_id_5d = out["z_id_pre"].view(B, T, -1, h, w)
            penalty_mask = torch.ones_like(f_id_3d, dtype=torch.bool) # 默认全惩罚
            for b in range(B):
                label = labels[b].item()
                if label >= 0:
                    start, end = raw_model.get_celeb_fields(label)
                    is_face_region = (mask_3d[b] > 0)
                    # 只有“对的人”在“对的区域(Mask)”是合法的 (penalty=0)
                    penalty_mask[b, start:end, :,  :, :] = ~is_face_region
            
            #K_id_D = raw_model.total_k_id_kernels
            # Keep reconstruction label-routed, but expose every partition to
            # cross-concept negatives through a memory-bounded leakage loss.
            loss_leak = partition_leakage_loss(
                z_id_pre=out["z_id_pre"],
                labels=labels,
                target_mask=mask_3d,
                n_id_per_partition=raw_model.n_id_per_celeb,
                k_id=raw_model.k_id,
                margin=0.01,
            )
            loss_contrast = partition_contrastive_loss(
                z_id_pre=out["z_id_pre"],
                labels=labels,
                target_mask=mask_3d,
                n_id_per_partition=raw_model.n_id_per_celeb,
                k_id=raw_model.k_id,
                margin=CONTRAST_MARGIN,
            )

            # 动态调整 partition_beta
            
            # C. 时间一致性 (Temporal Consistency)
            # 惩罚相邻帧激活值的剧烈跳变
            #f_id_diff = f_id_3d[:, :, 1:] - f_id_3d[:, :, :-1]
            #loss_temp = f_id_diff.pow(2).sum() / (B * (T-1) * h * w)
            # 目标：不仅约束 ID，也要约束 Gen，让 Gen 也要付出代价，不能随意闪烁
            
            # 1. 计算 ID 分支的稳定性 (Normalized by Channels)
            f_id_cur = f_id_3d[:, :, 1:]
            f_id_prev = f_id_3d[:, :, :-1]

            f_id_diff = f_id_cur - f_id_prev
            # 除以 total_id_kernels (通道数)，变成 "每个核的平均抖动"
            loss_temp_id = f_id_diff.pow(2).sum() / (B * (T-1) * h * w * raw_model.total_id_kernels + 1e-6)

            # 2. 计算 Gen 分支的稳定性 (新增!)
            f_gen_cur = f_gen_3d[:, :, 1:]
            f_gen_prev = f_gen_3d[:, :, :-1]

            f_gen_diff = f_gen_cur - f_gen_prev
            # 除以 n_gen
            loss_temp_gen = f_gen_diff.pow(2).sum() / (B * (T-1) * h * w * raw_model.n_gen + 1e-6)

            # 3. 组合 Loss
            # 我们可以给 Gen 更高的稳定性权重，逼迫它学习静态背景，
            # 从而把"动态的表情变化"挤压给 ID 分支 (如果 ID 是负责表情的话)
            # 或者简单地 1:1 加和
            loss_temp = loss_temp_id + loss_temp_gen

            # D. 辅助损失 (Auxiliary Loss) - 针对 Gen 分支
            residual_folded = (x_folded - out["x_recon"]).detach()
            #loss_aux = calculate_aux_loss(raw_model, out, residual_folded, branch="gen")

            # 计算 Gen 的 Aux
            loss_aux_gen = calculate_aux_loss(raw_model, out, residual_folded, branch="gen")

            # 计算 ID 的 Aux (新增!)
            # 注意：ID 分支应该只去拟合 "Gen分支搞不定的残差"，或者全图残差
            # 这里为了复活它，让它拟合全图残差
            valid_mask_spec = ~penalty_mask
            loss_aux_id = calculate_aux_loss(raw_model, out, residual_folded, branch="id",valid_mask=valid_mask_spec)

            loss_aux = loss_aux_gen + loss_aux_id

            # --- 5. 总损失与更新 ---
            x_recon_3d = out["x_recon"].view(B, T, D, h, w).permute(
                0, 2, 1, 3, 4
            )
            loss_global_recon = (x_recon_3d - x_pos_3d).pow(2).mean()

            if schedule["partition"] == 0.0:
                total_loss = loss_global_recon + W_AUX * loss_aux
            else:
                partition_weight = schedule["partition"]
                total_loss = (
                    loss_stage_1
                    + loss_stage_2
                    + schedule["leak"] * args.partition_beta * loss_leak
                    + partition_weight * W_CONTRAST * loss_contrast
                    + schedule["temp"] * W_TEMP * loss_temp
                    + W_AUX * loss_aux
                    + 0.1 * (1.0 - partition_weight) * loss_global_recon
                )

            require_finite_across_ranks(total_loss, "training loss", device)
            total_loss.backward()
            
            # SOTA: 梯度裁剪 (TopK 训练早期容易不稳)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            require_finite_across_ranks(gradient_norm, "gradient norm", device)
            
            optimizer.step()

            # --- 6. 日志 ---
            loss_metrics = distributed_mean_metrics(
                {
                    "loss/total": total_loss,
                    "loss/gen": loss_stage_1,
                    "loss/id": loss_stage_2,
                    "loss/bg": loss_gen_bg,
                    "loss/gen_sparse": loss_gen_sparse,
                    "loss/face_recon": loss_recon_celeb,
                    "loss/leak": loss_leak,
                    "loss/contrast": loss_contrast,
                    "loss/temp": loss_temp,
                    "loss/aux": loss_aux,
                    "loss/global_recon": loss_global_recon,
                    "train/gradient_norm": gradient_norm,
                },
                device,
            )
            if rank == 0:
                metrics = {
                    **loss_metrics,
                    "schedule/partition": schedule["partition"],
                    "schedule/leak": schedule["leak"],
                    "schedule/temp": schedule["temp"],
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
