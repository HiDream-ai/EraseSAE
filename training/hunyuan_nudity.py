import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.partitioned_sae import TopKConvPSAE, train_conv_sae_final
from model.checkpoint_io import resolve_checkpoint_run, resolve_layer_checkpoint
from configs.config_loader import build_training_parser, finalize_training_args
from training.distributed_utils import init_distributed_device, launch_training
from training.hunyuan_online_data import create_online_hunyuan_loader
from training.logging_utils import log_loader_summary, setup_run_logging
from training.mining_utils import quarantine_stale_attribution
from training.normalization_utils import prepare_online_normalization
from training.partitioned_attribution import run_partitioned_attribution
from training.partitioned_checkpoint import build_partitioned_sae

class UnifiedCategoricalDataset(Dataset):
    def __init__(self, root_dir, layer_name, target_celebs=None, mean=None, std=None, device="cuda"):
        """
        Args:
            root_dir: 数据根目录 (e.g. .../hunyuan_video/train)
            target_celebs: List[str] 或 None。如果不是 None，仅加载列表中的名人。
        """
        self.root_dir = Path(root_dir)
        self.layer_folder_name = layer_name.replace(".", "_")
        self.device = device
        
        self.mean = mean.to(device) if mean is not None else None
        self.std = std.to(device) if std is not None else None

        # -------------------------------------------------------
        # 1. 扫描目录结构 (新结构适配)
        # -------------------------------------------------------
        nsfw_root = self.root_dir / "nsfw"
        non_nsfw_root = self.root_dir / "no_nsfw"

        if not nsfw_root.exists():
            raise FileNotFoundError(f"Directory not found: {nsfw_root}")

        # 获取两个目录下的所有子文件夹名称
        dirs_nsfw = sorted([d.name for d in nsfw_root.iterdir() if d.is_dir()]) if nsfw_root.exists() else []
        dirs_non_nsfw = sorted([d.name for d in non_nsfw_root.iterdir() if d.is_dir()]) if non_nsfw_root.exists() else []

        # 只要在 target_celebs 里出现的，不管它在 nsfw 还是 no_nsfw 目录下，都算作 Specific Concept
        if target_celebs is not None and len(target_celebs) > 0:
            self.celebs = target_celebs # 直接使用参数列表作为类别列表 (例如 ['nudity', 'clothed'])
            print(f"🎯 [Target Concepts]: {self.celebs}")
        else:
            # 如果没指定，默认只用 nsfw 下的
            self.celebs = dirs_nsfw

        # 建立 概念名 -> ID 的映射 (0, 1, 2...)
        # 例如: {'nudity': 0, 'clothed': 1}
        self.celeb_to_id = {name: i for i, name in enumerate(self.celebs)}
        self.NON_TARGET_ID = -1
        

        # -------------------------------------------------------
        # 2. 收集样本 (分别遍历两个大目录)
        # -------------------------------------------------------
        self.all_samples = []
        def collect_from_dir(directory, is_nsfw_folder):
            if not directory.exists(): return
            
            # 遍历该目录下的子文件夹
            sub_dirs = [d for d in directory.iterdir() if d.is_dir()]
            for sub in sub_dirs:
                folder_name = sub.name
                
                # 🚀 [修改点 2]：判断当前文件夹是否属于目标概念
                if folder_name in self.celeb_to_id:
                    # 如果在目标列表中，赋予对应 ID (Specific 分支)
                    label = self.celeb_to_id[folder_name]
                    subject_name = folder_name
                else:
                    # 否则，视为纯背景 (Gen/Common 分支)
                    label = self.NON_TARGET_ID
                    subject_name = "background"

                # 加载文件
                layer_dir = sub / 'activation' / self.layer_folder_name
                if layer_dir.exists():
                    files = list(layer_dir.glob("*.pt"))
                    for f in files:
                        self.all_samples.append({
                            "file_path": f,
                            "label": label,
                            "subject_name": subject_name
                        })
        
        # 分别扫描两个根目录
        collect_from_dir(nsfw_root, is_nsfw_folder=True)
        collect_from_dir(non_nsfw_root, is_nsfw_folder=False)

        print(f"✅ Dataset Initialized:")
        print(f"   - Concepts: {self.celeb_to_id}")
        print(f"   - Total samples: {len(self.all_samples)}")

    def __len__(self):
        return len(self.all_samples)

    def _normalize(self, x):
        if self.mean is not None and self.std is not None:
            return (x - self.mean) / (self.std + 1e-8)
        return x

    def __getitem__(self, idx):
        # ... (这部分保持不变) ...
        sample_info = self.all_samples[idx]
        data = torch.load(sample_info['file_path'], map_location=self.device)
        
        orig_act = data['original_acts']
        mask = data['attnmap']
        
        orig_act = self._normalize(orig_act)
        t_val = data.get('timestep', 500)
        
        return {
            "orig_act": orig_act,
            "mask": mask,
            "label": sample_info['label'],
            "subject": sample_info['subject_name'],
            "timestep": t_val
        }
    

    
def create_dataloader_for_layer(data_root_dir, mean, std, layer_name: str, batch_size: int, rank: int, world_size: int, target_celebs=None):
    """
    为 DDP 模式创建 DataLoader，使用 DistributedSampler。
    """
    
    dataset = UnifiedCategoricalDataset(
        root_dir=data_root_dir, 
        layer_name=layer_name,
        mean=mean, 
        target_celebs=target_celebs,
        std=std,
    )
    
    # 🚀 DDP 核心: 使用 DistributedSampler
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )
    
    # 注意: batch_size 是 per-GPU batch size
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        sampler=sampler, # 使用 sampler 替代 shuffle=True
        num_workers=0, # DDP 模式下通常保持 0 或小值以避免 CPU 内存冲突
    )
    
    if rank == 0:
        print(f"\nDataLoader for {layer_name} created. (Rank 0)")
        print(f"Total files: {len(dataset)}")
        print(f"Per-GPU batch size: {batch_size}. Sampler ensures even distribution.")
        print(f"每个 epoch 总批次数 (per-GPU): {len(data_loader)}")
        
    return data_loader,dataset.celebs

class IncrementalStatistics:
    """
    在线（增量）计算激活值的均值和标准差。
    针对大规模 SAE 训练数据进行了优化。
    """
    def __init__(self, data_dim, device="cpu"):
        # 使用 float64 累加，防止数万个 Batch 后的舍入误差
        self.mean = torch.zeros(data_dim, dtype=torch.float64, device=device)
        self.M2 = torch.zeros(data_dim, dtype=torch.float64, device=device)
        self.count = 0
        self.data_dim = data_dim

    def update(self, new_data):
        """
        new_data: [B, N, D] 或 [Total_Tokens, D] 的 Tensor
        """
        # 1. 统一展平为 [Tokens, Dim] 并转为 float64
        # 建议在 CPU 上计算，以免统计阶段占用过多显存
        new_data = new_data.reshape(-1, self.data_dim).to(torch.float64).to(self.mean.device)
        
        n_b = new_data.shape[0]
        if n_b == 0:
            return

        # 2. 计算当前批次的统计量
        batch_mean = new_data.mean(dim=0)
        # 这里的 M2_b 是批次内样本到批次均值的平方和
        batch_M2 = ((new_data - batch_mean) ** 2).sum(dim=0)

        # 3. 经典的 Welford 分批更新公式
        new_count = self.count + n_b
        delta = batch_mean - self.mean
        
        # 更新均值
        self.mean += delta * (n_b / new_count)
        
        # 更新 M2 (Sum of Squares of Differences)
        # 结合了两个样本集的方差以及它们均值之间的距离
        self.M2 += batch_M2 + (delta ** 2) * (self.count * n_b / new_count)
        
        self.count = new_count

    def get_stats(self):
        """返回 (mean, std, dead_mask)"""
        if self.count < 2:
            return (self.mean.to(torch.float32), 
                    torch.ones(self.data_dim, dtype=torch.float32),
                    torch.zeros(self.data_dim, dtype=torch.bool))

        variance = self.M2 / (self.count - 1)
        std = torch.sqrt(variance)
        
        # 🚀 识别死维度：标准差极小意味着这个神经元几乎从未放电
        dead_mask = std < 1e-6
        
        # 对于死维度，将 std 设为 1.0 防止归一化时除以 0
        safe_std = std.clone()
        safe_std[dead_mask] = 1.0
        
        return (self.mean.to(torch.float32), 
                safe_std.to(torch.float32), 
                dead_mask)


def run_global_stats_collection(root_dir, layer_name, data_dim):
    """
    递归遍历 celebrity 和 non_celebrity 目录，计算该层的全局统计量
    """
    stats_calculator = IncrementalStatistics(data_dim, device="cpu")
    layer_fn = layer_name.replace(".", "_")
    root_path = Path(root_dir)
    
    # -------------------------------------------------------
    # 修改：分别定义两个匹配模式
    # -------------------------------------------------------
    # 1. 匹配 celebrity 下的所有: root/celebrity/*/activation/{layer}/*.pt
    pattern_celeb = f"nsfw/*/activation/{layer_fn}/*.pt"
    # 2. 匹配 non_celebrity 下的所有: root/non_celebrity/*/activation/{layer}/*.pt
    pattern_nc = f"no_nsfw/*/activation/{layer_fn}/*.pt"
    
    files_celeb = list(root_path.glob(pattern_celeb))
    files_nc = list(root_path.glob(pattern_nc))
    
    all_files = files_celeb + files_nc
    
    print(f"🔍 [Stats] 找到层 {layer_name} 的数据文件共 {len(all_files)} 个")
    print(f"   - Celebrity files: {len(files_celeb)}")
    print(f"   - Non-Celebrity files: {len(files_nc)}")

    if len(all_files) == 0:
        raise ValueError(f"No files found for layer {layer_name} in {root_dir}")

    for f_path in tqdm(all_files, desc=f"Calculating Stats [{layer_name}]"):
        data = torch.load(f_path, map_location="cpu")
        if 'original_acts' in data:
            stats_calculator.update(data['original_acts'])

    mean, std, dead_mask = stats_calculator.get_stats()
    
    # 保存结果到 root/global_stats
    save_dir = root_path / "nsfw_global_stats"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    save_path = save_dir / f"stats_{layer_fn}.pt"
    torch.save({
        'mean': mean, 
        'std': std, 
        'dead_mask': dead_mask,
        'n_samples': stats_calculator.count
    }, save_path)
    
    print(f"✅ 统计完成！发现 {dead_mask.sum().item()} 个死神经元。")
    return mean, std

def set_seed(seed: int):
    """固定所有随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def parse_target_layers(layer_str: str):
    """解析目标层字符串，支持格式如 "15-35:2" 或 "15,17,19" """
    if '-' in layer_str:
        # 范围格式：start-end:step
        parts = layer_str.split('-')
        start = int(parts[0])
        end_step = parts[1].split(':')
        end = int(end_step[0]) + 1  # 包含end
        step = int(end_step[1]) if len(end_step) > 1 else 1
        layers = [f"single_transformer_blocks.{i}.proj_out" for i in range(start, end, step)]
    elif ',' in layer_str:
        # 列表格式：15,17,19
        indices = [int(x.strip()) for x in layer_str.split(',')]
        layers = [f"single_transformer_blocks.{i}.proj_out" for i in indices]
    else:
        # 单个层
        layers = [f"single_transformer_blocks.{layer_str}.proj_out"]
    return layers



def train_offline(rank, world_size, args):
    # 设置设备
    """device="cuda"
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[INFO] Using dtype: {dtype}, device: {device}")"""

    local_rank, device = init_distributed_device(rank, world_size)
    setup_run_logging(args, Path(__file__).stem, rank)
    
    # 2. 只有 Rank 0 进行打印和统计量计算
    if rank == 0:
        print(f"[INFO] Initialized DDP with {world_size} processes.")
        print(f"[INFO] Using device: {device} (local_rank={local_rank})")
        
    # 确保每个进程都使用相同的随机种子
    set_seed(args.global_seed + rank)

    target_layers = args.target_layers_list # 解析层列表
    
    # 确定 target 和 generic 的切分点
    # 假设我们知道总共有 N 个文件，前 N/2 是 Target，后 N/2 是 Generic
    # 或者我们在 data collection 时可以通过文件名区分 batch_target_x.pt 和 batch_generic_x.pt
    # 简单起见，假设按索引切分 (需要您保证收集时的顺序)
    
    for layer_name in target_layers:
        layer_fn = layer_name.replace(".", "_")
        if rank == 0:
            print(f"\n>>> Processing Layer: {layer_name} (Rank 0)")
        quarantine_stale_attribution(args.sae_save_dir, layer_fn, rank)
        
        if args.activation_source == "online":
            layer_dataloader = create_online_hunyuan_loader(
                args=args,
                layer_name=layer_name,
                task="nudity",
                device=device,
                rank=rank,
                world_size=world_size,
            )
            classes = layer_dataloader.concepts
            log_loader_summary(args, "train", layer_name, layer_dataloader, rank)

            global_mean, global_std, dead_mask = prepare_online_normalization(
                args=args,
                layer_dataloader=layer_dataloader,
                layer_name=layer_name,
                layer_fn=layer_fn,
                data_dim=3072,
                rank=rank,
            )
        else:
            # 1. 路径与统计量处理 (所有进程指向统一的 global_layer_stats)
            stats_root = Path(args.data_save_dir) / "nsfw_global_stats"
            stats_path = stats_root / f"stats_{layer_fn}.pt"

            if rank == 0:
                stats_root.mkdir(exist_ok=True, parents=True)
                if not stats_path.exists():
                    global_mean, global_std = run_global_stats_collection(
                        root_dir=args.data_save_dir,
                        layer_name=layer_name,
                        data_dim=3072
                    )
                else:
                    stats_data = torch.load(stats_path, map_location="cpu")
                    global_mean, global_std = stats_data['mean'], stats_data['std']
                    print(f"已加载全局统计量: {stats_path}")

            dist.barrier()

            if rank != 0:
                stats_data = torch.load(stats_path, map_location="cpu")
                global_mean = stats_data['mean']
                global_std = stats_data['std']

            layer_dataloader, classes = create_dataloader_for_layer(
                data_root_dir=args.data_save_dir,
                layer_name=layer_name,
                mean=global_mean,
                std=global_std,
                batch_size=args.batch_size,
                rank=rank,
                world_size=world_size,
                target_celebs=args.target_celebs_list
            )

        sae_path = None
        if args.skip_train:
            sae_path = resolve_layer_checkpoint(args.sae_save_dir, layer_name)
            if rank == 0:
                print(
                    "[Checkpoint] Skipping training and running fresh "
                    f"attribution from {sae_path}."
                )

        sae = build_partitioned_sae(
            TopKConvPSAE,
            args,
            classes,
            d_model=3072,
            checkpoint=sae_path,
            layer_name=layer_name,
            verbose=(rank == 0),
        ).to(device)

        # 🚀 DDP 核心: 封装模型
        sae = torch.nn.parallel.DistributedDataParallel(sae, device_ids=[local_rank])
        
        # 4. 根据参数决定是否训练
        if not args.skip_train:
            train_conv_sae_final( 
                            args,
                            sae, 
                            layer_dataloader, 
                            device = device,
                            layer_name=layer_name,
                            )
        
        
        if args.skip_attribution:
            dist.barrier()
            if rank == 0:
                print(
                    "[Smoke] SAE training completed; attribution skipped. "
                    "This checkpoint is not inference-ready."
                )
            if hasattr(layer_dataloader, "unload"):
                layer_dataloader.unload()
            del layer_dataloader
            del sae
            torch.cuda.empty_cache()
            dist.barrier()
            continue

        # 🚀 2. 分布式特征挖掘 (适配卷积架构 & DDP汇总)
        dist.barrier()
        if rank == 0:
            print(f"\n🚀 [Phase 2] Starting Universal Feature Mining (Survival + Contrast)...")
        sae.eval()
        raw_model = sae.module if hasattr(sae, "module") else sae
        run_partitioned_attribution(
            args=args,
            data_loader=layer_dataloader,
            raw_model=raw_model,
            classes=classes,
            layer_name=layer_name,
            rank=rank,
            device=device,
        )

        if rank == 0:
            print(f"🧹 Cleaning up memory for layer {layer_name}...")

        # 1. 删除模型和 DDP 包装器
        if 'sae' in locals():
            del sae
        if 'raw_model' in locals():
            del raw_model
        
        # 2. 删除数据加载器 (你已经有了，但要确保引用彻底断开)
        if 'layer_dataloader' in locals():
            if hasattr(layer_dataloader, "unload"):
                layer_dataloader.unload()
            del layer_dataloader

        # 3. 删除特征挖掘阶段的大张量 (非常重要！这些都在 GPU 上)
        # 这些变量在下一轮循环会被重新定义，但为了安全先删掉
        # 3. 强制 Python 垃圾回收 (清理循环引用)
        import gc
        gc.collect()

        # 4. 清理 PyTorch 缓存 (将未使用的显存归还给 OS/分配器)
        torch.cuda.empty_cache()
        
        # 5. 等待所有进程完成清理，防止某个进程抢跑导致显存碎片
        dist.barrier()
        
        if rank == 0:
            print(f"✅ Memory cleaned. GPU allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB\n")

    # 循环结束后的清理
    dist.destroy_process_group()
        
    
def main(args):
    launch_training(args, train_offline)
if __name__ == "__main__":
    parser = build_training_parser(
        description="SAE-based feature erasure for HunyuanVideo",
        config_name="hunyuan_nudity",
    )
    args = parser.parse_args()
    finalize_training_args(args, parse_target_layers)
    if args.skip_train:
        args.sae_save_dir = str(
            resolve_checkpoint_run(args.sae_save_dir, args.target_layers_list)
        )

    # 固定种子
    set_seed(args.global_seed)
    print(f"[INFO] Global seed set to: {args.global_seed}")
    
    # 创建保存目录
    Path(args.sae_save_dir).mkdir(exist_ok=True, parents=True)

    print(f"SAE models will be saved to: {args.sae_save_dir}")
    
    print(f"Target layers: {args.target_layers_list}")
    
    main(args)
