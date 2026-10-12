"""
train_pyg_ddp.py - DDP 版本的 PyG 训练脚本

基于 train_pyg.py，使用 DistributedDataParallel 支持多 GPU 训练。
不修改 train_pyg.py，通过 import 复用现有函数/类。

启动方式:
    torchrun --standalone --nproc_per_node=4 train_pyg_ddp.py --scenario Collide ...
"""

import os
import sys
import argparse
import time
import random
import math
import glob
import re
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler

from torch_geometric.loader import DataLoader

from train_pyg import compute_loss_per_graph, str2bool
from data_pyg import PhysionPyGDataset
from models.GNS_pyg import create_gns_model


# ---------------------------------------------------------------------------
# DDP 初始化 / 清理
# ---------------------------------------------------------------------------

def setup_ddp():
    """通过 torchrun 环境变量初始化 DDP，返回 (rank, local_rank, world_size)"""
    if "RANK" not in os.environ:
        print("ERROR: 必须通过 torchrun 启动本脚本。", file=sys.stderr)
        print("示例: torchrun --standalone --nproc_per_node=4 train_pyg_ddp.py ...",
              file=sys.stderr)
        sys.exit(1)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_ddp():
    """销毁进程组"""
    dist.barrier()
    dist.destroy_process_group()


def parse_scenarios(raw_scenarios):
    scenarios = [s.strip() for s in str(raw_scenarios).split(",") if s.strip()]
    if not scenarios:
        raise ValueError("No valid scenarios parsed from --scenario")
    return scenarios


class DistributedScenarioBalancedSampler(Sampler):
    """
    Distributed sampler with scenario-balanced sampling:
    - each scenario appears with (near) equal counts in an epoch
    - each selected scenario then samples a random trial uniformly
    """

    def __init__(self, dataset, num_replicas, rank, shuffle=True, seed=0, drop_last=False):
        if num_replicas <= 0:
            raise ValueError(f"Invalid num_replicas: {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank: {rank}")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        if self.drop_last:
            self.num_samples = len(self.dataset) // self.num_replicas
        else:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.scenarios = sorted(list(self.dataset.scenario_to_trial_indices.keys()))
        if not self.scenarios:
            raise RuntimeError("No scenarios available for balanced sampling.")

        self.n_trials = self.dataset.n_trials
        self.samples_per_trial = self.dataset.samples_per_trial

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        n_scenarios = len(self.scenarios)
        scenario_ids = list(range(n_scenarios))

        # Build an exactly balanced scenario schedule up to total_size.
        scenario_schedule = scenario_ids * (self.total_size // n_scenarios)
        scenario_schedule += scenario_ids[:self.total_size % n_scenarios]

        if self.shuffle:
            perm = torch.randperm(len(scenario_schedule), generator=g).tolist()
            scenario_schedule = [scenario_schedule[i] for i in perm]

        indices = []
        for sid in scenario_schedule:
            scenario = self.scenarios[sid]
            trial_pool = self.dataset.scenario_to_trial_indices[scenario]
            trial_pos = int(torch.randint(len(trial_pool), (1,), generator=g).item())
            trial_idx = trial_pool[trial_pos]

            if self.samples_per_trial > 1:
                copy_id = int(torch.randint(self.samples_per_trial, (1,), generator=g).item())
            else:
                copy_id = 0

            idx = trial_idx + copy_id * self.n_trials
            indices.append(idx)

        # Shard by rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError(
                f"Sampler produced wrong shard size: got {len(indices)}, expected {self.num_samples}")
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def _extract_epoch_from_ckpt_path(path):
    m = re.search(r"ckpt_epoch_(\d+)\.pth$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _list_regular_ckpts(outf):
    paths = glob.glob(os.path.join(outf, "ckpt_epoch_*.pth"))
    with_epoch = [(_extract_epoch_from_ckpt_path(p), p) for p in paths]
    with_epoch = [(e, p) for (e, p) in with_epoch if e >= 0]
    with_epoch.sort(key=lambda x: x[0])
    return with_epoch


def _list_legacy_epoch_ckpts(outf):
    paths = glob.glob(os.path.join(outf, "model_epoch_*.pth"))
    with_epoch = []
    for p in paths:
        m = re.search(r"model_epoch_(\d+)\.pth$", os.path.basename(p))
        if m:
            with_epoch.append((int(m.group(1)), p))
    with_epoch.sort(key=lambda x: x[0])
    return with_epoch


def prune_regular_checkpoints(outf, max_keep_ckpt):
    max_keep_ckpt = max(1, int(max_keep_ckpt))
    ckpts = _list_regular_ckpts(outf)
    to_delete = ckpts[:-max_keep_ckpt]
    for _, path in to_delete:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def resolve_resume_checkpoint(outf, resume):
    if not resume:
        return None

    if resume == "auto":
        ckpts = _list_regular_ckpts(outf)
        if ckpts:
            return ckpts[-1][1]
        legacy_ckpts = _list_legacy_epoch_ckpts(outf)
        if legacy_ckpts:
            return legacy_ckpts[-1][1]
        return None

    if os.path.isabs(resume):
        path = resume
    else:
        path = os.path.join(outf, resume)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


# ---------------------------------------------------------------------------
# 分布式 DataLoader
# ---------------------------------------------------------------------------

def create_ddp_dataloaders(args, world_size, rank):
    """创建带 DistributedSampler 的 PyG DataLoader"""
    scenarios = parse_scenarios(args.scenario)

    dataset_kwargs = dict(
        neighbor_radius=args.neighbor_radius,
        noise_std=args.noise_std,
        training_fpt=args.training_fpt,
        floor_cheat=args.floor_cheat,
        subsample_limit=args.subsample_limit,
    )

    train_dataset = PhysionPyGDataset(
        args.data_root, scenarios, "train",
        stat_path=args.stat_path,
        train_samples_per_trial=args.train_samples_per_trial,
        **dataset_kwargs)
    valid_dataset = PhysionPyGDataset(
        args.data_root, scenarios, "valid",
        stat_path=args.stat_path, **dataset_kwargs)

    if args.sampling_mode == "scenario_balanced":
        train_sampler = DistributedScenarioBalancedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed, drop_last=False)
    elif args.sampling_mode == "trial_proportional":
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        raise ValueError(f"Unknown sampling_mode: {args.sampling_mode}")

    valid_sampler = DistributedSampler(
        valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0)
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=valid_sampler, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0)

    return train_loader, valid_loader, train_sampler, scenarios


# ---------------------------------------------------------------------------
# DDP 训练循环
# ---------------------------------------------------------------------------

def train_ddp(args):
    """DDP 主训练函数"""
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    # 固定随机种子 (每个 rank 使用不同种子以获得数据多样性)
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if is_main:
        print(f"=== DDP Training: {world_size} GPUs ===")
        os.makedirs(args.outf, exist_ok=True)

    # 等待 rank 0 创建目录
    dist.barrier()

    # 数据
    train_loader, valid_loader, train_sampler, scenarios = create_ddp_dataloaders(
        args, world_size, rank)

    if is_main:
        print(f"Scenarios: {scenarios}")
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Valid dataset size: {len(valid_loader.dataset)}")
        print(f"Train samples_per_trial: {train_loader.dataset.samples_per_trial}")
        print(f"Per-GPU batch size: {args.batch_size}, "
              f"Global batch size: {args.batch_size * world_size}")
        print(f"Sampling mode: {args.sampling_mode}")

    # 模型
    model = create_gns_model(
        nf_particle=args.nf_particle,
        nf_relation=args.nf_relation,
        nf_effect=args.nf_effect,
        pstep=args.pstep,
        residual=args.residual,
        use_layernorm=args.use_layernorm,
        deep_propagator=args.deep_propagator,
    )
    model = model.to(device)

    # 优化器
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, betas=(args.beta1, 0.999),
            weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # 学习率调度器
    if args.lr_schedule == "cosine_restart":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cosine_T0, T_mult=args.cosine_T_mult,
            eta_min=args.cosine_eta_min)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-6, verbose=is_main)

    # 断点恢复 (先载入原模型，再包装 DDP)
    start_epoch = 0
    best_valid_loss = float("inf")
    best_epoch = -1
    resume_path = resolve_resume_checkpoint(args.outf, args.resume)
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if args.resume_weights_only:
            # 只加载模型权重，optimizer/scheduler 使用新创建的
            if is_main:
                print(f"Loaded weights only from: {resume_path}")
                print(f"  (optimizer/scheduler reset to new settings)")
        else:
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_valid_loss = float(
                ckpt.get("best_valid_loss", ckpt.get("valid_loss", float("inf"))))
            best_epoch = int(ckpt.get("best_epoch", ckpt.get("epoch", -1)))
            if is_main:
                print(f"Resumed from: {resume_path}")
                print(f"  start_epoch={start_epoch}, best_valid_loss={best_valid_loss:.6f}, "
                      f"best_epoch={best_epoch}")
    elif is_main and args.resume:
        print("Resume requested but no checkpoint found for --resume auto. Start from scratch.")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

    optimizer.zero_grad()

    # 训练循环
    for epoch in range(start_epoch, args.n_epoch):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)

        # ---- 训练 ----
        model.train()
        losses = 0.0
        n_batches = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            n_batches += 1

            pred = model(data)
            target = data.y

            n_particles_list = data.n_particles
            if isinstance(n_particles_list, int):
                n_particles_list = [n_particles_list]
            elif torch.is_tensor(n_particles_list):
                n_particles_list = n_particles_list.tolist()

            loss = compute_loss_per_graph(pred, target, n_particles_list, device)
            loss = loss / args.forward_times

            loss.backward()
            # Step every forward_times batches.
            if (batch_idx + 1) % args.forward_times == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            current_loss = np.sqrt(loss.item() * args.forward_times)
            losses += np.sqrt(loss.item())

            if is_main and batch_idx % args.log_per_iter == 0:
                agg = losses / (batch_idx + 1)
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {current_loss:.6f}, Agg: {agg:.6f}")

        # Flush tail gradients if the last micro-batch group is incomplete.
        if n_batches > 0 and (n_batches % args.forward_times) != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = losses / max(n_batches, 1)

        # ---- 验证 (all_reduce 汇总) ----
        model.eval()
        valid_loss_sum = torch.tensor(0.0, device=device)
        valid_count = torch.tensor(0, device=device, dtype=torch.long)

        with torch.no_grad():
            for data in valid_loader:
                data = data.to(device)
                pred = model(data)
                target = data.y

                n_particles_list = data.n_particles
                if isinstance(n_particles_list, int):
                    n_particles_list = [n_particles_list]
                elif torch.is_tensor(n_particles_list):
                    n_particles_list = n_particles_list.tolist()

                loss = compute_loss_per_graph(pred, target, n_particles_list, device)
                loss = loss / args.forward_times

                valid_loss_sum += np.sqrt(loss.item())
                valid_count += 1

        # 跨 rank 汇总
        dist.all_reduce(valid_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_count, op=dist.ReduceOp.SUM)
        valid_loss = (valid_loss_sum / valid_count).item() if valid_count.item() > 0 else 0.0

        # 学习率调度
        if args.lr_schedule == "cosine_restart":
            scheduler.step(epoch)
        else:
            scheduler.step(valid_loss)

        epoch_time = time.time() - epoch_start

        if is_main:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} completed in {epoch_time:.1f}s")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Valid Loss: {valid_loss:.6f}")
            print(f"  LR: {current_lr:.2e}")

            # 更新 best 统计
            improved = valid_loss < best_valid_loss
            if improved:
                best_valid_loss = valid_loss
                best_epoch = epoch

            ckpt_payload = {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid_loss,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "valid_loss": valid_loss,
                "scenarios": scenarios,
            }

            # 保存最佳模型（与常规 checkpoint 分离）
            if improved:
                torch.save(ckpt_payload, os.path.join(args.outf, "best_model.pth"))
                print(f"  Saved best model (valid_loss: {valid_loss:.6f})")

            # 定期保存常规 checkpoint，并限制最多保留 N 个
            if epoch % args.ckp_per_epoch == 0:
                ckpt_path = os.path.join(args.outf, f"ckpt_epoch_{epoch}.pth")
                torch.save(ckpt_payload, ckpt_path)
                prune_regular_checkpoints(args.outf, args.max_keep_ckpt)

        # 所有 rank 同步后再进入下一个 epoch
        dist.barrier()

    cleanup_ddp()


# ---------------------------------------------------------------------------
# 参数解析 (与 train_pyg.py 完全一致)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GNS PyG model (DDP)")

    # 数据参数
    parser.add_argument("--data_root", type=str,
                        default="./data/preprocessed/train",
                        help="数据根目录")
    parser.add_argument("--stat_path", type=str, default=None,
                        help="统计文件路径 (stat.h5)")
    parser.add_argument("--scenario", type=str, default="Dominoes",
                        help="场景名称，多个用逗号分隔")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="每张卡的批次大小")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="数据加载线程数")
    parser.add_argument("--sampling_mode", type=str, default="scenario_balanced",
                        choices=["scenario_balanced", "trial_proportional"],
                        help="训练采样策略")
    parser.add_argument("--train_samples_per_trial", type=int, default=2,
                        help="训练集每个 trial 的 epoch 采样倍数")

    # 模型参数
    parser.add_argument("--nf_particle", type=int, default=200,
                        help="粒子特征维度")
    parser.add_argument("--nf_relation", type=int, default=300,
                        help="关系特征维度")
    parser.add_argument("--nf_effect", type=int, default=200,
                        help="效果特征维度")
    parser.add_argument("--pstep", type=int, default=10,
                        help="消息传递步数")
    parser.add_argument("--residual", type=str2bool, default=True,
                        help="是否使用残差连接 (true/false/1/0)")
    parser.add_argument("--use_layernorm", type=str2bool, default=False,
                        help="是否使用 LayerNorm (true/false/1/0)")
    parser.add_argument("--deep_propagator", type=str2bool, default=False,
                        help="是否使用深层 Propagator MLP (true/false/1/0)")

    # 数据处理参数
    parser.add_argument("--neighbor_radius", type=float, default=0.08,
                        help="邻居搜索半径")
    parser.add_argument("--noise_std", type=float, default=3e-4,
                        help="噪声标准差")
    parser.add_argument("--training_fpt", type=int, default=3,
                        help="训练帧间隔")
    parser.add_argument("--floor_cheat", type=str2bool, default=True,
                        help="是否使用 floor 节点 (true/false/1/0)")
    parser.add_argument("--subsample_limit", type=int, default=3000,
                        help="粒子子采样上限")

    # 训练参数
    parser.add_argument("--n_epoch", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1 (与原始 train.py 一致)")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw"],
                        help="优化器类型 (默认 adam, 保持原有行为)")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="权重衰减 (仅 adamw 有效, 默认 0 保持原有行为)")
    parser.add_argument("--grad_clip", type=float, default=0,
                        help="梯度裁剪 max_norm (0=不裁剪, 默认 0 保持原有行为)")
    parser.add_argument("--lr_schedule", type=str, default="plateau",
                        choices=["plateau", "cosine_restart"],
                        help="学习率调度策略 (默认 plateau, 保持原有行为)")
    parser.add_argument("--cosine_T0", type=int, default=50,
                        help="CosineAnnealingWarmRestarts 首次周期 (仅 cosine_restart)")
    parser.add_argument("--cosine_T_mult", type=int, default=2,
                        help="CosineAnnealingWarmRestarts 周期倍数 (仅 cosine_restart)")
    parser.add_argument("--cosine_eta_min", type=float, default=1e-6,
                        help="CosineAnnealingWarmRestarts 最小学习率 (仅 cosine_restart)")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子基数 (实际 seed = seed + rank)")
    parser.add_argument("--forward_times", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--log_per_iter", type=int, default=100,
                        help="日志打印频率")
    parser.add_argument("--ckp_per_epoch", type=int, default=1,
                        help="检查点保存频率")
    parser.add_argument("--max_keep_ckpt", type=int, default=5,
                        help="最多保留的常规 checkpoint 数量")
    parser.add_argument("--resume", type=str, default="",
                        help="恢复训练的 checkpoint 路径，或使用 auto 自动加载最新常规 checkpoint")
    parser.add_argument("--resume_weights_only", type=str2bool, default=False,
                        help="恢复时只加载模型权重，不加载 optimizer/scheduler (默认 false)")

    # 输出参数
    parser.add_argument("--outf", type=str, default="./output_pyg_ddp",
                        help="输出目录")

    args = parser.parse_args()
    if args.forward_times < 1:
        parser.error("--forward_times must be >= 1")
    if args.ckp_per_epoch < 1:
        parser.error("--ckp_per_epoch must be >= 1")
    if args.max_keep_ckpt < 1:
        parser.error("--max_keep_ckpt must be >= 1")
    if args.train_samples_per_trial < 1:
        parser.error("--train_samples_per_trial must be >= 1")

    train_ddp(args)


if __name__ == "__main__":
    main()
