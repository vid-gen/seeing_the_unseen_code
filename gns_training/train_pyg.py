"""
train_pyg.py - PyG 版本的训练脚本 (v4)

v4 关键特点:
- loss 按图平均再 batch 平均，与原始一致
- residual 默认 True
- 噪声缩放 noise_std / sqrt(n_velsteps)
"""

import os
import argparse
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch_scatter import scatter_mean

from data_pyg import PhysionPyGDataset, create_pyg_dataloaders
from models.GNS_pyg import GNS_PyG, create_gns_model


def str2bool(v):
    """Parse bool flags robustly (supports true/false/1/0)."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def compute_loss_per_graph(pred, target, n_particles_list, device):
    """
    按图计算 loss，再对 batch 平均

    原始 GNS: 每个 sample 等权
    如果直接对所有粒子均值，粒子多的场景权重更大

    Args:
        pred: (total_particles, 3) 预测速度
        target: (total_particles, 3) 目标速度
        n_particles_list: list of int, 每个图的粒子数
        device: torch device

    Returns:
        loss: scalar, batch 平均的 loss
    """
    if torch.is_tensor(n_particles_list):
        n_particles_list = n_particles_list.tolist()

    # 每个粒子的 loss (mean over 3D, 与源代码 nn.MSELoss 一致)
    loss_per_particle = ((pred - target) ** 2).mean(dim=-1)  # (total_particles,)

    # 构建粒子的图索引
    particle_graph_idx = []
    for graph_idx, n_p in enumerate(n_particles_list):
        particle_graph_idx.extend([graph_idx] * n_p)
    particle_graph_idx = torch.tensor(particle_graph_idx, device=device, dtype=torch.long)

    # 按图平均
    loss_per_graph = scatter_mean(loss_per_particle, particle_graph_idx, dim=0)

    # batch 平均
    loss = loss_per_graph.mean()

    return loss


def train_step(model, data, optimizer, device, forward_times, batch_idx):
    """单步训练"""
    model.train()

    data = data.to(device)

    # 前向传播
    pred = model(data)  # (total_particles, 3)
    target = data.y     # (total_particles, 3)

    # 获取每个图的粒子数
    n_particles_list = data.n_particles
    if isinstance(n_particles_list, int):
        n_particles_list = [n_particles_list]
    elif torch.is_tensor(n_particles_list):
        n_particles_list = n_particles_list.tolist()

    # 计算 loss (按图平均)
    loss = compute_loss_per_graph(pred, target, n_particles_list, device)
    loss = loss / forward_times

    # 反向传播
    loss.backward()
    if batch_idx % forward_times == 0 and batch_idx != 0:
        optimizer.step()
        optimizer.zero_grad()

    current_loss = np.sqrt(loss.item() * forward_times)
    return loss.item(), current_loss


def validate(model, valid_loader, device, forward_times):
    """验证"""
    model.eval()
    total_loss = 0
    n_batches = 0

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
            loss = loss / forward_times

            total_loss += np.sqrt(loss.item())
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0


def train(args):
    """主训练函数"""
    # 与原始 train.py 一致: 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建保存目录
    os.makedirs(args.outf, exist_ok=True)

    # 数据集
    scenarios = args.scenario.split(',')
    print(f"Scenarios: {scenarios}")

    train_loader, valid_loader = create_pyg_dataloaders(
        data_root=args.data_root,
        scenarios=scenarios,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stat_path=args.stat_path,
        neighbor_radius=args.neighbor_radius,
        noise_std=args.noise_std,
        training_fpt=args.training_fpt,
        floor_cheat=args.floor_cheat,
        subsample_limit=args.subsample_limit,
    )

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Valid dataset size: {len(valid_loader.dataset)}")

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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=3, verbose=True)

    # 与原始 train.py 一致: 梯度在循环外初始化
    optimizer.zero_grad()

    # 训练循环
    best_valid_loss = float('inf')
    for epoch in range(args.n_epoch):
        epoch_start = time.time()

        # 训练
        losses = 0.
        for batch_idx, data in enumerate(train_loader):
            loss, current_loss = train_step(
                model, data, optimizer, device, args.forward_times, batch_idx)
            losses += np.sqrt(loss)

            if batch_idx % args.log_per_iter == 0:
                agg = losses / (batch_idx + 1)
                print(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {current_loss:.6f}, Agg: {agg:.6f}")

        avg_train_loss = losses / len(train_loader)

        # 验证
        valid_loss = validate(model, valid_loader, device, args.forward_times)

        # 学习率调度
        scheduler.step(valid_loss)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} completed in {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Valid Loss: {valid_loss:.6f}")

        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'valid_loss': valid_loss,
            }, os.path.join(args.outf, 'best_model.pth'))
            print(f"  Saved best model (valid_loss: {valid_loss:.6f})")

        # 定期保存
        if epoch % args.ckp_per_epoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'valid_loss': valid_loss,
            }, os.path.join(args.outf, f'model_epoch_{epoch}.pth'))


def main():
    parser = argparse.ArgumentParser(description='Train GNS PyG model')

    # 数据参数
    parser.add_argument('--data_root', type=str,
                        default='./data/preprocessed/train',
                        help='数据根目录')
    parser.add_argument('--stat_path', type=str, default=None,
                        help='统计文件路径 (stat.h5)')
    parser.add_argument('--scenario', type=str, default='Dominoes',
                        help='场景名称，多个用逗号分隔')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载线程数')

    # 模型参数
    parser.add_argument('--nf_particle', type=int, default=200,
                        help='粒子特征维度')
    parser.add_argument('--nf_relation', type=int, default=300,
                        help='关系特征维度')
    parser.add_argument('--nf_effect', type=int, default=200,
                        help='效果特征维度')
    parser.add_argument('--pstep', type=int, default=10,
                        help='消息传递步数')
    parser.add_argument('--residual', type=str2bool, default=True,
                        help='是否使用残差连接 (true/false/1/0)')
    parser.add_argument('--use_layernorm', type=str2bool, default=False,
                        help='是否使用 LayerNorm (true/false/1/0)')
    parser.add_argument('--deep_propagator', type=str2bool, default=False,
                        help='是否使用深层 Propagator MLP (true/false/1/0)')

    # 数据处理参数
    parser.add_argument('--neighbor_radius', type=float, default=0.08,
                        help='邻居搜索半径')
    parser.add_argument('--noise_std', type=float, default=3e-4,
                        help='噪声标准差')
    parser.add_argument('--training_fpt', type=int, default=3,
                        help='训练帧间隔')
    parser.add_argument('--floor_cheat', type=str2bool, default=True,
                        help='是否使用 floor 节点 (true/false/1/0)')
    parser.add_argument('--subsample_limit', type=int, default=3000,
                        help='粒子子采样上限')

    # 训练参数
    parser.add_argument('--n_epoch', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1 (与原始 train.py 一致)')
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子 (与原始 train.py 一致)')
    parser.add_argument('--forward_times', type=int, default=2,
                        help='梯度累积步数 (与原始 train.py 一致)')
    parser.add_argument('--log_per_iter', type=int, default=100,
                        help='日志打印频率')
    parser.add_argument('--ckp_per_epoch', type=int, default=10,
                        help='检查点保存频率')

    # 输出参数
    parser.add_argument('--outf', type=str, default='./output_pyg',
                        help='输出目录')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
