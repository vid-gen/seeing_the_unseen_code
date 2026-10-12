"""
GNS_pyg.py - PyG 版本的 GNS (v4)

v4 关键特点：完全匹配原始 GNS 结构
- particle_encode 固定不变
- relation_encode 固定不变
- particle_effect 从 0 开始迭代
- particle_propagator 输入是 [particle_encode, effect_agg]
- 残差: relu(Wx + res)
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean


class ParticleEncoder(nn.Module):
    """粒子编码器 (支持可选 LayerNorm)"""
    def __init__(self, input_size, hidden_size, output_size, use_ln=False):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        if use_ln:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        if use_ln:
            layers.append(nn.LayerNorm(output_size))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RelationEncoder(nn.Module):
    """关系编码器 (支持可选 LayerNorm)"""
    def __init__(self, input_size, hidden_size, output_size, use_ln=False):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        if use_ln:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, hidden_size))
        if use_ln:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        if use_ln:
            layers.append(nn.LayerNorm(output_size))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Propagator(nn.Module):
    """传播器 (支持可选 LayerNorm 和深层 MLP)"""
    def __init__(self, input_size, output_size, residual=False,
                 use_ln=False, deep=False):
        super().__init__()
        self.residual = residual
        if deep:
            layers = [nn.Linear(input_size, output_size)]
            if use_ln:
                layers.append(nn.LayerNorm(output_size))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(output_size, output_size))
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        out = self.mlp(x)
        if self.residual and res is not None:
            return self.relu(out + res)
        else:
            return self.relu(out)


class ParticlePredictor(nn.Module):
    """粒子预测器 (支持可选 LayerNorm)"""
    def __init__(self, input_size, hidden_size, output_size, use_ln=False):
        super().__init__()
        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.ln_0 = nn.LayerNorm(hidden_size) if use_ln else nn.Identity()
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.ln_1 = nn.LayerNorm(hidden_size) if use_ln else nn.Identity()
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.ln_0(self.linear_0(x)))
        x = self.relu(self.ln_1(self.linear_1(x)))
        return self.linear_2(x)


class GNS_PyG(nn.Module):
    """
    PyG 版本的 GNS (v4) - 完全匹配原始结构

    原始结构:
    1. particle_encode = particle_encoder([attr, state*mask]) -> 固定
    2. relation_encode = relation_encoder([...]) -> 固定
    3. particle_effect = zeros -> 从 0 开始
    4. 循环:
       - effect_rel = relation_propagator([relation_encode, effect_r, effect_s])
       - effect_agg = aggregate(effect_rel)
       - effect = particle_propagator([particle_encode, effect_agg], res=effect_old)
       - particle_effect[particles] = effect
    5. output = predictor(particle_effect)[:n_particles]
    """

    def __init__(
        self,
        attr_dim: int = 3,
        state_dim: int = 6,
        relation_dim: int = 1,
        nf_particle: int = 200,
        nf_relation: int = 300,
        nf_effect: int = 200,
        pstep: int = 10,
        residual: bool = True,  # v4: 默认 True，与原始 train.py 一致
        use_layernorm: bool = False,
        deep_propagator: bool = False,
    ):
        super().__init__()

        self.attr_dim = attr_dim
        self.state_dim = state_dim
        self.nf_effect = nf_effect
        self.pstep = pstep
        self.residual = residual

        use_ln = use_layernorm

        # 粒子编码器
        # 输入: attr(3) + offset(6) + state*mask(6) = 15
        # 但 attr 在 forward 中会扩展为 attr + offset = 9
        # 所以输入维度 = attr_dim + state_dim * 2 = 3 + 6*2 = 15
        self.particle_encoder = ParticleEncoder(
            attr_dim + state_dim * 2, nf_particle, nf_effect, use_ln=use_ln)

        # 关系编码器
        # 输入: attr_r(9) + attr_s(9) + state_r_s_rel(4) + Ra(1) = 23
        # = 2*attr_dim + 2*state_dim + 4 + relation_dim
        # = 2*3 + 2*6 + 4 + 1 = 23
        self.relation_encoder = RelationEncoder(
            2 * attr_dim + 2 * state_dim + 4 + relation_dim,
            nf_relation, nf_relation, use_ln=use_ln)

        # 关系传播器
        # 输入: relation_encode(300) + receiver_effect(200) + sender_effect(200) = 700
        self.relation_propagator = Propagator(
            nf_relation + 2 * nf_effect, nf_effect, residual=False,
            use_ln=use_ln, deep=deep_propagator)

        # 粒子传播器
        # 输入: particle_encode(200) + effect_agg(200) = 400
        self.particle_propagator = Propagator(
            2 * nf_effect, nf_effect, residual=residual,
            use_ln=use_ln, deep=deep_propagator)

        # 粒子预测器
        self.particle_predictor = ParticlePredictor(
            nf_effect, nf_effect, 3, use_ln=use_ln)

    def forward(self, data):
        """
        完全匹配原始 GNS 的 forward 逻辑
        """
        x = data.x  # (N, 9) = attr(3) + vel_masked(6)
        edge_index = data.edge_index  # (2, E)
        edge_attr = data.edge_attr  # (E, 4)
        state = data.state  # (N, 6)
        batch = data.batch
        n_particles = data.n_particles

        device = x.device
        N = x.size(0)
        E = edge_index.size(1)

        # 提取 attr 和 vel_masked
        attr = x[:, :self.attr_dim]  # (N, 3)
        vel_masked = x[:, self.attr_dim:]  # (N, 6)

        # ========== 位置掩码 (与原始一致) ==========
        # pos_mask[0, :3] = 0，所以 state * pos_mask = [0,0,0, vx, vy, vz]
        # vel_masked 已经是这个格式

        # ========== 计算 offset (与原始一致) ==========
        offset = self._compute_offset(state, data)

        # 扩展 attr: attr + offset = (N, 9)
        attr_extended = torch.cat([attr, offset], dim=-1)

        # ========== 获取粒子索引 ==========
        # 原始代码中 node_r_idx = [0, 1, ..., n_particles-1] (只有粒子)
        # node_s_idx = [0, 1, ..., n_particles+n_shapes-1] (粒子 + floor)
        particle_mask = self._get_particle_mask(data)
        n_particles_total = particle_mask.sum().item()

        # ========== particle_encode (固定) ==========
        # 只对粒子计算，不对 floor 计算
        attr_r = attr_extended[particle_mask]  # (n_particles_total, 9)
        state_r = vel_masked[particle_mask]    # (n_particles_total, 6)，已经是 masked
        particle_encode = self.particle_encoder(
            torch.cat([attr_r, state_r], dim=-1))  # (n_particles_total, nf_effect)

        # ========== relation_encode (固定) ==========
        senders = edge_index[0]
        receivers = edge_index[1]

        attr_s = attr_extended[senders]    # (E, 9)
        attr_r_edge = attr_extended[receivers]  # (E, 9)

        # state_r_s_rel: 相对位置 + 距离 (已在 edge_attr 中)
        Ra = torch.zeros((E, 1), device=device)

        relation_encode = self.relation_encoder(
            torch.cat([attr_r_edge, attr_s, edge_attr, Ra], dim=-1))  # (E, nf_relation)

        # ========== particle_effect 初始化为 0 ==========
        # 对所有节点初始化，但只更新粒子
        particle_effect = torch.zeros((N, self.nf_effect), device=device)

        # ========== 消息传递循环 ==========
        for _ in range(self.pstep):
            # 获取边上的 effect
            effect_s = particle_effect[senders]    # (E, nf_effect)
            effect_r = particle_effect[receivers]  # (E, nf_effect)

            # 关系传播 (用固定的 relation_encode)
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], dim=-1))  # (E, nf_effect)

            # 聚合到粒子节点 (只聚合到粒子，不聚合到 floor)
            # 使用 scatter_add，receivers 中指向粒子的边会被聚合
            effect_agg = scatter_add(effect_rel, receivers, dim=0, dim_size=N)

            # 只取粒子的聚合结果
            effect_agg_particles = effect_agg[particle_mask]  # (n_particles_total, nf_effect)

            # 当前粒子的 effect
            effect_p_r = particle_effect[particle_mask]  # (n_particles_total, nf_effect)

            # 粒子传播 (用固定的 particle_encode)
            effect_p = self.particle_propagator(
                torch.cat([particle_encode, effect_agg_particles], dim=-1),
                res=effect_p_r)  # (n_particles_total, nf_effect)

            # 只更新粒子的 effect
            particle_effect[particle_mask] = effect_p

        # ========== 预测 ==========
        # 只对粒子预测
        pred = self.particle_predictor(particle_effect[particle_mask])

        return pred

    def _compute_offset(self, state, data):
        """计算到质心的偏移 (仅对 rigid)"""
        offset = torch.zeros_like(state)

        if not hasattr(data, 'object_id') or not hasattr(data, 'is_rigid'):
            return offset

        object_id = data.object_id
        is_rigid = data.is_rigid
        batch = data.batch

        max_obj = object_id.max().item() + 2
        unique_id = batch * max_obj + (object_id + 1)

        rigid_mask = is_rigid & (object_id >= 0)

        if rigid_mask.any():
            center = scatter_mean(state[rigid_mask], unique_id[rigid_mask], dim=0)
            particle_centers = center[unique_id[rigid_mask]]
            offset[rigid_mask] = state[rigid_mask] - particle_centers

        return offset

    def _get_particle_mask(self, data):
        """获取粒子节点的 mask (排除 floor)"""
        if hasattr(data, 'object_id'):
            return data.object_id >= 0
        else:
            batch = data.batch
            n_particles_list = data.n_particles
            if isinstance(n_particles_list, int):
                n_particles_list = [n_particles_list]
            elif torch.is_tensor(n_particles_list):
                n_particles_list = n_particles_list.tolist()

            mask = torch.zeros(data.x.size(0), dtype=torch.bool, device=data.x.device)
            batch_size = batch.max().item() + 1
            for b in range(batch_size):
                batch_mask = (batch == b)
                batch_indices = torch.where(batch_mask)[0]
                n_p = n_particles_list[b] if b < len(n_particles_list) else n_particles_list[0]
                mask[batch_indices[:n_p]] = True
            return mask


def create_gns_model(
    nf_particle: int = 200,
    nf_relation: int = 300,
    nf_effect: int = 200,
    pstep: int = 10,
    residual: bool = True,  # v4: 默认 True，与原始 train.py 一致
    use_layernorm: bool = False,
    deep_propagator: bool = False,
):
    """创建 GNS 模型"""
    return GNS_PyG(
        attr_dim=3,
        state_dim=6,
        relation_dim=1,
        nf_particle=nf_particle,
        nf_relation=nf_relation,
        nf_effect=nf_effect,
        pstep=pstep,
        residual=residual,
        use_layernorm=use_layernorm,
        deep_propagator=deep_propagator,
    )
