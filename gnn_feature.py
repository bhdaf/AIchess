"""
GNN 特征提取模块

将棋盘的 Image-like Tensor 转换为图结构，使用图神经网络建模棋子间的
攻击与防守关系（如：车看护炮、将受威胁等）。

实现使用纯 PyTorch（无需 torch_geometric），通过邻接矩阵和消息传递
实现图卷积操作。

用法：
    gnn = ChessGNN(node_features=16, hidden_dim=64, num_heads=4)
    graph_features = gnn(board_planes)  # (batch, feature_dim)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_chess_ai.game import (
    BOARD_HEIGHT, BOARD_WIDTH, NUM_ACTIONS, PIECE_TO_INDEX
)


class GraphConvLayer(nn.Module):
    """
    图卷积层（纯 PyTorch 实现）

    使用消息传递范式：h_i' = σ(W · Σ_j (a_ij · h_j) + b)

    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.self_linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, in_features) 节点特征
            adj: (batch, num_nodes, num_nodes) 邻接矩阵

        Returns:
            out: (batch, num_nodes, out_features)
        """
        # 消息聚合: 邻居特征加权求和
        neighbor_msg = torch.bmm(adj, x)  # (batch, num_nodes, in_features)
        neighbor_out = self.linear(neighbor_msg)

        # 自连接
        self_out = self.self_linear(x)

        # 合并
        out = neighbor_out + self_out

        # BatchNorm (reshape for BN)
        batch, nodes, feat = out.shape
        out = out.reshape(batch * nodes, feat)
        out = self.bn(out)
        out = out.reshape(batch, nodes, feat)

        return F.relu(out)


class GATLayer(nn.Module):
    """
    图注意力层（GAT）

    使用注意力机制动态计算边权重，建模棋子间的重要关系。

    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        num_heads: 注意力头数
    """

    def __init__(self, in_features, out_features, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        assert out_features % num_heads == 0

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, in_features)
            adj: (batch, num_nodes, num_nodes)

        Returns:
            out: (batch, num_nodes, out_features)
        """
        batch, num_nodes, _ = x.shape

        # 线性变换
        h = self.W(x)  # (batch, num_nodes, out_features)
        h = h.reshape(batch, num_nodes, self.num_heads, self.head_dim)

        # 计算注意力系数
        # a_src: (heads, head_dim) -> (1, 1, heads, head_dim)
        score_src = (h * self.a_src.unsqueeze(0).unsqueeze(0)).sum(-1)
        score_dst = (h * self.a_dst.unsqueeze(0).unsqueeze(0)).sum(-1)

        # (batch, num_nodes, heads) -> (batch, heads, num_nodes)
        score_src = score_src.permute(0, 2, 1)
        score_dst = score_dst.permute(0, 2, 1)

        # 注意力分数
        attn = score_src.unsqueeze(-1) + score_dst.unsqueeze(-2)
        attn = self.leaky_relu(attn)  # (batch, heads, N, N)

        # 用邻接矩阵遮蔽
        adj_expanded = adj.unsqueeze(1)  # (batch, 1, N, N)
        attn = attn.masked_fill(adj_expanded == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)  # handle all-masked rows

        # 聚合
        h_perm = h.permute(0, 2, 1, 3)  # (batch, heads, N, head_dim)
        out = torch.matmul(attn, h_perm)  # (batch, heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(batch, num_nodes, -1)

        # BatchNorm
        out = out.reshape(batch * num_nodes, -1)
        out = self.bn(out)
        out = out.reshape(batch, num_nodes, -1)

        return F.relu(out)


def build_chess_graph(board_planes):
    """
    将棋盘特征平面转换为图结构

    节点：棋盘上的每个位置（10x9=90个节点）
    边：棋子间的攻击/防守关系
      - 同色棋子之间：防守关系（互相保护）
      - 异色棋子直线上：攻击关系（车/炮威胁）
      - 空位相邻：移动关系

    Args:
        board_planes: (batch, 14, 10, 9) 特征平面

    Returns:
        node_features: (batch, 90, node_feat_dim) 节点特征
        adj_matrix: (batch, 90, 90) 邻接矩阵
    """
    batch_size = board_planes.shape[0]
    num_nodes = BOARD_HEIGHT * BOARD_WIDTH  # 90

    device = board_planes.device

    # 节点特征: 每个位置的 14 通道特征
    node_features = board_planes.reshape(batch_size, 14, num_nodes)
    node_features = node_features.permute(0, 2, 1)  # (batch, 90, 14)

    # 添加位置编码
    pos_x = torch.arange(BOARD_WIDTH, device=device).float() / BOARD_WIDTH
    pos_y = torch.arange(BOARD_HEIGHT, device=device).float() / BOARD_HEIGHT
    pos_grid_x = pos_x.unsqueeze(0).expand(BOARD_HEIGHT, -1).reshape(-1)
    pos_grid_y = pos_y.unsqueeze(1).expand(-1, BOARD_WIDTH).reshape(-1)
    pos_encoding = torch.stack([pos_grid_x, pos_grid_y], dim=-1)  # (90, 2)
    pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)

    node_features = torch.cat([node_features, pos_encoding], dim=-1)  # (batch, 90, 16)

    # 构建邻接矩阵（基于棋盘空间关系）
    adj_matrix = _build_spatial_adjacency(batch_size, board_planes, device)

    return node_features, adj_matrix


def _build_spatial_adjacency(batch_size, board_planes, device):
    """
    构建基于棋盘空间关系的邻接矩阵

    包含：
    1. 同行/同列关系（车、炮的攻击路线）
    2. 棋子间的邻近关系
    3. 自连接

    Args:
        batch_size: 批大小
        board_planes: (batch, 14, 10, 9)
        device: 计算设备

    Returns:
        adj: (batch, 90, 90) 归一化邻接矩阵
    """
    num_nodes = BOARD_HEIGHT * BOARD_WIDTH

    # 基础空间邻接（棋盘上的相邻关系）
    base_adj = torch.zeros(num_nodes, num_nodes, device=device)

    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            idx = y * BOARD_WIDTH + x
            # 上下左右邻居
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < BOARD_HEIGHT and 0 <= nx < BOARD_WIDTH:
                    nidx = ny * BOARD_WIDTH + nx
                    base_adj[idx, nidx] = 1.0

            # 同行连接（距离衰减，限制距离阈值）
            for nx in range(BOARD_WIDTH):
                if nx != x and abs(nx - x) <= 4:
                    nidx = y * BOARD_WIDTH + nx
                    dist = abs(nx - x)
                    base_adj[idx, nidx] = max(base_adj[idx, nidx], 1.0 / dist)

            # 同列连接（距离衰减，限制距离阈值）
            for ny in range(BOARD_HEIGHT):
                if ny != y and abs(ny - y) <= 5:
                    nidx = ny * BOARD_WIDTH + x
                    dist = abs(ny - y)
                    base_adj[idx, nidx] = max(base_adj[idx, nidx], 1.0 / dist)

            # 自连接
            base_adj[idx, idx] = 1.0

    # 归一化
    deg = base_adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
    base_adj = base_adj / deg

    # 扩展到 batch
    adj = base_adj.unsqueeze(0).expand(batch_size, -1, -1)

    return adj


class ChessGNN(nn.Module):
    """
    中国象棋图神经网络

    将棋盘转换为图结构，通过 GNN 提取棋子间的关系特征。

    Args:
        node_features: 节点特征维度（默认16 = 14通道 + 2位置编码）
        hidden_dim: 隐藏层维度
        output_dim: 输出特征维度
        num_heads: GAT 注意力头数
        num_layers: GNN 层数
    """

    def __init__(self, node_features=16, hidden_dim=64,
                 output_dim=128, num_heads=4, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(node_features, hidden_dim)

        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            self.gnn_layers.append(
                GATLayer(in_dim, hidden_dim, num_heads=num_heads)
            )

        # 全局池化后的输出
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, board_planes):
        """
        Args:
            board_planes: (batch, 14, 10, 9) 棋盘特征平面

        Returns:
            graph_features: (batch, output_dim) 图级别特征
        """
        # 构建图
        node_features, adj_matrix = build_chess_graph(board_planes)

        # 投影输入
        x = self.input_proj(node_features)

        # GNN 层
        for layer in self.gnn_layers:
            x = layer(x, adj_matrix)

        # 全局平均池化
        graph_features = x.mean(dim=1)  # (batch, hidden_dim)

        # 输出投影
        graph_features = self.output_proj(graph_features)

        return graph_features


class GNNPolicyValueNet(nn.Module):
    """
    集成 GNN 的策略价值网络

    在原有 ResNet 特征提取的基础上，增加 GNN 分支提取棋子关系特征，
    两个分支的特征融合后输入策略头和价值头。

    Args:
        num_channels: CNN 通道数
        num_res_blocks: 残差块数量
        gnn_hidden_dim: GNN 隐藏层维度
        gnn_output_dim: GNN 输出维度
        num_heads: GAT 注意力头数
    """

    def __init__(self, num_channels=128, num_res_blocks=4,
                 gnn_hidden_dim=64, gnn_output_dim=128, num_heads=4):
        super().__init__()
        from simple_chess_ai.model import ResBlock

        self.num_channels = num_channels

        # CNN 分支（与原 PolicyValueNet 相同）
        self.input_conv = nn.Conv2d(14, num_channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # GNN 分支
        self.gnn = ChessGNN(
            node_features=16,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_heads=num_heads
        )

        # 策略头（融合 CNN + GNN 特征）
        self.policy_conv = nn.Conv2d(num_channels, 4, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        cnn_policy_dim = 4 * BOARD_HEIGHT * BOARD_WIDTH
        self.policy_fc = nn.Linear(cnn_policy_dim + gnn_output_dim, NUM_ACTIONS)

        # 价值头（融合 CNN + GNN 特征）
        self.value_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        cnn_value_dim = 2 * BOARD_HEIGHT * BOARD_WIDTH
        self.value_fc1 = nn.Linear(cnn_value_dim + gnn_output_dim, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, 14, 10, 9) 棋盘特征

        Returns:
            policy: (batch, NUM_ACTIONS) 走法概率
            value: (batch, 1) 局面评估
        """
        # CNN 特征提取
        cnn_feat = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            cnn_feat = block(cnn_feat)

        # GNN 特征提取
        gnn_feat = self.gnn(x)  # (batch, gnn_output_dim)

        # 策略头
        p = F.relu(self.policy_bn(self.policy_conv(cnn_feat)))
        p = p.view(p.size(0), -1)
        p = torch.cat([p, gnn_feat], dim=-1)
        p = F.softmax(self.policy_fc(p), dim=1)

        # 价值头
        v = F.relu(self.value_bn(self.value_conv(cnn_feat)))
        v = v.view(v.size(0), -1)
        v = torch.cat([v, gnn_feat], dim=-1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
