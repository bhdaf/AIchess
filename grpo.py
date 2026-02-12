"""
GRPO (Group Relative Policy Optimization) 训练器

参考 DeepSeek R1 的思路，将传统的 Policy Gradient 替换为 GRPO 框架：
- 对同一局面采样多个动作（Group 采样机制）
- 利用组内相对收益作为奖励基准
- 不需要复杂的 Critic 网络，降低内存开销

用法：
    trainer = GRPOTrainer(model, group_size=8)
    loss = trainer.train_step(states, legal_actions_list)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_chess_ai.game import (
    ChessGame, NUM_ACTIONS, ACTION_LABELS, LABEL_TO_INDEX,
    flip_move, flip_policy
)


class GRPOTrainer:
    """
    GRPO 训练器

    核心思路：
    1. 对同一局面，从策略网络中采样 group_size 个动作
    2. 用价值网络（或环境回报）评估每个动作的收益
    3. 计算组内相对优势（advantage）= 个体收益 - 组内平均收益
    4. 用相对优势加权更新策略，无需单独的 Critic 网络

    Args:
        model: ChessModel 实例
        group_size: 每个局面采样的动作数
        clip_eps: PPO 风格裁剪系数
        kl_coeff: KL 散度正则化系数
        lr: 学习率
        use_fp16: 是否使用混合精度训练
    """

    def __init__(self, model, group_size=8, clip_eps=0.2,
                 kl_coeff=0.01, lr=1e-4, use_fp16=False):
        self.model = model
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        self.use_fp16 = use_fp16

        self.optimizer = torch.optim.Adam(
            model.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scaler = torch.amp.GradScaler('cuda') if use_fp16 else None

    def group_sample(self, policy_logits, legal_mask, group_size=None):
        """
        组采样机制：对同一局面采样多个动作

        Args:
            policy_logits: (batch, NUM_ACTIONS) 策略网络原始输出
            legal_mask: (batch, NUM_ACTIONS) 合法走法掩码 (1=合法, 0=非法)
            group_size: 每个局面采样动作数（默认使用 self.group_size）

        Returns:
            sampled_actions: (batch, group_size) 采样的动作索引
            sampled_log_probs: (batch, group_size) 对应的 log 概率
        """
        if group_size is None:
            group_size = self.group_size

        # 将非法动作的 logit 设为极小值
        masked_logits = policy_logits.clone()
        masked_logits[legal_mask == 0] = -1e9

        # 计算概率分布
        probs = F.softmax(masked_logits, dim=-1)

        batch_size = probs.shape[0]
        sampled_actions = torch.zeros(
            batch_size, group_size, dtype=torch.long, device=probs.device
        )
        sampled_log_probs = torch.zeros(
            batch_size, group_size, device=probs.device
        )

        # 对每个局面采样 group_size 个动作
        for g in range(group_size):
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            sampled_actions[:, g] = action
            sampled_log_probs[:, g] = log_prob

        return sampled_actions, sampled_log_probs

    def compute_group_advantage(self, rewards):
        """
        计算组内相对优势

        Args:
            rewards: (batch, group_size) 每个采样动作的收益

        Returns:
            advantages: (batch, group_size) 归一化的相对优势
        """
        # 组内均值和标准差
        group_mean = rewards.mean(dim=-1, keepdim=True)
        group_std = rewards.std(dim=-1, keepdim=True).clamp(min=1e-8)

        # 相对优势 = (个体收益 - 组均值) / 组标准差
        advantages = (rewards - group_mean) / group_std
        return advantages

    def evaluate_actions(self, model, states, actions):
        """
        评估采样动作的价值

        使用策略价值网络的价值头来估算每个动作的预期收益，
        而非使用单独的 Critic 网络。

        Args:
            model: ChessModel 实例
            states: (batch, 14, 10, 9) 棋盘特征
            actions: (batch, group_size) 动作索引

        Returns:
            rewards: (batch, group_size) 估算的收益
        """
        batch_size, group_size = actions.shape
        device = states.device

        # 获取当前策略和价值估计
        policy, value = model.model(states)

        # 用策略概率作为动作质量的代理指标
        # 高概率动作通常对应更好的收益
        log_policy = torch.log(policy + 1e-8)

        rewards = torch.zeros(batch_size, group_size, device=device)
        for g in range(group_size):
            action_indices = actions[:, g]
            # 组合价值估计和策略概率
            action_log_probs = log_policy.gather(1, action_indices.unsqueeze(1)).squeeze(1)
            rewards[:, g] = value.squeeze(-1) + action_log_probs

        return rewards

    def train_step(self, states, legal_masks, old_log_probs=None):
        """
        执行一步 GRPO 训练

        Args:
            states: numpy array (batch, 14, 10, 9) 棋盘特征
            legal_masks: numpy array (batch, NUM_ACTIONS) 合法走法掩码
            old_log_probs: 可选，旧策略的 log 概率（用于重要性采样）

        Returns:
            dict: 包含 loss, policy_loss, kl_loss 的训练指标
        """
        self.model.model.train()

        # 转换为 tensor
        if isinstance(states, np.ndarray):
            states = torch.FloatTensor(states).to(self.model.device)
        if isinstance(legal_masks, np.ndarray):
            legal_masks = torch.FloatTensor(legal_masks).to(self.model.device)

        amp_enabled = self.use_fp16 and states.device.type == 'cuda'

        with torch.amp.autocast('cuda', enabled=amp_enabled):
            # 1. 获取当前策略 logits
            policy, value = self.model.model(states)
            policy_logits = torch.log(policy + 1e-8)

            # 2. 组采样
            sampled_actions, sampled_log_probs = self.group_sample(
                policy_logits, legal_masks
            )

            # 3. 评估动作价值
            rewards = self.evaluate_actions(self.model, states, sampled_actions)

            # 4. 计算组内相对优势
            advantages = self.compute_group_advantage(rewards)

            # 5. 计算策略损失（GRPO 目标）
            # 加权 log 概率
            policy_loss = -(advantages.detach() * sampled_log_probs).mean()

            # 6. KL 散度正则化（防止策略更新过大）
            if old_log_probs is not None:
                if isinstance(old_log_probs, np.ndarray):
                    old_log_probs = torch.FloatTensor(old_log_probs).to(
                        self.model.device
                    )
                kl_div = F.kl_div(
                    F.log_softmax(policy_logits, dim=-1),
                    F.softmax(old_log_probs, dim=-1),
                    reduction='batchmean'
                )
            else:
                # 无旧策略时使用熵正则化
                entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=-1).mean()
                kl_div = -0.01 * entropy  # 鼓励探索

            total_loss = policy_loss + self.kl_coeff * kl_div

        # 反向传播
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        self.model.model.eval()

        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_div.item(),
        }


def generate_grpo_training_data(model, game, num_simulations=50):
    """
    为 GRPO 训练生成数据

    Args:
        model: ChessModel 实例
        game: ChessGame 实例
        num_simulations: MCTS 模拟次数（可以减少，因为 GRPO 本身有探索）

    Returns:
        states: 棋盘特征列表
        legal_masks: 合法走法掩码列表
    """
    states = []
    legal_masks = []

    planes = game.to_planes()
    states.append(planes)

    # 生成合法走法掩码
    legal_moves = game.get_legal_moves()
    if not game.red_to_move:
        legal_moves = [flip_move(m) for m in legal_moves]

    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for move in legal_moves:
        if move in LABEL_TO_INDEX:
            mask[LABEL_TO_INDEX[move]] = 1.0
    legal_masks.append(mask)

    return np.array(states), np.array(legal_masks)
