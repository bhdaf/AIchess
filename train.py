"""
训练管线

实现自对弈数据生成和模型训练的完整流程：
1. 自对弈：使用MCTS生成训练数据
2. 训练：用生成的数据训练神经网络
3. 循环：重复以上步骤持续提升

用法：
    python -m simple_chess_ai.train --num_games 100 --num_epochs 10
"""

import os
import json
import time
import argparse
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from simple_chess_ai.game import (
    ChessGame, NUM_ACTIONS, ACTION_LABELS, LABEL_TO_INDEX,
    flip_move, flip_policy, fen_to_planes
)
from simple_chess_ai.model import ChessModel
from simple_chess_ai.mcts import MCTS

# 默认模型保存路径
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, 'model.pth')
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), 'train_data')


def self_play_game(model, num_simulations=100, max_moves=200, temperature_threshold=30):
    """
    执行一局自对弈

    Args:
        model: ChessModel实例
        num_simulations: MCTS模拟次数
        max_moves: 最大步数（超过判和）
        temperature_threshold: 前N步使用温度1.0探索

    Returns:
        training_data: [(state_planes, policy_target, value_target), ...]
    """
    game = ChessGame()
    game.reset()
    mcts = MCTS(model, num_simulations=num_simulations)

    states = []
    policies = []
    players = []  # 记录每步的走子方

    move_count = 0

    while not game.done and move_count < max_moves:
        # 温度控制
        temperature = 1.0 if move_count < temperature_threshold else 0.1

        # MCTS搜索
        actions, probs = mcts.get_action_probs(game, temperature=temperature, add_noise=True)

        if not actions:
            break

        # 记录训练数据
        planes = game.to_planes()
        policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for action, prob in zip(actions, probs):
            if action in LABEL_TO_INDEX:
                policy_target[LABEL_TO_INDEX[action]] = prob

        states.append(planes)
        policies.append(policy_target)
        players.append(1 if game.red_to_move else -1)

        # 按概率选择走法
        action_idx = np.random.choice(len(actions), p=probs)
        chosen_action = actions[action_idx]

        # 执行走法
        if not game.red_to_move:
            actual_action = flip_move(chosen_action)
        else:
            actual_action = chosen_action
        game.step(actual_action)
        mcts.update_with_move(chosen_action)

        move_count += 1

    # 确定胜负
    if game.winner == 'red':
        winner = 1
    elif game.winner == 'black':
        winner = -1
    else:
        winner = 0

    # 生成训练数据
    training_data = []
    for state, policy, player in zip(states, policies, players):
        value = winner * player  # 从该玩家视角的评估值
        training_data.append((state, policy, value))

    return training_data, game.winner, move_count


def train_model(model, training_data, batch_size=256, epochs=5, lr=0.001):
    """
    用训练数据训练模型

    Args:
        model: ChessModel实例
        training_data: [(planes, policy, value), ...]
        batch_size: 批大小
        epochs: 训练轮数
        lr: 学习率

    Returns:
        avg_loss: 平均损失
    """
    if not training_data:
        return 0.0

    # 准备数据
    states = np.array([d[0] for d in training_data])
    policies = np.array([d[1] for d in training_data])
    values = np.array([d[2] for d in training_data], dtype=np.float32)

    dataset = TensorDataset(
        torch.FloatTensor(states),
        torch.FloatTensor(policies),
        torch.FloatTensor(values)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 设置训练模式
    model.model.train()
    optimizer = optim.Adam(model.model.parameters(), lr=lr, weight_decay=1e-4)

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for batch_states, batch_policies, batch_values in dataloader:
            batch_states = batch_states.to(model.device)
            batch_policies = batch_policies.to(model.device)
            batch_values = batch_values.to(model.device)

            # 前向传播
            pred_policies, pred_values = model.model(batch_states)

            # 策略损失：交叉熵
            policy_loss = -torch.mean(
                torch.sum(batch_policies * torch.log(pred_policies + 1e-8), dim=1)
            )
            # 价值损失：均方误差
            value_loss = torch.mean((batch_values - pred_values.squeeze()) ** 2)
            # 总损失
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    model.model.eval()
    return total_loss / max(num_batches, 1)


def run_training(num_games=50, num_simulations=100, num_epochs=5,
                 batch_size=256, lr=0.001, max_moves=200,
                 buffer_size=10000, model_path=None, save_interval=10):
    """
    运行完整的训练流程

    Args:
        num_games: 总自对弈局数
        num_simulations: MCTS模拟次数
        num_epochs: 每次训练的轮数
        batch_size: 批大小
        lr: 学习率
        max_moves: 每局最大步数
        buffer_size: 训练数据缓冲区大小
        model_path: 模型保存路径
        save_interval: 每隔多少局保存一次模型
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    # 初始化模型
    model = ChessModel(num_channels=128, num_res_blocks=4)
    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        model.load(model_path)
    else:
        print("创建新模型")
        model.build()

    # 训练数据缓冲区
    data_buffer = deque(maxlen=buffer_size)

    stats = {'red_wins': 0, 'black_wins': 0, 'draws': 0}

    print(f"\n{'='*60}")
    print(f"开始训练")
    print(f"自对弈局数: {num_games}")
    print(f"MCTS模拟次数: {num_simulations}")
    print(f"模型保存路径: {model_path}")
    print(f"{'='*60}\n")

    for game_idx in range(1, num_games + 1):
        start_time = time.time()

        # 自对弈
        data, winner, moves = self_play_game(
            model, num_simulations=num_simulations, max_moves=max_moves
        )
        data_buffer.extend(data)

        # 统计
        if winner == 'red':
            stats['red_wins'] += 1
        elif winner == 'black':
            stats['black_wins'] += 1
        else:
            stats['draws'] += 1

        elapsed = time.time() - start_time
        print(f"[第 {game_idx}/{num_games} 局] "
              f"胜方: {winner or '和棋'}, "
              f"步数: {moves}, "
              f"新增数据: {len(data)}, "
              f"缓冲区: {len(data_buffer)}, "
              f"耗时: {elapsed:.1f}s")

        # 训练（每局都训练，但数据足够时才有效）
        if len(data_buffer) >= batch_size:
            train_data = list(data_buffer)
            avg_loss = train_model(
                model, train_data, batch_size=batch_size,
                epochs=num_epochs, lr=lr
            )
            print(f"  训练完成，平均损失: {avg_loss:.4f}")

        # 定期保存模型
        if game_idx % save_interval == 0:
            model.save(model_path)
            print(f"  模型已保存到: {model_path}")

    # 最终保存
    model.save(model_path)
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"红方胜: {stats['red_wins']}, "
          f"黑方胜: {stats['black_wins']}, "
          f"和棋: {stats['draws']}")
    print(f"模型已保存到: {model_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='简化中国象棋AI - 训练')
    parser.add_argument('--num_games', type=int, default=50,
                        help='自对弈局数 (默认: 50)')
    parser.add_argument('--num_simulations', type=int, default=100,
                        help='每步MCTS模拟次数 (默认: 100)')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='每次训练轮数 (默认: 5)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='训练批大小 (默认: 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率 (默认: 0.001)')
    parser.add_argument('--max_moves', type=int, default=200,
                        help='每局最大步数 (默认: 200)')
    parser.add_argument('--buffer_size', type=int, default=10000,
                        help='训练数据缓冲区大小 (默认: 10000)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型保存路径')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='每隔多少局保存模型 (默认: 10)')

    args = parser.parse_args()
    run_training(**vars(args))


if __name__ == '__main__':
    main()
