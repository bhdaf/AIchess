"""
训练管线

实现自对弈数据生成和模型训练的完整流程：
1. 自对弈：使用MCTS生成训练数据
2. 训练：用生成的数据训练神经网络
3. 循环：重复以上步骤持续提升

支持训练模式：
- 标准训练（原版AlphaZero策略）
- GRPO训练（Group Relative Policy Optimization）
- FP16混合精度训练

用法：
    python -m simple_chess_ai.train --num_games 100 --num_epochs 10
    python -m simple_chess_ai.train --num_games 100 --use_grpo
    python -m simple_chess_ai.train --num_games 100 --use_fp16
"""

import os
import json
import time
import copy
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


def evaluate_models(model_a, model_b, n_games=20, num_simulations=50, max_moves=200):
    """
    通过对局评测两个模型，返回 model_a 的胜率。

    对弈时禁用 Dirichlet 噪声，使用温度=0 的确定性走法，
    并交替红黑方以减少先手优势偏差。

    Args:
        model_a: 候选新模型
        model_b: 基准模型
        n_games: 对局数
        num_simulations: MCTS模拟次数
        max_moves: 每局最大步数

    Returns:
        (winrate_a, wins_a, wins_b, draws)
    """
    wins_a = 0
    wins_b = 0
    draws = 0

    for game_idx in range(n_games):
        # 交替先手以减少先手优势偏差
        if game_idx % 2 == 0:
            red_model, black_model = model_a, model_b
            a_is_red = True
        else:
            red_model, black_model = model_b, model_a
            a_is_red = False

        game = ChessGame()
        game.reset()
        mcts_red = MCTS(red_model, num_simulations=num_simulations)
        mcts_black = MCTS(black_model, num_simulations=num_simulations)

        move_count = 0
        while not game.done and move_count < max_moves:
            mcts = mcts_red if game.red_to_move else mcts_black
            actions, probs = mcts.get_action_probs(game, temperature=0.0, add_noise=False)
            if not actions:
                break

            chosen_action = actions[int(np.argmax(probs))]
            actual_action = chosen_action if game.red_to_move else flip_move(chosen_action)
            game.step(actual_action)
            mcts.update_with_move(chosen_action)
            move_count += 1

        winner = game.winner
        if winner == 'draw' or winner is None:
            draws += 1
        elif (winner == 'red' and a_is_red) or (winner == 'black' and not a_is_red):
            wins_a += 1
        else:
            wins_b += 1

    total = wins_a + wins_b + draws
    winrate_a = wins_a / total if total > 0 else 0.0
    return winrate_a, wins_a, wins_b, draws


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


def train_model(model, training_data, batch_size=256, epochs=5, lr=0.001,
                use_fp16=False):
    """
    用训练数据训练模型

    Args:
        model: ChessModel实例
        training_data: [(planes, policy, value), ...]
        batch_size: 批大小
        epochs: 训练轮数
        lr: 学习率
        use_fp16: 是否使用FP16混合精度训练

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

    # FP16 混合精度
    amp_enabled = use_fp16 and model.device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if amp_enabled else None

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for batch_states, batch_policies, batch_values in dataloader:
            batch_states = batch_states.to(model.device)
            batch_policies = batch_policies.to(model.device)
            batch_values = batch_values.to(model.device)

            with torch.amp.autocast('cuda', enabled=amp_enabled):
                # 前向传播；模型输出 logits（策略头不含 softmax）
                pred_logits, pred_values = model.model(batch_states)

                # 策略损失：交叉熵（用 log_softmax 数值更稳定，避免 log(softmax+eps)）
                policy_loss = -torch.mean(
                    torch.sum(batch_policies * torch.nn.functional.log_softmax(pred_logits, dim=1), dim=1)
                )
                # 价值损失：均方误差
                value_loss = torch.mean((batch_values - pred_values.squeeze()) ** 2)
                # 总损失
                loss = policy_loss + value_loss

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    model.model.eval()
    return total_loss / max(num_batches, 1)


def _save_training_config(model_path, config):
    """将训练配置写入 saved_model/config.json，方便复现与追溯。"""
    import datetime
    model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
    os.makedirs(model_dir, exist_ok=True)
    config['_start_time'] = datetime.datetime.now().isoformat(timespec='seconds')
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"训练配置已保存到: {config_path}")


def run_training(num_games=50, num_simulations=100, num_epochs=5,
                 batch_size=256, lr=0.001, max_moves=200,
                 buffer_size=10000, model_path=None, save_interval=10,
                 use_grpo=False, grpo_group_size=8, use_fp16=False,
                 gating_interval=20, gating_games=20, gating_winrate=0.55,
                 seed=None, deterministic=False):
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
        use_grpo: 是否使用GRPO训练
        grpo_group_size: GRPO组采样大小
        use_fp16: 是否使用FP16混合精度训练
        gating_interval: 每隔多少局进行一次 gating 评测（0 表示禁用）
        gating_games: gating 评测对局数
        gating_winrate: gating 接受阈值（新模型胜率需超过此值）
        seed: 随机种子（None 表示不固定）
        deterministic: 是否启用 cuDNN 确定性模式（可能降低训练速度）
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    # ── 可复现性：设置随机种子 ──────────────────────────────────────────────
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # 启用 cuDNN 确定性选项；注意可能降低训练速度
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 初始化模型
    model = ChessModel(num_channels=128, num_res_blocks=4)
    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        model.load(model_path)
    else:
        print("创建新模型")
        model.build()

    # 保存本次训练的配置（供复现/追溯）
    _save_training_config(model_path, dict(
        num_games=num_games, num_simulations=num_simulations,
        num_epochs=num_epochs, batch_size=batch_size, lr=lr,
        max_moves=max_moves, buffer_size=buffer_size,
        save_interval=save_interval, use_grpo=use_grpo,
        grpo_group_size=grpo_group_size, use_fp16=use_fp16,
        gating_interval=gating_interval, gating_games=gating_games,
        gating_winrate=gating_winrate, seed=seed,
        deterministic=deterministic,
    ))

    # 记录基准模型权重（用于 gating 回滚）
    best_model_state = copy.deepcopy(model.model.state_dict())

    # GRPO 训练器
    grpo_trainer = None
    if use_grpo:
        from simple_chess_ai.grpo import GRPOTrainer
        grpo_trainer = GRPOTrainer(
            model, group_size=grpo_group_size,
            lr=lr, use_fp16=use_fp16
        )

    # 训练数据缓冲区
    data_buffer = deque(maxlen=buffer_size)

    stats = {'red_wins': 0, 'black_wins': 0, 'draws': 0}

    training_mode = "GRPO" if use_grpo else "Standard"
    fp16_str = " + FP16" if use_fp16 else ""

    print(f"\n{'='*60}")
    print(f"开始训练 ({training_mode}{fp16_str})")
    print(f"自对弈局数: {num_games}")
    print(f"MCTS模拟次数: {num_simulations}")
    if use_grpo:
        print(f"GRPO组大小: {grpo_group_size}")
    if gating_interval > 0:
        print(f"Gating: 每 {gating_interval} 局评测 {gating_games} 局, 阈值 {gating_winrate:.0%}")
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
            if use_grpo and grpo_trainer is not None:
                # GRPO 训练模式
                from simple_chess_ai.grpo import generate_grpo_training_data
                grpo_game = ChessGame()
                grpo_game.reset()
                grpo_states, grpo_masks = generate_grpo_training_data(
                    model, grpo_game
                )
                grpo_metrics = grpo_trainer.train_step(grpo_states, grpo_masks)
                print(f"  GRPO训练完成，损失: {grpo_metrics['loss']:.4f}, "
                      f"策略损失: {grpo_metrics['policy_loss']:.4f}")
            else:
                # 标准训练模式
                train_data = list(data_buffer)
                avg_loss = train_model(
                    model, train_data, batch_size=batch_size,
                    epochs=num_epochs, lr=lr, use_fp16=use_fp16
                )
                print(f"  训练完成，平均损失: {avg_loss:.4f}")

        # 定期保存模型 & gating 评测
        if game_idx % save_interval == 0:
            model.save(model_path)
            print(f"  模型已保存到: {model_path}")

        # Gating：定期评测新模型 vs 基准模型
        if gating_interval > 0 and game_idx % gating_interval == 0:
            print(f"  [Gating] 开始评测 (第 {game_idx} 局后)...")
            ref_model = ChessModel(
                num_channels=model.num_channels,
                num_res_blocks=model.num_res_blocks
            )
            ref_model.build()
            ref_model.model.load_state_dict(copy.deepcopy(best_model_state))
            ref_model.model.eval()

            winrate, wins, losses, draws_g = evaluate_models(
                model, ref_model,
                n_games=gating_games,
                num_simulations=max(num_simulations // 2, 20),
                max_moves=max_moves
            )
            print(f"  [Gating] 新模型胜率: {winrate:.2%} "
                  f"(胜 {wins} / 负 {losses} / 和 {draws_g}), "
                  f"阈值: {gating_winrate:.0%}")
            if winrate > gating_winrate:
                print(f"  [Gating] ✓ 新模型被接受，更新基准模型")
                best_model_state = copy.deepcopy(model.model.state_dict())
            else:
                print(f"  [Gating] ✗ 新模型被拒绝，回滚至基准模型")
                model.model.load_state_dict(copy.deepcopy(best_model_state))
                model.model.eval()

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
    parser.add_argument('--use_grpo', action='store_true',
                        help='使用GRPO训练模式')
    parser.add_argument('--grpo_group_size', type=int, default=8,
                        help='GRPO组采样大小 (默认: 8)')
    parser.add_argument('--use_fp16', action='store_true',
                        help='使用FP16混合精度训练')
    parser.add_argument('--gating_interval', type=int, default=20,
                        help='每隔多少局进行 gating 评测，0 表示禁用 (默认: 20)')
    parser.add_argument('--gating_games', type=int, default=20,
                        help='gating 评测对局数 (默认: 20)')
    parser.add_argument('--gating_winrate', type=float, default=0.55,
                        help='gating 接受阈值，新模型胜率需超过此值 (默认: 0.55)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子，设置后可复现数据生成序列 (默认: None)')
    parser.add_argument('--deterministic', action='store_true',
                        help='开启 cuDNN 确定性模式（配合 --seed 使用，可能降低训练速度）')

    args = parser.parse_args()
    run_training(**vars(args))


if __name__ == '__main__':
    main()
