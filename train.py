"""
训练管线

实现自对弈数据生成和模型训练的完整流程：
1. 自对弈：使用MCTS生成训练数据
2. 训练：用生成的数据训练神经网络
3. 循环：重复以上步骤持续提升

用法：
    python -m simple_chess_ai train --num_games 50 --num_simulations 100
"""

import os
import json
import time
import argparse
import datetime
import numpy as np
from collections import deque

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from simple_chess_ai.game import (
    ChessGame, NUM_ACTIONS, ACTION_LABELS, LABEL_TO_INDEX,
    flip_move, flip_policy, fen_to_planes
)
from simple_chess_ai.model import ChessModel
from simple_chess_ai.mcts import MCTS
from simple_chess_ai.export import (
    init_run_dir, append_self_play_jsonl, append_training_csv,
    append_evaluation_csv, load_evaluation_state, save_evaluation_state,
)

# 默认模型保存路径
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, 'model.pth')


def evaluate_models(model_a, model_b, n_games=20, num_simulations=50, max_moves=200):
    """
    通过对局评测两个模型，返回 model_a 的 score（draw=0.5 计分）。

    可作为独立工具函数使用，不在主训练循环中自动调用。

    Returns:
        (score, wins_a, wins_b, draws)
    """
    wins_a = 0
    wins_b = 0
    draws = 0

    for game_idx in range(n_games):
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
    score = (wins_a + 0.5 * draws) / total if total > 0 else 0.0
    return score, wins_a, wins_b, draws


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
    players = []

    move_count = 0

    while not game.done and move_count < max_moves:
        temperature = 1.0 if move_count < temperature_threshold else 0.1

        actions, probs = mcts.get_action_probs(game, temperature=temperature, add_noise=True)

        if not actions:
            break

        planes = game.to_planes()
        policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for action, prob in zip(actions, probs):
            if action in LABEL_TO_INDEX:
                policy_target[LABEL_TO_INDEX[action]] = prob

        states.append(planes)
        policies.append(policy_target)
        players.append(1 if game.red_to_move else -1)

        action_idx = np.random.choice(len(actions), p=probs)
        chosen_action = actions[action_idx]

        if not game.red_to_move:
            actual_action = flip_move(chosen_action)
        else:
            actual_action = chosen_action
        game.step(actual_action)
        mcts.update_with_move(chosen_action)

        move_count += 1

    if game.winner == 'red':
        winner = 1
    elif game.winner == 'black':
        winner = -1
    else:
        winner = 0

    training_data = []
    for state, policy, player in zip(states, policies, players):
        value = winner * player
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

    states = np.array([d[0] for d in training_data])
    policies = np.array([d[1] for d in training_data])
    values = np.array([d[2] for d in training_data], dtype=np.float32)

    dataset = TensorDataset(
        torch.FloatTensor(states),
        torch.FloatTensor(policies),
        torch.FloatTensor(values)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.model.train()
    optimizer = optim.Adam(model.model.parameters(), lr=lr, weight_decay=1e-4)

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for batch_states, batch_policies, batch_values in dataloader:
            batch_states = batch_states.to(model.device)
            batch_policies = batch_policies.to(model.device)
            batch_values = batch_values.to(model.device)

            pred_logits, pred_values = model.model(batch_states)

            # 策略损失：交叉熵
            policy_loss = -torch.mean(
                torch.sum(batch_policies * torch.nn.functional.log_softmax(pred_logits, dim=1), dim=1)
            )
            # 价值损失：均方误差
            value_loss = torch.mean((batch_values - pred_values.squeeze()) ** 2)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    model.model.eval()
    return total_loss / max(num_batches, 1)


def compute_elo_update(r_current, r_opponent, score, k=32):
    """
    计算 ELO 评分更新。

    使用 Logistic 期望值公式：

    .. code-block:: text

        expected = 1 / (1 + 10 ** ((R_opp - R_cur) / 400))
        R_cur_new = R_cur + K * (S - expected)

    Args:
        r_current (float): 当前模型的 ELO 评分。
        r_opponent (float): 对手模型的 ELO 评分。
        score (float): 本次对局的观测得分（win=1.0, draw=0.5, loss=0.0）。
        k (float): K 因子，控制评分变化幅度（默认 32）。

    Returns:
        tuple[float, float]: ``(new_rating, delta)`` 更新后的评分和变化量。
    """
    expected = 1.0 / (1.0 + 10.0 ** ((r_opponent - r_current) / 400.0))
    delta = k * (score - expected)
    return r_current + delta, delta


def run_training(num_games=50, num_simulations=100, num_epochs=5,
                 batch_size=256, lr=0.001, max_moves=200,
                 buffer_size=10000, model_path=None, save_interval=10,
                 quick=False,
                 eval_interval=0, eval_games=40, eval_simulations=50,
                 eval_opponent='previous', elo_k=32):
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
        quick: 快速模式，用于验证流程
        eval_interval: 每隔多少局进行一次评测（0 表示禁用）
        eval_games: 评测对局数
        eval_simulations: 评测每步MCTS模拟次数
        eval_opponent: 评测对手类型（'previous' / 'self'）
        elo_k: ELO K 因子
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if quick:
        num_games = 1
        num_simulations = 10
        num_epochs = 1
        batch_size = 16
        max_moves = 50

    model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'

    # 初始化模型
    model = ChessModel(num_channels=128, num_res_blocks=4)
    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        model.load(model_path)
    else:
        print("创建新模型")
        model.build()
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_path)
        print(f"初始模型已保存到: {model_path}")

    # 初始化数据导出目录
    training_config = dict(
        num_games=num_games, num_simulations=num_simulations,
        num_epochs=num_epochs, batch_size=batch_size, lr=lr,
        max_moves=max_moves, buffer_size=buffer_size,
        eval_interval=eval_interval, eval_games=eval_games,
        eval_simulations=eval_simulations, eval_opponent=eval_opponent,
        elo_k=elo_k,
    )
    run_dir = init_run_dir(config=training_config)
    print(f"日志目录: {run_dir}")

    # 初始化评测状态（含 ELO）
    eval_state = load_evaluation_state(run_dir)

    # 设置基准模型路径（用于 eval_opponent='previous'）
    baseline_path = os.path.join(run_dir, 'baseline.pth')
    if eval_interval > 0 and eval_opponent == 'previous':
        model.save(baseline_path)

    data_buffer = deque(maxlen=buffer_size)
    stats = {'red_wins': 0, 'black_wins': 0, 'draws': 0}

    print(f"\n{'='*60}")
    print(f"开始训练")
    print(f"自对弈局数: {num_games}")
    print(f"MCTS模拟次数: {num_simulations}")
    print(f"模型保存路径: {model_path}")
    if eval_interval > 0:
        print(f"评测间隔: 每 {eval_interval} 局, 对局数: {eval_games}, "
              f"对手: {eval_opponent}")
    print(f"{'='*60}\n")

    for game_idx in range(1, num_games + 1):
        start_time = time.time()

        data, winner, moves = self_play_game(
            model, num_simulations=num_simulations, max_moves=max_moves
        )
        data_buffer.extend(data)

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
              f"缓冲区: {len(data_buffer)}, "
              f"耗时: {elapsed:.1f}s")

        append_self_play_jsonl(run_dir, {
            'game_idx': game_idx,
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'winner': winner or 'draw',
            'num_moves': moves,
            'num_samples': len(data),
            'elapsed_s': round(elapsed, 2),
        })

        avg_loss = 0.0
        if len(data_buffer) >= batch_size:
            train_data = list(data_buffer)
            avg_loss = train_model(
                model, train_data, batch_size=batch_size,
                epochs=num_epochs, lr=lr
            )
            print(f"  训练完成，平均损失: {avg_loss:.4f}")

            append_training_csv(run_dir, {
                'game_idx': game_idx,
                'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                'loss': round(avg_loss, 6),
                'buffer_size': len(data_buffer),
                'elapsed_s': round(elapsed, 2),
            })

        if game_idx % save_interval == 0:
            model.save(model_path)
            print(f"  模型已保存到: {model_path}")

        # 定期评测
        if eval_interval > 0 and game_idx % eval_interval == 0:
            print(f"  [评测] 第 {game_idx} 局，开始评测...")
            if eval_opponent == 'self':
                opponent_model = ChessModel(num_channels=128, num_res_blocks=4)
                opponent_model.build()
                opponent_label = 'self'
            else:
                # 'previous': 对战上一个基准模型
                opponent_model = ChessModel(num_channels=128, num_res_blocks=4)
                if os.path.exists(baseline_path):
                    opponent_model.load(baseline_path)
                else:
                    opponent_model.build()
                opponent_label = 'previous'

            score, wins, losses, draws_eval = evaluate_models(
                model, opponent_model,
                n_games=eval_games,
                num_simulations=eval_simulations,
                max_moves=max_moves,
            )

            elo_cur = eval_state.get('elo_current', 1500.0)
            elo_opp = eval_state.get('elo_opponent', 1500.0)
            new_elo, elo_delta = compute_elo_update(elo_cur, elo_opp, score, k=elo_k)
            eval_state['elo_current'] = new_elo
            eval_state['last_game_idx'] = game_idx
            eval_state['last_opponent'] = opponent_label
            save_evaluation_state(run_dir, eval_state)

            print(f"  [评测] 胜: {wins}, 负: {losses}, 和: {draws_eval}, "
                  f"score: {score:.3f}, ELO: {new_elo:.1f} (Δ{elo_delta:+.1f})")

            append_evaluation_csv(run_dir, {
                'game_idx': game_idx,
                'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                'opponent': opponent_label,
                'eval_games': eval_games,
                'eval_sims': eval_simulations,
                'wins': wins,
                'losses': losses,
                'draws': draws_eval,
                'score': round(score, 6),
                'elo': round(new_elo, 2),
                'elo_delta': round(elo_delta, 2),
            })

            # 更新基准模型为当前检查点
            if eval_opponent == 'previous':
                model.save(baseline_path)

    model.save(model_path)
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"红方胜: {stats['red_wins']}, "
          f"黑方胜: {stats['black_wins']}, "
          f"和棋: {stats['draws']}")
    print(f"模型已保存到: {model_path}")
    print(f"日志目录: {run_dir}")
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
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型保存路径')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式：1局+1次训练，用于验证流程')
    parser.add_argument('--eval_interval', type=int, default=0,
                        help='每隔多少局进行一次评测（0 表示禁用，默认: 0）')
    parser.add_argument('--eval_games', type=int, default=40,
                        help='每次评测对局数 (默认: 40)')
    parser.add_argument('--eval_simulations', type=int, default=50,
                        help='评测每步MCTS模拟次数 (默认: 50)')
    parser.add_argument('--eval_opponent', type=str, default='previous',
                        choices=['previous', 'self'],
                        help='评测对手类型：previous=上一基准模型, self=随机初始模型 (默认: previous)')
    parser.add_argument('--elo_k', type=float, default=32,
                        help='ELO K 因子，控制评分变化幅度 (默认: 32)')

    args = parser.parse_args()
    run_training(**vars(args))


if __name__ == '__main__':
    main()
