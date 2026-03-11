"""
训练管线

实现自对弈数据生成和模型训练的完整流程：
1. 自对弈：使用MCTS生成训练数据
2. 训练：用生成的数据训练神经网络
3. 循环：重复以上步骤持续提升

用法：
    python -m AIchess train --num_games 50 --num_simulations 100
"""

import os
import json
import time
import argparse
import datetime
import logging
import random
import numpy as np
from collections import deque

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .game import (
    ChessGame, NUM_ACTIONS, ACTION_LABELS, LABEL_TO_INDEX,
    flip_move, flip_policy, fen_to_planes
)
from .model import ChessModel
from .mcts import MCTS
from .export import (
    init_run_dir, append_self_play_jsonl, append_training_csv,
    append_evaluation_csv, load_evaluation_state, save_evaluation_state,
)

logger = logging.getLogger(__name__)

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


def play_game_vs_opponent_collect_my_turn(
    model, opponent_agent, my_side='alternate', game_idx=0,
    num_simulations=100, max_moves=200, temperature_threshold=30,
    add_noise=True,
):
    """
    与对手 Agent 对弈，只收集我方回合的训练样本。

    对手回合调用 ``opponent_agent.get_move(game)``，不产生训练样本；
    我方回合通过 MCTS 获取动作概率分布，记录 ``(state_planes, policy_target, player_sign)``，
    终局后按胜负视角回填 value_target。

    对手走法容错：若 ``get_move`` 返回 ``None``（超时/非法），则随机选一步合法走法。

    Args:
        model: ChessModel 实例（当前训练模型）。
        opponent_agent: BaseAgent 子类实例，实现 ``new_game()``、``get_move(game)``。
        my_side: ``'red'``、``'black'`` 或 ``'alternate'``（按 game_idx 交替执红/黑）。
        game_idx: 当前局编号（从 0 开始，用于 alternate 模式确定执方）。
        num_simulations: 我方 MCTS 模拟次数。
        max_moves: 最大步数（超过判和）。
        temperature_threshold: 前 N 步使用温度 1.0 探索。
        add_noise: 是否向我方 MCTS 根节点注入 Dirichlet 噪声（训练时建议 True）。

    Returns:
        training_data: ``[(state_planes, policy_target, value_target), ...]``
        winner: ``'red'`` / ``'black'`` / ``'draw'`` / ``None``（对应原 game.winner）
        move_count: 实际步数（int）
        metadata: 描述字典，含 ``my_side``、``num_my_samples``
    """
    game = ChessGame()
    game.reset()

    # 确定我方执哪方
    if my_side == 'red':
        i_am_red = True
    elif my_side == 'black':
        i_am_red = False
    else:  # alternate
        i_am_red = (game_idx % 2 == 0)

    # 我方 MCTS（独立实例，不与对手共享）
    my_mcts = MCTS(model, num_simulations=num_simulations)

    opponent_agent.new_game()

    states = []
    policies = []
    players = []   # +1 = 红方，-1 = 黑方

    move_count = 0

    while not game.done and move_count < max_moves:
        is_my_turn = (game.red_to_move == i_am_red)

        if is_my_turn:
            temperature = 1.0 if move_count < temperature_threshold else 0.1
            actions, probs = my_mcts.get_action_probs(
                game, temperature=temperature, add_noise=add_noise
            )
            if not actions:
                logger.warning("我方 MCTS 无合法走法，局面：%s", game.get_fen())
                break

            # 记录训练样本（状态、策略目标、走棋方符号）
            planes = game.to_planes()
            policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for action, prob in zip(actions, probs):
                if action in LABEL_TO_INDEX:
                    policy_target[LABEL_TO_INDEX[action]] = prob

            states.append(planes)
            policies.append(policy_target)
            players.append(1 if game.red_to_move else -1)

            # 按概率采样走法
            action_idx = np.random.choice(len(actions), p=probs)
            chosen_action = actions[action_idx]  # MCTS 内部（红方视角）

            # 转换为实际棋盘坐标
            actual_action = (
                chosen_action if game.red_to_move else flip_move(chosen_action)
            )
            game.step(actual_action)
            my_mcts.update_with_move(chosen_action)

        else:
            # 对手回合：调用对手 Agent，不收集训练样本
            opponent_move = opponent_agent.get_move(game)

            if opponent_move is None:
                # 容错：对手无合法走法或超时，随机选一步合法走法
                legal = game.get_legal_moves()
                if not legal:
                    logger.warning("对手走法为 None 且棋盘已无合法走法，终止对局")
                    break
                opponent_move = random.choice(legal)
                logger.warning("对手走法为 None，改用随机走法: %s", opponent_move)

            game.step(opponent_move)

        move_count += 1

    # 计算终局胜负
    winner_raw = game.winner  # 'red' / 'black' / 'draw' / None
    if winner_raw == 'red':
        winner_val = 1
    elif winner_raw == 'black':
        winner_val = -1
    else:
        winner_val = 0

    # 回填 value_target（从我方视角看胜负）
    training_data = []
    for state, policy, player in zip(states, policies, players):
        value = winner_val * player
        training_data.append((state, policy, value))

    metadata = {
        'my_side': 'red' if i_am_red else 'black',
        'num_my_samples': len(training_data),
    }

    return training_data, winner_raw, move_count, metadata


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
                 eval_opponent='previous', elo_k=32,
                 # 对手池 / 课程学习参数
                 engine_path=None,
                 pikafish_movetime_weak=30,
                 pikafish_movetime_mid=60,
                 pikafish_movetime_full=100,
                 curriculum='default',
                 my_side='alternate',
                 eval_gate=0.55,
                 engine_options=None):
    """
    运行完整的训练流程

    Args:
        num_games: 总对局数
        num_simulations: MCTS模拟次数
        num_epochs: 每次训练的轮数
        batch_size: 批大小
        lr: 学习率
        max_moves: 每局最大步数
        buffer_size: 训练数据缓冲区大小
        model_path: 模型保存路径
        save_interval: 每隔多少局保存一次模型（同时加入历史 checkpoint 池）
        quick: 快速模式，用于验证流程
        eval_interval: 每隔多少局进行一次评测（0 表示禁用）
        eval_games: 评测对局数
        eval_simulations: 评测每步MCTS模拟次数
        eval_opponent: 评测对手类型（'previous' / 'self'）
        elo_k: ELO K 因子
        engine_path: Pikafish 等 UCI 引擎路径；为 None 时纯自对弈
        pikafish_movetime_weak: 弱强度引擎思考时间（ms）
        pikafish_movetime_mid: 中强度引擎思考时间（ms）
        pikafish_movetime_full: 全强度引擎思考时间（ms）
        curriculum: 课程学习策略（'default' 或 None/纯自对弈）
        my_side: 我方执哪方（'red'/'black'/'alternate'，仅对手池模式有效）
        eval_gate: 评测得分门控阈值（仅 eval_opponent='previous' 时生效，
                   score >= gate 才更新基准模型；默认 0.55）
        engine_options: 传给 UCI 引擎的选项字典
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
    use_opponent_pool = engine_path is not None
    training_config = dict(
        num_games=num_games, num_simulations=num_simulations,
        num_epochs=num_epochs, batch_size=batch_size, lr=lr,
        max_moves=max_moves, buffer_size=buffer_size,
        eval_interval=eval_interval, eval_games=eval_games,
        eval_simulations=eval_simulations, eval_opponent=eval_opponent,
        elo_k=elo_k,
        engine_path=engine_path,
        pikafish_movetime_weak=pikafish_movetime_weak,
        pikafish_movetime_mid=pikafish_movetime_mid,
        pikafish_movetime_full=pikafish_movetime_full,
        curriculum=curriculum,
        my_side=my_side,
        eval_gate=eval_gate,
    )
    run_dir = init_run_dir(config=training_config)
    print(f"日志目录: {run_dir}")

    # 初始化评测状态（含 ELO）
    eval_state = load_evaluation_state(run_dir)

    # 设置基准模型路径（用于 eval_opponent='previous'）
    baseline_path = os.path.join(run_dir, 'baseline.pth')
    if eval_interval > 0 and eval_opponent == 'previous':
        model.save(baseline_path)

    # 初始化对手池（仅 engine_path 非 None 时启用）
    pool = None
    if use_opponent_pool:
        from .opponent_pool import OpponentPool
        checkpoints_dir = os.path.join(run_dir, 'checkpoints')
        pool = OpponentPool(
            model=model,
            engine_path=engine_path,
            pikafish_movetime_weak=pikafish_movetime_weak,
            pikafish_movetime_mid=pikafish_movetime_mid,
            pikafish_movetime_full=pikafish_movetime_full,
            num_simulations=num_simulations,
            checkpoints_dir=checkpoints_dir,
            curriculum=curriculum,
            engine_options=engine_options or {},
        )

    data_buffer = deque(maxlen=buffer_size)
    stats = {'red_wins': 0, 'black_wins': 0, 'draws': 0}

    print(f"\n{'='*60}")
    print(f"开始训练")
    print(f"总对局数: {num_games}")
    print(f"MCTS模拟次数: {num_simulations}")
    print(f"模型保存路径: {model_path}")
    if use_opponent_pool:
        print(f"对手池模式: 启用（引擎: {engine_path}，课程: {curriculum}，"
              f"我方: {my_side}）")
    if eval_interval > 0:
        print(f"评测间隔: 每 {eval_interval} 局, 对局数: {eval_games}, "
              f"对手: {eval_opponent}, 门控: {eval_gate}")
    print(f"{'='*60}\n")

    try:
        for game_idx in range(1, num_games + 1):
            start_time = time.time()

            if use_opponent_pool and pool is not None:
                # 对手池模式：采样对手类型，收集我方回合数据
                opponent_type = pool.sample_opponent_type(game_idx, num_games)
                opponent_agent, opp_meta = pool.build_opponent(opponent_type)

                data, winner, moves, game_meta = play_game_vs_opponent_collect_my_turn(
                    model=model,
                    opponent_agent=opponent_agent,
                    my_side=my_side,
                    game_idx=game_idx - 1,
                    num_simulations=num_simulations,
                    max_moves=max_moves,
                )

                # 合并元数据
                opp_type_log = opp_meta.get('opponent_type', opponent_type)
                opp_strength_log = opp_meta.get('opponent_strength', '')
                engine_movetime_log = opp_meta.get('engine_movetime')
                my_side_log = game_meta.get('my_side', my_side)
            else:
                # 原有自对弈模式（不提供 engine_path 时）
                data, winner, moves = self_play_game(
                    model, num_simulations=num_simulations, max_moves=max_moves
                )
                opp_type_log = 'self_play'
                opp_strength_log = 'current'
                engine_movetime_log = None
                my_side_log = 'both'

            data_buffer.extend(data)

            if winner == 'red':
                stats['red_wins'] += 1
            elif winner == 'black':
                stats['black_wins'] += 1
            else:
                stats['draws'] += 1

            elapsed = time.time() - start_time
            print(f"[第 {game_idx}/{num_games} 局] "
                  f"对手: {opp_type_log}, "
                  f"胜方: {winner or '和棋'}, "
                  f"步数: {moves}, "
                  f"缓冲区: {len(data_buffer)}, "
                  f"耗时: {elapsed:.1f}s")

            jsonl_record = {
                'game_idx': game_idx,
                'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                'winner': winner or 'draw',
                'num_moves': moves,
                'num_samples': len(data),
                'elapsed_s': round(elapsed, 2),
                'opponent_type': opp_type_log,
                'opponent_strength': opp_strength_log,
                'my_side': my_side_log,
                'engine_movetime': engine_movetime_log,
            }
            append_self_play_jsonl(run_dir, jsonl_record)

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
                # 将 checkpoint 加入历史池（对手池模式）
                if pool is not None:
                    pool.add_checkpoint(model_path)

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

                # 评测门控：仅 score >= eval_gate 时才更新基准模型
                if eval_opponent == 'previous':
                    if score >= eval_gate:
                        model.save(baseline_path)
                        print(f"  [评测] score {score:.3f} >= gate {eval_gate}，"
                              f"基准模型已更新")
                    else:
                        print(f"  [评测] score {score:.3f} < gate {eval_gate}，"
                              f"保持基准模型不变")

    finally:
        # 确保引擎子进程被正确关闭
        if pool is not None:
            pool.close()

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
                        help='对局数 (默认: 50)')
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
    # 对手池 / 课程学习参数
    parser.add_argument('--engine_path', type=str, default=None,
                        help='Pikafish 等 UCI 引擎路径；提供后启用对手池训练模式')
    parser.add_argument('--pikafish_movetime_weak', type=int, default=30,
                        help='弱强度引擎思考时间 ms (默认: 30)')
    parser.add_argument('--pikafish_movetime_mid', type=int, default=60,
                        help='中强度引擎思考时间 ms (默认: 60)')
    parser.add_argument('--pikafish_movetime_full', type=int, default=100,
                        help='全强度引擎思考时间 ms (默认: 100)')
    parser.add_argument('--curriculum', type=str, default='default',
                        help="课程学习策略：'default' 或 'none'（全自对弈，默认: default）")
    parser.add_argument('--my_side', type=str, default='alternate',
                        choices=['red', 'black', 'alternate'],
                        help='我方执哪方（对手池模式有效，默认: alternate）')
    parser.add_argument('--eval_gate', type=float, default=0.55,
                        help='评测门控阈值：score >= gate 才更新基准模型 (默认: 0.55)')

    args = parser.parse_args()
    run_training(**vars(args))


if __name__ == '__main__':
    main()
