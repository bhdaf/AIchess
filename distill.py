"""
蒸馏训练管线（阶段 A：Pikafish 策略蒸馏）

两阶段训练流程说明
=================
阶段 A（本模块）—— 策略蒸馏 warm start：
  1. 让 Pikafish 在每一步提供 bestmove 作为 policy_target（one-hot 编码）。
  2. value_target 固定为 0（不训练 value head），避免引入错误的价值信号。
  3. 输出蒸馏后的模型权重，供阶段 B（RL 微调）加载。

阶段 B（train.py）—— RL 微调：
  使用 ``--init_from_distill`` 参数加载阶段 A 产出的权重，再运行自对弈 RL 训练。

用法示例
--------
阶段 A 蒸馏::

    python -m AIchess distill \\
        --engine_path /path/to/pikafish \\
        --out_model saved_model/model_distill.pth \\
        --n_games 200 \\
        --movetime_ms 100

阶段 B RL 微调（使用蒸馏权重为起点，并放宽重复判和阈值）::

    python -m AIchess train \\
        --init_from_distill saved_model/model_distill.pth \\
        --repetition_draw_threshold 6 \\
        --engine_path /path/to/pikafish \\
        --num_games 500 \\
        --num_simulations 200
"""

import os
import json
import time
import argparse
import datetime
import logging
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .game import (
    ChessGame, NUM_ACTIONS, LABEL_TO_INDEX,
    flip_move, fen_to_planes,
)
from .model import ChessModel
from .export import init_run_dir, append_self_play_jsonl, append_training_csv

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')
DEFAULT_DISTILL_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, 'model_distill.pth')


def generate_distill_game(engine_agent, max_moves: int = 200):
    """
    用 Pikafish 为双方走子，在每一步收集 (state_planes, policy_target) 样本。

    策略目标为 one-hot 编码：Pikafish bestmove 对应的动作索引置 1，其余置 0。
    value_target 固定为 0（不训练价值头）。

    黑方走法自动 flip_move 转换为红方视角，与 ACTION_LABELS 体系一致。

    Args:
        engine_agent: 已调用 ``start()`` 的 PikafishAgent 实例（或任意 BaseAgent）。
        max_moves:    局内最大步数；超过后停止收集。

    Returns:
        training_data: ``[(state_planes, policy_target, value_target), ...]``
                        其中 value_target 恒为 0.0。
        winner:        ``game.winner``（``'red'`` / ``'black'`` / ``'draw'`` / ``None``）。
        move_count:    实际步数。
        terminate_reason: ``game.terminate_reason``（字符串或 ``None``）。
    """
    game = ChessGame()
    game.reset()
    engine_agent.new_game()

    training_data = []
    move_count = 0

    while not game.done and move_count < max_moves:
        # 记录当前局面（始终为当前走棋方视角）
        state_planes = game.to_planes()

        # 从引擎获取 bestmove（board 坐标）
        bestmove = engine_agent.get_move(game)

        if bestmove is None:
            # 引擎无合法走法或超时，回退到随机合法走法
            legal = game.get_legal_moves()
            if not legal:
                break
            bestmove = random.choice(legal)
            logger.warning("引擎未返回走法，改用随机走法: %s", bestmove)

        # 将 bestmove 映射到 ACTION_LABELS（红方视角）
        # board 坐标：红方走法直接使用；黑方走法需 flip_move
        if game.red_to_move:
            policy_move = bestmove
        else:
            policy_move = flip_move(bestmove)

        policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if policy_move in LABEL_TO_INDEX:
            policy_target[LABEL_TO_INDEX[policy_move]] = 1.0
        else:
            # 走法不在动作空间内（极少数情况，如规则差异导致引擎走出内部不支持的走法）。
            # 跳过本步的训练样本，但仍执行走法以推进游戏，避免中断对局。
            # 这种局部缺失通常不影响蒸馏质量，因为 LABEL_TO_INDEX 覆盖所有合法走法。
            logger.warning("bestmove %r (policy: %r) 不在 LABEL_TO_INDEX 中，跳过", bestmove, policy_move)
            # 仍需执行走法推进游戏
            game.step(bestmove)
            move_count += 1
            continue

        # value_target = 0（蒸馏阶段不训练价值头）
        training_data.append((state_planes, policy_target, 0.0))

        game.step(bestmove)
        move_count += 1

    return training_data, game.winner, move_count, game.terminate_reason


def distill_model(model, training_data, batch_size: int = 256,
                  epochs: int = 5, lr: float = 0.001,
                  value_loss_weight: float = 0.0):
    """
    用蒸馏数据训练模型（策略蒸馏 + 可选价值损失）。

    默认 ``value_loss_weight=0.0``，即只训练策略头，不更新价值头。

    Args:
        model:             ChessModel 实例。
        training_data:     ``[(planes, policy, value), ...]``
        batch_size:        批大小（默认 256）。
        epochs:            训练轮数（默认 5）。
        lr:                学习率（默认 0.001）。
        value_loss_weight: 价值损失权重（蒸馏阶段建议 0.0；默认 0.0）。

    Returns:
        avg_loss (float): 平均总损失。
    """
    if not training_data:
        return 0.0

    states = np.array([d[0] for d in training_data])
    policies = np.array([d[1] for d in training_data])
    values = np.array([d[2] for d in training_data], dtype=np.float32)

    dataset = TensorDataset(
        torch.FloatTensor(states),
        torch.FloatTensor(policies),
        torch.FloatTensor(values),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.model.train()
    optimizer = optim.Adam(model.model.parameters(), lr=lr, weight_decay=1e-4)

    total_loss = 0.0
    num_batches = 0

    for _epoch in range(epochs):
        for batch_states, batch_policies, batch_values in dataloader:
            batch_states = batch_states.to(model.device)
            batch_policies = batch_policies.to(model.device)
            batch_values = batch_values.to(model.device)

            pred_logits, pred_values = model.model(batch_states)

            # 策略损失：交叉熵（one-hot target）
            policy_loss = -torch.mean(
                torch.sum(
                    batch_policies * torch.nn.functional.log_softmax(pred_logits, dim=1),
                    dim=1,
                )
            )
            # 价值损失（蒸馏阶段权重 = 0，不更新价值头）
            value_loss = torch.mean((batch_values - pred_values.squeeze()) ** 2)
            loss = policy_loss + value_loss_weight * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    model.model.eval()
    return total_loss / max(num_batches, 1)


def run_distill(
    engine_path: str,
    out_model: str = DEFAULT_DISTILL_MODEL_PATH,
    model_path: str = None,
    n_games: int = 200,
    max_moves: int = 200,
    movetime_ms: int = 100,
    batch_size: int = 256,
    epochs: int = 5,
    lr: float = 0.001,
    value_loss_weight: float = 0.0,
    buffer_size: int = 20000,
    save_interval: int = 20,
    engine_options: dict = None,
):
    """
    运行完整的蒸馏流程（阶段 A）。

    Args:
        engine_path:         Pikafish 可执行文件路径（必须）。
        out_model:           蒸馏模型输出路径（默认 saved_model/model_distill.pth）。
        model_path:          基础模型路径；若指定且存在，则在其权重上继续蒸馏；
                             否则创建新模型。
        n_games:             蒸馏对局数（默认 200）。
        max_moves:           每局最大步数（默认 200）。
        movetime_ms:         引擎每步思考时间（ms，默认 100）。
        batch_size:          批大小（默认 256）。
        epochs:              每批训练轮数（默认 5）。
        lr:                  学习率（默认 0.001）。
        value_loss_weight:   价值损失权重（蒸馏阶段建议 0.0；默认 0.0）。
        buffer_size:         训练数据缓冲区大小（默认 20000）。
        save_interval:       每隔多少局保存一次模型（默认 20）。
        engine_options:      传给 UCI 引擎的选项字典（如 ``{"UCI_Elo": "1500"}``）。
    """
    from collections import deque
    from .pikafish_agent import PikafishAgent

    out_dir = os.path.dirname(out_model) if os.path.dirname(out_model) else '.'
    os.makedirs(out_dir, exist_ok=True)

    # 初始化模型
    model = ChessModel(num_channels=128, num_res_blocks=4)
    if model_path and os.path.exists(model_path):
        print(f"从已有权重继续蒸馏: {model_path}")
        model.load(model_path)
    else:
        print("创建新模型（随机初始化）")
        model.build()

    # 初始化日志目录
    distill_config = dict(
        mode='distill',
        engine_path=engine_path,
        n_games=n_games,
        max_moves=max_moves,
        movetime_ms=movetime_ms,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        value_loss_weight=value_loss_weight,
        buffer_size=buffer_size,
    )
    run_dir = init_run_dir(config=distill_config)
    print(f"蒸馏日志目录: {run_dir}")

    data_buffer = deque(maxlen=buffer_size)
    stats = {'red_wins': 0, 'black_wins': 0, 'draws': 0}

    print(f"\n{'='*60}")
    print(f"阶段 A：Pikafish 策略蒸馏")
    print(f"引擎: {engine_path}，思考时间: {movetime_ms} ms")
    print(f"总局数: {n_games}，价值损失权重: {value_loss_weight}")
    print(f"蒸馏模型输出: {out_model}")
    print(f"{'='*60}\n")

    engine_opts = engine_options or {}

    with PikafishAgent(engine_path, movetime_ms=movetime_ms, options=engine_opts) as agent:
        for game_idx in range(1, n_games + 1):
            start_time = time.time()

            data, winner, moves, terminate_reason = generate_distill_game(
                agent, max_moves=max_moves
            )
            data_buffer.extend(data)

            if winner == 'red':
                stats['red_wins'] += 1
            elif winner == 'black':
                stats['black_wins'] += 1
            else:
                stats['draws'] += 1

            elapsed = time.time() - start_time
            print(f"[蒸馏 {game_idx}/{n_games}] "
                  f"步数: {moves}, "
                  f"原因: {terminate_reason or '-'}, "
                  f"样本: {len(data)}, "
                  f"缓冲区: {len(data_buffer)}, "
                  f"耗时: {elapsed:.1f}s")

            jsonl_record = {
                'game_idx': game_idx,
                'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                'winner': winner or 'draw',
                'terminate_reason': terminate_reason,
                'num_moves': moves,
                'num_samples': len(data),
                'elapsed_s': round(elapsed, 2),
                'mode': 'distill',
            }
            append_self_play_jsonl(run_dir, jsonl_record)

            avg_loss = 0.0
            if len(data_buffer) >= batch_size:
                avg_loss = distill_model(
                    model, list(data_buffer),
                    batch_size=batch_size,
                    epochs=epochs,
                    lr=lr,
                    value_loss_weight=value_loss_weight,
                )
                print(f"  蒸馏训练完成，平均策略损失: {avg_loss:.4f}")

                append_training_csv(run_dir, {
                    'game_idx': game_idx,
                    'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                    'loss': round(avg_loss, 6),
                    'buffer_size': len(data_buffer),
                    'elapsed_s': round(elapsed, 2),
                    'mode': 'distill',
                })

            if game_idx % save_interval == 0:
                model.save(out_model)
                print(f"  蒸馏模型已保存到: {out_model}")

    model.save(out_model)
    print(f"\n{'='*60}")
    print(f"蒸馏完成！")
    print(f"红方胜: {stats['red_wins']}, "
          f"黑方胜: {stats['black_wins']}, "
          f"和棋: {stats['draws']}")
    print(f"蒸馏模型已保存到: {out_model}")
    print(f"日志目录: {run_dir}")
    print(f"{'='*60}")
    print(f"\n下一步：RL 微调（阶段 B）")
    print(f"  python -m AIchess train \\")
    print(f"      --init_from_distill {out_model} \\")
    print(f"      --repetition_draw_threshold 6 \\")
    print(f"      --num_games 500 --num_simulations 200")


def main():
    parser = argparse.ArgumentParser(
        description='简化中国象棋AI - 阶段A 策略蒸馏（Pikafish → 神经网络）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（阶段 A 蒸馏）:
  python -m AIchess distill \\
      --engine_path /path/to/pikafish \\
      --out_model saved_model/model_distill.pth \\
      --n_games 200 --movetime_ms 100

示例（阶段 B RL 微调，使用蒸馏权重为起点）:
  python -m AIchess train \\
      --init_from_distill saved_model/model_distill.pth \\
      --repetition_draw_threshold 6 \\
      --engine_path /path/to/pikafish \\
      --num_games 500 --num_simulations 200
        """,
    )
    parser.add_argument('--engine_path', type=str, required=True,
                        help='Pikafish 等 UCI 引擎可执行文件路径（必须）')
    parser.add_argument('--out_model', type=str, default=DEFAULT_DISTILL_MODEL_PATH,
                        help=f'蒸馏模型输出路径 (默认: {DEFAULT_DISTILL_MODEL_PATH})')
    parser.add_argument('--model_path', type=str, default=None,
                        help='基础模型路径；若指定且存在，则在其权重上继续蒸馏')
    parser.add_argument('--n_games', type=int, default=200,
                        help='蒸馏对局数 (默认: 200)')
    parser.add_argument('--max_moves', type=int, default=200,
                        help='每局最大步数 (默认: 200)')
    parser.add_argument('--movetime_ms', type=int, default=100,
                        help='引擎每步思考时间 ms (默认: 100)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='训练批大小 (默认: 256)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='每批训练轮数 (默认: 5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率 (默认: 0.001)')
    parser.add_argument('--value_loss_weight', type=float, default=0.0,
                        help='价值损失权重；蒸馏阶段建议 0.0（默认: 0.0）')
    parser.add_argument('--buffer_size', type=int, default=20000,
                        help='训练数据缓冲区大小 (默认: 20000)')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='每隔多少局保存一次模型 (默认: 20)')

    args = parser.parse_args()
    run_distill(
        engine_path=args.engine_path,
        out_model=args.out_model,
        model_path=args.model_path,
        n_games=args.n_games,
        max_moves=args.max_moves,
        movetime_ms=args.movetime_ms,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        value_loss_weight=args.value_loss_weight,
        buffer_size=args.buffer_size,
        save_interval=args.save_interval,
    )


if __name__ == '__main__':
    main()
