"""
蒸馏训练管线（阶段 A：Pikafish 策略蒸馏）

两阶段训练流程说明
=================
阶段 A（本模块）—— 策略蒸馏 warm start：
  1. 弱引擎（weak agent）为双方走子，生成对局分布。
  2. 强引擎（teacher agent）在每个局面上通过 MultiPV 给出 soft policy_target
     （概率分布，float32，sum≈1，严禁 one-hot）。
  3. value_target 固定为 0（不训练 value head），避免引入错误的价值信号。
  4. 输出蒸馏后的模型权重，供阶段 B（RL 微调）加载。

阶段 B（train.py）—— RL 微调：
  使用 ``--init_from_distill`` 参数加载阶段 A 产出的权重，再运行自对弈 RL 训练。

用法示例
--------
阶段 A 蒸馏（弱-vs-弱 走子，强引擎标注）::

    python -m AIchess distill \\
        --engine_path /path/to/pikafish_weak \\
        --teacher_engine_path /path/to/pikafish_strong \\
        --out_model saved_model/model_distill.pth \\
        --n_games 200 \\
        --movetime_ms 100 \\
        --teacher_movetime_ms 300 \\
        --multipv_k 5

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


# ---------------------------------------------------------------------------
# Soft policy helpers
# ---------------------------------------------------------------------------

def _build_soft_policy_from_candidates(
    candidates: list,
    is_black_to_move: bool,
    temperature: float = 1.0,
) -> "np.ndarray | None":
    """
    将 MultiPV 候选列表转换为 soft policy 概率分布。

    Args:
        candidates:       ``[(internal_move, score_cp), ...]``，按排名从高到低。
        is_black_to_move: 当前是否轮到黑方走子（用于 flip_move 坐标变换）。
        temperature:      Softmax 温度（1.0 = 原始 cp 分；越高越平；越低越尖锐）。

    Returns:
        ``np.ndarray`` of shape ``(NUM_ACTIONS,)``，float32，sum ≈ 1；
        若没有任何候选映射到动作空间则返回 ``None``。
    """
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    indices: list = []
    scores: list = []

    for internal_mv, score_cp in candidates:
        policy_mv = flip_move(internal_mv) if is_black_to_move else internal_mv
        if policy_mv not in LABEL_TO_INDEX:
            continue
        indices.append(LABEL_TO_INDEX[policy_mv])
        scores.append(float(score_cp))

    if not indices:
        return None

    arr = np.array(scores, dtype=np.float64) / 100.0 / max(temperature, 1e-6)
    arr -= arr.max()  # numerical stability
    probs = np.exp(arr)
    probs /= probs.sum()

    for idx, prob in zip(indices, probs):
        policy[idx] += float(prob)

    return policy


def _build_soft_policy_fallback(
    bestmove: str,
    legal_moves: list,
    is_black_to_move: bool,
    peak_logit: float = 5.0,
) -> "np.ndarray | None":
    """
    当 MultiPV 不可用时，通过单个 bestmove + 所有合法走法构造 soft policy。

    bestmove 被赋予高 logit（``peak_logit``），其余合法走法赋 logit 0，
    经 softmax 后得到"尖锐但非 one-hot"的概率分布。

    Args:
        bestmove:         最优走法（内部格式）。
        legal_moves:      当前所有合法走法（内部格式）列表。
        is_black_to_move: 当前是否轮到黑方走子。
        peak_logit:       bestmove 的 logit（默认 5.0；控制分布的尖锐程度）。

    Returns:
        ``np.ndarray`` of shape ``(NUM_ACTIONS,)``，float32，sum ≈ 1；
        若 bestmove 不在动作空间则返回 ``None``。
    """
    policy_bestmove = flip_move(bestmove) if is_black_to_move else bestmove
    if policy_bestmove not in LABEL_TO_INDEX:
        return None

    best_idx = LABEL_TO_INDEX[policy_bestmove]

    # Collect valid legal move indices
    legal_indices: list = []
    for m in legal_moves:
        pm = flip_move(m) if is_black_to_move else m
        if pm in LABEL_TO_INDEX:
            legal_indices.append(LABEL_TO_INDEX[pm])

    if best_idx not in legal_indices:
        legal_indices.append(best_idx)

    # Deduplicate while preserving order
    seen: set = set()
    unique_indices: list = []
    for idx in legal_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    logits = np.zeros(len(unique_indices), dtype=np.float64)
    best_pos = unique_indices.index(best_idx)
    logits[best_pos] = peak_logit

    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for i, idx in enumerate(unique_indices):
        policy[idx] = float(probs[i])

    return policy


# ---------------------------------------------------------------------------
# Core game generation
# ---------------------------------------------------------------------------

def generate_distill_game(
    weak_agent,
    teacher_agent=None,
    max_moves: int = 200,
    multipv_k: int = 5,
    teacher_temperature: float = 1.0,
    anti_repetition_window: int = 6,
    repetition_draw_threshold: int = 6,
):
    """
    弱引擎走子 + 强引擎（teacher）标注 soft policy_target。

    policy_target 始终是概率分布（float32，长度 NUM_ACTIONS，sum≈1），
    严禁 one-hot。teacher 优先使用 MultiPV；失败时回退到"软标注策略"（见
    :func:`_build_soft_policy_fallback`）。

    黑方走法自动 ``flip_move`` 转换为红方视角，与 ACTION_LABELS 体系一致。

    Args:
        weak_agent:              实际走子的 Agent（``BaseAgent`` 子类）。
        teacher_agent:           提供 soft policy 标注的 Agent；若为 ``None``
                                 则复用 ``weak_agent``（统一 Agent 模式）。
        max_moves:               局内最大步数；超过后停止收集。
        multipv_k:               Teacher MultiPV 候选数量（默认 5）。
        teacher_temperature:     Softmax 温度（默认 1.0）。
        anti_repetition_window:  反重复检测窗口（最近 N 步内同一走法出现
                                 ≥2 次则换随机走法；0=禁用）。
        repetition_draw_threshold: ChessGame 重复局面判和阈值（默认 6，
                                   比标准阈值宽松，以减少蒸馏局过早结束）。

    Returns:
        training_data: ``[(state_planes, policy_target, value_target), ...]``
                        其中 value_target 恒为 0.0，policy_target 为 soft 分布。
        winner:        ``game.winner``（``'red'`` / ``'black'`` / ``'draw'`` / ``None``）。
        move_count:    实际步数。
        terminate_reason: ``game.terminate_reason``（字符串或 ``None``）。
    """
    game = ChessGame(repetition_draw_threshold=repetition_draw_threshold)
    game.reset()
    weak_agent.new_game()

    _teacher = teacher_agent if teacher_agent is not None else weak_agent
    if _teacher is not weak_agent:
        _teacher.new_game()

    training_data = []
    move_count = 0
    recent_moves: list = []  # for anti-repetition

    while not game.done and move_count < max_moves:
        state_planes = game.to_planes()
        legal = game.get_legal_moves()
        if not legal:
            break

        # ------------------------------------------------------------------
        # 1. Weak agent: choose the actual move
        # ------------------------------------------------------------------
        actual_move = weak_agent.get_move(game)
        if actual_move is None:
            actual_move = random.choice(legal)
            logger.warning("弱引擎未返回走法，改用随机走法: %s", actual_move)

        # Anti-repetition: if this move appeared ≥2 times in recent window,
        # substitute with a random alternative to escape back-and-forth loops.
        if anti_repetition_window > 0:
            window = recent_moves[-anti_repetition_window:]
            if window.count(actual_move) >= 2:
                alternatives = [m for m in legal if m != actual_move]
                if alternatives:
                    old_move = actual_move
                    actual_move = random.choice(alternatives)
                    logger.debug(
                        "反重复替换：%s → %s（窗口=%s）",
                        old_move, actual_move, window,
                    )

        recent_moves.append(actual_move)
        if len(recent_moves) > anti_repetition_window:
            recent_moves = recent_moves[-anti_repetition_window:]

        # ------------------------------------------------------------------
        # 2. Teacher: produce soft policy for the current position
        # ------------------------------------------------------------------
        policy: "np.ndarray | None" = None

        # Try MultiPV-based soft policy (PikafishAgent with get_soft_policy)
        if hasattr(_teacher, 'get_soft_policy'):
            try:
                policy = _teacher.get_soft_policy(
                    game, k=multipv_k, temperature=teacher_temperature,
                )
            except Exception as exc:
                logger.warning("get_soft_policy 失败，回退到软标注策略: %s", exc)
                policy = None

        # Soft fallback: ask teacher for its best move, build peaked distribution
        if policy is None:
            teacher_move = _teacher.get_move(game)
            if teacher_move is None:
                teacher_move = actual_move  # last resort
            policy = _build_soft_policy_fallback(
                teacher_move, legal, not game.red_to_move,
            )

        if policy is None:
            # Even fallback failed (teacher move not in action space).
            # Skip this sample but continue the game.
            logger.warning(
                "教师策略构建失败（走法: %r），跳过本步样本", actual_move
            )
            game.step(actual_move)
            move_count += 1
            continue

        # Sanity check: policy should sum to ≈ 1 with all non-negative values.
        policy_sum = float(policy.sum())
        if policy_sum < 1e-6:
            logger.warning("policy 全零，跳过本步样本")
            game.step(actual_move)
            move_count += 1
            continue
        if abs(policy_sum - 1.0) > 1e-3:
            policy = policy / policy_sum  # re-normalize

        training_data.append((state_planes, policy, 0.0))

        game.step(actual_move)
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

            # 策略损失：交叉熵（soft target 通用形式）
            # CE = -mean(sum(target * log_softmax(logits)))
            # 同时适用于 one-hot 和 soft label；soft label 使 KL 散度最小化。
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
    teacher_engine_path: str = None,
    teacher_movetime_ms: int = 300,
    teacher_options: dict = None,
    multipv_k: int = 5,
    teacher_temperature: float = 1.0,
    anti_repetition_window: int = 6,
    repetition_draw_threshold: int = 6,
):
    """
    运行完整的蒸馏流程（阶段 A）。

    Args:
        engine_path:              弱引擎可执行文件路径（必须）；用于实际走子。
        out_model:                蒸馏模型输出路径（默认 saved_model/model_distill.pth）。
        model_path:               基础模型路径；若指定且存在，则在其权重上继续蒸馏；
                                  否则创建新模型。
        n_games:                  蒸馏对局数（默认 200）。
        max_moves:                每局最大步数（默认 200）。
        movetime_ms:              弱引擎每步思考时间（ms，默认 100）。
        batch_size:               批大小（默认 256）。
        epochs:                   每批训练轮数（默认 5）。
        lr:                       学习率（默认 0.001）。
        value_loss_weight:        价值损失权重（蒸馏阶段建议 0.0；默认 0.0）。
        buffer_size:              训练数据缓冲区大小（默认 20000）。
        save_interval:            每隔多少局保存一次模型（默认 20）。
        engine_options:           传给弱引擎的选项字典（如 ``{"UCI_Elo": "1500"}``）。
        teacher_engine_path:      强引擎路径；若为 ``None`` 则复用弱引擎（unified 模式）。
        teacher_movetime_ms:      强引擎每步思考时间（ms，默认 300）。
        teacher_options:          传给强引擎的选项字典。
        multipv_k:                Teacher MultiPV 候选数量（默认 5）。
        teacher_temperature:      Softmax 温度（默认 1.0）。
        anti_repetition_window:   反重复检测窗口（默认 6）。
        repetition_draw_threshold: 重复局面判和阈值（默认 6）。
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
        teacher_engine_path=teacher_engine_path or engine_path,
        n_games=n_games,
        max_moves=max_moves,
        movetime_ms=movetime_ms,
        teacher_movetime_ms=teacher_movetime_ms,
        multipv_k=multipv_k,
        teacher_temperature=teacher_temperature,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        value_loss_weight=value_loss_weight,
        buffer_size=buffer_size,
        anti_repetition_window=anti_repetition_window,
    )
    run_dir = init_run_dir(config=distill_config)
    print(f"蒸馏日志目录: {run_dir}")

    data_buffer = deque(maxlen=buffer_size)
    stats = {'red_wins': 0, 'black_wins': 0, 'draws': 0}

    print(f"\n{'='*60}")
    print(f"阶段 A：Pikafish 策略蒸馏（soft target）")
    print(f"弱引擎: {engine_path}，思考时间: {movetime_ms} ms")
    teacher_desc = teacher_engine_path or engine_path
    print(f"Teacher:  {teacher_desc}，思考时间: {teacher_movetime_ms} ms，MultiPV: {multipv_k}")
    print(f"总局数: {n_games}，价值损失权重: {value_loss_weight}")
    print(f"蒸馏模型输出: {out_model}")
    print(f"{'='*60}\n")

    weak_opts = engine_options or {}
    strong_opts = teacher_options or {}

    def _run_games(weak_agent, teacher_agent):
        for game_idx in range(1, n_games + 1):
            start_time = time.time()

            data, winner, moves, terminate_reason = generate_distill_game(
                weak_agent,
                teacher_agent=teacher_agent,
                max_moves=max_moves,
                multipv_k=multipv_k,
                teacher_temperature=teacher_temperature,
                anti_repetition_window=anti_repetition_window,
                repetition_draw_threshold=repetition_draw_threshold,
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

    use_separate_teacher = (
        teacher_engine_path is not None
        and teacher_engine_path != engine_path
    )

    if use_separate_teacher:
        with PikafishAgent(
            engine_path, movetime_ms=movetime_ms, options=weak_opts
        ) as weak_agent:
            with PikafishAgent(
                teacher_engine_path,
                movetime_ms=teacher_movetime_ms,
                options=strong_opts,
                multipv=multipv_k,
            ) as teacher_agent:
                _run_games(weak_agent, teacher_agent)
    else:
        # Unified mode: same engine instance acts as both weak player and teacher.
        # MultiPV is set dynamically inside get_soft_policy().
        with PikafishAgent(
            engine_path, movetime_ms=movetime_ms, options=weak_opts
        ) as agent:
            _run_games(agent, None)

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
        description='简化中国象棋AI - 阶段A 策略蒸馏（Pikafish → 神经网络，soft target）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（阶段 A 蒸馏，弱-vs-弱 走子，强引擎标注）:
  python -m AIchess distill \\
      --engine_path /path/to/pikafish_weak \\
      --teacher_engine_path /path/to/pikafish_strong \\
      --out_model saved_model/model_distill.pth \\
      --n_games 200 --movetime_ms 100 --multipv_k 5

示例（阶段 B RL 微调，使用蒸馏权重为起点）:
  python -m AIchess train \\
      --init_from_distill saved_model/model_distill.pth \\
      --repetition_draw_threshold 6 \\
      --engine_path /path/to/pikafish \\
      --num_games 500 --num_simulations 200
        """,
    )
    parser.add_argument('--engine_path', type=str, required=True,
                        help='弱引擎（走子方）可执行文件路径（必须）')
    parser.add_argument('--teacher_engine_path', type=str, default=None,
                        help='强引擎（teacher）可执行文件路径；省略时复用弱引擎')
    parser.add_argument('--out_model', type=str, default=DEFAULT_DISTILL_MODEL_PATH,
                        help=f'蒸馏模型输出路径 (默认: {DEFAULT_DISTILL_MODEL_PATH})')
    parser.add_argument('--model_path', type=str, default=None,
                        help='基础模型路径；若指定且存在，则在其权重上继续蒸馏')
    parser.add_argument('--n_games', type=int, default=200,
                        help='蒸馏对局数 (默认: 200)')
    parser.add_argument('--max_moves', type=int, default=200,
                        help='每局最大步数 (默认: 200)')
    parser.add_argument('--movetime_ms', type=int, default=100,
                        help='弱引擎每步思考时间 ms (默认: 100)')
    parser.add_argument('--teacher_movetime_ms', type=int, default=300,
                        help='强引擎每步思考时间 ms (默认: 300)')
    parser.add_argument('--multipv_k', type=int, default=5,
                        help='Teacher MultiPV 候选数量 (默认: 5，最小 2)')
    parser.add_argument('--teacher_temperature', type=float, default=1.0,
                        help='Softmax 温度，控制 soft target 尖锐程度 (默认: 1.0)')
    parser.add_argument('--anti_repetition_window', type=int, default=6,
                        help='反重复检测窗口大小（0=禁用，默认: 6）')
    parser.add_argument('--repetition_draw_threshold', type=int, default=6,
                        help='重复局面判和阈值（默认: 6）')
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
        teacher_engine_path=args.teacher_engine_path,
        teacher_movetime_ms=args.teacher_movetime_ms,
        multipv_k=args.multipv_k,
        teacher_temperature=args.teacher_temperature,
        anti_repetition_window=args.anti_repetition_window,
        repetition_draw_threshold=args.repetition_draw_threshold,
    )


if __name__ == '__main__':
    main()
