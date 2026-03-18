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


# ---------------------------------------------------------------------------
# 和棋过滤辅助工具
# ---------------------------------------------------------------------------

def classify_draw(terminate_reason, num_moves, max_moves):
    """将一局和棋按终止原因分为三类：maxmoves / repetition / other。

    Args:
        terminate_reason: game.terminate_reason 字符串（可能为 None）。
        num_moves:        实际步数（int）。
        max_moves:        训练时设置的最大步数上限（int）。

    Returns:
        str: ``'maxmoves'``、``'repetition'`` 或 ``'other'``。
    """
    # repetition 类：重复局面 / 长将 / 长捉
    if terminate_reason in ('repetition', 'perpetual_check', 'perpetual_chase'):
        return 'repetition'
    # max-moves 类：显式达到步数上限，或无终止原因但步数到上限
    if terminate_reason is None or terminate_reason in ('-', 'max_moves', 'move_limit'):
        if num_moves >= max_moves:
            return 'maxmoves'
    # 其他（stalemate 等）
    return 'other'


class DrawFilter:
    """按概率决定是否保留一局和棋的训练数据。

    对于非和棋（win/loss）的对局，``should_keep`` 始终返回 True。
    对于和棋对局，根据 draw_category 和对应概率进行随机抽样。

    抽样使用独立的 ``random.Random`` 实例，以确保可复现且不影响其他
    随机数流（如 MCTS 噪声）。
    """

    def __init__(self,
                 keep_draw_maxmoves_prob=0.25,
                 keep_draw_other_prob=0.10,
                 keep_draw_repetition_prob=0.05,
                 seed=0):
        self._rng = random.Random(seed)
        self._probs = {
            'maxmoves': keep_draw_maxmoves_prob,
            'repetition': keep_draw_repetition_prob,
            'other': keep_draw_other_prob,
        }
        # 统计信息
        self.counts = {k: 0 for k in self._probs}
        self.kept = {k: 0 for k in self._probs}

    def should_keep(self, draw_category):
        """返回 True 时保留该和棋对局的训练数据。"""
        # 将未知类别归入 other
        if draw_category not in self._probs:
            draw_category = 'other'
        prob = self._probs[draw_category]
        self.counts[draw_category] += 1
        keep = self._rng.random() < prob
        if keep:
            self.kept[draw_category] += 1
        return keep

    def summary_str(self):
        """返回当前统计信息的单行字符串，用于日志输出。"""
        parts = []
        for cat in ('maxmoves', 'repetition', 'other'):
            total = self.counts.get(cat, 0)
            k = self.kept.get(cat, 0)
            pct = f'{k/total*100:.0f}%' if total > 0 else '-'
            parts.append(f'{cat}: {k}/{total}({pct})')
        return '  '.join(parts)


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


def self_play_game(model, num_simulations=100, max_moves=200, temperature_threshold=30,
                   repetition_draw_threshold=3, mcts_cache_size=0):
    """
    执行一局自对弈

    Args:
        model: ChessModel 实例，或实现了 ``predict_with_mask(planes, legal_indices)``
               接口的对象（如 :class:`~AIchess.inference_server.RemoteEvaluator`）
        num_simulations: MCTS模拟次数
        max_moves: 最大步数（超过判和）
        temperature_threshold: 前N步使用温度1.0探索
        repetition_draw_threshold: 重复局面判和阈值（传递给 ChessGame，默认 3）
        mcts_cache_size: MCTS 局面评估缓存大小（0=禁用；>0 可减少重复推理，
                         与 RemoteEvaluator 搭配使用时可降低 IPC 开销）

    Returns:
        training_data: [(state_planes, policy_target, value_target), ...]
    """
    game = ChessGame(repetition_draw_threshold=repetition_draw_threshold)
    game.reset()
    mcts = MCTS(model, num_simulations=num_simulations, cache_size=mcts_cache_size)

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

    return training_data, game.winner, move_count, game.terminate_reason


def play_game_vs_opponent_collect_my_turn(
    model, opponent_agent, my_side='alternate', game_idx=0,
    num_simulations=100, max_moves=200, temperature_threshold=30,
    add_noise=True, repetition_draw_threshold=3,
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
        repetition_draw_threshold: 重复局面判和阈值（传递给 ChessGame，默认 3）。

    Returns:
        training_data: ``[(state_planes, policy_target, value_target), ...]``
        winner: ``'red'`` / ``'black'`` / ``'draw'`` / ``None``（对应原 game.winner）
        move_count: 实际步数（int）
        metadata: 描述字典，含 ``my_side``、``num_my_samples``
    """
    game = ChessGame(repetition_draw_threshold=repetition_draw_threshold)
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
        'terminate_reason': game.terminate_reason,
    }

    return training_data, winner_raw, move_count, metadata


# ---------------------------------------------------------------------------
# 多进程自对弈 Worker 进程函数
# ---------------------------------------------------------------------------

def self_play_worker(
    worker_id,
    request_queue,
    response_queue,
    data_queue,
    shutdown_event,
    num_simulations=100,
    max_moves=200,
    temperature_threshold=30,
    repetition_draw_threshold=3,
    mcts_cache_size=0,
    seed=None,
):
    """
    自对弈 worker 进程入口函数（在独立子进程中运行）。

    使用 :class:`~AIchess.inference_server.RemoteEvaluator` 与推理服务通信，
    本进程**不**持有任何 GPU 资源，只负责运行 MCTS 搜索逻辑与棋局管理。

    Args:
        worker_id:               本 worker 的编号（与 response_queue 对应）
        request_queue:           所有 worker 共用的推理请求队列
        response_queue:          本 worker 专属的推理响应队列
        data_queue:              向主进程发送自对弈结果的队列
        shutdown_event:          主进程设置此事件时 worker 优雅退出
        num_simulations:         每步 MCTS 模拟次数
        max_moves:               每局最大步数
        temperature_threshold:   前 N 步使用温度 1.0 探索
        repetition_draw_threshold: 重复局面判和阈值
        mcts_cache_size:         MCTS 局面缓存大小（0=禁用）
        seed:                    基础随机数种子（每个 worker 加上自身 id）
    """
    import logging as _logging
    # Spawn context creates fresh processes; configure logging only if not yet configured.
    root_logger = _logging.getLogger()
    if not root_logger.handlers:
        _logging.basicConfig(
            level=_logging.INFO,
            format=f"%(asctime)s [Worker-{worker_id}] %(levelname)s %(message)s",
        )
    _logger = _logging.getLogger(__name__)

    # 设置可复现种子（每个 worker 独立）
    if seed is not None:
        import random as _random
        _random.seed(seed + worker_id)
        np.random.seed((seed + worker_id) & 0xFFFFFFFF)

    from .inference_server import RemoteEvaluator, InferenceServerShutdownError
    evaluator = RemoteEvaluator(worker_id, request_queue, response_queue)

    games_played = 0
    t_start = time.time()

    while not shutdown_event.is_set():
        try:
            training_data, winner, move_count, terminate_reason = self_play_game(
                evaluator,
                num_simulations=num_simulations,
                max_moves=max_moves,
                temperature_threshold=temperature_threshold,
                repetition_draw_threshold=repetition_draw_threshold,
                mcts_cache_size=mcts_cache_size,
            )
        except InferenceServerShutdownError:
            _logger.info("推理服务已关闭，worker 退出")
            break
        except RuntimeError as exc:
            _logger.error("Worker 运行时错误: %s", exc)
            break
        except Exception as exc:
            import traceback
            _logger.error("Worker 未预期错误: %s\n%s", exc, traceback.format_exc())
            break

        games_played += 1

        try:
            data_queue.put(
                {
                    "type": "game_data",
                    "worker_id": worker_id,
                    "training_data": training_data,
                    "winner": winner,
                    "move_count": move_count,
                    "terminate_reason": terminate_reason,
                },
                timeout=60.0,
            )
        except Exception as exc:
            _logger.warning("无法发送游戏数据到主进程: %s", exc)

        # 每 5 局打印一次吞吐量日志
        if games_played % 5 == 0:
            elapsed = time.time() - t_start
            _logger.info(
                "已完成 %d 局，速率 %.2f 局/s",
                games_played, games_played / max(elapsed, 1e-6),
            )

    # 通知主进程本 worker 已完成
    try:
        data_queue.put(
            {"type": "worker_done", "worker_id": worker_id, "games_played": games_played},
            timeout=5.0,
        )
    except Exception:
        pass

    _logger.info("Worker %d 已退出，共完成 %d 局", worker_id, games_played)


# ---------------------------------------------------------------------------
# 并行自对弈训练循环
# ---------------------------------------------------------------------------

def _run_parallel_training(
    model,
    model_path,
    num_games,
    num_simulations,
    num_epochs,
    batch_size,
    lr,
    max_moves,
    buffer_size,
    save_interval,
    run_dir,
    data_buffer,
    stats,
    draw_filter,
    eval_state,
    baseline_path,
    pool,
    eval_interval,
    eval_games,
    eval_simulations,
    eval_opponent,
    elo_k,
    eval_gate,
    repetition_draw_threshold,
    num_selfplay_workers,
    inference_batch_size_max,
    inference_flush_ms,
    inference_precision,
    use_tf32,
    mcts_cache_size,
    worker_seed,
):
    """
    并行自对弈训练主循环（生产者-消费者流水线）。

    * N 个 worker 进程持续生成自对弈数据并写入 data_queue。
    * 1 个推理服务进程独占 GPU，为 worker 提供批量推理。
    * 主进程从 data_queue 收集数据，写入 ReplayBuffer，触发训练，
      并在每次保存 checkpoint 后通知推理服务重新加载最新权重。
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")

    # 队列大小上限（避免内存爆炸）
    _REQ_MAXSIZE  = 512
    _RESP_MAXSIZE = 256
    _DATA_MAXSIZE = 200

    request_queue    = ctx.Queue(maxsize=_REQ_MAXSIZE)
    response_queues  = [ctx.Queue(maxsize=_RESP_MAXSIZE) for _ in range(num_selfplay_workers)]
    data_queue       = ctx.Queue(maxsize=_DATA_MAXSIZE)
    shutdown_event   = ctx.Event()
    model_update_q   = ctx.Queue(maxsize=4)

    # ── 启动推理服务进程 ──────────────────────────────────────────────────────
    from .inference_server import (
        run_inference_server,
        RELOAD_MODEL_MSG,
        SHUTDOWN_MSG as _INF_SHUTDOWN,
    )
    server_proc = ctx.Process(
        target=run_inference_server,
        args=(
            model_path,
            request_queue,
            response_queues,
            shutdown_event,
            model_update_q,
        ),
        kwargs=dict(
            batch_size_max=inference_batch_size_max,
            flush_ms=inference_flush_ms,
            precision=inference_precision,
            use_tf32=use_tf32,
        ),
        daemon=True,
        name="InferenceServer",
    )
    server_proc.start()
    print(f"[并行模式] 推理服务进程已启动 (PID={server_proc.pid})")

    # ── 启动 worker 进程 ──────────────────────────────────────────────────────
    worker_procs = []
    for wid in range(num_selfplay_workers):
        wp = ctx.Process(
            target=self_play_worker,
            args=(
                wid,
                request_queue,
                response_queues[wid],
                data_queue,
                shutdown_event,
            ),
            kwargs=dict(
                num_simulations=num_simulations,
                max_moves=max_moves,
                temperature_threshold=30,
                repetition_draw_threshold=repetition_draw_threshold,
                mcts_cache_size=mcts_cache_size,
                seed=worker_seed,
            ),
            daemon=True,
            name=f"SelfPlayWorker-{wid}",
        )
        wp.start()
        worker_procs.append(wp)
    print(f"[并行模式] {num_selfplay_workers} 个 self-play worker 已启动")

    # ── 主循环：收集数据 + 训练 ──────────────────────────────────────────────
    games_collected = 0
    workers_done = 0
    t_loop_start = time.time()
    games_per_s_window_start = time.time()
    games_per_s_window_count = 0

    try:
        while games_collected < num_games:
            # 从 data_queue 取一局结果（超时后继续检查 shutdown）
            try:
                item = data_queue.get(timeout=2.0)
            except Exception:
                # 检查所有 worker 是否已退出
                if all(not wp.is_alive() for wp in worker_procs):
                    print("[并行模式] 所有 worker 进程已退出，停止收集")
                    break
                continue

            if item.get("type") == "worker_done":
                workers_done += 1
                print(f"  [并行模式] Worker {item['worker_id']} 结束，"
                      f"共 {item['games_played']} 局")
                if workers_done >= num_selfplay_workers:
                    break
                continue

            if item.get("type") != "game_data":
                continue

            # ── 处理一局游戏数据 ─────────────────────────────────────────────
            games_collected += 1
            games_per_s_window_count += 1
            data    = item["training_data"]
            winner  = item["winner"]
            moves   = item["move_count"]
            reason  = item["terminate_reason"]
            wid     = item["worker_id"]

            is_draw = (winner == "draw" or winner is None)
            kept_for_training = True
            draw_category = None
            if is_draw:
                draw_category = classify_draw(reason, moves, max_moves)
                kept_for_training = draw_filter.should_keep(draw_category)

            if kept_for_training:
                data_buffer.extend(data)

            if winner == "red":
                stats["red_wins"] += 1
            elif winner == "black":
                stats["black_wins"] += 1
            else:
                stats["draws"] += 1

            # 局/秒统计（每 20 局）
            now = time.time()
            if games_per_s_window_count >= 20:
                elapsed_w = now - games_per_s_window_start
                gps = games_per_s_window_count / max(elapsed_w, 1e-6)
                print(f"  [并行模式] 吞吐：{gps:.2f} 局/s (最近 {games_per_s_window_count} 局)")
                games_per_s_window_start = now
                games_per_s_window_count = 0

            filter_info = (
                f", draw_cat={draw_category}, kept={kept_for_training}" if is_draw else ""
            )
            print(
                f"[第 {games_collected}/{num_games} 局] "
                f"worker={wid}, 胜方={winner or '和棋'}, 步数={moves}, "
                f"原因={reason or '-'}{filter_info}, "
                f"缓冲区={len(data_buffer)}"
            )

            jsonl_record = {
                "game_idx": games_collected,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "winner": winner or "draw",
                "terminate_reason": reason,
                "num_moves": moves,
                "num_samples": len(data),
                "elapsed_s": round(time.time() - t_loop_start, 2),
                "opponent_type": "self_play_parallel",
                "opponent_strength": "current",
                "my_side": "both",
                "engine_movetime": None,
                "draw_category": draw_category,
                "kept_for_training": kept_for_training,
            }
            append_self_play_jsonl(run_dir, jsonl_record)

            # 每 50 局输出和棋过滤汇总
            if games_collected % 50 == 0:
                total_draws = sum(draw_filter.counts.values())
                total_kept  = sum(draw_filter.kept.values())
                print(
                    f"  [和棋过滤汇总 @{games_collected}局] "
                    f"draw总数={total_draws}, 进入buffer={total_kept} | "
                    f"{draw_filter.summary_str()}"
                )

            # ── 训练 ─────────────────────────────────────────────────────────
            avg_loss = 0.0
            if len(data_buffer) >= batch_size:
                train_data = list(data_buffer)
                avg_loss = train_model(
                    model, train_data, batch_size=batch_size,
                    epochs=num_epochs, lr=lr,
                )
                print(f"  训练完成，平均损失: {avg_loss:.4f}")

                append_training_csv(run_dir, {
                    "game_idx": games_collected,
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    "loss": round(avg_loss, 6),
                    "buffer_size": len(data_buffer),
                    "elapsed_s": round(time.time() - t_loop_start, 2),
                })

            # ── 定期保存 + 通知推理服务重载 ──────────────────────────────────
            if games_collected % save_interval == 0:
                model.save(model_path)
                print(f"  模型已保存到: {model_path}")
                try:
                    model_update_q.put_nowait(RELOAD_MODEL_MSG)
                except Exception:
                    pass

            # ── 定期评测 ─────────────────────────────────────────────────────
            if eval_interval > 0 and games_collected % eval_interval == 0:
                _do_eval(
                    model, model_path, baseline_path, run_dir,
                    eval_opponent, eval_games, eval_simulations,
                    max_moves, elo_k, eval_gate, eval_state,
                    game_idx=games_collected,
                )

    finally:
        # ── 优雅关闭 ─────────────────────────────────────────────────────────
        shutdown_event.set()
        # 通知服务进程关闭
        try:
            model_update_q.put(_INF_SHUTDOWN, timeout=2.0)
        except Exception:
            pass
        # 等待子进程退出（最多 15 秒）
        for wp in worker_procs:
            wp.join(timeout=15)
            if wp.is_alive():
                wp.terminate()
        server_proc.join(timeout=15)
        if server_proc.is_alive():
            server_proc.terminate()
        print("[并行模式] 所有子进程已退出")


def _do_eval(
    model, model_path, baseline_path, run_dir,
    eval_opponent, eval_games, eval_simulations,
    max_moves, elo_k, eval_gate, eval_state,
    game_idx,
):
    """执行一次模型评测并更新 ELO 状态（供主循环复用）。"""
    print(f"  [评测] 第 {game_idx} 局，开始评测...")
    if eval_opponent == "self":
        opponent_model = ChessModel(num_channels=128, num_res_blocks=4)
        opponent_model.build()
        opponent_label = "self"
    else:
        opponent_model = ChessModel(num_channels=128, num_res_blocks=4)
        if os.path.exists(baseline_path):
            opponent_model.load(baseline_path)
        else:
            opponent_model.build()
        opponent_label = "previous"

    score, wins, losses, draws_eval = evaluate_models(
        model, opponent_model,
        n_games=eval_games,
        num_simulations=eval_simulations,
        max_moves=max_moves,
    )

    elo_cur = eval_state.get("elo_current", 1500.0)
    elo_opp = eval_state.get("elo_opponent", 1500.0)
    new_elo, elo_delta = compute_elo_update(elo_cur, elo_opp, score, k=elo_k)
    eval_state["elo_current"] = new_elo
    eval_state["last_game_idx"] = game_idx
    eval_state["last_opponent"] = opponent_label
    save_evaluation_state(run_dir, eval_state)

    print(
        f"  [评测] 胜: {wins}, 负: {losses}, 和: {draws_eval}, "
        f"score: {score:.3f}, ELO: {new_elo:.1f} (Δ{elo_delta:+.1f})"
    )

    append_evaluation_csv(run_dir, {
        "game_idx": game_idx,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "opponent": opponent_label,
        "eval_games": eval_games,
        "eval_sims": eval_simulations,
        "wins": wins,
        "losses": losses,
        "draws": draws_eval,
        "score": round(score, 6),
        "elo": round(new_elo, 2),
        "elo_delta": round(elo_delta, 2),
    })

    if eval_opponent == "previous":
        if score >= eval_gate:
            model.save(baseline_path)
            print(f"  [评测] score {score:.3f} >= gate {eval_gate}，基准模型已更新")
        else:
            print(f"  [评测] score {score:.3f} < gate {eval_gate}，保持基准模型不变")


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
                 pikafish_elo_weak=None,
                 pikafish_elo_mid=None,
                 pikafish_elo_full=None,
                 curriculum='default',
                 my_side='alternate',
                 eval_gate=0.55,
                 engine_options=None,
                 # 阶段 B：从蒸馏模型启动 / 放宽重复判和
                 init_from_distill=None,
                 repetition_draw_threshold=3,
                 # 和棋过滤参数
                 keep_draw_maxmoves_prob=0.25,
                 keep_draw_other_prob=0.10,
                 keep_draw_repetition_prob=0.05,
                 draw_filter_seed=0,
                 # ── 并行自对弈 + GPU batch inference ────────────────────────
                 num_selfplay_workers=0,
                 inference_batch_size_max=64,
                 inference_flush_ms=5.0,
                 inference_precision='bf16',
                 use_tf32=False,
                 mcts_cache_size=0,
                 worker_seed=42,
                 device='cuda'):
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
        pikafish_elo_weak: 弱强度引擎目标 Elo；设置后以 UCI_Elo 控制强度，movetime 仍用作搜索时间上限
        pikafish_elo_mid: 中强度引擎目标 Elo
        pikafish_elo_full: 全强度引擎目标 Elo
        curriculum: 课程学习策略（'default' 或 None/纯自对弈）
        my_side: 我方执哪方（'red'/'black'/'alternate'，仅对手池模式有效）
        eval_gate: 评测得分门控阈值（仅 eval_opponent='previous' 时生效，
                   score >= gate 才更新基准模型；默认 0.55）
        engine_options: 传给 UCI 引擎的选项字典
        init_from_distill: 蒸馏模型路径；若指定，则**优先**以该权重为 RL 微调起点，
                           并覆盖写入 model_path（若 model_path 已存在旧权重，会被替换）。
                           典型用法：先运行 ``distill`` 命令生成蒸馏权重，再用此参数
                           启动阶段 B RL 微调。
        repetition_draw_threshold: 重复局面判和阈值（>= 3）；训练时可调大（如 5 或 6）
                                   以减少短局重复判和，获得更丰富的训练信号（默认 3）。
        keep_draw_maxmoves_prob: 达到步数上限的和棋保留概率（默认 0.25）。
        keep_draw_other_prob: 其他原因和棋（stalemate 等）保留概率（默认 0.10）。
        keep_draw_repetition_prob: 重复局面和棋（repetition/perpetual_check/perpetual_chase）
                                   保留概率（默认 0.05）。
        draw_filter_seed: 和棋过滤随机数种子，用于可复现抽样（默认 0）。
        num_selfplay_workers: 并行 self-play worker 进程数（>0 时启用并行模式；
                              默认 0 = 使用原有单进程模式）。
                              推荐：Linux + RTX 4070 下设置 6~8。
        inference_batch_size_max: 推理服务最大 batch 大小（默认 64）。
        inference_flush_ms: 推理服务 flush 超时（毫秒，默认 5.0）。
        inference_precision: 推理精度：'bf16'（推荐）/ 'fp16' / 'fp32'（默认 'bf16'）。
        use_tf32: 是否为 matmul/cudnn 启用 TF32（默认 False）。
        mcts_cache_size: MCTS 局面评估缓存大小（默认 0=禁用）。
        worker_seed: worker 进程随机数基础种子（默认 42；每个 worker 加上自身 id）。
        device: 计算设备（'cuda' / 'cpu'）；'cpu' 时强制走单进程原逻辑。
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
    if init_from_distill and os.path.exists(init_from_distill):
        # 阶段 B：以蒸馏权重为起点（忽略 model_path 中可能存在的旧权重）
        print(f"从蒸馏模型加载权重: {init_from_distill}")
        model.load(init_from_distill)
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_path)
        print(f"蒸馏权重已复制到: {model_path}")
    elif os.path.exists(model_path):
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
        pikafish_elo_weak=pikafish_elo_weak,
        pikafish_elo_mid=pikafish_elo_mid,
        pikafish_elo_full=pikafish_elo_full,
        curriculum=curriculum,
        my_side=my_side,
        eval_gate=eval_gate,
        init_from_distill=init_from_distill,
        repetition_draw_threshold=repetition_draw_threshold,
        keep_draw_maxmoves_prob=keep_draw_maxmoves_prob,
        keep_draw_other_prob=keep_draw_other_prob,
        keep_draw_repetition_prob=keep_draw_repetition_prob,
        draw_filter_seed=draw_filter_seed,
        num_selfplay_workers=num_selfplay_workers,
        inference_batch_size_max=inference_batch_size_max,
        inference_flush_ms=inference_flush_ms,
        inference_precision=inference_precision,
        use_tf32=use_tf32,
        mcts_cache_size=mcts_cache_size,
    )
    run_dir = init_run_dir(config=training_config)
    print(f"日志目录: {run_dir}")
    print(f"重复判和阈值: {repetition_draw_threshold}")

    # 初始化和棋过滤器
    draw_filter = DrawFilter(
        keep_draw_maxmoves_prob=keep_draw_maxmoves_prob,
        keep_draw_other_prob=keep_draw_other_prob,
        keep_draw_repetition_prob=keep_draw_repetition_prob,
        seed=draw_filter_seed,
    )

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
            pikafish_elo_weak=pikafish_elo_weak,
            pikafish_elo_mid=pikafish_elo_mid,
            pikafish_elo_full=pikafish_elo_full,
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

    # 并行模式：num_selfplay_workers > 0 且 device != 'cpu' 且无对手池
    _use_parallel = (
        num_selfplay_workers > 0
        and device != 'cpu'
        and not use_opponent_pool
    )
    if _use_parallel:
        print(f"并行模式: workers={num_selfplay_workers}, "
              f"batch_max={inference_batch_size_max}, "
              f"flush_ms={inference_flush_ms}, "
              f"precision={inference_precision}")
    print(f"{'='*60}\n")

    # ── 并行自对弈训练路径 ────────────────────────────────────────────────────
    if _use_parallel:
        try:
            _run_parallel_training(
                model=model,
                model_path=model_path,
                num_games=num_games,
                num_simulations=num_simulations,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr=lr,
                max_moves=max_moves,
                buffer_size=buffer_size,
                save_interval=save_interval,
                run_dir=run_dir,
                data_buffer=data_buffer,
                stats=stats,
                draw_filter=draw_filter,
                eval_state=eval_state,
                baseline_path=baseline_path,
                pool=pool,
                eval_interval=eval_interval,
                eval_games=eval_games,
                eval_simulations=eval_simulations,
                eval_opponent=eval_opponent,
                elo_k=elo_k,
                eval_gate=eval_gate,
                repetition_draw_threshold=repetition_draw_threshold,
                num_selfplay_workers=num_selfplay_workers,
                inference_batch_size_max=inference_batch_size_max,
                inference_flush_ms=inference_flush_ms,
                inference_precision=inference_precision,
                use_tf32=use_tf32,
                mcts_cache_size=mcts_cache_size,
                worker_seed=worker_seed,
            )
        finally:
            if pool is not None:
                pool.close()
        model.save(model_path)
        print(f"\n{'='*60}")
        print(f"训练完成！")
        print(f"红方胜: {stats['red_wins']}, "
              f"黑方胜: {stats['black_wins']}, "
              f"和棋: {stats['draws']}")
        total_draws_final = sum(draw_filter.counts.values())
        total_kept_final = sum(draw_filter.kept.values())
        print(f"和棋过滤汇总: draw总数={total_draws_final}, "
              f"进入buffer={total_kept_final} | {draw_filter.summary_str()}")
        print(f"模型已保存到: {model_path}")
        print(f"日志目录: {run_dir}")
        print(f"{'='*60}")
        return

    # ── 原有单进程串行训练路径 ────────────────────────────────────────────────
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
                    repetition_draw_threshold=repetition_draw_threshold,
                )

                # 合并元数据
                opp_type_log = opp_meta.get('opponent_type', opponent_type)
                opp_strength_log = opp_meta.get('opponent_strength', '')
                engine_movetime_log = opp_meta.get('engine_movetime')
                my_side_log = game_meta.get('my_side', my_side)
                terminate_reason_log = game_meta.get('terminate_reason')
            else:
                # 原有自对弈模式（不提供 engine_path 时）
                data, winner, moves, terminate_reason_log = self_play_game(
                    model, num_simulations=num_simulations, max_moves=max_moves,
                    repetition_draw_threshold=repetition_draw_threshold,
                )
                opp_type_log = 'self_play'
                opp_strength_log = 'current'
                engine_movetime_log = None
                my_side_log = 'both'

            # 和棋过滤：win/loss 全部保留，draw 按概率过滤
            # winner 为 'draw' 或 None（无合法走法提前终止）时均视为和棋
            is_draw = (winner == 'draw' or winner is None)
            if is_draw:
                draw_category = classify_draw(terminate_reason_log, moves, max_moves)
                kept_for_training = draw_filter.should_keep(draw_category)
            else:
                draw_category = None
                kept_for_training = True

            if kept_for_training:
                data_buffer.extend(data)

            if winner == 'red':
                stats['red_wins'] += 1
            elif winner == 'black':
                stats['black_wins'] += 1
            else:
                stats['draws'] += 1

            elapsed = time.time() - start_time
            filter_info = (
                f', draw_cat={draw_category}, kept={kept_for_training}'
                if is_draw else ''
            )
            print(f"[第 {game_idx}/{num_games} 局] "
                  f"对手: {opp_type_log}, "
                  f"我方: {my_side_log}, "
                  f"胜方: {winner or '和棋'}, "
                  f"步数: {moves}, "
                  f"原因: {terminate_reason_log or '-'}"
                  f"{filter_info}, "
                  f"缓冲区: {len(data_buffer)}, "
                  f"耗时: {elapsed:.1f}s")

            jsonl_record = {
                'game_idx': game_idx,
                'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                'winner': winner or 'draw',
                'terminate_reason': terminate_reason_log,
                'num_moves': moves,
                'num_samples': len(data),
                'elapsed_s': round(elapsed, 2),
                'opponent_type': opp_type_log,
                'opponent_strength': opp_strength_log,
                'my_side': my_side_log,
                'engine_movetime': engine_movetime_log,
                'draw_category': draw_category,
                'kept_for_training': kept_for_training,
            }
            append_self_play_jsonl(run_dir, jsonl_record)

            # 每 50 局输出一次和棋过滤汇总
            if game_idx % 50 == 0:
                total_draws = sum(draw_filter.counts.values())
                total_kept = sum(draw_filter.kept.values())
                print(f"  [和棋过滤汇总 @{game_idx}局] "
                      f"draw总数={total_draws}, 进入buffer={total_kept} | "
                      f"{draw_filter.summary_str()}")

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
    total_draws_final = sum(draw_filter.counts.values())
    total_kept_final = sum(draw_filter.kept.values())
    print(f"和棋过滤汇总: draw总数={total_draws_final}, "
          f"进入buffer={total_kept_final} | {draw_filter.summary_str()}")
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
    parser.add_argument('--pikafish_elo_weak', type=int, default=None,
                        help='弱强度引擎目标 Elo（如 1000）；设置后以 UCI_Elo 控制引擎强度，'
                             'movetime 仍作为搜索时间上限（不再通过 movetime 控制强度）')
    parser.add_argument('--pikafish_elo_mid', type=int, default=None,
                        help='中强度引擎目标 Elo（如 1500）')
    parser.add_argument('--pikafish_elo_full', type=int, default=None,
                        help='全强度引擎目标 Elo（如 2000）')
    parser.add_argument('--curriculum', type=str, default='default',
                        help="课程学习策略：'default' 或 'none'（全自对弈，默认: default）")
    parser.add_argument('--my_side', type=str, default='alternate',
                        choices=['red', 'black', 'alternate'],
                        help='我方执哪方（对手池模式有效，默认: alternate）')
    parser.add_argument('--eval_gate', type=float, default=0.55,
                        help='评测门控阈值：score >= gate 才更新基准模型 (默认: 0.55)')
    # 阶段 B：从蒸馏模型启动 / 放宽重复判和
    parser.add_argument('--init_from_distill', type=str, default=None,
                        help='蒸馏模型路径；指定后以蒸馏权重为起点进行 RL 微调 '
                             '（优先级高于 --model_path 中已有权重）')
    parser.add_argument('--repetition_draw_threshold', type=int, default=3,
                        help='重复局面判和阈值（>= 3）；训练时可调大（如 5 或 6）'
                             '以减少短局重复判和，获得更丰富的训练信号 (默认: 3)')
    # 和棋过滤参数
    parser.add_argument('--keep_draw_maxmoves_prob', type=float, default=0.25,
                        help='达到步数上限的和棋保留概率 (默认: 0.25)')
    parser.add_argument('--keep_draw_other_prob', type=float, default=0.10,
                        help='其他原因和棋（stalemate 等）保留概率 (默认: 0.10)')
    parser.add_argument('--keep_draw_repetition_prob', type=float, default=0.05,
                        help='重复局面和棋（repetition/perpetual_check 等）保留概率 (默认: 0.05)')
    parser.add_argument('--draw_filter_seed', type=int, default=0,
                        help='和棋过滤随机数种子，用于可复现抽样 (默认: 0)')
    # ── 并行自对弈 + GPU batch inference ────────────────────────────────────
    parser.add_argument(
        '--num_selfplay_workers', type=int, default=0,
        help='并行 self-play worker 进程数（0=单进程原逻辑；推荐 Linux+RTX4070 设 6~8，默认: 0）'
    )
    parser.add_argument(
        '--inference_batch_size_max', type=int, default=64,
        help='推理服务最大 batch 大小（默认: 64）'
    )
    parser.add_argument(
        '--inference_flush_ms', type=float, default=5.0,
        help='推理服务 flush 超时（毫秒，默认: 5.0）'
    )
    parser.add_argument(
        '--inference_precision', type=str, default='bf16',
        choices=['bf16', 'fp16', 'fp32'],
        help="推理精度（默认: 'bf16'；RTX4070 支持 bf16）"
    )
    parser.add_argument(
        '--use_tf32', action='store_true',
        help='为 matmul/cudnn 启用 TF32（RTX 30/40 系列 GPU 有效，默认关闭）'
    )
    parser.add_argument(
        '--mcts_cache_size', type=int, default=0,
        help='MCTS 局面评估缓存大小（0=禁用；与并行模式搭配可降低 IPC 开销，默认: 0）'
    )
    parser.add_argument(
        '--worker_seed', type=int, default=42,
        help='worker 进程随机数基础种子（每个 worker 加上自身 id 保证独立，默认: 42）'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help="计算设备（'cpu' 时强制走单进程原逻辑，默认: 'cuda'）"
    )

    args = parser.parse_args()
    run_training(**vars(args))


if __name__ == '__main__':
    main()
