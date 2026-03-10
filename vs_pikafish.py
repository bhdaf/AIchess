"""
与 Pikafish 引擎对战并在对战中学习

Pikafish 是一个免费的强力中国象棋 UCI 引擎，派生自 Stockfish。
GitHub: https://github.com/official-pikafish/Pikafish

协议说明
---------
Pikafish 使用标准 UCI 协议（Universal Chess Interface）。
主要命令序列：

  1. ``uci``              → 引擎回复 ``uciok``
  2. ``isready``          → 引擎回复 ``readyok``
  3. ``ucinewgame``       → 开始新局（建议每局前发送）
  4. ``position fen <FEN>``  → 设置当前局面（完整 FEN，含走子方）
  5. ``go movetime <ms>`` → 开始搜索，引擎回复 ``bestmove <move>``
  6. ``quit``             → 退出

走法格式
---------
Pikafish 走法使用 ICCS 格式：``<from_file><from_rank><to_file><to_rank>``

  - file（列）：'a'–'i'，对应棋盘列 0–8（a=0, i=8）
  - rank（行）：'0'–'9'，对应棋盘行 0–9（0=红方底线, 9=黑方底线）

示例：``b2e2`` 表示从第 1 列第 2 行走到第 4 列第 2 行
（即红方中炮开局：炮从 b2 平移到 e2）。

AIchess 内部走法格式为 ``x0y0x1y1``（每位一个字符），与 ICCS 数字部分一致，
列号用数字而非字母表示。转换关系：

  AIchess ``1242`` ↔ Pikafish ``b2e2``

FEN 格式
---------
Pikafish 需要完整 FEN，形如：
  ``rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1``

其中：
  - 棋盘部分与 AIchess 的 ``get_fen()`` 返回结果相同
  - 走子方：'w'（红方/先手）或 'b'（黑方/后手）
  - 其余字段固定为 ``- - 0 1``（象棋引擎通常不使用吃过路卒/可用王车易位字段）

时间控制
---------
默认采用 ``go movetime <ms>``（固定每步毫秒数），而非 ``go depth <n>``。

原因：
  - movetime 直接限制挂钟时间，使对局总耗时可预测，便于批量训练
  - depth 搜索深度受局面复杂度和硬件影响，同一 depth 实际耗时差异很大
  - 训练时建议从 movetime=100（0.1 秒）起步，可用 ``--movetime`` 调整

训练数据收集策略
-----------------
训练数据**只记录学习方（AIchess 模型）的回合**，对手（Pikafish）的回合不记录。

原因：
  - 学习目标是提升 AIchess 自身的策略与价值估计
  - 记录双方会把 Pikafish 的走法强度混入 policy target，
    导致 AIchess 模仿超出自身能力的分布，梯度信号反而更嘈杂
  - 只记录己方步骤，policy target 来自 MCTS 访问计数分布，
    value target 来自终局胜负回填，与自对弈格式完全一致

用法
----
对战一局（不训练）::

    python -m AIchess vs_pikafish --engine /path/to/pikafish

边对战边训练（AlphaZero 风格）::

    python -m AIchess vs_pikafish \\
        --engine /path/to/pikafish \\
        --model_path saved_model/model.pth \\
        --num_games 50 \\
        --movetime 500 \\
        --num_simulations 100 \\
        --train

学习方执黑（AIchess 为黑方，Pikafish 为红方）::

    python -m AIchess vs_pikafish --engine /path/to/pikafish --ai_color black
"""

import argparse
import os
import subprocess
import sys
import time
import datetime

import numpy as np

from .game import (
    ChessGame, NUM_ACTIONS, ACTION_LABELS, LABEL_TO_INDEX,
    flip_move, flip_policy, fen_to_planes, INIT_FEN,
)
from .model import ChessModel
from .mcts import MCTS
from .export import init_run_dir, append_self_play_jsonl

# 默认模型路径
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 'saved_model', 'model.pth'
)


# ---------------------------------------------------------------------------
# 走法格式转换
# ---------------------------------------------------------------------------

def aicoord_to_uci(move: str) -> str:
    """将 AIchess 内部走法格式转换为 Pikafish UCI 格式。

    AIchess 格式：``x0y0x1y1``（每位一个十进制字符，列用数字 0–8）
    Pikafish 格式：``<file><rank><file><rank>``（列用字母 a–i，行用数字 0–9）

    Examples::

        >>> aicoord_to_uci("1242")
        'b2e2'
        >>> aicoord_to_uci("0019")
        'a0b9'
    """
    x0, y0, x1, y1 = int(move[0]), int(move[1]), int(move[2]), int(move[3])
    return f"{chr(ord('a') + x0)}{y0}{chr(ord('a') + x1)}{y1}"


def uci_to_aicoord(move: str) -> str:
    """将 Pikafish UCI 走法格式转换为 AIchess 内部格式。

    Examples::

        >>> uci_to_aicoord("b2e2")
        '1242'
    """
    x0 = ord(move[0]) - ord('a')
    y0 = int(move[1])
    x1 = ord(move[2]) - ord('a')
    y1 = int(move[3])
    return f"{x0}{y0}{x1}{y1}"


def game_full_fen(game: ChessGame) -> str:
    """生成 Pikafish 所需的完整 FEN 字符串。

    Pikafish 要求完整 FEN，包含走子方和其他字段，例如：
      ``rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1``

    走子方：'w' 表示红方，'b' 表示黑方。
    其余字段（en passant、半步数、全步数）对中国象棋无意义，固定为 ``- - 0 1``。
    """
    side = 'w' if game.red_to_move else 'b'
    return f"{game.get_fen()} {side} - - 0 1"


# ---------------------------------------------------------------------------
# UCI 引擎包装器
# ---------------------------------------------------------------------------

class UCIEngine:
    """轻量级 UCI 引擎进程包装器，专为 Pikafish 设计。

    使用标准 subprocess 管道与引擎通信，实现最小 UCI 命令集：
    uci / isready / ucinewgame / position / go / stop / quit。

    Example::

        engine = UCIEngine("/path/to/pikafish")
        engine.start()
        engine.new_game()
        move = engine.get_best_move("rnbakabnr/... w - - 0 1", movetime=500)
        print(move)   # e.g. 'b2e2'
        engine.quit()
    """

    def __init__(self, engine_path: str, threads: int = 1, hash_mb: int = 64):
        """
        Args:
            engine_path: Pikafish 可执行文件的路径。
            threads: 引擎搜索线程数（对于训练批量对局建议设为 1–2）。
            hash_mb: 置换表大小（MB）。
        """
        self.engine_path = engine_path
        self.threads = threads
        self.hash_mb = hash_mb
        self._proc: subprocess.Popen | None = None

    def start(self):
        """启动引擎进程并完成 UCI 握手。"""
        self._proc = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok", timeout=10.0)
        # 配置引擎选项
        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send("isready")
        self._wait_for("readyok", timeout=10.0)

    def new_game(self):
        """通知引擎开始新对局，并等待引擎就绪。"""
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", timeout=10.0)

    def get_best_move(self, full_fen: str, movetime: int = 1000) -> str | None:
        """向引擎发送局面并搜索最佳走法。

        Args:
            full_fen: 完整 FEN 字符串（含走子方字段），例如由 :func:`game_full_fen` 生成。
            movetime: 每步思考时间（毫秒）。默认 1000ms。

        Returns:
            UCI 格式的最佳走法字符串（如 ``'b2e2'``），若引擎无走法则返回 None。
        """
        self._send(f"position fen {full_fen}")
        self._send(f"go movetime {movetime}")
        return self._read_best_move(timeout=movetime / 1000.0 + 5.0)

    def quit(self):
        """安全退出引擎进程。"""
        if self._proc and self._proc.poll() is None:
            try:
                self._send("quit")
                self._proc.wait(timeout=3.0)
            except Exception:
                self._proc.terminate()
        self._proc = None

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _send(self, command: str):
        """向引擎标准输入发送命令（自动追加换行）。"""
        if self._proc and self._proc.stdin:
            self._proc.stdin.write(command + "\n")
            self._proc.stdin.flush()

    def _wait_for(self, token: str, timeout: float = 10.0):
        """读取引擎输出直到找到包含 *token* 的行或超时。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                break
            line = self._proc.stdout.readline().rstrip()
            if token in line:
                return
        raise TimeoutError(
            f"引擎未在 {timeout}s 内返回 '{token}'（引擎路径: {self.engine_path}）"
        )

    def _read_best_move(self, timeout: float = 10.0) -> str | None:
        """读取引擎输出，提取 ``bestmove`` 行中的走法。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                break
            line = self._proc.stdout.readline().rstrip()
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    return parts[1]
                return None
        return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.quit()


# ---------------------------------------------------------------------------
# 单局对战函数
# ---------------------------------------------------------------------------

def play_game_vs_engine(
    model: ChessModel,
    engine: UCIEngine,
    ai_color: str = 'red',
    num_simulations: int = 100,
    movetime: int = 1000,
    max_moves: int = 200,
    temperature_threshold: int = 30,
) -> tuple[list, str | None, int]:
    """与 Pikafish 对战一局，返回训练数据（只含学习方回合）。

    Args:
        model: AIchess 策略价值网络模型。
        engine: 已启动的 :class:`UCIEngine` 实例。
        ai_color: 学习方颜色，``'red'`` 或 ``'black'``。
        num_simulations: MCTS 模拟次数（每步）。
        movetime: Pikafish 每步思考时间（毫秒）。
        max_moves: 每局最大步数（超过判和）。
        temperature_threshold: 前 N 步 AI 使用温度 1.0 探索，之后使用 0.1。

    Returns:
        tuple ``(training_data, winner, move_count)``：

        - ``training_data``: ``[(state_planes, policy_target, value_target), ...]``，
          **只包含学习方（AIchess）回合**的样本。
        - ``winner``: ``'red'`` / ``'black'`` / ``'draw'`` / ``None``（超步数）。
        - ``move_count``: 实际步数。
    """
    game = ChessGame()
    game.reset()
    mcts = MCTS(model, num_simulations=num_simulations)
    engine.new_game()

    ai_is_red = (ai_color == 'red')

    # 记录中间状态（只记录 AI 方回合）
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    players: list[int] = []   # +1 = AI 方视角

    move_count = 0

    while not game.done and move_count < max_moves:
        is_ai_turn = (game.red_to_move == ai_is_red)

        if is_ai_turn:
            # AI 方：用 MCTS 决策
            temperature = 1.0 if move_count < temperature_threshold else 0.1
            actions, probs = mcts.get_action_probs(
                game, temperature=temperature, add_noise=True
            )
            if not actions:
                break

            # 构造 policy target（从当前走子方视角的动作空间）
            planes = game.to_planes()
            policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for action, prob in zip(actions, probs):
                if action in LABEL_TO_INDEX:
                    policy_target[LABEL_TO_INDEX[action]] = prob

            states.append(planes)
            policies.append(policy_target)
            players.append(1)   # AI 方视角 value = winner * 1

            # 采样走法
            action_idx = np.random.choice(len(actions), p=probs)
            chosen_action = actions[action_idx]
            # 转换为棋盘真实坐标（MCTS 动作是从当前走子方视角）
            actual_action = chosen_action if game.red_to_move else flip_move(chosen_action)
            game.step(actual_action)
            mcts.update_with_move(chosen_action)

        else:
            # Pikafish 方：通过 UCI 获取走法
            full_fen = game_full_fen(game)
            uci_move = engine.get_best_move(full_fen, movetime=movetime)
            if uci_move is None:
                break
            ai_move = uci_to_aicoord(uci_move)
            # 验证走法合法性（防止格式不兼容）
            legal = game.get_legal_moves()
            if ai_move not in legal:
                # 走法不合法，对局终止（通常不应发生）
                print(
                    f"  [警告] Pikafish 走法 {uci_move} ({ai_move}) "
                    f"不在合法走法列表中，终止对局",
                    file=sys.stderr,
                )
                break
            game.step(ai_move)
            # 更新 MCTS 树根：对手的走法从对手视角看是正向坐标，
            # 但 MCTS 始终以当前走子方视角存储动作，
            # 所以需把对手走法翻转成 MCTS 期望的格式
            opponent_action_for_mcts = flip_move(ai_move) if ai_is_red else ai_move
            mcts.update_with_move(opponent_action_for_mcts)

        move_count += 1

    # 回填终局 value
    if game.winner == 'red':
        winner_val = 1 if ai_is_red else -1
    elif game.winner == 'black':
        winner_val = -1 if ai_is_red else 1
    else:
        winner_val = 0

    training_data = [
        (state, policy, float(winner_val))
        for state, policy in zip(states, policies)
    ]

    return training_data, game.winner, move_count


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

def run_vs_pikafish(
    engine_path: str,
    model_path: str | None = None,
    ai_color: str = 'red',
    num_games: int = 1,
    num_simulations: int = 100,
    movetime: int = 1000,
    max_moves: int = 200,
    train: bool = False,
    batch_size: int = 256,
    num_epochs: int = 5,
    lr: float = 0.001,
    threads: int = 1,
    hash_mb: int = 64,
):
    """运行 AIchess 对战 Pikafish 的完整流程（可选训练）。

    Args:
        engine_path: Pikafish 可执行文件路径。
        model_path: AIchess 模型文件路径（.pth）。
        ai_color: 学习方颜色（``'red'`` 或 ``'black'``）。
        num_games: 对战局数。
        num_simulations: AI 每步 MCTS 模拟次数。
        movetime: Pikafish 每步思考时间（毫秒）。
        max_moves: 每局最大步数。
        train: 是否在对战后用收集到的数据训练模型。
        batch_size: 训练批大小（仅 ``train=True`` 时有效）。
        num_epochs: 每轮训练 epoch 数（仅 ``train=True`` 时有效）。
        lr: 学习率（仅 ``train=True`` 时有效）。
        threads: 引擎线程数。
        hash_mb: 引擎置换表大小（MB）。
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    # 加载/初始化模型
    model = ChessModel()
    if os.path.exists(model_path):
        model.load(model_path)
        print(f"模型已加载: {model_path}")
    else:
        model.build()
        print("未找到模型，使用随机初始化模型")

    # 初始化日志目录
    run_dir = init_run_dir(config=dict(
        engine_path=engine_path,
        ai_color=ai_color,
        num_games=num_games,
        num_simulations=num_simulations,
        movetime=movetime,
        max_moves=max_moves,
        train=train,
    ))
    print(f"日志目录: {run_dir}")

    all_data: list = []
    stats = {'ai_wins': 0, 'engine_wins': 0, 'draws': 0}

    print(f"\n{'='*60}")
    print(f"开始 AIchess vs Pikafish 对战")
    print(f"学习方颜色: {ai_color}")
    print(f"对战局数: {num_games}")
    print(f"AI MCTS 模拟次数: {num_simulations}")
    print(f"Pikafish 思考时间: {movetime}ms/步")
    print(f"训练模式: {'启用' if train else '禁用'}")
    print(f"{'='*60}\n")

    with UCIEngine(engine_path, threads=threads, hash_mb=hash_mb) as engine:
        engine.start()

        for game_idx in range(1, num_games + 1):
            start_time = time.time()

            data, winner, moves = play_game_vs_engine(
                model=model,
                engine=engine,
                ai_color=ai_color,
                num_simulations=num_simulations,
                movetime=movetime,
                max_moves=max_moves,
            )
            all_data.extend(data)

            ai_is_red = (ai_color == 'red')
            if winner == ('red' if ai_is_red else 'black'):
                stats['ai_wins'] += 1
                result_str = "AI 胜"
            elif winner == ('black' if ai_is_red else 'red'):
                stats['engine_wins'] += 1
                result_str = "Pikafish 胜"
            else:
                stats['draws'] += 1
                result_str = "和棋"

            elapsed = time.time() - start_time
            print(
                f"[第 {game_idx}/{num_games} 局] {result_str}, "
                f"步数: {moves}, 样本: {len(data)}, "
                f"耗时: {elapsed:.1f}s"
            )

            append_self_play_jsonl(run_dir, {
                'game_idx': game_idx,
                'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                'winner': winner or 'draw',
                'num_moves': moves,
                'num_samples': len(data),
                'elapsed_s': round(elapsed, 2),
                'ai_color': ai_color,
                'movetime_ms': movetime,
            })

    # 训练
    if train and all_data:
        from .train import train_model
        print(f"\n共收集 {len(all_data)} 条训练样本（仅 AI 回合），开始训练...")
        avg_loss = train_model(
            model, all_data,
            batch_size=batch_size,
            epochs=num_epochs,
            lr=lr,
        )
        model_dir = os.path.dirname(model_path) or '.'
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_path)
        print(f"训练完成，平均损失: {avg_loss:.4f}，模型已保存到: {model_path}")

    total = stats['ai_wins'] + stats['engine_wins'] + stats['draws']
    print(f"\n{'='*60}")
    print(f"对战结束")
    print(f"AI 胜: {stats['ai_wins']}, Pikafish 胜: {stats['engine_wins']}, "
          f"和棋: {stats['draws']}")
    if total > 0:
        print(f"AI 胜率: {stats['ai_wins'] / total * 100:.1f}%")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AIchess vs Pikafish — 对战并（可选）训练学习',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m AIchess vs_pikafish --engine ./pikafish\n"
            "  python -m AIchess vs_pikafish --engine ./pikafish --train "
            "--num_games 20 --movetime 500"
        ),
    )
    parser.add_argument(
        '--engine', required=True,
        help='Pikafish 可执行文件路径',
    )
    parser.add_argument(
        '--model_path', type=str, default=None,
        help='AIchess 模型文件路径（.pth）',
    )
    parser.add_argument(
        '--ai_color', type=str, default='red',
        choices=['red', 'black'],
        help='学习方（AIchess）执哪方',
    )
    parser.add_argument(
        '--num_games', type=int, default=1,
        help='对战局数',
    )
    parser.add_argument(
        '--num_simulations', type=int, default=100,
        help='AI 每步 MCTS 模拟次数',
    )
    parser.add_argument(
        '--movetime', type=int, default=1000,
        help='Pikafish 每步思考时间（毫秒）。'
             '使用 movetime 而非 depth 可使对局时长可预测，适合批量训练。',
    )
    parser.add_argument(
        '--max_moves', type=int, default=200,
        help='每局最大步数（超过判和）',
    )
    parser.add_argument(
        '--train', action='store_true',
        help='对战结束后用收集到的数据训练模型（只训练 AI 方回合）',
    )
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='训练批大小（仅 --train 时有效）',
    )
    parser.add_argument(
        '--num_epochs', type=int, default=5,
        help='训练 epoch 数（仅 --train 时有效）',
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='学习率（仅 --train 时有效）',
    )
    parser.add_argument(
        '--threads', type=int, default=1,
        help='Pikafish 搜索线程数',
    )
    parser.add_argument(
        '--hash_mb', type=int, default=64,
        help='Pikafish 置换表大小（MB）',
    )

    args = parser.parse_args()
    run_vs_pikafish(
        engine_path=args.engine,
        model_path=args.model_path,
        ai_color=args.ai_color,
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        movetime=args.movetime,
        max_moves=args.max_moves,
        train=args.train,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        threads=args.threads,
        hash_mb=args.hash_mb,
    )


if __name__ == '__main__':
    main()
