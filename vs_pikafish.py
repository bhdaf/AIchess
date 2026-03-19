"""
与 Pikafish 引擎对弈脚本

让训练好的 AI 模型与外部 UCI 引擎（Pikafish 等）进行对弈，
并将对局结果输出为 PGN 格式及 JSON 日志。

用法::

    python -m AIchess vs_pikafish \\
        --engine_path /path/to/pikafish \\
        [--model_path path/to/model.pth] \\
        [--n_games 10] \\
        [--elo 1500] \\
        [--ai_side red|black|both] \\
        [--num_simulations 50] \\
        [--out runs/vs_pikafish] \\
        [--pgn_path game.pgn] \\
        [--verbose]

输出：
    - 标准输出：对局摘要（胜/负/和）
    - ``<out>/vs_pikafish_log.jsonl``：每局 JSONL 记录
    - ``<out>/vs_pikafish.pgn``（或 --pgn_path 指定路径）：PGN 文件
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Optional

import numpy as np

from .game import flip_move
from .pikafish_agent import PikafishAgent
from .mcts import MCTSNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PGN 工具
# ---------------------------------------------------------------------------

def _pgn_header(
    event: str,
    date_str: str,
    red_name: str,
    black_name: str,
    result_str: str,
) -> str:
    """生成 PGN 文件头（Chinese Chess / 象棋 PGN 扩展）。"""
    return (
        f'[Event "{event}"]\n'
        f'[Date "{date_str}"]\n'
        f'[Red "{red_name}"]\n'
        f'[Black "{black_name}"]\n'
        f'[Result "{result_str}"]\n'
        f'[Variant "Xiangqi"]\n'
        '\n'
    )


def _result_str(winner: Optional[str]) -> str:
    """将内部胜负结果转换为 PGN 结果字符串。"""
    if winner == "red":
        return "1-0"
    elif winner == "black":
        return "0-1"
    else:
        return "1/2-1/2"


def moves_to_pgn_body(moves_uci: list[str], result: str) -> str:
    """将 UCI ICCS 走法列表格式化为 PGN 走法正文。"""
    tokens = []
    for i, move in enumerate(moves_uci):
        if i % 2 == 0:
            tokens.append(f"{i // 2 + 1}.")
        tokens.append(move)
    tokens.append(result)
    return " ".join(tokens) + "\n"


# ---------------------------------------------------------------------------
# 核心对局函数
# ---------------------------------------------------------------------------

def play_one_game(
    ai_agent,          # BaseAgent（MCTS）
    engine_agent,      # PikafishAgent
    ai_plays_red: bool,
    max_moves: int,
    verbose: bool,
) -> dict:
    """
    进行一局对弈，返回包含结果和走法列表的字典。

    Args:
        ai_agent: 我方 AI（:class:`~AIchess.mcts.MCTS` 封装的 Agent）。
        engine_agent: 对手引擎 Agent（:class:`~AIchess.pikafish_agent.PikafishAgent`）。
        ai_plays_red: ``True`` 表示 AI 执红，``False`` 表示 AI 执黑。
        max_moves: 最大步数（防止超长对局）。
        verbose: 是否在控制台打印每步走法和棋盘。

    Returns:
        包含以下字段的字典：

        - ``winner``: ``"red"`` / ``"black"`` / ``"draw"`` / ``"max_moves"``
        - ``num_moves``: 实际步数
        - ``ai_plays_red``: AI 执哪方
        - ``moves_internal``: 内部格式走法列表
        - ``moves_uci``: UCI ICCS 格式走法列表
        - ``terminate_reason``: 终局原因
    """
    from .game import ChessGame
    from .pikafish_agent import internal_to_uci

    game = ChessGame().reset()
    ai_agent.new_game()
    engine_agent.new_game()

    moves_internal: list[str] = []
    moves_uci: list[str] = []

    if verbose:
        game.print_board()

    while not game.done and game.num_moves < max_moves:
        red_to_move = game.red_to_move
        is_ai_turn = (ai_plays_red == red_to_move)
        side_name = "红方(AI)" if (red_to_move and ai_plays_red) else (
            "红方(引擎)" if red_to_move else (
                "黑方(AI)" if ai_plays_red is False else "黑方(引擎)"
            )
        )

        if is_ai_turn:
            # AI 通过 MCTS 选择走法
            actions, probs = ai_agent.get_action_probs(
                game, mode="eval"
            )
            if not actions:
                logger.warning("AI 无合法走法，局面：%s", game.get_fen())
                break
            # 选择访问次数最多的走法（probs[best_idx]=1.0，其余为0）
            best_idx = int(np.argmax(probs))
            move_mcts = actions[best_idx]  # MCTS 内部视角（始终为红方坐标）
            # 转换为实际棋盘坐标（黑方走子时需翻转）
            move = move_mcts if game.red_to_move else flip_move(move_mcts)
        else:
            # 引擎走法（PikafishAgent.get_move 返回内部格式）
            move = engine_agent.get_move(game)
            if move is None:
                # 引擎无走法（规则不一致或对局结束）
                logger.warning(
                    "引擎无法给出走法，视为认输。局面：%s", game.get_fen()
                )
                break

        # 记录走法
        try:
            uci_move = internal_to_uci(move)
        except ValueError:
            uci_move = move  # 降级：原样记录
        moves_internal.append(move)
        moves_uci.append(uci_move)

        if verbose:
            print(f"  {side_name}: {move} ({uci_move})")

        _, _, done, _ = game.step(move)

        if verbose and done:
            game.print_board()

    winner = game.winner or ("max_moves" if game.num_moves >= max_moves else "draw")
    terminate_reason = getattr(game, "terminate_reason", None) or (
        "max_moves" if game.num_moves >= max_moves else None
    )

    result = {
        "winner": winner,
        "num_moves": game.num_moves,
        "ai_plays_red": ai_plays_red,
        "moves_internal": moves_internal,
        "moves_uci": moves_uci,
        "terminate_reason": terminate_reason,
    }
    return result


# ---------------------------------------------------------------------------
# MCTSAgent 适配器（将 MCTS 包装为 BaseAgent 兼容对象）
# ---------------------------------------------------------------------------

class _MCTSAgentAdapter:
    """轻量适配器，让 MCTS 对象实现 new_game() 接口。"""

    def __init__(self, mcts):
        self._mcts = mcts

    def new_game(self) -> None:
        self._mcts.root = MCTSNode()

    def get_action_probs(self, game, temperature=0.0, add_noise=False,
                         mode=None):
        return self._mcts.get_action_probs(
            game, temperature=temperature, add_noise=add_noise, mode=mode
        )


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="让 AI 模型与 Pikafish 等 UCI 引擎对弈，输出 PGN/日志",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--engine_path", required=True,
        help="UCI 引擎可执行文件路径（如 /usr/local/bin/pikafish）"
    )
    parser.add_argument(
        "--model_path", default=None,
        help="AI 模型 .pth 文件路径；省略则使用随机初始化模型（仅用于测试）"
    )
    parser.add_argument(
        "--n_games", type=int, default=10,
        help="对局总数"
    )
    parser.add_argument(
        "--elo", type=int, default=None,
        help="引擎目标 Elo（如 1500）；通过 UCI_LimitStrength + UCI_Elo 控制强度"
    )
    parser.add_argument(
        "--skill_level", type=int, default=None,
        metavar="0-20",
        help="引擎技能等级（0-20）；仅在未设置 --elo 时生效"
    )
    parser.add_argument(
        "--ai_side", choices=["red", "black", "both"], default="both",
        help="AI 执哪方：red/black/both（both 时交替执红黑）"
    )
    parser.add_argument(
        "--num_simulations", type=int, default=50,
        help="AI 每步 MCTS 模拟次数"
    )
    parser.add_argument(
        "--max_moves", type=int, default=300,
        help="每局最大步数（防止超长对局）"
    )
    parser.add_argument(
        "--out", default=None,
        help="输出目录；日志和 PGN 将写入此目录"
    )
    parser.add_argument(
        "--pgn_path", default=None,
        help="PGN 输出文件路径；省略时写入 <out>/vs_pikafish.pgn"
    )
    parser.add_argument(
        "--engine_options", nargs="*", default=[],
        metavar="NAME=VALUE",
        help="传给引擎的 UCI 选项，如 UCI_Elo=1500 Skill\\ Level=5"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="打印每步走法和棋盘"
    )
    parser.add_argument(
        "--debug_mcts", action="store_true",
        help="打印 MCTS 调试信息（网络输出、根节点统计等）"
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # 解析引擎选项
    engine_options: dict[str, str] = {}
    for opt in args.engine_options:
        if "=" in opt:
            k, v = opt.split("=", 1)
            engine_options[k.strip()] = v.strip()
        else:
            logger.warning("忽略无效引擎选项（缺少 '='）：%s", opt)

    # 创建输出目录
    out_dir = args.out or os.path.join(
        "runs",
        f"vs_pikafish_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)

    pgn_path = args.pgn_path or os.path.join(out_dir, "vs_pikafish.pgn")
    log_path = os.path.join(out_dir, "vs_pikafish_log.jsonl")

    # 加载 AI 模型
    from .model import ChessModel
    from .mcts import MCTS

    model = ChessModel()
    if args.model_path:
        loaded = model.load(args.model_path)
        if not loaded:
            print(f"[警告] 无法加载模型 {args.model_path}，使用随机初始化权重。",
                  file=sys.stderr)
            model.build()
    else:
        print("[信息] 未指定 --model_path，使用随机初始化权重（仅用于一致性验证）。",
              file=sys.stderr)
        model.build()

    mcts = MCTS(model, num_simulations=args.num_simulations,
                debug_mcts=args.debug_mcts)
    ai_agent = _MCTSAgentAdapter(mcts)

    # 统计
    stats = {"wins": 0, "losses": 0, "draws": 0, "invalid": 0}
    date_str = datetime.datetime.now().strftime("%Y.%m.%d")
    event_name = "AIchess vs Pikafish"

    # 打开输出文件
    pgn_file = open(pgn_path, "w", encoding="utf-8")
    log_file = open(log_path, "w", encoding="utf-8")

    try:
        with PikafishAgent(
            args.engine_path,
            elo=args.elo,
            skill_level=args.skill_level,
            options=engine_options,
        ) as engine_agent:

            for game_idx in range(args.n_games):
                # 决定 AI 执哪方
                if args.ai_side == "red":
                    ai_plays_red = True
                elif args.ai_side == "black":
                    ai_plays_red = False
                else:  # both：奇局 AI 执红，偶局 AI 执黑
                    ai_plays_red = (game_idx % 2 == 0)

                print(
                    f"[局 {game_idx + 1}/{args.n_games}] "
                    f"AI={'红方' if ai_plays_red else '黑方'} ...",
                    end=" ", flush=True
                )

                result = play_one_game(
                    ai_agent=ai_agent,
                    engine_agent=engine_agent,
                    ai_plays_red=ai_plays_red,
                    max_moves=args.max_moves,
                    verbose=args.verbose,
                )

                # 判断 AI 胜负
                winner = result["winner"]
                if winner == "max_moves" or winner == "draw":
                    outcome = "draw"
                    stats["draws"] += 1
                elif (winner == "red") == ai_plays_red:
                    outcome = "win"
                    stats["wins"] += 1
                else:
                    outcome = "loss"
                    stats["losses"] += 1

                print(f"结果={outcome} ({winner}) 步数={result['num_moves']}")

                # 写入 PGN
                red_name = "AIchess" if ai_plays_red else "Pikafish"
                black_name = "Pikafish" if ai_plays_red else "AIchess"
                pgn_result = _result_str(winner if winner in ("red", "black") else None)
                pgn_file.write(
                    _pgn_header(event_name, date_str, red_name, black_name, pgn_result)
                )
                pgn_file.write(
                    moves_to_pgn_body(result["moves_uci"], pgn_result)
                )
                pgn_file.write("\n")
                pgn_file.flush()

                # 写入 JSONL 日志
                log_record = {
                    "game_idx": game_idx + 1,
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    "ai_plays_red": ai_plays_red,
                    "winner": winner,
                    "outcome": outcome,
                    "num_moves": result["num_moves"],
                    "terminate_reason": result["terminate_reason"],
                    "moves_uci": result["moves_uci"],
                }
                log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                log_file.flush()

    finally:
        pgn_file.close()
        log_file.close()

    # 输出汇总
    total = args.n_games
    print(
        f"\n=== 对战汇总 ({total} 局) ===\n"
        f"  AI 胜：{stats['wins']}  负：{stats['losses']}  和：{stats['draws']}\n"
        f"  胜率：{stats['wins'] / total:.1%}  得分率："
        f"{(stats['wins'] + 0.5 * stats['draws']) / total:.1%}\n"
        f"  日志：{log_path}\n"
        f"  PGN： {pgn_path}"
    )


if __name__ == "__main__":
    main()
