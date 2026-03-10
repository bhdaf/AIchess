"""
AIchess vs Pikafish 对弈模块

让训练好的 AIchess 模型与 Pikafish 引擎进行对弈。
支持：
  - AI（AIchess 模型）执红方或黑方
  - 指定 Pikafish 引擎路径
  - 配置 Pikafish 思考时间或搜索深度
  - 记录对局结果

用法::

    python -m AIchess vs_pikafish \\
        --engine_path /path/to/pikafish \\
        [--model_path path/to/model.pth] \\
        [--ai_color red|black] \\
        [--movetime_ms 100] \\
        [--depth N] \\
        [--num_simulations 200] \\
        [--n_games 1] \\
        [--max_moves 300] \\
        [--verbose]

时间控制说明：
  - 默认使用 --movetime_ms 100（每步 100 毫秒），适合快速测试
  - 若指定 --depth，则使用固定深度搜索（忽略 --movetime_ms）
  - 建议训练评测时使用 movetime_ms=50~200；正式测试时使用 movetime_ms=1000 或 depth=15
"""

import argparse
import os
import sys
import numpy as np

from .game import ChessGame, flip_move
from .model import ChessModel
from .mcts import MCTS
from .pikafish import PikafishEngine

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_model', 'model.pth')
DEFAULT_MOVETIME_MS = 100

_RESULT_LABELS = {'ai_win': 'AI 胜', 'engine_win': '引擎胜', 'draw': '和棋'}


def play_one_game(ai_model, engine: PikafishEngine, ai_is_red: bool,
                  num_simulations: int = 200, max_moves: int = 300,
                  verbose: bool = False):
    """进行一局 AI vs Pikafish 对弈。

    Args:
        ai_model: 已加载的 ChessModel 实例
        engine: 已启动的 PikafishEngine 实例
        ai_is_red: True 表示 AI 执红方（先手），False 表示 AI 执黑方
        num_simulations: AI MCTS 模拟次数
        max_moves: 最大步数（防止无限循环）
        verbose: 是否打印每步走法

    Returns:
        str: 'ai_win' | 'engine_win' | 'draw'
    """
    game = ChessGame()
    game.reset()
    mcts = MCTS(ai_model, num_simulations=num_simulations)
    engine.new_game()

    move_count = 0

    if verbose:
        print(f"\n=== 对局开始：AI 执{'红' if ai_is_red else '黑'}方 ===")
        game.print_board()

    while not game.done and move_count < max_moves:
        is_ai_turn = (game.red_to_move == ai_is_red)

        if is_ai_turn:
            # AI 使用 MCTS 决策
            actions, probs = mcts.get_action_probs(game, temperature=0.1)
            if not actions:
                break
            best_idx = np.argmax(probs)
            action = actions[best_idx]

            # MCTS 动作是当前玩家视角；需要转换为棋盘实际走法
            if not game.red_to_move:
                actual_action = flip_move(action)
            else:
                actual_action = action

            game.step(actual_action)
            mcts.update_with_move(action)

            if verbose:
                print(f"[{'红' if ai_is_red else '黑'} AI ] 走法: {actual_action}")
        else:
            # Pikafish 决策
            try:
                engine_action = engine.get_best_move(game)
            except RuntimeError as e:
                if verbose:
                    print(f"引擎错误: {e}")
                break

            game.step(engine_action)
            # 通知 MCTS 对手走法（转换为当前玩家视角）
            if not game.red_to_move:
                mcts_action = flip_move(engine_action)
            else:
                mcts_action = engine_action
            mcts.update_with_move(mcts_action)

            if verbose:
                print(f"[{'黑' if ai_is_red else '红'} 引擎] 走法: {engine_action}")

        move_count += 1

        if verbose and not game.done:
            game.print_board()

    if verbose:
        game.print_board()

    # 判断结果
    if game.winner == 'red':
        result = 'ai_win' if ai_is_red else 'engine_win'
    elif game.winner == 'black':
        result = 'engine_win' if ai_is_red else 'ai_win'
    else:
        result = 'draw'

    if verbose:
        labels = _RESULT_LABELS
        print(f"结果: {labels[result]}（原因: {game.terminate_reason or '未知'}）")

    return result


def run_vs_pikafish(engine_path: str, model_path: str = None,
                    ai_color: str = 'red', movetime_ms: int = DEFAULT_MOVETIME_MS,
                    depth: int = None, num_simulations: int = 200,
                    n_games: int = 1, max_moves: int = 300,
                    verbose: bool = False):
    """运行 AIchess vs Pikafish 对弈。

    Args:
        engine_path: Pikafish 可执行文件路径
        model_path: AIchess 模型文件路径（.pth）；若为 None 则使用默认路径或随机模型
        ai_color: AI 执子颜色，'red'（先手）或 'black'（后手）
        movetime_ms: 引擎每步思考时间（毫秒），默认 100
        depth: 引擎搜索深度（若指定则忽略 movetime_ms）
        num_simulations: AI MCTS 模拟次数，默认 200
        n_games: 对局总数，默认 1
        max_moves: 每局最大步数，默认 300
        verbose: 是否打印棋盘和走法

    Returns:
        dict: {'ai_wins': int, 'engine_wins': int, 'draws': int, 'total': int}
    """
    # 加载 AI 模型
    model = ChessModel()
    resolved_path = model_path or DEFAULT_MODEL_PATH
    if os.path.exists(resolved_path):
        model.load(resolved_path)
        print(f"模型已加载: {resolved_path}")
    else:
        model.build()
        print("未找到训练好的模型，使用随机模型（仅供测试）")

    ai_is_red = (ai_color == 'red')
    stats = {'ai_wins': 0, 'engine_wins': 0, 'draws': 0, 'total': n_games}

    with PikafishEngine(engine_path, movetime_ms=movetime_ms, depth=depth) as engine:
        print(f"Pikafish 引擎已启动: {engine_path}")
        if depth is not None:
            print(f"时间控制: 固定深度 depth={depth}")
        else:
            print(f"时间控制: movetime={movetime_ms}ms / 步")
        print(f"AI 执{'红' if ai_is_red else '黑'}方，共 {n_games} 局\n")

        for game_idx in range(n_games):
            if n_games > 1:
                print(f"--- 第 {game_idx + 1}/{n_games} 局 ---")

            result = play_one_game(
                ai_model=model,
                engine=engine,
                ai_is_red=ai_is_red,
                num_simulations=num_simulations,
                max_moves=max_moves,
                verbose=verbose,
            )

            if result == 'ai_win':
                stats['ai_wins'] += 1
            elif result == 'engine_win':
                stats['engine_wins'] += 1
            else:
                stats['draws'] += 1

            if n_games > 1 or not verbose:
                print(f"第 {game_idx + 1} 局: {_RESULT_LABELS[result]}")

    print(f"\n=== 对战结果 ({n_games} 局) ===")
    print(f"AI 胜:   {stats['ai_wins']}")
    print(f"引擎胜:  {stats['engine_wins']}")
    print(f"和棋:    {stats['draws']}")
    if n_games > 0:
        score = (stats['ai_wins'] + 0.5 * stats['draws']) / n_games
        print(f"AI 得分: {score:.3f}")
        stats['score'] = score

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='AIchess vs Pikafish - 模型与引擎对弈',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--engine_path', required=True,
                        help='Pikafish 可执行文件路径')
    parser.add_argument('--model_path', type=str, default=None,
                        help='AIchess 模型文件路径（.pth）')
    parser.add_argument('--ai_color', type=str, default='red',
                        choices=['red', 'black'],
                        help='AI 执子颜色（red=先手，black=后手）')
    parser.add_argument('--movetime_ms', type=int, default=DEFAULT_MOVETIME_MS,
                        help='引擎每步思考时间（毫秒）；若指定 --depth 则忽略此参数')
    parser.add_argument('--depth', type=int, default=None,
                        help='引擎搜索深度（指定后忽略 --movetime_ms）')
    parser.add_argument('--num_simulations', type=int, default=200,
                        help='AI MCTS 模拟次数')
    parser.add_argument('--n_games', type=int, default=1,
                        help='对局总数')
    parser.add_argument('--max_moves', type=int, default=300,
                        help='每局最大步数')
    parser.add_argument('--verbose', action='store_true',
                        help='打印棋盘和每步走法')
    args = parser.parse_args()

    run_vs_pikafish(
        engine_path=args.engine_path,
        model_path=args.model_path,
        ai_color=args.ai_color,
        movetime_ms=args.movetime_ms,
        depth=args.depth,
        num_simulations=args.num_simulations,
        n_games=args.n_games,
        max_moves=args.max_moves,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
