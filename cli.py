"""
命令行对弈界面

纯文本界面的人机对弈模式。

用法:
    python -m AIchess play_cli [--model_path path/to/model.pth]
"""

import os
import sys
import argparse
import numpy as np

from .game import ChessGame, flip_move
from .model import ChessModel
from .mcts import MCTS

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_model', 'model.pth')


def main():
    parser = argparse.ArgumentParser(description='简化中国象棋AI - 命令行界面')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型文件路径')
    parser.add_argument('--num_simulations', type=int, default=200,
                        help='AI搜索模拟次数 (默认: 200)')
    parser.add_argument('--human_color', type=str, default='red',
                        choices=['red', 'black'],
                        help='人类执哪方 (默认: red)')
    args = parser.parse_args()

    model_path = args.model_path or DEFAULT_MODEL_PATH

    # 加载模型
    model = ChessModel()
    if os.path.exists(model_path):
        model.load(model_path)
        print(f"模型已加载: {model_path}")
    else:
        model.build()
        print("未找到训练好的模型，使用随机模型")

    game = ChessGame()
    game.reset()
    mcts = MCTS(model, num_simulations=args.num_simulations)

    human_is_red = args.human_color == 'red'

    print("\n===== 简化中国象棋AI =====")
    print(f"你执{'红' if human_is_red else '黑'}方")
    print("走法格式: x0 y0 x1 y1 (例如: 4 0 4 1 表示帅向前走一步)")
    print("输入 'q' 退出, 'r' 重新开始\n")

    while True:
        game.print_board()

        if game.done:
            if game.winner == 'red':
                print("红方胜！")
            elif game.winner == 'black':
                print("黑方胜！")
            else:
                print("和棋！")

            cmd = input("再来一局? (y/n): ").strip().lower()
            if cmd == 'y':
                game = ChessGame()
                game.reset()
                mcts = MCTS(model, num_simulations=args.num_simulations)
                continue
            else:
                break

        is_human = (game.red_to_move and human_is_red) or \
                   (not game.red_to_move and not human_is_red)

        if is_human:
            turn = "红方" if game.red_to_move else "黑方"
            user_input = input(f"[{turn}] 请输入走法 (x0 y0 x1 y1): ").strip()

            if user_input.lower() == 'q':
                print("再见！")
                break
            elif user_input.lower() == 'r':
                game = ChessGame()
                game.reset()
                mcts = MCTS(model, num_simulations=args.num_simulations)
                continue

            parts = user_input.replace(',', ' ').split()
            if len(parts) != 4:
                print("格式错误！请输入4个数字: x0 y0 x1 y1")
                continue

            try:
                x0, y0, x1, y1 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                print("请输入数字！")
                continue

            move = f"{x0}{y0}{x1}{y1}"
            legal_moves = game.get_legal_moves()
            if move not in legal_moves:
                print(f"非法走法！合法走法示例: {legal_moves[:5]}")
                continue

            red_before = game.red_to_move
            game.step(move)
            mcts_move = move if red_before else flip_move(move)
            mcts.update_with_move(mcts_move)
            print(f"你走了: {move}")
        else:
            print("AI思考中...")
            actions, probs = mcts.get_action_probs(
                game, temperature=0.1, reset_root=False
            )

            if not actions:
                print("AI无合法走法")
                break

            best_idx = np.argmax(probs)
            action = actions[best_idx]

            if not game.red_to_move:
                actual_action = flip_move(action)
            else:
                actual_action = action

            game.step(actual_action)
            mcts.update_with_move(action)
            print(f"AI走了: {actual_action}")

    print("游戏结束")


if __name__ == '__main__':
    main()
