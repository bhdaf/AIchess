"""
推理模式命令行界面

展示AI的思维链推理过程，在每次走子前输出分析。

用法:
    python -m simple_chess_ai reason [--model_path path/to/model.pth]
"""

import os
import argparse

from simple_chess_ai.game import ChessGame, flip_move
from simple_chess_ai.model import ChessModel
from simple_chess_ai.reasoning import ChessReasoner

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_model', 'model.pth')


def main():
    parser = argparse.ArgumentParser(description='简化中国象棋AI - 推理模式')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型文件路径')
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
    reasoner = ChessReasoner(model)

    human_is_red = args.human_color == 'red'

    print("\n===== 中国象棋AI - 推理模式 =====")
    print(f"你执{'红' if human_is_red else '黑'}方")
    print("走法格式: x0 y0 x1 y1")
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
                continue
            else:
                break

        is_human = (game.red_to_move and human_is_red) or \
                   (not game.red_to_move and not human_is_red)

        if is_human:
            turn = "红方" if game.red_to_move else "黑方"
            user_input = input(f"[{turn}] 请输入走法 (x0 y0 x1 y1): ").strip()

            if user_input.lower() == 'q':
                break
            elif user_input.lower() == 'r':
                game = ChessGame()
                game.reset()
                continue

            parts = user_input.replace(',', ' ').split()
            if len(parts) != 4:
                print("格式错误！")
                continue

            try:
                x0, y0, x1, y1 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                print("请输入数字！")
                continue

            move = f"{x0}{y0}{x1}{y1}"
            legal_moves = game.get_legal_moves()
            if move not in legal_moves:
                print(f"非法走法！")
                continue

            game.step(move)
            print(f"你走了: {move}")
        else:
            print("\nAI推理中...")
            reasoning_text, action, policy = reasoner.reason_and_act(game)
            print(f"  思维链: {reasoning_text}")

            if not game.red_to_move:
                actual_action = flip_move(action)
            else:
                actual_action = action

            game.step(actual_action)
            print(f"  AI走了: {actual_action}\n")

    print("游戏结束")


if __name__ == '__main__':
    main()
