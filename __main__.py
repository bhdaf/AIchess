"""
简化中国象棋AI - 主入口

用法:
    训练模型:
        python -m simple_chess_ai train [--num_games 50] [--num_simulations 100]

    图形界面对弈:
        python -m simple_chess_ai play [--model_path path/to/model.pth]

    命令行对弈:
        python -m simple_chess_ai play_cli [--model_path path/to/model.pth]
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='简化中国象棋AI',
        usage='python -m simple_chess_ai {train,play,play_cli} [options]'
    )
    parser.add_argument('command', choices=['train', 'play', 'play_cli'],
                        help='命令: train(训练), play(图形界面), play_cli(命令行)')

    args, remaining = parser.parse_known_args()

    if args.command == 'train':
        from simple_chess_ai.train import main as train_main
        sys.argv = [sys.argv[0]] + remaining
        train_main()
    elif args.command == 'play':
        from simple_chess_ai.gui import main as gui_main
        sys.argv = [sys.argv[0]] + remaining
        gui_main()
    elif args.command == 'play_cli':
        from simple_chess_ai.cli import main as cli_main
        sys.argv = [sys.argv[0]] + remaining
        cli_main()


if __name__ == '__main__':
    main()
