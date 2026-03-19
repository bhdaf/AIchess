"""
简化中国象棋AI - 主入口

用法:
    训练模型:
        python -m AIchess train [--num_games 50] [--num_simulations 100]

    策略蒸馏（阶段 A）：
        python -m AIchess distill --engine_path /path/to/pikafish \\
            [--out_model saved_model/model_distill.pth] [--n_games 200]

    图形界面对弈:
        python -m AIchess play [--model_path path/to/model.pth]

    命令行对弈:
        python -m AIchess play_cli [--model_path path/to/model.pth]

    模型评测:
        python -m AIchess eval --model_a A.pth --model_b B.pth [--n_games 200]

    绘制训练曲线:
        python -m AIchess plot --run_dir runs/run_YYYYMMDD_HHMMSS

    与 Pikafish/UCI 引擎对弈:
        python -m AIchess vs_pikafish --engine_path /path/to/pikafish \\
            [--model_path model.pth] [--n_games 10] [--elo 1500]
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='简化中国象棋AI',
        usage='python -m AIchess {train,distill,play,play_cli,eval,plot,vs_pikafish} [options]'
    )
    parser.add_argument('command',
                        choices=['train', 'distill', 'play', 'play_cli', 'eval',
                                 'plot', 'vs_pikafish'],
                        help='命令: train(训练), distill(策略蒸馏), play(图形界面), '
                             'play_cli(命令行), eval(模型评测), plot(绘制训练曲线), '
                             'vs_pikafish(与UCI引擎对弈)')

    args, remaining = parser.parse_known_args()

    if args.command == 'train':
        from .train import main as train_main
        sys.argv = [sys.argv[0]] + remaining
        train_main()
    elif args.command == 'distill':
        from .distill import main as distill_main
        sys.argv = [sys.argv[0]] + remaining
        distill_main()
    elif args.command == 'play':
        from .gui import main as gui_main
        sys.argv = [sys.argv[0]] + remaining
        gui_main()
    elif args.command == 'play_cli':
        from .cli import main as cli_main
        sys.argv = [sys.argv[0]] + remaining
        cli_main()
    elif args.command == 'eval':
        from .eval import main as eval_main
        sys.argv = [sys.argv[0]] + remaining
        eval_main()
    elif args.command == 'plot':
        from .plot import main as plot_main
        sys.argv = [sys.argv[0]] + remaining
        plot_main()
    elif args.command == 'vs_pikafish':
        from .vs_pikafish import main as vs_pikafish_main
        sys.argv = [sys.argv[0]] + remaining
        vs_pikafish_main()


if __name__ == '__main__':
    main()
