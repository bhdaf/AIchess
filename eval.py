"""
评测模块

对两个模型进行对局评测，输出胜率统计。支持独立 CLI 调用和作为库函数使用。

用法::

    python -m simple_chess_ai eval \\
        --model_a path/to/model_a.pth \\
        --model_b path/to/model_b.pth \\
        [--n_games 200] [--num_simulations 50] [--max_moves 200] \\
        [--seed 0] [--out runs/run_YYYYMMDD_HHMMSS]
"""

import argparse
import datetime
import json
import os
import random
import sys

import numpy as np
import torch

from simple_chess_ai.model import ChessModel
from simple_chess_ai.train import evaluate_models
from simple_chess_ai.export import append_evaluation_csv


def set_seeds(seed):
    """设置所有随机种子以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_eval(model_a_path, model_b_path, n_games=200, num_simulations=50,
             max_moves=200, seed=0, out=None):
    """
    加载两个模型并进行评测。

    Args:
        model_a_path (str): 模型A的路径。
        model_b_path (str): 模型B的路径。
        n_games (int): 对局数（建议为偶数以保证红黑均衡）。
        num_simulations (int): 每步MCTS模拟次数。
        max_moves (int): 每局最大步数。
        seed (int): 随机种子。
        out (str | None): 运行目录路径。若提供则将结果追加写入
            ``evaluation_metrics.csv``。

    Returns:
        dict: 包含 ``wins_a``, ``wins_b``, ``draws``, ``score`` 的字典，
            其中 ``score = (wins_a + 0.5 * draws) / total``。
    """
    set_seeds(seed)

    model_a = ChessModel()
    model_a.load(model_a_path)

    model_b = ChessModel()
    model_b.load(model_b_path)

    score, wins_a, wins_b, draws = evaluate_models(
        model_a, model_b,
        n_games=n_games,
        num_simulations=num_simulations,
        max_moves=max_moves,
    )

    result = {
        'wins_a': wins_a,
        'wins_b': wins_b,
        'draws': draws,
        'score': score,
    }

    if out is not None:
        os.makedirs(out, exist_ok=True)
        row = {
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'model_a': model_a_path,
            'model_b': model_b_path,
            'n_games': n_games,
            'num_simulations': num_simulations,
            'max_moves': max_moves,
            'seed': seed,
            'wins_a': wins_a,
            'wins_b': wins_b,
            'draws': draws,
            'score': round(score, 6),
        }
        append_evaluation_csv(out, row)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='模型评测 - 对两个模型进行对局评测',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_a', required=True,
                        help='模型A的路径（.pth文件）')
    parser.add_argument('--model_b', required=True,
                        help='模型B的路径（.pth文件）')
    parser.add_argument('--n_games', type=int, default=200,
                        help='对局数（建议为偶数以保证红黑均衡）')
    parser.add_argument('--num_simulations', type=int, default=50,
                        help='每步MCTS模拟次数')
    parser.add_argument('--max_moves', type=int, default=200,
                        help='每局最大步数')
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子（保证可复现）')
    parser.add_argument('--out', type=str, default=None,
                        help='运行目录路径；若提供则将结果追加写入 evaluation_metrics.csv')

    args = parser.parse_args()

    result = run_eval(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        n_games=args.n_games,
        num_simulations=args.num_simulations,
        max_moves=args.max_moves,
        seed=args.seed,
        out=args.out,
    )

    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()
