"""
绘图模块

从运行目录中的 CSV 文件生成训练曲线图（ELO、胜率、损失）。

依赖 ``matplotlib``（可选）。若未安装，将打印明确的错误提示。

用法::

    python -m simple_chess_ai plot \\
        --run_dir runs/run_YYYYMMDD_HHMMSS \\
        [--out_dir output/] \\
        [--format png]
"""

import argparse
import csv
import os
import sys


def _require_matplotlib():
    """导入 matplotlib，未安装时打印友好错误提示并退出。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print(
            "错误：未找到 matplotlib。请先安装：\n"
            "    pip install matplotlib\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _read_csv(path):
    """读取 CSV 文件，返回字典列表。文件不存在时返回空列表。"""
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def plot_elo(run_dir, out_dir, fmt='png'):
    """
    绘制 ELO 曲线，并保存到 ``out_dir/elo.<fmt>``。

    数据来源：``run_dir/evaluation_metrics.csv``（需包含 ``game_idx`` 和 ``elo`` 列）。
    若数据不足，跳过并打印提示。

    Args:
        run_dir (str): 运行目录路径。
        out_dir (str): 图片输出目录。
        fmt (str): 图片格式（默认 ``'png'``）。

    Returns:
        str | None: 输出文件路径，若数据不足则返回 None。
    """
    plt = _require_matplotlib()
    rows = _read_csv(os.path.join(run_dir, 'evaluation_metrics.csv'))
    rows = [r for r in rows if r.get('elo') not in (None, '')]
    if not rows:
        print("跳过 elo 图：evaluation_metrics.csv 中没有 elo 数据。")
        return None

    x = [int(r['game_idx']) for r in rows]
    y = [float(r['elo']) for r in rows]

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', markersize=3)
    ax.set_xlabel('game_idx')
    ax.set_ylabel('ELO')
    ax.set_title('ELO Rating over Training')
    ax.grid(True)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f'elo.{fmt}')
    fig.savefig(out_path, format=fmt)
    plt.close(fig)
    print(f"已保存: {out_path}")
    return out_path


def plot_score(run_dir, out_dir, fmt='png'):
    """
    绘制胜率（score）曲线，并保存到 ``out_dir/score.<fmt>``。

    数据来源：``run_dir/evaluation_metrics.csv``（需包含 ``game_idx`` 和 ``score`` 列）。

    Args:
        run_dir (str): 运行目录路径。
        out_dir (str): 图片输出目录。
        fmt (str): 图片格式（默认 ``'png'``）。

    Returns:
        str | None: 输出文件路径，若数据不足则返回 None。
    """
    plt = _require_matplotlib()
    rows = _read_csv(os.path.join(run_dir, 'evaluation_metrics.csv'))
    rows = [r for r in rows if r.get('score') not in (None, '')]
    if not rows:
        print("跳过 score 图：evaluation_metrics.csv 中没有 score 数据。")
        return None

    x = [int(r['game_idx']) for r in rows]
    y = [float(r['score']) for r in rows]

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', markersize=3, color='tab:orange')
    ax.axhline(0.5, linestyle='--', color='gray', linewidth=0.8)
    ax.set_xlabel('game_idx')
    ax.set_ylabel('Score')
    ax.set_title('Win Rate (Score) over Training')
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f'score.{fmt}')
    fig.savefig(out_path, format=fmt)
    plt.close(fig)
    print(f"已保存: {out_path}")
    return out_path


def plot_loss(run_dir, out_dir, fmt='png'):
    """
    绘制训练损失曲线，并保存到 ``out_dir/loss.<fmt>``。

    数据来源：``run_dir/training_metrics.csv``（需包含 ``game_idx`` 和 ``loss`` 列）。

    Args:
        run_dir (str): 运行目录路径。
        out_dir (str): 图片输出目录。
        fmt (str): 图片格式（默认 ``'png'``）。

    Returns:
        str | None: 输出文件路径，若数据不足则返回 None。
    """
    plt = _require_matplotlib()
    rows = _read_csv(os.path.join(run_dir, 'training_metrics.csv'))
    rows = [r for r in rows if r.get('loss') not in (None, '')]
    if not rows:
        print("跳过 loss 图：training_metrics.csv 中没有 loss 数据。")
        return None

    x = [int(r['game_idx']) for r in rows]
    y = [float(r['loss']) for r in rows]

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', markersize=3, color='tab:green')
    ax.set_xlabel('game_idx')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss over Training')
    ax.grid(True)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f'loss.{fmt}')
    fig.savefig(out_path, format=fmt)
    plt.close(fig)
    print(f"已保存: {out_path}")
    return out_path


def run_plot(run_dir, out_dir=None, fmt='png'):
    """
    从运行目录生成全部曲线图。

    Args:
        run_dir (str): 运行目录路径（包含 CSV 文件）。
        out_dir (str | None): 图片输出目录。默认与 ``run_dir`` 相同。
        fmt (str): 图片格式（默认 ``'png'``）。

    Returns:
        list[str]: 成功生成的图片路径列表。
    """
    if out_dir is None:
        out_dir = run_dir
    os.makedirs(out_dir, exist_ok=True)

    outputs = []
    for fn in (plot_elo, plot_score, plot_loss):
        path = fn(run_dir, out_dir, fmt)
        if path is not None:
            outputs.append(path)
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description='绘图 - 从运行目录生成训练曲线图',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--run_dir', required=True,
                        help='运行目录路径（包含 training_metrics.csv / evaluation_metrics.csv）')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='图片输出目录（默认与 run_dir 相同）')
    parser.add_argument('--format', type=str, default='png',
                        dest='fmt',
                        help='图片格式（png / pdf / svg 等）')

    args = parser.parse_args()
    run_plot(run_dir=args.run_dir, out_dir=args.out_dir, fmt=args.fmt)


if __name__ == '__main__':
    main()
