"""
数据与图片导出管线

提供自对弈数据落盘、训练指标记录、gating 评测记录以及可视化图片生成功能。
所有输出写入带时间戳的运行目录，方便多次实验的结果对比与复现。

默认输出目录结构::

    simple_chess_ai/runs/<run_YYYYMMDD_HHMMSS>/
        config.json              训练超参数及元信息
        self_play.jsonl          每局自对弈记录（每行一个 JSON 对象）
        training_metrics.csv     训练指标（game_idx、loss、buffer_size 等）
        gating_metrics.csv       gating 评测结果（winrate、是否接受等）
        plots/
            loss_curve.png       训练损失曲线
            winrate_curve.png    gating 胜率曲线

用法示例::

    from simple_chess_ai.export import init_run_dir, append_self_play_jsonl, \\
        append_training_csv, append_gating_csv, plot_curves

    run_dir = init_run_dir(config=my_config)
    append_self_play_jsonl(run_dir, {"game_idx": 1, "winner": "red", ...})
    append_training_csv(run_dir, {"game_idx": 1, "loss": 2.34, ...})
    append_gating_csv(run_dir, {"game_idx": 20, "winrate": 0.6, "accepted": True, ...})
    plot_curves(run_dir)
"""

import os
import csv
import json
import datetime
import traceback

# 默认运行目录根路径
DEFAULT_RUNS_DIR = os.path.join(os.path.dirname(__file__), 'runs')


def init_run_dir(runs_dir=None, config=None):
    """
    初始化带时间戳的运行目录，保存训练配置。

    在 ``runs_dir`` 下创建 ``run_<YYYYMMDD_HHMMSS>`` 子目录，
    同时自动创建 ``plots/`` 子目录，并将 ``config`` 写为 ``config.json``。

    Args:
        runs_dir (str | None): 根目录路径。默认为 ``simple_chess_ai/runs/``。
        config (dict | None): 训练配置字典，写入 ``config.json``；为 ``None`` 时跳过。

    Returns:
        str: 本次运行的输出目录绝对路径。
    """
    if runs_dir is None:
        runs_dir = DEFAULT_RUNS_DIR

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(runs_dir, f'run_{timestamp}')

    try:
        os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    except OSError:
        traceback.print_exc()

    if config is not None:
        config_with_meta = dict(config)
        config_with_meta['_run_dir'] = run_dir
        config_with_meta['_start_time'] = datetime.datetime.now().isoformat(timespec='seconds')
        config_path = os.path.join(run_dir, 'config.json')
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_with_meta, f, indent=2, ensure_ascii=False)
        except OSError:
            traceback.print_exc()

    return run_dir


def append_self_play_jsonl(run_dir, record):
    """
    将一局自对弈记录追加写入 ``self_play.jsonl``。

    每次调用追加一行，每行是完整的 JSON 对象，格式示例::

        {"game_idx": 1, "winner": "red", "num_moves": 85,
         "num_samples": 85, "elapsed_s": 12.3,
         "timestamp": "2024-01-01T10:00:00"}

    Args:
        run_dir (str): 运行目录路径（由 :func:`init_run_dir` 返回）。
        record (dict): 游戏记录字典，建议包含 ``game_idx``、``winner``、
                       ``num_moves``、``num_samples``、``elapsed_s`` 字段。
    """
    path = os.path.join(run_dir, 'self_play.jsonl')
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except OSError:
        traceback.print_exc()


def append_training_csv(run_dir, row):
    """
    将训练指标一行追加写入 ``training_metrics.csv``。

    首次写入时自动添加表头。CSV 字段示例::

        game_idx, timestamp, loss, buffer_size, elapsed_s

    Args:
        run_dir (str): 运行目录路径。
        row (dict): 指标字典，字段名作为 CSV 列名。建议包含 ``game_idx``、
                    ``loss``、``buffer_size``、``elapsed_s`` 字段。
    """
    path = os.path.join(run_dir, 'training_metrics.csv')
    write_header = not os.path.exists(path)
    try:
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except OSError:
        traceback.print_exc()


def append_gating_csv(run_dir, row):
    """
    将 gating 评测结果一行追加写入 ``gating_metrics.csv``。

    首次写入时自动添加表头。CSV 字段示例::

        game_idx, timestamp, winrate, wins, losses, draws, accepted

    Args:
        run_dir (str): 运行目录路径。
        row (dict): 评测结果字典。建议包含 ``game_idx``、``winrate``、
                    ``wins``、``losses``、``draws``、``accepted`` 字段。
    """
    path = os.path.join(run_dir, 'gating_metrics.csv')
    write_header = not os.path.exists(path)
    try:
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except OSError:
        traceback.print_exc()


def plot_curves(run_dir):
    """
    从 CSV 文件自动生成 PNG 曲线图，保存到 ``plots/`` 子目录。

    生成的图片:

    - ``plots/loss_curve.png``    — 训练损失随局数的变化曲线
    - ``plots/winrate_curve.png`` — gating 评测胜率曲线（绿点=接受，红点=拒绝）

    Args:
        run_dir (str): 运行目录路径，需包含 ``training_metrics.csv``
                       和/或 ``gating_metrics.csv``。

    Note:
        依赖 ``matplotlib``。若未安装，函数会打印提示并直接返回，不会抛出异常。
        安装方式: ``pip install matplotlib``。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无显示设备时的 headless 后端
        import matplotlib.pyplot as plt
    except ImportError:
        print("提示: 未安装 matplotlib，跳过图表生成。安装方式: pip install matplotlib")
        return

    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ── 训练损失曲线 ────────────────────────────────────────────────────────
    training_csv = os.path.join(run_dir, 'training_metrics.csv')
    if os.path.exists(training_csv):
        try:
            game_indices, losses = [], []
            with open(training_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    loss_val = row.get('loss', '')
                    if loss_val and float(loss_val) > 0:
                        game_indices.append(int(row['game_idx']))
                        losses.append(float(loss_val))

            if game_indices:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(game_indices, losses, color='tab:blue',
                        linewidth=1.5, label='Total Loss')
                ax.set_xlabel('Game Index')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                out_path = os.path.join(plots_dir, 'loss_curve.png')
                plt.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"  图表已保存: {out_path}")
        except Exception:
            traceback.print_exc()

    # ── Gating 胜率曲线 ─────────────────────────────────────────────────────
    gating_csv = os.path.join(run_dir, 'gating_metrics.csv')
    if os.path.exists(gating_csv):
        try:
            game_indices, winrates, accepted_flags = [], [], []
            with open(gating_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    game_indices.append(int(row['game_idx']))
                    # 支持新字段 score（draw=0.5 计分）及旧字段 winrate
                    score_val = row.get('score') or row.get('winrate', '0')
                    winrates.append(float(score_val))
                    accepted_flags.append(row.get('accepted', '').lower() == 'true')

            if game_indices:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(game_indices, winrates, color='tab:orange',
                        linewidth=1.5, label='Win Rate vs Baseline')
                # 接受/拒绝点用绿/红标注
                for x, y, acc in zip(game_indices, winrates, accepted_flags):
                    ax.scatter(x, y, color='green' if acc else 'red',
                               zorder=5, s=50)
                ax.axhline(0.5, color='gray', linestyle='--',
                           alpha=0.5, label='50% baseline')
                ax.set_xlabel('Game Index')
                ax.set_ylabel('Win Rate')
                ax.set_title('Gating Win Rate Curve')
                ax.set_ylim(0, 1)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                out_path = os.path.join(plots_dir, 'winrate_curve.png')
                plt.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"  图表已保存: {out_path}")
        except Exception:
            traceback.print_exc()


def save_board_screenshot(surface, save_dir=None, filename=None):
    """
    将 Pygame surface 保存为 PNG 截图。

    Args:
        surface: ``pygame.Surface`` 对象（即 ``pygame.display.get_surface()``）。
        save_dir (str | None): 截图保存目录。默认为 ``simple_chess_ai/runs/screenshots/``。
        filename (str | None): 文件名（不含路径）。默认自动生成带时间戳的名称。

    Returns:
        str | None: 保存路径；失败时返回 ``None``。
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'runs', 'screenshots')

    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError:
        traceback.print_exc()
        return None

    if filename is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]
        filename = f'screenshot_{ts}.png'

    out_path = os.path.join(save_dir, filename)
    try:
        import pygame
        pygame.image.save(surface, out_path)
        return out_path
    except Exception:
        traceback.print_exc()
        return None
