"""
数据导出工具

提供自对弈数据落盘、训练指标记录功能。
所有输出写入带时间戳的运行目录。

默认输出目录结构::

    simple_chess_ai/runs/<run_YYYYMMDD_HHMMSS>/
        config.json              训练超参数及元信息
        self_play.jsonl          每局自对弈记录（每行一个 JSON 对象）
        training_metrics.csv     训练指标（game_idx、loss、buffer_size 等）
        evaluation_metrics.csv   评测指标（game_idx、score、elo 等）
        evaluation_state.json    ELO 状态（当前评分、基准模型信息等）
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
        os.makedirs(run_dir, exist_ok=True)
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

    Args:
        run_dir (str): 运行目录路径（由 :func:`init_run_dir` 返回）。
        record (dict): 游戏记录字典。
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

    Args:
        run_dir (str): 运行目录路径。
        row (dict): 指标字典，字段名作为 CSV 列名。
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


def append_evaluation_csv(run_dir, row):
    """
    将评测指标一行追加写入 ``evaluation_metrics.csv``。

    Args:
        run_dir (str): 运行目录路径。
        row (dict): 指标字典，字段名作为 CSV 列名。
            推荐字段：``game_idx``, ``timestamp``, ``opponent``,
            ``eval_games``, ``eval_sims``, ``wins``, ``losses``,
            ``draws``, ``score``, ``elo``, ``elo_delta``。
    """
    path = os.path.join(run_dir, 'evaluation_metrics.csv')
    write_header = not os.path.exists(path)
    try:
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except OSError:
        traceback.print_exc()


def load_evaluation_state(run_dir):
    """
    从 ``evaluation_state.json`` 加载 ELO 状态。

    Args:
        run_dir (str): 运行目录路径。

    Returns:
        dict: ELO 状态字典。若文件不存在则返回默认初始状态
            ``{'elo_current': 1500.0, 'elo_opponent': 1500.0}``。
    """
    path = os.path.join(run_dir, 'evaluation_state.json')
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            traceback.print_exc()
    return {'elo_current': 1500.0, 'elo_opponent': 1500.0}


def save_evaluation_state(run_dir, state):
    """
    将 ELO 状态保存到 ``evaluation_state.json``。

    Args:
        run_dir (str): 运行目录路径。
        state (dict): ELO 状态字典（包含 ``elo_current``、``elo_opponent`` 等键）。
    """
    path = os.path.join(run_dir, 'evaluation_state.json')
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except OSError:
        traceback.print_exc()
