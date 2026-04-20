# AIchess（中国象棋 AI）

一个以 **AlphaZero 风格 MCTS + 策略价值网络** 为核心的中国象棋项目，支持：

- 自对弈强化学习训练（`train.py`）
- 阶段 A 策略蒸馏（`distill.py`，可接入 Pikafish）
- 模型评测与 ELO 跟踪（`eval.py` + `export.py`）
- 图形界面/命令行人机对弈（`gui.py` / `cli.py`）
- 与 Pikafish 等 UCI 引擎对局（`vs_pikafish.py`）
- 训练曲线绘图（`plot.py`）

---

## 1. 环境要求

- Python 3.10+（建议）
- 核心依赖：
  - `numpy`
  - `torch`
- 可选依赖：
  - `pygame`（图形界面）
  - `matplotlib`（绘图）
- 外部引擎（可选）：
  - Pikafish 或其他兼容 UCI 的中国象棋引擎

### 快速安装

```bash
pip install numpy torch pygame matplotlib
```

> 若只需要训练/评测，可不安装 `pygame`；若不需要绘图，可不安装 `matplotlib`。

---

## 2. 项目结构（核心文件）

```text
AIchess/
├── __main__.py            # 统一命令入口：train/distill/play/play_cli/eval/plot/vs_pikafish
├── game.py                # 棋盘规则、走法生成、状态编码
├── model.py               # 策略价值网络与模型保存/加载
├── mcts.py                # 蒙特卡洛树搜索
├── train.py               # 强化学习训练主流程（含并行 self-play）
├── distill.py             # 策略蒸馏（teacher soft target）
├── eval.py                # 双模型对弈评测
├── vs_pikafish.py         # 与 UCI 引擎对弈，输出 PGN/JSONL
├── gui.py                 # Pygame 图形界面
├── cli.py                 # 命令行对弈
├── export.py              # 训练/评测数据落盘
├── plot.py                # 训练曲线绘图
└── tests.py               # 单元测试
```

---

## 3. 运行方式

从仓库父目录运行（让 `AIchess` 作为包）：

```bash
cd /home/runner/work/AIchess
python -m AIchess <command> [options]
```

### 可用 command

- `train`：强化学习训练
- `distill`：策略蒸馏
- `play`：图形界面对弈
- `play_cli`：命令行对弈
- `eval`：模型对局评测
- `plot`：绘图
- `vs_pikafish`：与 UCI 引擎对弈

---

## 4. 常用命令

## 4.1 训练（RL）

```bash
python -m AIchess train \
  --num_games 50 \
  --num_simulations 100 \
  --num_epochs 5 \
  --batch_size 256 \
  --lr 0.001
```

常用参数：

- `--quick`：最小闭环快速验证
- `--model_path`：模型保存/加载路径
- `--eval_interval`：周期性评测间隔（0 为关闭）
- `--init_from_distill`：从蒸馏模型启动 RL 微调
- `--num_selfplay_workers`：并行自对弈 worker 数（0=单进程）
- `--device {cuda,cpu}`：训练设备

---

## 4.2 蒸馏（Distill）

```bash
python -m AIchess distill \
  --engine_path /path/to/pikafish_weak \
  --teacher_engine_path /path/to/pikafish_strong \
  --out_model AIchess/saved_model/model_distill.pth \
  --n_games 200 \
  --multipv_k 5
```

常用参数：

- `--trajectory_source {weak,teacher,mixed}`
- `--teacher_temperature`
- `--teacher_min_top1_prob`
- `--teacher_low_quality_action {fallback_onehot,fallback_topk_sharpen,skip_sample}`
- `--distill_value_mode {zero,game_outcome}`

---

## 4.3 模型评测

```bash
python -m AIchess eval \
  --model_a /path/to/model_a.pth \
  --model_b /path/to/model_b.pth \
  --n_games 200 \
  --num_simulations 50 \
  --out AIchess/runs/eval_run
```

---

## 4.4 人机对弈

### 图形界面（Pygame）

```bash
python -m AIchess play \
  --model_path /path/to/model.pth \
  --num_simulations 200 \
  --human_color red
```

### 命令行

```bash
python -m AIchess play_cli \
  --model_path /path/to/model.pth \
  --num_simulations 200 \
  --human_color red
```

---

## 4.5 与 Pikafish 对弈（输出 PGN）

```bash
python -m AIchess vs_pikafish \
  --engine_path /path/to/pikafish \
  --model_path /path/to/model.pth \
  --n_games 10 \
  --num_simulations 50 \
  --ai_side both \
  --out AIchess/runs/vs_pikafish_run
```

输出文件：

- `vs_pikafish_log.jsonl`
- `vs_pikafish.pgn`

---

## 4.6 绘图

```bash
python -m AIchess plot \
  --run_dir AIchess/runs/run_YYYYMMDD_HHMMSS \
  --format png
```

默认生成：

- `elo.png`
- `score.png`
- `loss.png`

---

## 5. 训练输出说明

训练/蒸馏运行目录通常位于 `AIchess/runs/run_时间戳/`，常见文件：

- `config.json`：运行配置
- `self_play.jsonl`：每局对弈记录
- `training_history.csv`：训练过程指标
- `evaluation_metrics.csv`：评测指标（含 score/ELO）
- `evaluation_state.json`：ELO 状态

模型默认保存在：

- `AIchess/saved_model/model.pth`（RL）
- `AIchess/saved_model/model_distill.pth`（蒸馏）

---

## 6. 测试

在仓库父目录执行：

```bash
cd /home/runner/work/AIchess
python -m AIchess.tests
```

---

## 7. 常见问题

1. **`ModuleNotFoundError: No module named 'AIchess'`**
   - 原因：在 `AIchess` 目录内直接运行模块。
   - 解决：切到父目录 `/home/runner/work/AIchess` 再执行 `python -m AIchess ...`。

2. **无法使用引擎相关命令**
   - 检查 `--engine_path` 是否可执行、UCI 引擎是否可正常启动。

