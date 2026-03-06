# 简化版中国象棋 AI（AlphaZero 风格）

轻量、易理解、可在普通电脑上跑通的中国象棋智能体。

## 特点

- **完整中国象棋规则**：蹩马腿、塞象眼、飞将、长将/长捉、三次重复局面判和等
- **轻量级 CNN 策略价值网络**：128 通道、4 个残差块，参数量约为原版 AlphaZero 的 1/4
- **基于 MCTS 的自对弈与决策**：蒙特卡洛树搜索，带 Dirichlet 噪声增强探索
- **单机自博弈训练脚本**：自对弈生成数据 → 训练神经网络，循环迭代
- **内置 ELO 评分系统**：训练过程中自动追踪模型强度变化
- **图形界面（Pygame）与命令行对弈模式**

## 目录结构

```
simple_chess_ai/
├── __init__.py        # 包初始化
├── __main__.py        # 主入口（train / play / play_cli / eval / plot）
├── game.py            # 完整中国象棋规则与状态管理
├── model.py           # CNN 策略价值网络（PolicyValueNet + ChessModel）
├── mcts.py            # 蒙特卡洛树搜索（MCTS）
├── train.py           # 自对弈 + AlphaZero 风格训练管线（含 ELO 评测）
├── eval.py            # 独立模型评测模块
├── plot.py            # 训练曲线绘图模块（需要 matplotlib）
├── export.py          # 日志记录（JSONL / CSV / ELO 状态）
├── gui.py             # Pygame 图形界面
├── cli.py             # 命令行文字界面
├── tests.py           # 单元测试
└── README.md
```

## 依赖

- Python 3.8+
- PyTorch >= 1.9
- NumPy
- Pygame（仅图形界面需要）
- matplotlib（仅绘图需要，可选）

```bash
pip install torch numpy
pip install pygame        # 可选，图形界面
pip install matplotlib    # 可选，训练曲线绘图
```

## 快速开始

### 1. 训练模型

```bash
# 默认训练（50 局自对弈）
python -m simple_chess_ai train

# 自定义参数
python -m simple_chess_ai train --num_games 100 --num_simulations 200

# 快速验证流程（1 局 + 1 次训练）
python -m simple_chess_ai train --quick
```

### 2. 带评测的训练

开启 `--eval_interval` 后，训练循环将每隔指定局数自动评测当前模型，
并将 ELO 分数和胜率记录到运行目录的 `evaluation_metrics.csv`。

```bash
# 每 10 局对战上一基准模型评测一次
python -m simple_chess_ai train \
    --num_games 100 \
    --eval_interval 10 \
    --eval_games 20 \
    --eval_simulations 50 \
    --eval_opponent previous \
    --elo_k 32
```

### 3. 对弈

```bash
# 图形界面（需要 Pygame）
python -m simple_chess_ai play --model_path saved_model/model.pth

# 命令行界面
python -m simple_chess_ai play_cli --model_path saved_model/model.pth

# 执黑方
python -m simple_chess_ai play --human_color black
```

### 4. 独立模型评测

```bash
python -m simple_chess_ai eval \
    --model_a saved_model/model_v1.pth \
    --model_b saved_model/model_v2.pth \
    --n_games 200 \
    --num_simulations 50 \
    --seed 42 \
    --out runs/run_20240101_120000
```

输出（JSON 到 stdout）：

```json
{"wins_a": 110, "wins_b": 72, "draws": 18, "score": 0.595}
```

若提供 `--out`，结果同时追加到该目录的 `evaluation_metrics.csv`。

### 5. 绘制训练曲线

```bash
python -m simple_chess_ai plot \
    --run_dir runs/run_20240101_120000 \
    --out_dir runs/run_20240101_120000/plots \
    --format png
```

生成三张图：

| 文件名 | 内容 |
|--------|------|
| `elo.png` | ELO 评分随训练进度的变化曲线 |
| `score.png` | 胜率（score）随训练进度的变化曲线 |
| `loss.png` | 训练损失随训练进度的变化曲线 |

> **注意**：绘图需要安装 `matplotlib`。未安装时会打印清晰的安装提示。

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_games` | 50 | 自对弈总局数 |
| `--num_simulations` | 100 | 每步 MCTS 模拟次数（越大越强，越慢）|
| `--num_epochs` | 5 | 每轮训练的 epoch 数 |
| `--batch_size` | 256 | 训练批大小 |
| `--lr` | 0.001 | 学习率 |
| `--max_moves` | 200 | 每局最大步数，超过判和 |
| `--model_path` | saved_model/model.pth | 模型保存路径 |
| `--eval_interval` | 0 | 评测间隔（0 = 禁用）|
| `--eval_games` | 40 | 每次评测对局数 |
| `--eval_simulations` | 50 | 评测每步 MCTS 模拟次数 |
| `--eval_opponent` | previous | 评测对手：`previous`（上一基准）/ `self`（随机初始）|
| `--elo_k` | 32 | ELO K 因子 |

## 对弈参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | saved_model/model.pth | 模型文件路径 |
| `--num_simulations` | 200 | AI 搜索模拟次数 |
| `--human_color` | red | 人类执哪方（red / black）|

## 运行目录输出说明

训练/评测结果统一保存在 `runs/run_<时间戳>/` 目录下：

| 文件 | 说明 |
|------|------|
| `config.json` | 训练超参数及元信息 |
| `self_play.jsonl` | 每局自对弈记录（每行一个 JSON 对象）|
| `training_metrics.csv` | 训练损失指标 |
| `evaluation_metrics.csv` | 评测指标（见下表）|
| `evaluation_state.json` | ELO 持久化状态 |
| `baseline.pth` | 当前基准模型权重（`eval_opponent=previous` 时生成）|

### evaluation_metrics.csv 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `game_idx` | int | 触发评测时的训练局数序号 |
| `timestamp` | str | 评测时间（ISO 8601）|
| `opponent` | str | 对手类型（`previous` / `random_init`）|
| `eval_games` | int | 本次评测对局数 |
| `eval_sims` | int | 本次评测每步 MCTS 模拟次数 |
| `wins` | int | 当前模型胜局数 |
| `losses` | int | 当前模型负局数 |
| `draws` | int | 和棋数 |
| `score` | float | 得分 = (wins + 0.5×draws) / total |
| `elo` | float | 更新后的 ELO 评分 |
| `elo_delta` | float | 本次 ELO 变化量 |

## ELO 算法

使用标准 Logistic 期望值公式：

```
expected = 1 / (1 + 10^((R_opp - R_cur) / 400))
R_cur_new = R_cur + K × (S - expected)
```

- 初始评分：1500
- K 因子通过 `--elo_k` 控制（默认 32）
- ELO 状态持久化到 `evaluation_state.json`，训练可中断后恢复

## 原理简述

### 棋盘编码

每个局面编码为 **14 通道 × 10 行 × 9 列** 的特征平面：前 7 通道表示红方各类棋子的位置（车/马/象/仕/帅/炮/兵），后 7 通道表示黑方对应棋子，以当前走子方视角呈现。

### 网络结构

输入特征经过 **初始卷积层 → 4 个残差块 → 两个输出头**：
- **策略头**：输出所有合法走法的概率分布（logits → softmax）
- **价值头**：输出当前局面的胜负评估，范围 [-1, 1]

### MCTS 搜索

每步决策运行若干次模拟：从根节点出发，使用 **UCB 公式**（结合网络先验概率和访问次数）选择子节点，到达叶节点后用神经网络估值，再反向传播更新统计。自对弈时在根节点注入 **Dirichlet 噪声**增强探索多样性。

### 自对弈训练

训练循环：
1. **自对弈**：当前模型与自身博弈，记录每步的局面、MCTS 搜索概率和最终胜负；
2. **训练**：从数据缓冲区采样，用 **策略损失（交叉熵）+ 价值损失（MSE）** 更新网络；
3. **评测**（可选）：定期与基准模型对战，记录 ELO 和胜率；
4. 重复以上步骤，模型能力持续提升。

训练日志（每局自对弈记录、每轮 loss）保存在 `runs/run_<时间戳>/` 目录下。

## 运行测试

```bash
cd ..  # 进入 simple_chess_ai 包的父目录
python -m unittest simple_chess_ai.tests -v
```
