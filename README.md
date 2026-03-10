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
AIchess/
├── __init__.py        # 包初始化
├── __main__.py        # 主入口（train / play / play_cli / eval / plot / vs_pikafish）
├── game.py            # 完整中国象棋规则与状态管理
├── model.py           # CNN 策略价值网络（PolicyValueNet + ChessModel）
├── mcts.py            # 蒙特卡洛树搜索（MCTS）
├── train.py           # 自对弈 + AlphaZero 风格训练管线（含 ELO 评测）
├── vs_pikafish.py     # Pikafish 引擎集成：UCI 包装器 + 对战 + 训练
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
python -m AIchess train

# 自定义参数
python -m AIchess train --num_games 100 --num_simulations 200

# 快速验证流程（1 局 + 1 次训练）
python -m AIchess train --quick
```

### 2. 带评测的训练

开启 `--eval_interval` 后，训练循环将每隔指定局数自动评测当前模型，
并将 ELO 分数和胜率记录到运行目录的 `evaluation_metrics.csv`。

```bash
# 每 10 局对战上一基准模型评测一次
python -m AIchess train \
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
python -m AIchess play --model_path saved_model/model.pth

# 命令行界面
python -m AIchess play_cli --model_path saved_model/model.pth

# 执黑方
python -m AIchess play --human_color black
```

### 4. 独立模型评测

```bash
python -m AIchess eval \
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
python -m AIchess plot \
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

### 6. 对战 Pikafish 引擎

[Pikafish](https://github.com/official-pikafish/Pikafish) 是基于 Stockfish 的强力中国象棋 UCI 引擎。
通过 `vs_pikafish` 命令可以让 AIchess 与 Pikafish 对战，并可选地用对战数据训练模型。

**前置准备**：从 [Pikafish Releases](https://github.com/official-pikafish/Pikafish/releases) 下载对应平台的可执行文件，并确保 `.nnue` 权重文件与可执行文件在同一目录。

```bash
# 仅对战一局（不训练）
python -m AIchess vs_pikafish --engine ./pikafish

# 对战 20 局并训练（AI 执红方）
python -m AIchess vs_pikafish \
    --engine ./pikafish \
    --model_path saved_model/model.pth \
    --num_games 20 \
    --movetime 500 \
    --num_simulations 100 \
    --train

# AI 执黑方
python -m AIchess vs_pikafish --engine ./pikafish --ai_color black
```

#### vs_pikafish 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--engine` | 必填 | Pikafish 可执行文件路径 |
| `--model_path` | saved_model/model.pth | AIchess 模型文件路径 |
| `--ai_color` | red | 学习方颜色（red / black） |
| `--num_games` | 1 | 对战局数 |
| `--num_simulations` | 100 | AI 每步 MCTS 模拟次数 |
| `--movetime` | 1000 | Pikafish 每步思考时间（毫秒） |
| `--max_moves` | 200 | 每局最大步数（超过判和） |
| `--train` | 否 | 启用后对战结束用收集数据训练模型 |
| `--batch_size` | 256 | 训练批大小（仅 `--train` 时有效） |
| `--num_epochs` | 5 | 训练 epoch 数（仅 `--train` 时有效） |
| `--lr` | 0.001 | 学习率（仅 `--train` 时有效） |
| `--threads` | 1 | Pikafish 搜索线程数 |
| `--hash_mb` | 64 | Pikafish 置换表大小（MB） |

## Pikafish 集成设计说明

### 协议：标准 UCI

Pikafish 使用标准 **UCI（Universal Chess Interface）** 协议，与 Stockfish 命令集完全兼容。
通信通过子进程标准输入/输出进行，主要命令序列：

```
→ uci
← id name Pikafish ...
← uciok
→ isready
← readyok
→ ucinewgame
→ isready
← readyok
→ position fen <FEN>
→ go movetime <ms>
← info depth ...
← bestmove <move>
→ quit
```

更多命令详见 [Pikafish UCI 文档](https://github.com/official-pikafish/Pikafish/wiki/UCI-&-Commands)。

### 走法格式转换

| 格式 | 示例 | 说明 |
|------|------|------|
| AIchess 内部格式 | `1242` | `x0y0x1y1`，列用数字 0–8，行用数字 0–9 |
| Pikafish UCI 格式 | `b2e2` | `<file><rank><file><rank>`，列用字母 a–i，行用数字 0–9 |

转换规则（`vs_pikafish.py` 中的 `aicoord_to_uci` / `uci_to_aicoord`）：
- AIchess 列 `x` → Pikafish file `chr(ord('a') + x)`
- 行号（rank）两种格式相同，无需转换

### FEN 格式

AIchess 的 `get_fen()` 返回棋盘部分，Pikafish 需要完整 FEN：

```
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
```

完整 FEN = `<棋盘FEN> <走子方> - - 0 1`，其中走子方为 `w`（红方）或 `b`（黑方）。

### 时间控制：movetime（推荐）

默认使用 `go movetime <ms>` 而非 `go depth <n>`，原因：

- **可预测性**：movetime 直接限制挂钟时间，总对局时长稳定，适合批量训练
- **硬件无关性**：depth 搜索实际耗时随局面复杂度和硬件差异很大
- **训练建议**：从 `--movetime 100`（0.1 秒）起步，逐步增大以提升对手强度
- 有 GPU 时建议配合增大 `--num_simulations`（AI 方 MCTS）来提升 AI 决策质量

### 训练数据收集策略

训练数据**只记录学习方（AIchess）的回合**，Pikafish 的走法不进入训练集。

**数据格式**（与自对弈完全一致）：
- `state`：14 通道特征平面，当前走子方视角，形状 `(14, 10, 9)`
- `policy_target`：MCTS 访问计数归一化分布，形状 `(NUM_ACTIONS,)`
- `value_target`：终局胜负回填，AI 方胜 = +1，负 = -1，和 = 0

**为什么只记录 AI 方回合**：
- 学习目标是提升 AIchess 自身的策略与价值估计，只有己方回合的数据对此有直接贡献
- 记录 Pikafish 的走法会把远超 AIchess 当前水平的分布混入 policy target，造成梯度信号嘈杂
- 这与 AlphaZero 自对弈格式保持一致，可与 `train` 命令产生的数据无缝合并

### 课程学习建议（进阶）

直接对战全力 Pikafish 可能导致长期失败、价值估计始终为 -1。建议逐步提升对手强度：

1. 初期：`--movetime 50`（约 ELO 2000–2500），确保 AI 能偶尔赢棋
2. 中期：`--movetime 500`，混合 50% 自对弈（`train` 命令）
3. 后期：`--movetime 2000+`，主要依赖 vs_pikafish 数据

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
cd ..  # 进入 AIchess 包的父目录
python -m unittest AIchess.tests -v
```
