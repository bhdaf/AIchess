# 简化版中国象棋 AI（AlphaZero 风格）

轻量、易理解、可在普通电脑上跑通的中国象棋智能体。

## 特点

- **完整中国象棋规则**：蹩马腿、塞象眼、飞将、长将/长捉、三次重复局面判和等
- **轻量级 CNN 策略价值网络**：128 通道、4 个残差块，参数量约为原版 AlphaZero 的 1/4
- **基于 MCTS 的自对弈与决策**：蒙特卡洛树搜索，带 Dirichlet 噪声增强探索
- **单机自博弈训练脚本**：自对弈生成数据 → 训练神经网络，循环迭代
- **内置 ELO 评分系统**：训练过程中自动追踪模型强度变化
- **图形界面（Pygame）与命令行对弈模式**

## 特性亮点（新增）

- **对手池 + 课程学习**：训练时可引入 Pikafish 等 UCI 引擎作为对手，按三阶段课程学习自动调配自对弈/弱/中/全强度引擎/历史版本对手
- **引擎进程复用**：PikafishAgent 保持子进程常驻，避免每局重启开销
- **我方样本收集**：引擎回合不产生训练样本，只收集我方 MCTS 回合的数据
- **评测门控**：`--eval_gate` 参数，仅当评测得分 ≥ 阈值时才更新基准模型

## 目录结构

```
AIchess/
├── __init__.py          # 包初始化
├── __main__.py          # 主入口（train / play / play_cli / eval / plot / vs_pikafish）
├── game.py              # 完整中国象棋规则与状态管理
├── model.py             # CNN 策略价值网络（PolicyValueNet + ChessModel）
├── mcts.py              # 蒙特卡洛树搜索（MCTS）
├── train.py             # 自对弈 + AlphaZero 风格训练管线（含 ELO 评测、对手池）
├── agents.py            # MCTSAgent 封装（BaseAgent 兼容接口）
├── opponent_pool.py     # OpponentPool 对手池（课程学习调度）
├── pikafish_agent.py    # BaseAgent 接口 + PikafishAgent UCI 引擎封装
├── uci.py               # UCIEngine 通用 UCI 协议封装
├── vs_pikafish.py       # 与 Pikafish 对弈脚本（输出 PGN / JSONL）
├── eval.py              # 独立模型评测模块
├── plot.py              # 训练曲线绘图模块（需要 matplotlib）
├── export.py            # 日志记录（JSONL / CSV / ELO 状态）
├── gui.py               # Pygame 图形界面
├── cli.py               # 命令行文字界面
├── tests.py             # 单元测试
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

### 3. 引入 Pikafish 对手池训练

提供 `--engine_path` 后，训练将自动启用对手池模式，按课程学习调度对抗不同强度对手：

```bash
# 最小示例（使用默认课程学习 schedule）
python -m AIchess train \
    --engine_path /path/to/pikafish \
    --num_games 80 \
    --num_simulations 60 \
    --num_epochs 2 \
    --eval_interval 0 \
    --my_side alternate \
    --curriculum default

# 完整参数示例（Kaggle P100 建议预设，约 1-2 小时）
python -m AIchess train \
    --engine_path /path/to/pikafish \
    --num_games 200 \
    --num_simulations 60 \
    --num_epochs 2 \
    --batch_size 256 \
    --max_moves 200 \
    --my_side alternate \
    --curriculum default \
    --pikafish_movetime_weak 30 \
    --pikafish_movetime_mid 60 \
    --pikafish_movetime_full 100 \
    --eval_interval 20 \
    --eval_games 20 \
    --eval_gate 0.55 \
    --save_interval 10
```

**课程学习阶段（`--curriculum default`）：**

| 训练进度 | 自对弈 | 弱引擎 | 中引擎 | 强引擎 | 历史版本 |
|---------|--------|--------|--------|--------|---------|
| 前 1/3  | 100%   | —      | —      | —      | —       |
| 中 1/3  | 50%    | 30%    | —      | —      | 20%     |
| 后 1/3  | 30%    | 20%    | 20%    | 10%    | 20%     |

**数据收集策略：**
- 只收集我方回合（`--my_side`）的训练样本
- 引擎对手回合不产生训练数据
- 终局胜负按我方视角回填 value_target

**评测门控（`--eval_gate`）：**
- 仅当评测 score ≥ gate（默认 0.55）时才更新基准模型
- 配合 `--eval_opponent previous` 使用，防止策略退化

**新增 CLI 参数一览：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--engine_path` | None | Pikafish 等 UCI 引擎路径；提供后启用对手池 |
| `--pikafish_movetime_weak` | 30 | 弱强度思考时间（ms） |
| `--pikafish_movetime_mid` | 60 | 中强度思考时间（ms） |
| `--pikafish_movetime_full` | 100 | 全强度思考时间（ms） |
| `--curriculum` | default | 课程策略：`default` 或 `none`（纯自对弈） |
| `--my_side` | alternate | 我方执哪方：`red`/`black`/`alternate` |
| `--eval_gate` | 0.55 | 评测门控阈值 |

### 4. 对弈

```bash
# 图形界面（需要 Pygame）
python -m AIchess play --model_path saved_model/model.pth

# 命令行界面
python -m AIchess play_cli --model_path saved_model/model.pth

# 执黑方
python -m AIchess play --human_color black
```

### 5. 独立模型评测

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

### 6. 绘制训练曲线

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

---

## Pikafish 对战 + 训练扩展

本节说明如何将 AIchess 与外部 UCI 引擎（如 [Pikafish](https://github.com/official-pikafish/Pikafish)）
对接，并基于对战结果持续进化模型。

### 新增文件概览

| 文件 | 作用 |
|------|------|
| `uci.py` | 通用 UCI 引擎子进程封装（启动/握手/走子/退出） |
| `pikafish_agent.py` | `BaseAgent` 接口 + `PikafishAgent` 实现（含格式转换） |
| `vs_pikafish.py` | 对战脚本：AI 模型 vs UCI 引擎，输出 PGN + JSONL 日志 |

### 快速开始：与 Pikafish 对弈

```bash
# 下载并编译 Pikafish（或从 release 下载预编译版本）
# https://github.com/official-pikafish/Pikafish/releases

# AI 执红，与 Pikafish（100 ms 每步）对弈 10 局
python -m AIchess vs_pikafish \
    --engine_path /path/to/pikafish \
    --model_path saved_model/model.pth \
    --n_games 10 \
    --movetime 100 \
    --ai_side both \
    --num_simulations 50 \
    --out runs/vs_pikafish_001

# 调低引擎强度（通过 UCI 选项）
python -m AIchess vs_pikafish \
    --engine_path /path/to/pikafish \
    --model_path saved_model/model.pth \
    --n_games 20 \
    --movetime 50 \
    --engine_options "UCI_LimitStrength=true" "UCI_Elo=1500" \
    --out runs/vs_pikafish_elo1500
```

输出文件：

| 文件 | 说明 |
|------|------|
| `<out>/vs_pikafish.pgn` | 所有对局 PGN（可用 Arena/CuteChess 等工具打开） |
| `<out>/vs_pikafish_log.jsonl` | 每局 JSONL 记录（含步数、走法、胜负原因） |

### 坐标格式转换（已实现）

| 场景 | 内部格式 | UCI ICCS 格式 |
|------|----------|---------------|
| 帅从 e0 走到 e1 | `"4041"` | `"e0e1"` |
| 红车从 a0 走到 a2 | `"0002"` | `"a0a2"` |
| 黑炮从 b7 走到 e7 | `"1747"` | `"b7e7"` |
| 红马从 h0 走到 g2 | `"7062"` | `"h0g2"` |

映射规则（`pikafish_agent.py`）：
- 列 x (0–8) → UCI 文件 a–i（`chr(ord('a') + x)`）
- 行 y (0–9) → UCI 阶 0–9（直接对应，红方底线=0，黑方底线=9）

FEN 转换：`game.get_fen()` 返回棋盘部分；`game_to_uci_fen(game)` 追加走子方
(`w`/`b`) 及其他 UCCI 字段，生成完整 UCI FEN。

> **⚠ 一致性风险**：若 Pikafish 的规则实现（长将/长捉/三次重复等）与本仓库
> `game.py` 存在差异，引擎走法可能在本地验证时被判为非法。`PikafishAgent.get_move()`
> 会记录警告并返回 `None`；调用方应提供备用走法（如随机合法走法）。
> 验证方法：运行 100 局快速对局，检查 `vs_pikafish_log.jsonl` 中
> `terminate_reason` 是否出现异常终止。

### 统一 Agent 接口

`BaseAgent`（`pikafish_agent.py`）定义了所有棋手的公共接口：

```python
class BaseAgent(ABC):
    def get_move(self, game: ChessGame) -> str | None: ...
    def new_game(self) -> None: ...          # 可选
    def update_move(self, move: str) -> None: ...  # 可选
```

借助此接口，可以在同一个 `play_game(red_agent, black_agent)` 循环中
混合 `MCTS`、`PikafishAgent`、`RandomAgent` 等不同类型的棋手，
无需修改训练主流程。

### 对手池 + 课程学习方案（训练扩展设计）

#### 对手池结构

```text
OpponentPool
├── self_play       权重 α（当前模型自对弈）
├── pikafish_weak   权重 β（限制强度的 Pikafish，如 Elo≈1500）
├── pikafish_mid    权重 γ（中等强度，如 Elo≈2000）
├── pikafish_full   权重 δ（全强度，movetime=100ms）
└── historical      权重 ε（随机选历史检查点，防止策略坍塌）
```

#### 课程学习阶段

| 阶段 | 训练局数 | self_play | pikafish_weak | pikafish_mid | pikafish_full |
|------|----------|-----------|---------------|--------------|---------------|
| 初期 | 0–500    | 80% | 20% | 0%  | 0%  |
| 中期 | 500–2000 | 50% | 20% | 30% | 0%  |
| 后期 | 2000+    | 20% | 10% | 30% | 40% |

#### 数据收集策略

仅收集 **我方回合** 的训练样本，引擎回合不产生训练数据：

```python
# 对局循环（伪代码）
for step in game:
    if is_my_turn:
        actions, probs = mcts.get_action_probs(game, add_noise=True)
        move = sample(actions, probs)
        buffer.append((game.to_planes(), probs, None))  # value 待回填
    else:
        move = engine_agent.get_move(game)
    game.step(move)

# 回填 value（终局胜负）
outcome = +1 if winner == my_side else -1 if winner != "draw" else 0
for sample in buffer:
    sample.value = outcome
    outcome = -outcome  # 翻转视角
```

#### 训练循环伪代码

```python
# generate_games → buffer → train_step → checkpoint → eval_gate
for iteration in range(max_iterations):
    # 1. 生成对局数据（可多进程并行）
    opponent = opponent_pool.sample(curriculum_stage(iteration))
    games_data = parallel_generate_games(current_model, opponent, n=N)
    replay_buffer.extend(games_data)

    # 2. 训练
    for epoch in range(num_epochs):
        batch = replay_buffer.sample(batch_size)
        loss = policy_loss(batch) + value_loss(batch)
        optimizer.step(loss)

    # 3. 检查点
    if iteration % save_interval == 0:
        save_checkpoint(current_model, iteration)
        opponent_pool.add_historical(current_model.copy())

    # 4. 评测门控（与上一检查点对战，胜率 > 55% 才更新基准）
    if iteration % eval_interval == 0:
        score = evaluate(current_model, baseline_model)
        if score > 0.55:
            baseline_model = current_model.copy()
            print(f"[迭代 {iteration}] 模型升级，得分 {score:.2%}")
```

#### 性能优化建议

| 优化点 | 建议 |
|--------|------|
| 并行自对弈 | 使用 `multiprocessing.Pool` 并行生成对局数据 |
| 引擎思考时间 | 训练期从 `movetime=10ms` 起步，强化期再逐步增加 |
| MCTS 模拟次数 | 训练期用 50–100，推理期用 200–400 |
| 位置缓存 | 开启 `MCTS(cache_size=2000)` 减少重复评估 |
| 引擎进程复用 | `PikafishAgent` 保持进程常驻，避免每局重启 |

### 风险清单与缓解策略

| 风险 | 严重性 | 验证方法 | 缓解策略 |
|------|--------|----------|----------|
| 规则不一致（长将/长捉/重复等） | 高 | 跑 100 局快速对局，检查异常终止率 | 在 `PikafishAgent.get_move()` 中添加合法性验证；规则差异较大时考虑关闭引擎的特殊规则 |
| FEN 格式不兼容 | 中 | 比对 `game_to_uci_fen()` 输出与引擎期望格式 | 查阅 Pikafish 文档；必要时调整 FEN 字段顺序 |
| 一直输导致梯度贫瘠 | 高 | 监控 value loss 是否趋近于常数 | 严格执行课程学习；初期以弱引擎为主要对手 |
| 速度瓶颈 | 中 | 测量每局耗时 | 降低 movetime + num_simulations；多进程并行 |
| 策略坍塌（遗忘） | 中 | 定期与历史模型对战 | 历史模型池 + 混合 self-play 比例 |
| 数据分布偏移 | 低 | 监控训练数据来源比例 | 限制 replay_buffer 中引擎对局数据的比例上限 |

### 验证步骤（里程碑）

```
M1：走法一致性 ─ 与 Pikafish 完整下完 100 局，无异常终止
M2：对战脚本   ─ `vs_pikafish` 命令生成可读 PGN，步数分布合理
M3：课程学习   ─ AI 对弱引擎（Elo≈1500）胜率从随机水平提升到 >30%
M4：强度提升   ─ AI 的 ELO 较训练前提升 200+ 点
M5：全强度对战 ─ 对全强度 Pikafish 胜率稳定在 >5%（非随机）
```
