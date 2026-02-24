# 简化中国象棋AI (Simple Chinese Chess AI)

基于 [ChineseChess-AlphaZero](https://github.com/bhdaf/ChineseChess-AlphaZero) 项目简化而来的中国象棋智能体。

## 特点

- **简化的网络结构**: 128通道、4个残差块（原项目256通道、7个残差块），参数量减少约75%
- **一体化训练**: 自对弈和训练在同一个脚本中完成，无需分布式协调
- **图形化界面**: 使用Pygame绘制棋盘和棋子（纯代码绘制，无需外部图片资源）
- **完整规则**: 支持中国象棋全部规则（蹩马腿、塞象眼、飞将等），含走后自家被将军过滤
- **GRPO训练**: 支持Group Relative Policy Optimization，参考DeepSeek R1思路
- **GNN特征提取**: 图神经网络建模棋子间攻击/防守关系
- **推理增强**: Chain-of-Thought思维链推理模块
- **FP16混合精度**: 支持混合精度训练，降低显存占用
- **MCTS Dirichlet噪声**: 自对弈训练时向 root 节点注入噪声，增强探索
- **模型评测门控（Gating）**: 定期对比新旧模型，只有达到胜率阈值才接受更新

## 项目结构

```
simple_chess_ai/
├── __init__.py        # 包初始化
├── __main__.py        # 主入口
├── game.py            # 游戏逻辑（棋盘、规则、走法生成）
├── model.py           # 策略价值网络（PyTorch）
├── mcts.py            # 蒙特卡洛树搜索
├── train.py           # 训练管线（自对弈+训练，支持GRPO和FP16）
├── grpo.py            # GRPO训练器（Group Relative Policy Optimization）
├── gnn_feature.py     # GNN特征提取（图卷积/图注意力网络）
├── reasoning.py       # 推理增强模块（Chain-of-Thought）
├── reasoning_cli.py   # 推理模式命令行界面
├── gui.py             # Pygame图形界面
├── cli.py             # 命令行界面
├── tests.py           # 单元测试
└── README.md          # 说明文档
```

## 依赖

- Python 3.7+
- PyTorch >= 1.9.0
- NumPy
- Pygame（图形界面需要）

安装依赖：
```bash
pip install torch numpy pygame
```

## 使用方法

### 1. 训练模型

```bash
# 基本训练（50局自对弈）
python -m simple_chess_ai train

# 自定义参数
python -m simple_chess_ai train \
    --num_games 100 \
    --num_simulations 200 \
    --num_epochs 10 \
    --batch_size 256 \
    --lr 0.001

# 使用GRPO训练模式
python -m simple_chess_ai train --use_grpo --grpo_group_size 8

# 使用FP16混合精度训练（需要CUDA GPU）
python -m simple_chess_ai train --use_fp16

# 启用 gating（每20局评测，新模型胜率>55%才接受）
python -m simple_chess_ai train \
    --gating_interval 20 \
    --gating_games 20 \
    --gating_winrate 0.55

# 禁用 gating
python -m simple_chess_ai train --gating_interval 0

# 组合使用
python -m simple_chess_ai train --use_grpo --use_fp16

# 可复现训练（设置随机种子）
python -m simple_chess_ai train --seed 42

# 可复现训练 + cuDNN 确定性模式（速度可能降低）
python -m simple_chess_ai train --seed 42 --deterministic

# 指定模型保存路径
python -m simple_chess_ai train --model_path my_model.pth
```

训练参数说明：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_games` | 50 | 自对弈局数 |
| `--num_simulations` | 100 | 每步MCTS模拟次数 |
| `--num_epochs` | 5 | 每次训练轮数 |
| `--batch_size` | 256 | 训练批大小 |
| `--lr` | 0.001 | 学习率 |
| `--max_moves` | 200 | 每局最大步数 |
| `--buffer_size` | 10000 | 训练数据缓冲区大小 |
| `--model_path` | 自动 | 模型保存路径 |
| `--save_interval` | 10 | 每隔多少局保存模型 |
| `--use_grpo` | False | 使用GRPO训练模式 |
| `--grpo_group_size` | 8 | GRPO组采样大小 |
| `--use_fp16` | False | 使用FP16混合精度训练 |
| `--gating_interval` | 20 | 每隔多少局进行 gating 评测（0=禁用） |
| `--gating_games` | 20 | gating 评测对局数 |
| `--gating_winrate` | 0.55 | gating 接受阈值（新模型胜率需超过此值） |
| `--seed` | None | 随机种子（设置后可复现数据生成序列） |
| `--deterministic` | False | 开启 cuDNN 确定性模式（配合 `--seed` 使用，**注意可能降低训练速度**） |

### 推荐训练参数

| 场景 | 推荐参数 | 说明 |
|------|----------|------|
| **CPU 快速验证** | `--num_games 50 --num_simulations 50 --gating_interval 0` | 最快完成，用于验证流程 |
| **CPU 标准训练** | `--num_games 500 --num_simulations 100 --gating_interval 50 --gating_games 10` | 适合4核CPU，每局约60s |
| **GPU 标准训练** | `--num_games 2000 --num_simulations 200 --num_epochs 10 --gating_interval 50` | RTX 3060推荐配置 |
| **GPU 强化训练** | `--num_games 5000 --num_simulations 400 --use_fp16 --gating_interval 100 --gating_games 30` | 追求更高棋力 |
| **GRPO模式** | `--use_grpo --grpo_group_size 8 --num_games 2000 --gating_interval 50` | 更稳定的策略优化 |

### 2. 图形界面对弈

```bash
# 使用默认模型
python -m simple_chess_ai play

# 指定模型和参数
python -m simple_chess_ai play \
    --model_path my_model.pth \
    --num_simulations 400 \
    --human_color red
```

操作说明：
- **点击棋子**选择要走的棋子（绿色高亮显示合法目标）
- **点击目标位置**完成走子
- **按R键**或**点击重新开始按钮**重新开始
- 红方在下方，黑方在上方

### 3. 命令行对弈

```bash
python -m simple_chess_ai play_cli
```

走法输入格式：`x0 y0 x1 y1`（列0-8 行0-9），例如：
- `4 0 4 1` — 帅向前一步
- `1 0 2 2` — 马走日

### 4. 推理模式对弈

```bash
python -m simple_chess_ai reason
```

推理模式会在每次AI走子前输出思维链分析，展示AI的推理过程。

## 技术架构

### 神经网络
- **输入**: 14×10×9 特征平面（7种棋子×2方）
- **结构**: 初始卷积 → 4个残差块 → 策略头+价值头
- **策略头**: 输出原始 logits（不含 softmax），由调用方按需处理：
  - `ChessModel.predict()` — 对全体走法做 softmax（向后兼容）
  - `ChessModel.predict_with_mask(planes, legal_indices)` — 先将非法走法
    logit 置为极小值（-1e9）再 softmax，使概率质量仅落在合法走法上；
    MCTS 扩展节点时默认使用此方法
  - 训练损失使用 `F.log_softmax` + KL 交叉熵，数值更稳定
- **价值头**: 输出局面评估值 [-1, 1]

### GNN特征提取（gnn_feature.py）
- 将棋盘转换为图结构（90个节点，对应棋盘位置）
- 使用GAT（Graph Attention Network）建模棋子间关系
- 自动编码攻击路线（同行/同列距离衰减权重）
- GNN特征与CNN特征融合后输入策略头和价值头

### GRPO训练（grpo.py）
- 参考DeepSeek R1的Group Relative Policy Optimization
- 对同一局面采样多个动作（组采样机制）
- 计算组内相对优势作为奖励基准（无需Critic网络）
- 比传统PPO减少约30%内存开销

### 推理增强（reasoning.py）
- 基于规则的棋盘态势分析
- 生成结构化思维链（CoT）
- 威胁分析、物质评估、走法决策
- 可通过GRPO强化推理路径准确性

### MCTS搜索与 Dirichlet 噪声
- 使用PUCT算法平衡探索与利用
- 神经网络引导搜索方向
- 支持温度参数控制探索程度
- **root 重置 vs 复用**
  - `reset_root=True`（默认）：每次调用 `get_action_probs()` 前重置 root，
    保证搜索基于当前局面，无旧统计污染。适合 GUI/CLI/推理场景。
  - `reset_root=False`（树复用模式）：保留上次子树的访问统计，配合
    `update_with_move(action)` 手动推进 root，减少重复计算。适合对性能要求
    高的训练场景（`self_play_game` 默认使用树复用）。
- **局面缓存**：在 `MCTS` 构造时传入 `cache_size=N`（N>0），单次搜索内遇到
  相同局面（FEN 相同）时复用网络评估结果，减少重复推理；`cache_size=0`（默认）
  表示禁用缓存。
- **训练时**（`add_noise=True`）：在第一次模拟扩展 root 节点后，按 AlphaZero 标准方式
  向 root 的子节点先验概率注入 Dirichlet 噪声：
  `prior = (1 - w) * prior + w * noise`，其中 `w = dirichlet_weight`（默认 0.25）
- **评测/对弈时**（`add_noise=False`）：不注入噪声，保证稳定的评测结果

### 走后自家被将军过滤（game.py）
- `get_legal_moves()` 现在会过滤所有走后导致己方将/帅处于被将军状态的走法
- 新增辅助方法：`_find_king(for_red)`、`_is_attacked(x, y, for_red)`、
  `_is_in_check(for_red)`、`_move_leaves_king_in_check(move)`
- 实现采用"临时执行+还原"策略，避免深拷贝开销
- 攻击检测涵盖：车、炮、马、兵/卒、飞将（将帅对面）

### 模型评测门控（Gating，train.py）
- 新增 `evaluate_models(model_a, model_b, n_games, num_simulations)` 函数
- 每隔 `gating_interval` 局，评测当前候选模型 vs. 基准模型（禁用噪声，temperature=0）
- 交替红黑方，减少先手优势偏差
- 若新模型胜率 > `gating_winrate`（默认 55%），接受并更新基准；否则回滚
- 保存的模型始终是被接受的基准模型

### FP16混合精度训练
- 使用`torch.amp.autocast`和`torch.amp.GradScaler`
- 在CUDA GPU上减少约40%显存占用
- 训练速度提升约30-50%（取决于GPU型号）

### 训练流程
1. **自对弈**: 使用当前模型+MCTS生成对局（root 注入 Dirichlet 噪声）
2. **数据收集**: 记录每步的棋盘状态、搜索概率、最终胜负
3. **网络训练**: 标准模式或GRPO模式优化
4. **Gating评测**: 每隔 N 局对比候选模型与基准模型，决定是否接受更新
5. **循环**: 不断生成新数据并训练

## 预估训练时间和算力开销

### 标准训练模式
| 配置 | 硬件 | 每局耗时 | 100局训练 | 收敛所需局数 |
|------|------|----------|-----------|-------------|
| 默认(128ch, 100sim) | CPU (4核) | ~60s | ~100min | ~5000-10000 |
| 默认(128ch, 100sim) | GPU (RTX 3060) | ~15s | ~25min | ~5000-10000 |
| 大规模(128ch, 400sim) | GPU (RTX 3060) | ~45s | ~75min | ~3000-5000 |

### GRPO训练模式
| 配置 | 硬件 | 每局耗时 | 100局训练 | 收敛所需局数 |
|------|------|----------|-----------|-------------|
| GRPO(group=8) | CPU (4核) | ~65s | ~108min | ~3000-7000 |
| GRPO(group=8) | GPU (RTX 3060) | ~18s | ~30min | ~3000-7000 |
| GRPO+FP16 | GPU (RTX 3060) | ~12s | ~20min | ~3000-7000 |

### GNN增强模式（使用GNNPolicyValueNet）
- 额外显存占用: ~200-400MB
- 推理延迟增加: ~20-30%
- 预期棋力提升: ~50-100 Elo

### 关于LLM集成
完整的LLM集成（如接入大语言模型做推理）需要额外8-16GB显存，
在资源紧张的情况下不建议使用。当前的轻量级推理模块（reasoning.py）
是一个实用的替代方案，仅需几十MB额外内存。

## 与原项目的对比

| 特性 | 原项目 | 本项目 |
|------|--------|--------|
| 网络通道数 | 256 | 128 |
| 残差块数 | 7 | 4 |
| 训练方式 | 分布式（多进程） | 单进程（支持GRPO） |
| MCTS模拟数 | 800 | 100-200 |
| Dirichlet噪声 | 有 | 有（已修复，正确注入root） |
| 走后将军过滤 | 有 | 有（新增） |
| 模型评测门控 | 有 | 有（新增，可配置阈值） |
| 图形界面 | 需要图片资源 | 纯代码绘制 |
| 配置系统 | 复杂（多种模式） | 简化（命令行参数） |
| GNN特征 | 无 | 支持（GAT） |
| 推理增强 | 无 | 支持（CoT） |
| 混合精度 | 无 | 支持（FP16） |
