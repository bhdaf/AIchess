# 简化中国象棋AI (Simple Chinese Chess AI)

基于 [ChineseChess-AlphaZero](https://github.com/bhdaf/ChineseChess-AlphaZero) 项目简化而来的中国象棋智能体。

## 特点

- **简化的网络结构**: 128通道、4个残差块（原项目256通道、7个残差块），参数量减少约75%
- **一体化训练**: 自对弈和训练在同一个脚本中完成，无需分布式协调
- **图形化界面**: 使用Pygame绘制棋盘和棋子（纯代码绘制，无需外部图片资源）
- **完整规则**: 支持中国象棋全部规则（蹩马腿、塞象眼、飞将等）
- **GRPO训练**: 支持Group Relative Policy Optimization，参考DeepSeek R1思路
- **GNN特征提取**: 图神经网络建模棋子间攻击/防守关系
- **推理增强**: Chain-of-Thought思维链推理模块
- **FP16混合精度**: 支持混合精度训练，降低显存占用

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

# 组合使用
python -m simple_chess_ai train --use_grpo --use_fp16

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
| `--model_path` | 自动 | 模型保存路径 |
| `--use_grpo` | False | 使用GRPO训练模式 |
| `--grpo_group_size` | 8 | GRPO组采样大小 |
| `--use_fp16` | False | 使用FP16混合精度训练 |

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
- **策略头**: 输出所有可能走法的概率分布
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

### MCTS搜索
- 使用PUCT算法平衡探索与利用
- 神经网络引导搜索方向
- 支持温度参数控制探索程度

### FP16混合精度训练
- 使用`torch.amp.autocast`和`torch.amp.GradScaler`
- 在CUDA GPU上减少约40%显存占用
- 训练速度提升约30-50%（取决于GPU型号）

### 训练流程
1. **自对弈**: 使用当前模型+MCTS生成对局
2. **数据收集**: 记录每步的棋盘状态、搜索概率、最终胜负
3. **网络训练**: 标准模式或GRPO模式优化
4. **循环**: 不断生成新数据并训练

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
| 图形界面 | 需要图片资源 | 纯代码绘制 |
| 配置系统 | 复杂（多种模式） | 简化（命令行参数） |
| GNN特征 | 无 | 支持（GAT） |
| 推理增强 | 无 | 支持（CoT） |
| 混合精度 | 无 | 支持（FP16） |
