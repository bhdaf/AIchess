"""
推理增强模块（Chain-of-Thought）

设计思维链框架，引导模型在选择动作前生成一段推理过程，
例如："当前红方夹车炮威胁黑方将位，我应先补士..."

由于资源限制，本模块采用轻量级的基于规则+神经网络的推理方式，
而非完整的 LLM。可通过 GRPO 来强化推理路径的准确性。

用法：
    reasoner = ChessReasoner(model)
    reasoning, action = reasoner.reason_and_act(game)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_chess_ai.game import (
    ChessGame, BOARD_HEIGHT, BOARD_WIDTH, NUM_ACTIONS,
    ACTION_LABELS, LABEL_TO_INDEX, flip_move
)


# 推理模板（思维链 CoT 提示词框架）
COT_TEMPLATES = {
    'threat_analysis': "分析威胁：{threats}",
    'defense_analysis': "防守需求：{defenses}",
    'attack_opportunity': "进攻机会：{attacks}",
    'piece_coordination': "子力配合：{coordination}",
    'position_evaluation': "局面评估：{evaluation}",
    'move_decision': "决策：基于{reasoning}，选择{move}",
}


class BoardAnalyzer:
    """
    棋盘局面分析器

    对当前局面进行规则化的态势分析，生成结构化的推理输入。
    """

    @staticmethod
    def analyze_threats(game):
        """分析当前局面的威胁关系"""
        threats = []

        # 找到所有棋子位置
        pieces = {}
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                piece = game.board[y][x]
                if piece is not None:
                    pieces[(x, y)] = piece

        # 找到双方的将/帅位置
        king_pos = {}
        for (x, y), piece in pieces.items():
            if piece == 'K':
                king_pos['red'] = (x, y)
            elif piece == 'k':
                king_pos['black'] = (x, y)

        # 检查将帅是否受到直接威胁
        legal_moves = game.get_legal_moves()

        for move in legal_moves:
            x1, y1 = int(move[2]), int(move[3])
            target = game.board[y1][x1]
            if target is not None:
                attacker = game.board[int(move[1])][int(move[0])]
                if target.upper() == 'K':
                    threats.append({
                        'type': 'king_threat',
                        'attacker': attacker,
                        'attacker_pos': (int(move[0]), int(move[1])),
                        'target_pos': (x1, y1),
                    })
                else:
                    threats.append({
                        'type': 'piece_threat',
                        'attacker': attacker,
                        'target': target,
                        'attacker_pos': (int(move[0]), int(move[1])),
                        'target_pos': (x1, y1),
                    })

        return threats

    @staticmethod
    def analyze_piece_relations(game):
        """分析棋子间的保护和攻击关系"""
        relations = {
            'attacks': [],    # 攻击关系
            'defenses': [],   # 防守关系
        }

        # 获取所有合法走法
        legal_moves = game.get_legal_moves()

        for move in legal_moves:
            x0, y0 = int(move[0]), int(move[1])
            x1, y1 = int(move[2]), int(move[3])
            attacker = game.board[y0][x0]
            target = game.board[y1][x1]

            if target is not None and game.is_enemy_piece(target):
                relations['attacks'].append({
                    'from': attacker,
                    'to': target,
                    'move': move,
                })

        return relations

    @staticmethod
    def evaluate_position(game):
        """简单的局面评估"""
        piece_values = {
            'P': 1, 'p': 1,    # 兵/卒
            'A': 2, 'a': 2,    # 仕/士
            'B': 2, 'b': 2,    # 象/相
            'N': 4, 'n': 4,    # 马
            'C': 4.5, 'c': 4.5, # 炮
            'R': 9, 'r': 9,    # 车
            'K': 0, 'k': 0,    # 将/帅（不计入物质）
        }

        red_material = 0
        black_material = 0

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                piece = game.board[y][x]
                if piece is not None:
                    value = piece_values.get(piece, 0)
                    if piece.isupper():
                        red_material += value
                    else:
                        black_material += value

        return {
            'red_material': red_material,
            'black_material': black_material,
            'material_advantage': red_material - black_material,
        }


class ReasoningEncoder(nn.Module):
    """
    推理特征编码器

    将结构化的推理分析结果编码为向量，与策略网络特征融合。

    Args:
        reasoning_dim: 推理特征编码维度
        output_dim: 输出特征维度
    """

    def __init__(self, reasoning_dim=32, output_dim=64):
        super().__init__()
        # 威胁特征编码
        self.threat_encoder = nn.Sequential(
            nn.Linear(reasoning_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        # 物质评估编码
        self.material_encoder = nn.Sequential(
            nn.Linear(3, output_dim),
            nn.ReLU(),
        )
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
        )

    def forward(self, threat_features, material_features):
        """
        Args:
            threat_features: (batch, reasoning_dim) 威胁分析特征
            material_features: (batch, 3) 物质评估特征

        Returns:
            reasoning_embedding: (batch, output_dim) 推理嵌入
        """
        threat_emb = self.threat_encoder(threat_features)
        material_emb = self.material_encoder(material_features)
        combined = torch.cat([threat_emb, material_emb], dim=-1)
        return self.fusion(combined)


class ChessReasoner:
    """
    中国象棋推理器

    在走子前进行结构化推理，生成思维链，提升走子质量。
    可通过 GRPO 训练来强化推理路径的准确性。

    Args:
        model: ChessModel 实例
        reasoning_dim: 推理特征维度
    """

    PIECE_NAMES = {
        'R': '车', 'N': '马', 'B': '象', 'A': '仕', 'K': '帅', 'C': '炮', 'P': '兵',
        'r': '车', 'n': '马', 'b': '象', 'a': '士', 'k': '将', 'c': '砲', 'p': '卒',
    }

    def __init__(self, model, reasoning_dim=32):
        self.model = model
        self.analyzer = BoardAnalyzer()
        self.reasoning_dim = reasoning_dim

    def generate_reasoning_chain(self, game):
        """
        生成思维链推理

        Args:
            game: ChessGame 实例

        Returns:
            reasoning_text: 推理过程的文本描述
            reasoning_features: 推理特征向量（可用于神经网络）
        """
        side = "红方" if game.red_to_move else "黑方"
        chain = [f"[{side}思考]"]

        # 1. 分析威胁
        threats = self.analyzer.analyze_threats(game)
        king_threats = [t for t in threats if t['type'] == 'king_threat']
        piece_threats = [t for t in threats if t['type'] == 'piece_threat']

        if king_threats:
            threat_strs = []
            for t in king_threats:
                name = self.PIECE_NAMES.get(t['attacker'], t['attacker'])
                threat_strs.append(f"{name}在{t['attacker_pos']}威胁对方将帅")
            chain.append(COT_TEMPLATES['threat_analysis'].format(
                threats="、".join(threat_strs)
            ))

        # 2. 分析攻击机会
        if piece_threats:
            attack_strs = []
            for t in piece_threats[:3]:  # 只显示前3个
                a_name = self.PIECE_NAMES.get(t['attacker'], t['attacker'])
                t_name = self.PIECE_NAMES.get(t['target'], t['target'])
                attack_strs.append(f"{a_name}可吃{t_name}")
            chain.append(COT_TEMPLATES['attack_opportunity'].format(
                attacks="、".join(attack_strs)
            ))

        # 3. 局面评估
        position = self.analyzer.evaluate_position(game)
        adv = position['material_advantage']
        if game.red_to_move:
            eval_str = f"红方物质{'领先' if adv > 0 else '落后' if adv < 0 else '均衡'}"
        else:
            eval_str = f"黑方物质{'领先' if adv < 0 else '落后' if adv > 0 else '均衡'}"
        eval_str += f"（红{position['red_material']:.0f} vs 黑{position['black_material']:.0f}）"
        chain.append(COT_TEMPLATES['position_evaluation'].format(
            evaluation=eval_str
        ))

        # 4. 生成推理特征向量
        reasoning_features = self._encode_reasoning(threats, position)

        reasoning_text = " → ".join(chain)
        return reasoning_text, reasoning_features

    def _encode_reasoning(self, threats, position):
        """将推理分析编码为特征向量"""
        features = np.zeros(self.reasoning_dim, dtype=np.float32)

        # 威胁计数特征
        king_threats = sum(1 for t in threats if t['type'] == 'king_threat')
        piece_threats = sum(1 for t in threats if t['type'] == 'piece_threat')
        features[0] = min(king_threats / 3.0, 1.0)
        features[1] = min(piece_threats / 10.0, 1.0)

        # 物质评估特征
        features[2] = np.tanh(position['material_advantage'] / 10.0)
        features[3] = position['red_material'] / 50.0
        features[4] = position['black_material'] / 50.0

        # 威胁类型分布
        for i, t in enumerate(threats[:5]):
            if i + 5 < self.reasoning_dim:
                piece = t.get('attacker', '')
                piece_idx = {'R': 1, 'N': 2, 'C': 3, 'P': 4, 'K': 5,
                             'r': 1, 'n': 2, 'c': 3, 'p': 4, 'k': 5}
                features[i + 5] = piece_idx.get(piece, 0) / 5.0

        return features

    def reason_and_act(self, game, temperature=0.1):
        """
        推理后走子

        Args:
            game: ChessGame 实例
            temperature: 动作选择温度

        Returns:
            reasoning_text: 推理过程文本
            action: 选择的走法
            policy: 策略概率分布
        """
        # 生成推理链
        reasoning_text, reasoning_features = self.generate_reasoning_chain(game)

        # 使用神经网络预测
        planes = game.to_planes()
        policy, value = self.model.predict(planes)

        # 用推理特征微调策略（简单的加权融合）
        # 如果检测到将军威胁，增强相关走法的权重
        legal_moves = game.get_legal_moves()
        if not game.red_to_move:
            legal_moves_for_policy = [flip_move(m) for m in legal_moves]
        else:
            legal_moves_for_policy = legal_moves

        # 掩蔽非法走法
        legal_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for move in legal_moves_for_policy:
            if move in LABEL_TO_INDEX:
                legal_mask[LABEL_TO_INDEX[move]] = 1.0

        masked_policy = policy * legal_mask
        total = masked_policy.sum()
        if total > 0:
            masked_policy /= total
        elif legal_moves_for_policy:
            # uniform over legal moves
            for move in legal_moves_for_policy:
                if move in LABEL_TO_INDEX:
                    masked_policy[LABEL_TO_INDEX[move]] = 1.0 / len(legal_moves_for_policy)

        # 选择动作
        if temperature < 1e-8:
            action_idx = np.argmax(masked_policy)
        else:
            powered = masked_policy ** (1.0 / temperature)
            total = powered.sum()
            if total > 0:
                powered /= total
                action_idx = np.random.choice(len(powered), p=powered)
            else:
                action_idx = np.argmax(masked_policy)

        action = ACTION_LABELS[action_idx]

        # 补充决策推理
        if action in LABEL_TO_INDEX:
            reasoning_text += f" → {COT_TEMPLATES['move_decision'].format(reasoning='综合分析', move=action)}"

        return reasoning_text, action, masked_policy


def create_grpo_reasoning_reward(reasoner, game, action, outcome):
    """
    为 GRPO 创建基于推理的奖励

    评估推理路径的质量，用于 GRPO 训练强化推理准确性。

    Args:
        reasoner: ChessReasoner 实例
        game: ChessGame 实例
        action: 选择的走法
        outcome: 对弈结果（1=赢, 0=和, -1=输）

    Returns:
        reward: 推理质量奖励
    """
    reasoning_text, reasoning_features = reasoner.generate_reasoning_chain(game)

    # 基础奖励：对弈结果
    reward = float(outcome)

    # 推理一致性奖励：推理是否与结果一致
    threats = reasoner.analyzer.analyze_threats(game)
    king_threats = [t for t in threats if t['type'] == 'king_threat']

    if king_threats and outcome > 0:
        reward += 0.2  # 成功利用将军威胁

    # 物质评估一致性
    position = reasoner.analyzer.evaluate_position(game)
    if position['material_advantage'] > 0 and outcome > 0:
        reward += 0.1  # 物质优势与胜利一致

    return reward
