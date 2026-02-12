"""
简化蒙特卡洛树搜索 (MCTS)

实现基于神经网络引导的MCTS，用于走法选择。
"""

import math
import numpy as np

from simple_chess_ai.game import (
    ChessGame, ACTION_LABELS, LABEL_TO_INDEX, NUM_ACTIONS,
    flip_move, flip_policy
)


class MCTSNode:
    """MCTS树节点"""

    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.children = {}  # action_str -> MCTSNode

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MCTS:
    """
    蒙特卡洛树搜索

    使用神经网络评估叶子节点，通过PUCT算法选择走法。

    Args:
        model: ChessModel实例
        num_simulations: 每步搜索的模拟次数
        c_puct: 探索系数
        dirichlet_alpha: Dirichlet噪声参数（用于自对弈训练）
        dirichlet_weight: 噪声权重
    """

    def __init__(self, model, num_simulations=200, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_weight=0.25):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.root = MCTSNode()

    def get_action_probs(self, game, temperature=1.0, add_noise=False):
        """
        运行MCTS搜索并返回走法概率分布

        Args:
            game: ChessGame实例
            temperature: 温度参数，控制探索程度
            add_noise: 是否添加Dirichlet噪声（训练时使用）

        Returns:
            actions: 走法列表
            probs: 对应的概率列表
        """
        # 运行模拟
        for _ in range(self.num_simulations):
            game_copy = game.copy()
            self._simulate(game_copy, self.root)

        # 从根节点提取走法概率
        actions = list(self.root.children.keys())
        visits = [self.root.children[a].visit_count for a in actions]

        if not actions:
            return [], []

        if temperature < 1e-8:
            # 选择访问次数最多的走法
            best_idx = np.argmax(visits)
            probs = [0.0] * len(actions)
            probs[best_idx] = 1.0
        else:
            # 按温度计算概率
            visits_arr = np.array(visits, dtype=np.float64)
            visits_temp = visits_arr ** (1.0 / temperature)
            total = visits_temp.sum()
            if total > 0:
                probs = (visits_temp / total).tolist()
            else:
                probs = [1.0 / len(actions)] * len(actions)

        return actions, probs

    def update_with_move(self, action):
        """用选择的走法更新树（复用子树）"""
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = MCTSNode()

    def _simulate(self, game, node):
        """
        运行一次MCTS模拟

        1. 选择: 沿着树选择最优子节点
        2. 扩展: 到达叶子节点时用神经网络扩展
        3. 回传: 将评估值回传到路径上的节点
        """
        path = [node]

        # 选择阶段：沿树向下
        while node.children and not game.done:
            action, node = self._select_child(node)
            # 需要根据当前走棋方调整走法
            if not game.red_to_move:
                actual_action = flip_move(action)
            else:
                actual_action = action
            game.step(actual_action)
            path.append(node)

        # 评估叶子节点
        if game.done:
            if game.winner == 'draw':
                value = 0.0
            else:
                # 从当前走子方视角看：对手赢了 = -1
                value = -1.0
        else:
            # 用神经网络评估
            planes = game.to_planes()
            policy, value = self.model.predict(planes)

            # 获取合法走法
            legal_moves = game.get_legal_moves()
            if not game.red_to_move:
                legal_moves_flipped = [flip_move(m) for m in legal_moves]
            else:
                legal_moves_flipped = legal_moves

            # 扩展节点
            total_prior = 0.0
            for move in legal_moves_flipped:
                if move in LABEL_TO_INDEX:
                    idx = LABEL_TO_INDEX[move]
                    prior = policy[idx]
                    node.children[move] = MCTSNode(prior=prior)
                    total_prior += prior

            # 归一化先验概率
            if total_prior > 0 and node.children:
                for child in node.children.values():
                    child.prior /= total_prior
            elif node.children:
                uniform = 1.0 / len(node.children)
                for child in node.children.values():
                    child.prior = uniform

            value = -value  # 翻转值（父节点视角）

        # 回传
        for i in range(len(path) - 1, -1, -1):
            path[i].visit_count += 1
            path[i].total_value += value
            value = -value  # 交替翻转

    def _select_child(self, node):
        """使用PUCT算法选择最优子节点"""
        best_score = -float('inf')
        best_action = None
        best_child = None

        sqrt_total = math.sqrt(node.visit_count)

        for action, child in node.children.items():
            # PUCT公式
            q = child.q_value
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child
