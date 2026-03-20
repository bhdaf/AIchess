"""
简化蒙特卡洛树搜索 (MCTS)

实现基于神经网络引导的MCTS，用于走法选择。

行为说明
--------
* reset_root=True（默认）：每次 get_action_probs() 调用前重置 root，保证搜索
  基于当前局面，适合推理/GUI/CLI 场景。
* reset_root=False：保留上次搜索的子树（树复用模式），结合 update_with_move()
  使用，可在连续走子时复用统计数据，适合对性能要求高的训练循环。
* cache_size > 0：在单次 get_action_probs() 调用内，对相同局面的神经网络评估
  结果进行缓存，减少重复推理；cache_size=0 表示禁用缓存（默认）。
"""

import math
from typing import Optional

import numpy as np

from .game import (
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
        cache_size: 局面评估缓存大小（0=禁用；>0 时在单次搜索内复用重复局面
                    的网络输出，减少推理次数）
    """

    def __init__(self, model, num_simulations=200, c_puct=1.5,
                 dirichlet_alpha=0.15, dirichlet_weight=0.25, cache_size=0,
                 debug_mcts: bool = False):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.cache_size = cache_size  # 0 = 禁用缓存
        self.debug_mcts = debug_mcts
        self.root = MCTSNode()
        self._cache_hits = 0  # 当次搜索命中次数（供日志统计）

    def get_action_probs(self, game, temperature=1.0, add_noise=False,
                         reset_root=True, mode: Optional[str] = None):
        """
        运行MCTS搜索并返回走法概率分布

        Args:
            game: ChessGame实例
            temperature: 温度参数，控制探索程度
            add_noise: 是否添加Dirichlet噪声（训练时使用）
            reset_root: 搜索前是否重置 root（默认 True）。
                        True  — 每次搜索从空树开始，保证与当前局面匹配；
                        False — 保留上次子树统计（树复用模式），须配合
                                update_with_move() 手动推进 root。
            mode: 运行模式，可选值：
                  "eval"      — 强制 add_noise=False, temperature=0.0（确定性）
                  "self_play" — 强制 add_noise=True，temperature 由调用方提供
                  None        — 使用调用方传入的 temperature/add_noise（兼容原有行为）

        Returns:
            actions: 走法列表
            probs: 对应的概率列表
        """
        # 根据 mode 覆盖 add_noise / temperature
        if mode == "eval":
            add_noise = False
            temperature = 0.0
        elif mode == "self_play":
            add_noise = True
        # mode is None → keep caller-provided values

        # 每次搜索前重置 root（默认开启，保证正确性）
        if reset_root:
            self.root = MCTSNode()

        # 建立本次搜索的局面缓存（cache_size>0 时启用，减少重复网络推理）
        episode_cache = {} if self.cache_size > 0 else None
        self._cache_hits = 0

        # 运行模拟；第一次模拟后若需要则向 root 注入 Dirichlet 噪声
        for i in range(self.num_simulations):
            game_copy = game.copy()
            self._simulate(game_copy, self.root, episode_cache)
            # 在第一次模拟扩展 root 后注入噪声（仅 root 节点，仅训练时）
            if i == 0 and add_noise and self.root.children:
                self._add_dirichlet_noise(self.root)

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

        # 打印根节点统计（仅调试模式）
        if self.debug_mcts:
            self._print_root_stats(actions, visits)

        return actions, probs

    def _add_dirichlet_noise(self, node):
        """向节点的子节点先验概率注入 Dirichlet 噪声（仅用于 root 节点）"""
        children = list(node.children.values())
        n = len(children)
        if n == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        for child, eta in zip(children, noise):
            child.prior = (
                (1 - self.dirichlet_weight) * child.prior
                + self.dirichlet_weight * eta
            )

        if self.debug_mcts:
            self._print_root_priors_after_noise(node)

    def _print_root_priors_after_noise(self, node):
        """打印注入噪声后根节点各走法的先验概率（调试用）"""
        items = sorted(
            node.children.items(), key=lambda kv: kv[1].prior, reverse=True
        )
        print("Root priors after Dirichlet noise:")
        for action, child in items:
            print(f"  {action} P={child.prior:.4f}")

    def _print_root_stats(self, actions, visits):
        """打印根节点各走法的访问次数、Q 值和先验概率（调试用）"""
        combined = sorted(
            zip(actions, visits),
            key=lambda av: av[1],
            reverse=True,
        )
        print("Root visit counts:")
        for action, vc in combined:
            child = self.root.children[action]
            print(
                f"  {action} {vc:4d} visits  Q={child.q_value:.4f}  P={child.prior:.4f}"
            )

    def update_with_move(self, action):
        """用选择的走法更新树（复用子树）"""
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = MCTSNode()

    def _simulate(self, game, node, cache=None):
        """
        运行一次MCTS模拟

        1. 选择: 沿着树选择最优子节点
        2. 扩展: 到达叶子节点时用神经网络扩展
        3. 回传: 将评估值回传到路径上的节点

        Args:
            game: 当前局面（会被就地修改）
            node: 当前 root 节点
            cache: 可选的局面 -> (policy, value) 缓存字典（减少重复推理）
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
        # value_for_current_player：以"当前将要走子的一方"为正方向
        #   +1.0 = 该方赢，-1.0 = 该方输（被将死），0.0 = 平局
        if game.done:
            if game.winner == 'draw':
                value_for_current_player = 0.0
            else:
                # 到达终局时，轮到走子的一方已被将死，故为 -1.0
                value_for_current_player = -1.0
        else:
            # 获取合法走法（在网络调用前计算，用于掩码）
            legal_moves = game.get_legal_moves()
            if not game.red_to_move:
                legal_moves_flipped = [flip_move(m) for m in legal_moves]
            else:
                legal_moves_flipped = legal_moves

            # 构造合法走法的索引列表（用于 predict_with_mask）
            legal_indices = [LABEL_TO_INDEX[m] for m in legal_moves_flipped
                             if m in LABEL_TO_INDEX]

            # 局面缓存：key = 当前玩家视角的 FEN，避免重复网络调用
            cache_key = game.get_observation() if cache is not None else None

            if cache is not None and cache_key in cache:
                # 命中缓存：直接复用已计算的 policy/value
                policy, value_for_current_player = cache[cache_key]
                self._cache_hits += 1
            else:
                # 先对非法走法 mask 再 softmax，数值更稳定
                planes = game.to_planes()
                policy, value_for_current_player = self.model.predict_with_mask(
                    planes, legal_indices)
                # 将结果存入缓存（未超上限时）
                if cache is not None and len(cache) < self.cache_size:
                    cache[cache_key] = (policy, value_for_current_player)

                if self.debug_mcts:
                    self._print_network_output(
                        policy, value_for_current_player, legal_moves_flipped)

            # 扩展节点
            total_prior = 0.0
            for move in legal_moves_flipped:
                if move in LABEL_TO_INDEX:
                    idx = LABEL_TO_INDEX[move]
                    prior = policy[idx]
                    node.children[move] = MCTSNode(prior=prior)
                    total_prior += prior

            # 归一化先验概率（predict_with_mask 已做 mask+softmax，此处为保险）
            if total_prior > 0 and node.children:
                for child in node.children.values():
                    child.prior /= total_prior
            elif node.children:
                uniform = 1.0 / len(node.children)
                for child in node.children.values():
                    child.prior = uniform

        # 回传
        # Q 值约定：node.total_value / node.visit_count 表示"选择该节点的那方玩家"
        # 的视角价值（与 _select_child 直接最大化 child.q_value 的逻辑对应）。
        # path[-1] 由其父节点选出，因此应存储父方视角 = -value_for_current_player；
        # 之后每上升一层，双方交替，视角再翻转一次。
        backprop_value = -value_for_current_player
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += backprop_value
            backprop_value = -backprop_value

    def _print_network_output(self, policy, value, legal_moves_flipped):
        """打印神经网络预测输出（调试用）"""
        print(f"value: {value:.4f}")
        # policy 已经是合法走法上的 softmax 概率；对合法走法重新归一化（安全检查）
        legal_probs = []
        for move in legal_moves_flipped:
            if move in LABEL_TO_INDEX:
                idx = LABEL_TO_INDEX[move]
                legal_probs.append((move, float(policy[idx])))
        total = sum(p for _, p in legal_probs)
        if total > 0:
            legal_probs = [(m, p / total) for m, p in legal_probs]
        legal_probs.sort(key=lambda mp: mp[1], reverse=True)
        print("Top policy moves:")
        for move, prob in legal_probs[:10]:
            print(f"  {move} {prob:.4f}")

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
