"""
Agent 封装模块

提供 MCTSAgent，将 MCTS 包装为与 BaseAgent 接口兼容的对象，
用于训练循环中的对手池和数据收集。

BaseAgent 接口已在 pikafish_agent.py 中定义：
    - get_move(game) -> Optional[str]：返回实际棋盘坐标的走法
    - new_game()：通知 Agent 开始新局
    - update_move(move)：在对方走子后更新 Agent 状态
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .pikafish_agent import BaseAgent
from .game import flip_move

logger = logging.getLogger(__name__)


class MCTSAgent(BaseAgent):
    """
    封装 MCTS 搜索的 Agent，实现 BaseAgent 接口。

    提供两种主要用途：
    1. 作为训练对手（通过 get_move 返回实际棋盘坐标走法）
    2. 作为训练数据收集方（通过 get_action_probs 返回动作概率分布）

    Args:
        model: ChessModel 实例
        num_simulations: MCTS 模拟次数
        temperature_threshold: 前 N 步使用温度 1.0 探索，之后切换为 0.1

    Note:
        MCTS 内部始终以"红方视角"表示动作（flip_move 转换）。
        get_move() 返回值是实际棋盘坐标（与 PikafishAgent 格式一致）。
        get_action_probs() 返回的 actions 是 MCTS 内部视角（红方坐标），
        与 self_play_game 中的 chosen_action 一致。
    """

    def __init__(self, model, num_simulations: int = 100,
                 temperature_threshold: int = 30) -> None:
        from .mcts import MCTS
        self._model = model
        self._num_simulations = num_simulations
        self._temperature_threshold = temperature_threshold
        self._mcts = MCTS(model, num_simulations=num_simulations)
        self._move_count = 0

    # ------------------------------------------------------------------
    # BaseAgent 接口
    # ------------------------------------------------------------------

    def new_game(self) -> None:
        """重置 MCTS 树和步数计数器，准备开始新局。"""
        from .mcts import MCTS
        self._mcts = MCTS(self._model, num_simulations=self._num_simulations)
        self._move_count = 0

    def get_move(self, game) -> Optional[str]:
        """
        以贪心策略（temperature=0）返回最佳走法（实际棋盘坐标格式）。

        Returns:
            内部格式走法字符串（实际棋盘坐标），如 ``"7072"``；
            无合法走法时返回 ``None``。
        """
        actions, probs = self._mcts.get_action_probs(
            game, temperature=0.0, add_noise=False
        )
        if not actions:
            return None

        best_idx = int(np.argmax(probs))
        chosen_action = actions[best_idx]  # MCTS 内部（红方视角）

        # 转换为实际棋盘坐标
        actual_action = chosen_action if game.red_to_move else flip_move(chosen_action)
        self._move_count += 1
        return actual_action

    def update_move(self, move_mcts_perspective: str) -> None:
        """
        以红方视角的走法推进 MCTS 树（树复用模式）。

        Args:
            move_mcts_perspective: 红方坐标系格式的走法字符串
                （与 get_action_probs 返回的 actions 元素格式相同）。
        """
        self._mcts.update_with_move(move_mcts_perspective)
        self._move_count += 1

    # ------------------------------------------------------------------
    # 训练数据收集接口
    # ------------------------------------------------------------------

    def get_action_probs(self, game, temperature: Optional[float] = None,
                         add_noise: bool = False):
        """
        运行 MCTS 并返回 ``(actions, probs)``，用于训练样本收集。

        Args:
            game: :class:`~AIchess.game.ChessGame` 实例。
            temperature: 温度参数；为 ``None`` 时按 temperature_threshold 自动切换。
            add_noise: 是否向根节点注入 Dirichlet 噪声（训练我方时应为 True）。

        Returns:
            (actions, probs)：actions 以 MCTS 内部视角（红方坐标）表示。
        """
        if temperature is None:
            temperature = (
                1.0 if self._move_count < self._temperature_threshold else 0.1
            )
        return self._mcts.get_action_probs(
            game, temperature=temperature, add_noise=add_noise
        )
