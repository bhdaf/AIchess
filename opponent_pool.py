"""
对手池模块

提供 OpponentPool，支持按权重采样对手类型以及课程学习阶段切换，
使训练不再只有 self-play，而是可以对接：
  - 当前模型自对弈（self_play）
  - Pikafish 弱/中/全强度引擎对手（pikafish_weak / pikafish_mid / pikafish_full）
  - 历史 checkpoint 对手（historical）

课程学习 Schedule（'default'）：
  - 前 1/3 局：全部自对弈
  - 中 1/3 局：自对弈 50% + pikafish_weak 30% + historical 20%
  - 后 1/3 局：自对弈 30% + pikafish_weak 20% + pikafish_mid 20%
               + pikafish_full 10% + historical 20%

强度控制（elo 模式）：
  - 设置 pikafish_elo_weak/mid/full（如 1000/1500/2000），
    引擎通过 UCI_LimitStrength + UCI_Elo 选项控制强度。
"""

from __future__ import annotations

import logging
import os
import random
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

# 默认课程学习阶段权重：(进度上限, 权重字典)
_DEFAULT_CURRICULUM_STAGES = [
    (1 / 3, {'self_play': 1.0}),
    (2 / 3, {'self_play': 0.5, 'pikafish_weak': 0.3, 'historical': 0.2}),
    (1.0,   {'self_play': 0.3, 'pikafish_weak': 0.2, 'pikafish_mid': 0.2,
              'pikafish_full': 0.1, 'historical': 0.2}),
]


class OpponentPool:
    """
    对手池，按课程学习权重采样对手，并管理引擎子进程和历史 checkpoint。

    Args:
        model: ChessModel 实例（当前训练模型）。
        engine_path: Pikafish 引擎可执行文件路径；为 ``None`` 时禁用引擎对手。
        pikafish_elo_weak: 弱强度引擎目标 Elo（如 1000）；通过 UCI_LimitStrength + UCI_Elo 控制强度。
        pikafish_elo_mid: 中强度引擎目标 Elo（如 1500）。
        pikafish_elo_full: 全强度引擎目标 Elo（如 2000）。
        num_simulations: MCTSAgent 模拟次数（用于 self_play 和 historical）。
        checkpoints_dir: 保存历史 checkpoint 的目录；为 ``None`` 时禁用历史对手。
        curriculum: 课程学习策略名称（``'default'``）或 ``None``（全自对弈）。
        engine_options: 传给 UCI 引擎的选项字典，如 ``{"Hash": "256"}``。
        max_history: 历史 checkpoint 池最大容量（超出时删除最旧的）。
    """

    def __init__(
        self,
        model,
        engine_path: Optional[str] = None,
        pikafish_elo_weak: Optional[int] = None,
        pikafish_elo_mid: Optional[int] = None,
        pikafish_elo_full: Optional[int] = None,
        num_simulations: int = 100,
        checkpoints_dir: Optional[str] = None,
        curriculum: Optional[str] = 'default',
        engine_options: Optional[dict] = None,
        max_history: int = 10,
    ) -> None:
        self.model = model
        self.engine_path = engine_path
        self.pikafish_elo_weak = pikafish_elo_weak
        self.pikafish_elo_mid = pikafish_elo_mid
        self.pikafish_elo_full = pikafish_elo_full
        self.num_simulations = num_simulations
        self.checkpoints_dir = checkpoints_dir
        self.curriculum = curriculum
        self.engine_options = engine_options or {}
        self.max_history = max_history

        # 历史 checkpoint 路径列表（按添加顺序排列）
        self._historical_checkpoints: list[str] = []

        # Pikafish 引擎实例（按强度缓存，常驻子进程）
        self._pikafish_agents: dict[str, object] = {}

        # 确定当前配置下可用的对手类型
        self._available_types = self._get_available_types()

    # ------------------------------------------------------------------
    # 可用类型查询
    # ------------------------------------------------------------------

    def _get_available_types(self) -> set:
        """返回当前配置下可用的对手类型集合。"""
        types = {'self_play'}
        if self.engine_path:
            types.update({'pikafish_weak', 'pikafish_mid', 'pikafish_full'})
        if self.checkpoints_dir:
            types.add('historical')
        return types

    # ------------------------------------------------------------------
    # 课程学习权重
    # ------------------------------------------------------------------

    def _get_weights(self, game_idx: int, num_games: int) -> dict:
        """根据训练进度和课程阶段计算对手权重字典。"""
        if not self.curriculum or self.curriculum == 'none':
            return {'self_play': 1.0}

        progress = game_idx / max(num_games, 1)
        stage_weights: dict = {'self_play': 1.0}
        for fraction_end, weights in _DEFAULT_CURRICULUM_STAGES:
            if progress <= fraction_end:
                stage_weights = weights
                break

        # 过滤不可用类型
        filtered = {k: v for k, v in stage_weights.items()
                    if k in self._available_types}
        # 历史对手须有 checkpoint 才可用
        if 'historical' in filtered and not self._historical_checkpoints:
            del filtered['historical']
        if not filtered:
            filtered = {'self_play': 1.0}

        # 归一化
        total = sum(filtered.values())
        return {k: v / total for k, v in filtered.items()}

    # ------------------------------------------------------------------
    # 采样接口
    # ------------------------------------------------------------------

    def sample_opponent_type(self, game_idx: int, num_games: int) -> str:
        """
        根据课程阶段权重采样一种对手类型。

        Args:
            game_idx: 当前对局编号（从 1 开始）。
            num_games: 总对局数。

        Returns:
            对手类型字符串，如 ``'self_play'``、``'pikafish_weak'`` 等。
        """
        weights = self._get_weights(game_idx, num_games)
        types = list(weights.keys())
        probs = [weights[t] for t in types]
        return random.choices(types, weights=probs, k=1)[0]

    def build_opponent(self, opponent_type: str):
        """
        根据类型构建对手 Agent 实例。

        Args:
            opponent_type: 对手类型字符串。

        Returns:
            (agent, metadata)：agent 是 BaseAgent 子类实例，metadata 是描述字典。
        """
        from .agents import MCTSAgent

        if opponent_type == 'self_play':
            agent = MCTSAgent(self.model, num_simulations=self.num_simulations)
            metadata = {
                'opponent_type': 'self_play',
                'opponent_strength': 'current',
                'engine_elo': None,
            }

        elif opponent_type in ('pikafish_weak', 'pikafish_mid', 'pikafish_full'):
            agent = self._get_pikafish_agent(opponent_type)
            elo = self._elo_for(opponent_type)
            metadata = {
                'opponent_type': opponent_type,
                'opponent_strength': opponent_type,
                'engine_elo': elo,
            }

        elif opponent_type == 'historical':
            agent = self._get_historical_agent()
            if agent is None:
                # 历史池为空，回退到自对弈
                agent = MCTSAgent(self.model, num_simulations=self.num_simulations)
                metadata = {
                    'opponent_type': 'self_play',
                    'opponent_strength': 'current',
                    'engine_elo': None,
                    'fallback_from': 'historical',
                }
            else:
                metadata = {
                    'opponent_type': 'historical',
                    'opponent_strength': 'historical',
                    'engine_elo': None,
                }

        else:
            raise ValueError(f"未知对手类型：{opponent_type!r}")

        return agent, metadata

    # ------------------------------------------------------------------
    # Pikafish 引擎管理
    # ------------------------------------------------------------------

    def _elo_for(self, strength: str) -> Optional[int]:
        """返回指定强度对应的目标 Elo；未配置时返回 ``None``。"""
        return {
            'pikafish_weak': self.pikafish_elo_weak,
            'pikafish_mid':  self.pikafish_elo_mid,
            'pikafish_full': self.pikafish_elo_full,
        }[strength]

    def _get_pikafish_agent(self, strength: str):
        """获取指定强度的 PikafishAgent（延迟创建、复用常驻子进程）。"""
        if strength not in self._pikafish_agents:
            if not self.engine_path:
                raise ValueError("engine_path 未配置，无法使用 Pikafish 对手")
            elo = self._elo_for(strength)
            from .pikafish_agent import PikafishAgent
            agent = PikafishAgent(
                self.engine_path,
                elo=elo,
                options=self.engine_options,
            )
            agent.start()
            self._pikafish_agents[strength] = agent
            if elo is not None:
                logger.info("启动 Pikafish 引擎（%s，elo=%d）", strength, elo)
            else:
                logger.info("启动 Pikafish 引擎（%s）", strength)
        return self._pikafish_agents[strength]

    # ------------------------------------------------------------------
    # 历史 checkpoint 管理
    # ------------------------------------------------------------------

    def _get_historical_agent(self):
        """随机选择一个历史 checkpoint，创建 MCTSAgent；池为空时返回 None。"""
        if not self._historical_checkpoints:
            return None

        ckpt_path = random.choice(self._historical_checkpoints)
        from .model import ChessModel
        from .agents import MCTSAgent

        hist_model = ChessModel(
            num_channels=self.model.num_channels,
            num_res_blocks=self.model.num_res_blocks,
        )
        hist_model.build()
        loaded = hist_model.load(ckpt_path)
        if not loaded:
            logger.warning("无法加载历史 checkpoint %s，回退到自对弈", ckpt_path)
            return None
        return MCTSAgent(hist_model, num_simulations=self.num_simulations)

    def add_checkpoint(self, model_path: str) -> None:
        """
        将模型 checkpoint 复制到历史池目录并记录。

        若历史池超过 max_history，则删除最旧的 checkpoint。

        Args:
            model_path: 源模型 .pth 文件路径。
        """
        if not self.checkpoints_dir:
            return
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        idx = len(self._historical_checkpoints)
        ckpt_name = f"ckpt_{idx:04d}.pth"
        dest_path = os.path.join(self.checkpoints_dir, ckpt_name)

        try:
            shutil.copy2(model_path, dest_path)
            # 同时复制配置文件（如存在）
            config_src = model_path.replace('.pth', '_config.json')
            if os.path.exists(config_src):
                shutil.copy2(config_src,
                             dest_path.replace('.pth', '_config.json'))
            self._historical_checkpoints.append(dest_path)
            logger.info("已保存历史 checkpoint: %s", dest_path)
        except OSError as exc:
            logger.warning("保存历史 checkpoint 失败: %s", exc)
            return

        # 清理过旧的 checkpoint
        while len(self._historical_checkpoints) > self.max_history:
            old_path = self._historical_checkpoints.pop(0)
            for suffix in ('', '_config.json'):
                try:
                    p = old_path if not suffix else old_path.replace('.pth', suffix)
                    if os.path.exists(p):
                        os.remove(p)
                        logger.debug("删除旧 checkpoint: %s", p)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭所有 Pikafish 引擎子进程。"""
        for strength, agent in list(self._pikafish_agents.items()):
            try:
                agent.quit()
                logger.info("已关闭 Pikafish 引擎（%s）", strength)
            except (OSError, RuntimeError, AttributeError) as exc:
                logger.warning("关闭 Pikafish 引擎失败（%s）: %s", strength, exc)
        self._pikafish_agents.clear()

    def __enter__(self) -> "OpponentPool":
        return self

    def __exit__(self, *_) -> None:
        self.close()
