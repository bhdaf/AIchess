"""
Pikafish/UCI 引擎 Agent 封装

提供 ``UCIAgent`` 基类（统一接口，与 MCTS Agent 对等）以及
``PikafishAgent`` 具体实现，包含：

* 内部走法格式（``"x0y0x1y1"``）↔ UCI ICCS 格式（``"a0b2"``）的双向转换
* ``game.get_fen()``（棋盘部分）→ 完整 UCI FEN 字符串
* UCI ICCS 走法 → 内部走法格式

坐标映射（已验证与 INIT_FEN 一致）：
    列 x (0‑8)  →  UCI file  a‑i  （a = col 0, i = col 8）
    行 y (0‑9)  →  UCI rank  0‑9  （0 = 红方底线, 9 = 黑方底线）

示例::

    agent = PikafishAgent("/path/to/pikafish", elo=1500)
    agent.start()
    agent.new_game()
    move = agent.get_move(game)   # 返回内部格式走法，如 "7072"
    agent.quit()

或作为上下文管理器::

    with PikafishAgent("/path/to/pikafish", elo=1500) as agent:
        agent.new_game()
        move = agent.get_move(game)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 固定的引擎每步思考时间（ms）；强度由 elo 参数（UCI_LimitStrength + UCI_Elo）控制
_DEFAULT_MOVETIME_MS = 100


# ---------------------------------------------------------------------------
# 统一 Agent 接口
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    棋手 Agent 统一接口。

    ``get_move(game)`` 始终接收 :class:`~AIchess.game.ChessGame` 实例，
    返回内部格式走法字符串（``"x0y0x1y1"``）。
    子类可选择性地覆盖 ``new_game()``、``update_move()`` 以支持状态跟踪。
    """

    @abstractmethod
    def get_move(self, game: "ChessGame") -> Optional[str]:  # type: ignore[name-defined]
        """
        给定当前局面，返回一步合法走法（内部格式）。

        Returns:
            走法字符串，如 ``"7072"``；无合法走法时返回 ``None``。
        """

    def new_game(self) -> None:
        """可选：通知 Agent 开始新局。"""

    def update_move(self, move: str) -> None:
        """可选：在对方走子后更新 Agent 状态（用于树复用等）。"""


# ---------------------------------------------------------------------------
# UCI 走法 / FEN 转换工具
# ---------------------------------------------------------------------------

def internal_to_uci(move: str) -> str:
    """
    内部走法格式 → UCI ICCS 格式。

    内部格式：``"x0y0x1y1"``，坐标均为单个数字字符（列 0‑8，行 0‑9）。
    UCI ICCS 格式：``"<file0><rank0><file1><rank1>"``，文件为 ``a``‑``i``。

    Args:
        move: 内部格式走法，如 ``"7072"``（车从 h0 到 h2）。

    Returns:
        UCI ICCS 格式走法，如 ``"h0h2"``。

    Raises:
        ValueError: 走法字符串格式不正确时。
    """
    if len(move) != 4 or not move.isdigit():
        raise ValueError(
            f"内部走法格式错误：期望 4 位数字字符串，实际为 {move!r}"
        )
    x0, y0, x1, y1 = int(move[0]), int(move[1]), int(move[2]), int(move[3])
    file0 = chr(ord('a') + x0)
    file1 = chr(ord('a') + x1)
    return f"{file0}{y0}{file1}{y1}"


def uci_to_internal(uci_move: str) -> str:
    """
    UCI ICCS 格式走法 → 内部走法格式。

    Args:
        uci_move: UCI ICCS 格式走法，如 ``"h0h2"``。

    Returns:
        内部格式走法，如 ``"7072"``。

    Raises:
        ValueError: 走法字符串格式不正确时。
    """
    if len(uci_move) != 4:
        raise ValueError(
            f"UCI 走法格式错误：期望 4 字符字符串，实际为 {uci_move!r}"
        )
    file0, rank0, file1, rank1 = uci_move[0], uci_move[1], uci_move[2], uci_move[3]
    if not (file0.isalpha() and file1.isalpha() and rank0.isdigit() and rank1.isdigit()):
        raise ValueError(
            f"UCI 走法格式错误（非字母/数字组合）：{uci_move!r}"
        )
    x0 = ord(file0.lower()) - ord('a')
    x1 = ord(file1.lower()) - ord('a')
    if not (0 <= x0 <= 8 and 0 <= x1 <= 8):
        raise ValueError(
            f"UCI 走法列超出范围（a‑i）：{uci_move!r}"
        )
    return f"{x0}{rank0}{x1}{rank1}"


def game_to_uci_fen(game) -> str:
    """
    将 :class:`~AIchess.game.ChessGame` 转换为完整 UCI FEN 字符串。

    UCI FEN 格式（UCCI 标准）：
        ``<board> <side> <check?> <step_count> <move_count>``

    其中 ``<side>`` 为 ``w``（红方/先手）或 ``b``（黑方/后手）。

    Args:
        game: :class:`~AIchess.game.ChessGame` 实例（已处于某一局面）。

    Returns:
        完整 UCI FEN 字符串，如
        ``"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"``。
    """
    board_fen = game.get_fen()
    side = "w" if game.red_to_move else "b"
    # 使用保守的兜底值；UCCI 引擎通常忽略 step/move count 字段
    move_number = max(1, (game.num_moves // 2) + 1)
    halfmove_clock = 0  # 无捉/将信息时置零
    return f"{board_fen} {side} - - {halfmove_clock} {move_number}"


# ---------------------------------------------------------------------------
# Pikafish / 通用 UCI Agent
# ---------------------------------------------------------------------------


def parse_multipv_info(info_lines: list) -> list:
    """
    解析 MultiPV 搜索返回的 info 行，提取各候选走法及其评分。

    每行格式（节选）：
        ``info depth N ... multipv K score cp SCORE ... pv MOVE ...``

    ``score mate N`` 情形将映射为 ``±10000`` cp。

    Args:
        info_lines: UCIEngine 搜索期间输出的 info 行列表。

    Returns:
        按 multipv 排名（1=最优）排序的列表，每项为
        ``(uci_move: str, score_cp: int)``。
    """
    candidates: dict = {}
    
    # 倒序遍历列表：从最新的 info 往回看
    for line in reversed(info_lines):
        if 'multipv' not in line:
            continue
            
        parts = line.split()
        try:
            multipv_rank = int(parts[parts.index('multipv') + 1])
        except (ValueError, IndexError):
            continue

        # 如果这个排名的“最高深度”已经存过了，就跳过（因为我们是倒着看的）
        if multipv_rank in candidates:
            continue

        # --- 以下是解析 score 和 pv 的逻辑（保持不变） ---
        score_cp = None
        try:
            if 'score' in parts:
                score_idx = parts.index('score')
                score_type = parts[score_idx + 1]
                score_val = parts[score_idx + 2]
                if score_type == 'cp':
                    score_cp = int(score_val)
                elif score_type == 'mate':
                    score_cp = 10000 if int(score_val) > 0 else -10000
        except: pass

        pv_move = None
        try:
            if 'pv' in parts:
                pv_idx = parts.index('pv')
                pv_move = parts[pv_idx + 1]
        except: pass
        # ----------------------------------------------

        if score_cp is not None and pv_move is not None:
            candidates[multipv_rank] = (pv_move, score_cp)

    # 返回按 rank 排序的结果
    return [candidates[k] for k in sorted(candidates.keys())]

class PikafishAgent(BaseAgent):
    """
    Pikafish（或任意兼容 UCI 协议的中国象棋引擎）Agent。

    该 Agent 在每步时：
      1. 调用 :func:`game_to_uci_fen` 将当前局面转换为 UCI FEN
      2. 通过 :class:`~AIchess.uci.UCIEngine` 发送给引擎
      3. 引擎返回 ICCS 格式走法，经 :func:`uci_to_internal` 转换为内部格式

    引擎强度由 ``elo`` 参数（``UCI_LimitStrength + UCI_Elo``）控制。

    Args:
        engine_path (str): 引擎可执行文件路径（如 ``"/usr/local/bin/pikafish"``）。
        depth (int | None): 固定深度搜索；若指定则以 ``go depth`` 替代默认的
            ``go movetime``。
        elo (int | None): 目标 Elo 评分（如 ``1500``）；设置后自动启用
            ``UCI_LimitStrength=true`` 和 ``UCI_Elo=<elo>``。同时设置 ``elo``
            和 ``skill_level`` 时，``elo`` 生效而 ``skill_level`` 被忽略。
        skill_level (int | None): 引擎技能等级（0‑20）；设置后发送
            ``Skill Level=<skill_level>``。仅在 ``elo`` 未指定时生效。
        options (dict | None): 在握手后通过 ``setoption`` 发送给引擎的选项字典，
            例如 ``{"Hash": "256"}``。该字典中的选项优先级高于 ``elo``/
            ``skill_level`` 自动应用的选项，可用于覆盖默认行为。
        init_timeout (float): UCI 握手超时秒数。
        move_timeout (float): 等待 bestmove 的额外超时秒数。

    Example::

        with PikafishAgent("/path/to/pikafish", elo=1500) as agent:
            game = ChessGame().reset()
            agent.new_game()
            move = agent.get_move(game)   # "7072"
            game.step(move)
    """

    def __init__(
        self,
        engine_path: str,
        depth: Optional[int] = None,
        elo: Optional[int] = None,
        skill_level: Optional[int] = None,
        options: Optional[dict] = None,
        init_timeout: float = 10.0,
        move_timeout: float = 5.0,
        multipv: int = 1,
    ) -> None:
        from .uci import UCIEngine  # 延迟导入，降低无引擎时的依赖成本

        self.engine_path = engine_path
        self.depth = depth
        self.elo = elo
        self.skill_level = skill_level
        self.options = options or {}
        self.multipv = max(1, int(multipv))
        self._engine = UCIEngine(
            engine_path,
            init_timeout=init_timeout,
            move_timeout=move_timeout,
        )

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动引擎并应用 setoption 配置。"""
        self._engine.start()
        # 先应用 Elo/Skill Level 强度设置（用户 options 可覆盖）
        if self.elo is not None:
            # 启用 UCI 强度限制并设置目标 Elo
            self._engine.set_option("UCI_LimitStrength", "true")
            self._engine.set_option("UCI_Elo", str(self.elo))
            logger.debug("Pikafish 强度控制：UCI_Elo=%d", self.elo)
        elif self.skill_level is not None:
            # Skill Level 控制（0-20），不启用 UCI_LimitStrength
            self._engine.set_option("Skill Level", str(self.skill_level))
            logger.debug("Pikafish 强度控制：Skill Level=%d", self.skill_level)
        # 用户自定义选项（优先级最高，可覆盖上述设置）
        for name, value in self.options.items():
            self._engine.set_option(name, str(value))
        if self.multipv > 1:
            self._engine.set_option("MultiPV", str(self.multipv))

    def quit(self) -> None:
        """关闭引擎子进程。"""
        self._engine.quit()

    def __enter__(self) -> "PikafishAgent":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.quit()

    # ------------------------------------------------------------------
    # BaseAgent 接口
    # ------------------------------------------------------------------

    def new_game(self) -> None:
        """通知引擎开始新局。"""
        self._engine.new_game()

    def get_move(self, game) -> Optional[str]:
        """
        给定当前局面，调用引擎并返回内部格式走法。

        Args:
            game: :class:`~AIchess.game.ChessGame` 实例。

        Returns:
            内部格式走法字符串（如 ``"7072"``），或引擎无合法走法时
            返回 ``None``。
        """
        uci_fen = game_to_uci_fen(game)
        self._engine.set_position(uci_fen)

        if self.depth is not None:
            uci_move = self._engine.go_depth(self.depth)
        else:
            uci_move = self._engine.go_movetime(_DEFAULT_MOVETIME_MS)

        if uci_move is None:
            logger.warning("引擎未返回走法（局面：%s）", uci_fen)
            return None

        try:
            internal_move = uci_to_internal(uci_move)
        except ValueError as exc:
            logger.error("无法解析引擎走法 %r：%s", uci_move, exc)
            return None

        # 校验：走法必须在当前局面的合法走法中
        legal = game.get_legal_moves()
        if internal_move not in legal:
            logger.warning(
                "引擎走法 %r（内部：%r）不在合法走法列表中；"
                "这可能是 FEN/规则不一致导致的。",
                uci_move, internal_move,
            )
            # TODO: 若规则存在系统性差异，请在此处添加走法映射逻辑。
            # 目前作为容错措施，返回 None 让调用方选择备用走法
            return None

        return internal_move

    def get_move_with_info(self, game) -> tuple:
        """
        给定当前局面，调用引擎并返回 ``(内部格式走法, info_lines)``。

        info_lines 包含搜索期间引擎输出的所有 ``info`` 行，可用于 MultiPV 解析。

        Args:
            game: :class:`~AIchess.game.ChessGame` 实例。

        Returns:
            ``(internal_move, info_lines)``：走法字符串或 ``None``；info 行列表。
        """
        uci_fen = game_to_uci_fen(game)
        self._engine.set_position(uci_fen)

        if self.depth is not None:
            uci_move, info_lines = self._engine.go_depth_with_info(self.depth)
        else:
            uci_move, info_lines = self._engine.go_movetime_with_info(_DEFAULT_MOVETIME_MS)

        if uci_move is None:
            logger.warning("引擎未返回走法（局面：%s）", uci_fen)
            return None, info_lines

        try:
            internal_move = uci_to_internal(uci_move)
        except ValueError as exc:
            logger.error("无法解析引擎走法 %r：%s", uci_move, exc)
            return None, info_lines

        legal = game.get_legal_moves()
        if internal_move not in legal:
            logger.warning(
                "引擎走法 %r（内部：%r）不在合法走法列表中。",
                uci_move, internal_move,
            )
            return None, info_lines

        return internal_move, info_lines

    def get_policy_and_value(self, game, k: int = 5,
            temperature: float = 1.0) -> "tuple[np.ndarray, float, str] | tuple[None, None, None]":
            """
            单次搜索同时获取 Soft Policy, Value Score 和 Best Move。
            
            Returns:
                (policy, value, best_move_internal): 
                    policy: np.ndarray (NUM_ACTIONS,), sum≈1。
                    value: float, 红方视角分数。
                    best_move_internal: str, 内部格式的最佳走法 (如 "7072")。
                如果失败，返回 None。
            """
            from .game import NUM_ACTIONS, LABEL_TO_INDEX, flip_move

            # 1. 设置 MultiPV 并执行一次搜索
            actual_k = max(k, self.multipv)
            if actual_k != self.multipv:
                self._engine.set_option("MultiPV", str(actual_k))

            # _move 这里是 UCI 格式的最佳走法
            best_move_uci, info_lines = self.get_move_with_info(game)

            if actual_k != self.multipv:
                self._engine.set_option("MultiPV", str(self.multipv))

            candidates = parse_multipv_info(info_lines)
            if not candidates:
                return None, None, None

            # 2. 提取最佳走法 并转换为内部格式
            best_move_internal = None
            try:
                if best_move_uci:
                    best_move_internal = uci_to_internal(best_move_uci)
            except ValueError:
                pass
            
            # 3. 提取价值分数
            _, score_cp = candidates[0]
            is_black = not game.red_to_move
            value_red_view = -float(score_cp) if is_black else float(score_cp)

            # 4. 构建 Soft Policy
            policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
            indices = []
            scores = []

            for uci_mv, score_cp_iter in candidates:
                try:
                    internal_mv = uci_to_internal(uci_mv)
                except ValueError:
                    continue
                
                policy_mv = flip_move(internal_mv) if is_black else internal_mv
                if policy_mv not in LABEL_TO_INDEX:
                    continue
                
                indices.append(LABEL_TO_INDEX[policy_mv])
                scores.append(float(score_cp_iter))

            if not indices:
                # 如果策略构建失败，但走法和分数是有的，仍然可以返回（或者选择全返回None）
                # 这里为了安全，返回 None
                return None, value_red_view, best_move_internal

            # Softmax 计算概率
            arr = np.array(scores, dtype=np.float64) / 100.0 / max(temperature, 1e-6)
            arr -= arr.max()
            probs = np.exp(arr)
            probs /= probs.sum()

            for idx, prob in zip(indices, probs):
                policy[idx] += float(prob)

            return policy, value_red_view, best_move_internal