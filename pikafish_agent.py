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

    agent = PikafishAgent("/path/to/pikafish", movetime_ms=100)
    agent.start()
    agent.new_game()
    move = agent.get_move(game)   # 返回内部格式走法，如 "7072"
    agent.quit()

或作为上下文管理器::

    with PikafishAgent("/path/to/pikafish") as agent:
        agent.new_game()
        move = agent.get_move(game)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


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

class PikafishAgent(BaseAgent):
    """
    Pikafish（或任意兼容 UCI 协议的中国象棋引擎）Agent。

    该 Agent 在每步时：
      1. 调用 :func:`game_to_uci_fen` 将当前局面转换为 UCI FEN
      2. 通过 :class:`~AIchess.uci.UCIEngine` 发送给引擎
      3. 引擎返回 ICCS 格式走法，经 :func:`uci_to_internal` 转换为内部格式

    Args:
        engine_path (str): 引擎可执行文件路径（如 ``"/usr/local/bin/pikafish"``）。
        movetime_ms (int): 每步思考时间（毫秒），默认 100 ms。
        depth (int | None): 固定深度搜索；若指定则忽略 ``movetime_ms``。
        options (dict | None): 在握手后通过 ``setoption`` 发送给引擎的选项字典，
            例如 ``{"UCI_Elo": "1500", "Skill Level": "5"}``。
        init_timeout (float): UCI 握手超时秒数。
        move_timeout (float): 等待 bestmove 的额外超时秒数。

    Example::

        with PikafishAgent("/path/to/pikafish", movetime_ms=50) as agent:
            game = ChessGame().reset()
            agent.new_game()
            move = agent.get_move(game)   # "7072"
            game.step(move)
    """

    def __init__(
        self,
        engine_path: str,
        movetime_ms: int = 100,
        depth: Optional[int] = None,
        options: Optional[dict] = None,
        init_timeout: float = 10.0,
        move_timeout: float = 5.0,
    ) -> None:
        from .uci import UCIEngine  # 延迟导入，降低无引擎时的依赖成本

        self.engine_path = engine_path
        self.movetime_ms = movetime_ms
        self.depth = depth
        self.options = options or {}
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
        for name, value in self.options.items():
            self._engine.set_option(name, str(value))

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
            uci_move = self._engine.go_movetime(self.movetime_ms)

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
