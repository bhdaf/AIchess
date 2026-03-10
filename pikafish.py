"""
Pikafish UCI 引擎封装

Pikafish 是一个基于 Stockfish 的标准 UCI 中国象棋引擎。
本模块提供：
  - 走法格式转换（内部格式 x0y0x1y1 ↔ Pikafish UCI 格式 <file><rank><file><rank>）
  - FEN 格式转换（棋盘 FEN ↔ 完整 UCI FEN）
  - PikafishEngine 类：子进程 UCI 通信封装

协议说明（来自 https://github.com/official-pikafish/Pikafish ）：
  - 协议：标准 UCI（Universal Chess Interface），非 UCCI
  - 启动后先发送 `uci`，等待 `uciok`
  - 每局开始发送 `ucinewgame`，再 `isready`，等待 `readyok`
  - 设置棋盘：`position fen <fen> moves <m1> <m2> ...`
  - 搜索：`go movetime <ms>` 或 `go depth <n>`
  - 引擎响应 `bestmove <move>` 表示最优走法

走法坐标格式（Pikafish）：
  - 文件（列）：a~i 对应 x=0~8
  - 排（行）：  0~9 对应 y=0~9（0 为红方底线，9 为黑方底线）
  - 示例：b2e2  ←→  内部 "1224"（x0=1,y0=2 → x1=4,y1=2）

FEN 格式：
  - Pikafish 完整 FEN = 棋盘部分 + " w - - 0 1"（红方先行）
                     或 棋盘部分 + " b - - 0 1"（黑方先行）
  - 棋盘部分与本项目内部的 get_fen() 完全相同
"""

import subprocess
import threading
import time
import os


# 文件字母：a=列0, b=列1, ..., i=列8
_FILE_LETTERS = 'abcdefghi'


def internal_move_to_uci(move: str) -> str:
    """将内部走法格式 'x0y0x1y1' 转换为 Pikafish UCI 格式 '<file><rank><file><rank>'。

    Args:
        move: 4 字符字符串，如 '1224'

    Returns:
        UCI 格式走法，如 'b2e2'

    Examples:
        >>> internal_move_to_uci('1242')
        'b2e2'
        >>> internal_move_to_uci('4041')
        'e0e1'
    """
    x0, y0, x1, y1 = int(move[0]), int(move[1]), int(move[2]), int(move[3])
    return f"{_FILE_LETTERS[x0]}{y0}{_FILE_LETTERS[x1]}{y1}"


def uci_move_to_internal(uci_move: str) -> str:
    """将 Pikafish UCI 格式走法 '<file><rank><file><rank>' 转换为内部格式 'x0y0x1y1'。

    Args:
        uci_move: UCI 格式走法，如 'b2e2'

    Returns:
        内部格式走法，如 '1224'

    Examples:
        >>> uci_move_to_internal('b2e2')
        '1242'
        >>> uci_move_to_internal('e0e1')
        '4041'
    """
    x0 = _FILE_LETTERS.index(uci_move[0])
    y0 = int(uci_move[1])
    x1 = _FILE_LETTERS.index(uci_move[2])
    y1 = int(uci_move[3])
    return f"{x0}{y0}{x1}{y1}"


def board_fen_to_uci_fen(board_fen: str, red_to_move: bool = True,
                          halfmove_clock: int = 0, fullmove_number: int = 1) -> str:
    """将棋盘 FEN 字符串转换为 Pikafish UCI 完整 FEN。

    Pikafish 的完整 FEN 格式为：
        <棋盘部分> <走子方> - - <半步数> <全步数>

    Args:
        board_fen: 棋盘部分 FEN，如 'rnbakabnr/9/1c5c1/...'
        red_to_move: 是否红方先行（True=红方，False=黑方）
        halfmove_clock: 半步计数（用于50步规则，象棋中通常为0）
        fullmove_number: 全步数（从1开始）

    Returns:
        完整 UCI FEN 字符串，如 'rnbakabnr/9/... w - - 0 1'

    Examples:
        >>> board_fen_to_uci_fen('rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR')
        'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'
    """
    side = 'w' if red_to_move else 'b'
    return f"{board_fen} {side} - - {halfmove_clock} {fullmove_number}"


class PikafishEngine:
    """Pikafish UCI 引擎子进程封装。

    管理与 Pikafish 可执行文件的通信，实现标准 UCI 协议。

    Args:
        engine_path: Pikafish 可执行文件的路径
        movetime_ms: 每步思考时间（毫秒），默认 100ms
        depth: 搜索深度（若指定则忽略 movetime_ms），默认 None
        threads: 引擎线程数，默认 1
        hash_mb: 哈希表大小（MB），默认 16

    Examples:
        >>> engine = PikafishEngine('/path/to/pikafish', movetime_ms=100)
        >>> engine.start()
        >>> move = engine.get_best_move(game)
        >>> engine.quit()
    """

    def __init__(self, engine_path: str, movetime_ms: int = 100,
                 depth: int = None, threads: int = 1, hash_mb: int = 16):
        self.engine_path = engine_path
        self.movetime_ms = movetime_ms
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self._process = None
        self._lock = threading.Lock()

    def start(self):
        """启动引擎子进程并完成 UCI 握手。

        Raises:
            FileNotFoundError: 若引擎文件不存在
            RuntimeError: 若 UCI 握手失败
        """
        if not os.path.isfile(self.engine_path):
            raise FileNotFoundError(
                f"Pikafish 引擎文件不存在: {self.engine_path}\n"
                "请从 https://github.com/official-pikafish/Pikafish/releases 下载引擎"
            )
        self._process = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # UCI 握手
        self._send('uci')
        self._wait_for('uciok', timeout=10)
        # 设置参数
        self._send(f'setoption name Threads value {self.threads}')
        self._send(f'setoption name Hash value {self.hash_mb}')
        # 初始化完毕
        self._send('isready')
        self._wait_for('readyok', timeout=10)

    def new_game(self):
        """通知引擎开始新一局对弈，并等待引擎就绪。"""
        self._send('ucinewgame')
        self._send('isready')
        self._wait_for('readyok', timeout=10)

    def get_best_move(self, game) -> str:
        """根据当前棋盘状态获取引擎最佳走法。

        Args:
            game: ChessGame 实例，用于获取当前 FEN 和走子方

        Returns:
            内部格式走法字符串（如 '1224'），即引擎建议的最佳走法

        Raises:
            RuntimeError: 若引擎未返回有效走法
        """
        uci_fen = board_fen_to_uci_fen(game.get_fen(), game.red_to_move,
                                        halfmove_clock=0,
                                        fullmove_number=game.num_moves // 2 + 1)
        self._send(f'position fen {uci_fen}')

        if self.depth is not None:
            self._send(f'go depth {self.depth}')
        else:
            self._send(f'go movetime {self.movetime_ms}')

        uci_move = self._read_bestmove(timeout=max(self.movetime_ms / 1000.0 * 3 + 5, 15))
        if uci_move is None or uci_move == '(none)':
            raise RuntimeError("引擎未返回有效走法（可能已将死或和棋）")
        return uci_move_to_internal(uci_move)

    def quit(self):
        """优雅关闭引擎子进程。"""
        if self._process is not None:
            try:
                self._send('quit')
                self._process.wait(timeout=3)
            except Exception:
                self._process.kill()
            finally:
                self._process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.quit()

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _send(self, command: str):
        """向引擎发送一条 UCI 指令。"""
        if self._process is None:
            raise RuntimeError("引擎未启动，请先调用 start()")
        self._process.stdin.write(command + '\n')
        self._process.stdin.flush()

    def _readline(self, timeout: float = 5.0) -> str:
        """从引擎读取一行输出（带超时）。"""
        line = ''
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process.stdout.readable():
                line = self._process.stdout.readline()
                if line:
                    return line.rstrip('\n')
            time.sleep(0.001)
        return ''

    def _wait_for(self, keyword: str, timeout: float = 10.0):
        """等待引擎输出包含指定关键字的行。

        Raises:
            RuntimeError: 超时未收到期望响应
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._readline(timeout=max(deadline - time.time(), 0.1))
            if keyword in line:
                return
        raise RuntimeError(
            f"等待引擎响应 '{keyword}' 超时（{timeout}s）。"
            "请确认引擎路径正确且引擎可正常执行。"
        )

    def _read_bestmove(self, timeout: float = 15.0) -> str:
        """读取引擎 bestmove 响应，返回 UCI 格式走法字符串。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._readline(timeout=max(deadline - time.time(), 0.1))
            if line.startswith('bestmove'):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
        return None
