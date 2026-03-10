"""
UCI (Universal Chess Interface) 引擎封装

提供与任意支持 UCI/UCCI 协议的象棋引擎（如 Pikafish）通信的最小封装。
协议流程：
  1. 启动子进程
  2. 发送 "uci"，等待 "uciok"
  3. 发送 "isready"，等待 "readyok"
  4. 每局开始时发送 "ucinewgame"
  5. 每步之前发送 "position fen <FEN>"
  6. 发送 "go movetime <ms>"，等待 "bestmove <move>"
  7. 退出时发送 "quit"
"""

import subprocess
import threading
import time
import logging

logger = logging.getLogger(__name__)


class UCIEngine:
    """
    通用 UCI 引擎子进程封装。

    Args:
        engine_path (str): 引擎可执行文件的路径。
        init_timeout (float): 等待 ``uciok`` / ``readyok`` 的超时秒数。
        move_timeout (float): 等待 ``bestmove`` 的超时秒数（额外余量）。

    Example::

        with UCIEngine("/path/to/pikafish") as engine:
            engine.new_game()
            engine.set_position("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/"
                                 "P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1")
            best = engine.go_movetime(100)   # 思考 100 ms
            print(best)                      # e.g. "h2e2"
    """

    def __init__(self, engine_path: str, init_timeout: float = 10.0,
                 move_timeout: float = 5.0):
        self.engine_path = engine_path
        self.init_timeout = init_timeout
        self.move_timeout = move_timeout
        self._proc: subprocess.Popen | None = None
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._stop_reader = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动引擎子进程并完成 UCI 握手。"""
        self._proc = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._stop_reader = False
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True
        )
        self._reader_thread.start()

        self._send("uci")
        self._wait_for("uciok", timeout=self.init_timeout)

        self._send("isready")
        self._wait_for("readyok", timeout=self.init_timeout)

        logger.debug("UCI engine started: %s", self.engine_path)

    def quit(self) -> None:
        """向引擎发送 quit 并等待子进程退出。"""
        if self._proc and self._proc.poll() is None:
            try:
                self._send("quit")
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._stop_reader = True
        if self._reader_thread:
            self._reader_thread.join(timeout=2)
        self._proc = None
        logger.debug("UCI engine stopped.")

    def __enter__(self) -> "UCIEngine":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.quit()

    # ------------------------------------------------------------------
    # 高层 API
    # ------------------------------------------------------------------

    def new_game(self) -> None:
        """通知引擎开始新局（清空内部缓存）。"""
        self._send("ucinewgame")
        # 重新确认就绪，保证引擎处理完 ucinewgame
        self._send("isready")
        self._wait_for("readyok", timeout=self.init_timeout)

    def set_position(self, fen: str) -> None:
        """
        设置棋盘位置。

        Args:
            fen: 完整 UCI FEN，例如
                 ``"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"``
        """
        self._send(f"position fen {fen}")

    def go_movetime(self, movetime_ms: int) -> str | None:
        """
        以固定思考时间搜索并返回最佳走法。

        Args:
            movetime_ms: 思考时间（毫秒）。

        Returns:
            引擎返回的走法字符串（ICCS 格式，如 ``"h2e2"``），或在超时 / 错误
            时返回 ``None``。
        """
        with self._lock:
            self._lines.clear()
        self._send(f"go movetime {movetime_ms}")
        timeout = movetime_ms / 1000.0 + self.move_timeout
        return self._wait_for_bestmove(timeout=timeout)

    def go_depth(self, depth: int) -> str | None:
        """
        以固定深度搜索并返回最佳走法。

        Args:
            depth: 搜索深度。

        Returns:
            走法字符串或 ``None``。
        """
        with self._lock:
            self._lines.clear()
        self._send(f"go depth {depth}")
        return self._wait_for_bestmove(timeout=60.0)

    def set_option(self, name: str, value: str) -> None:
        """设置引擎选项，例如 UCI_Elo / Skill Level 等。"""
        self._send(f"setoption name {name} value {value}")

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _send(self, cmd: str) -> None:
        """向引擎写入一行命令。"""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("UCI engine is not running.")
        logger.debug(">>> %s", cmd)
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _read_loop(self) -> None:
        """在后台线程中持续读取引擎输出。"""
        while not self._stop_reader:
            try:
                line = self._proc.stdout.readline()
                if not line:
                    break
                line = line.rstrip()
                logger.debug("<<< %s", line)
                with self._lock:
                    self._lines.append(line)
            except Exception:
                break

    def _wait_for(self, keyword: str, timeout: float) -> None:
        """阻塞直到输出中出现包含 keyword 的行，或超时后抛出异常。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                for line in self._lines:
                    if keyword in line:
                        return
            time.sleep(0.01)
        raise TimeoutError(
            f"UCI engine did not respond with '{keyword}' within {timeout}s"
        )

    def _wait_for_bestmove(self, timeout: float) -> str | None:
        """阻塞等待 bestmove 行并解析走法。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                for line in self._lines:
                    if line.startswith("bestmove"):
                        parts = line.split()
                        move = parts[1] if len(parts) >= 2 else None
                        # "bestmove (none)" 表示引擎无合法走法
                        return None if move in (None, "(none)") else move
            time.sleep(0.01)
        logger.warning("Timed out waiting for bestmove (%.1fs)", timeout)
        return None
