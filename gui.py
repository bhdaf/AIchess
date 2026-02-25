"""
中国象棋图形界面

使用Pygame实现的人机对弈界面，特点：
- 使用程序绘制棋盘和棋子（无需外部图片资源）
- 支持鼠标点击选择和移动棋子
- 显示合法走法提示
- AI使用MCTS搜索走法

用法:
    python -m simple_chess_ai.gui [--model_path path/to/model.pth]
"""

import os
import sys
import math
import threading
import argparse

import numpy as np

from simple_chess_ai.game import (
    ChessGame, BOARD_HEIGHT, BOARD_WIDTH, flip_move
)
from simple_chess_ai.model import ChessModel
from simple_chess_ai.mcts import MCTS
from simple_chess_ai.export import save_board_screenshot

# 默认模型路径
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_model', 'model.pth')

# 颜色定义
COLOR_BG = (240, 217, 181)       # 棋盘背景色
COLOR_LINE = (0, 0, 0)           # 线条颜色
COLOR_RED = (200, 30, 30)        # 红方棋子颜色
COLOR_BLACK = (30, 30, 30)       # 黑方棋子颜色
COLOR_PIECE_BG = (255, 235, 200) # 棋子背景色
COLOR_SELECT = (0, 180, 0)       # 选中高亮色
COLOR_HINT = (100, 200, 100)     # 走法提示色
COLOR_LAST_MOVE = (200, 200, 50) # 上一步标记色
COLOR_TEXT = (80, 50, 30)        # 文字颜色
COLOR_BUTTON = (180, 140, 100)   # 按钮颜色
COLOR_BUTTON_TEXT = (255, 255, 255)  # 按钮文字颜色
COLOR_STATUS_BG = (60, 40, 20)   # 状态栏背景色
COLOR_STATUS_TEXT = (255, 255, 255)  # 状态栏文字色
COLOR_RIVER = (200, 230, 255)    # 河界颜色

# 棋子中文名称
PIECE_NAMES_CN = {
    'R': '車', 'N': '馬', 'B': '相', 'A': '仕', 'K': '帥', 'C': '炮', 'P': '兵',
    'r': '車', 'n': '馬', 'b': '象', 'a': '士', 'k': '將', 'c': '砲', 'p': '卒',
}

# 棋子英文名称（无中文字体时使用）
PIECE_NAMES_EN = {
    'R': 'R', 'N': 'N', 'B': 'B', 'A': 'A', 'K': 'K', 'C': 'C', 'P': 'P',
    'r': 'R', 'n': 'N', 'b': 'B', 'a': 'A', 'k': 'K', 'c': 'C', 'p': 'P',
}

# 布局常量
CELL_SIZE = 60       # 格子大小
MARGIN = 50          # 棋盘边距
PIECE_RADIUS = 26    # 棋子半径
INFO_WIDTH = 200     # 右侧信息栏宽度


def _try_load_chinese_font(pygame, size):
    """尝试加载支持中文的字体，失败则返回默认字体"""
    # 尝试常见的中文字体名称
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'Noto Sans SC', 'PingFang SC',
        'Heiti SC', 'STHeiti', 'WenQuanYi Zen Hei',
        'AR PL UMing CN', 'AR PL UKai CN',
    ]
    for font_name in chinese_fonts:
        try:
            font = pygame.font.SysFont(font_name, size)
            # 测试渲染中文和ASCII，比较结果来判断是否真正支持中文
            test_cn = font.render('車', True, (0, 0, 0))
            test_en = font.render('?', True, (0, 0, 0))
            # 如果中文字符和问号渲染宽度相同，说明中文字符被fallback为方块
            if test_cn.get_width() > test_en.get_width():
                return font, True
        except Exception:
            continue
    return pygame.font.Font(None, size), False


def board_to_pixel(x, y):
    """棋盘坐标 -> 像素坐标（y坐标翻转，让红方在下方）"""
    px = MARGIN + x * CELL_SIZE
    py = MARGIN + (BOARD_HEIGHT - 1 - y) * CELL_SIZE
    return px, py


def pixel_to_board(px, py):
    """像素坐标 -> 棋盘坐标"""
    x = round((px - MARGIN) / CELL_SIZE)
    y = BOARD_HEIGHT - 1 - round((py - MARGIN) / CELL_SIZE)
    if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
        return x, y
    return None, None


class ChessGUI:
    """
    中国象棋图形界面

    Args:
        model_path: 模型文件路径
        num_simulations: AI搜索模拟次数
        human_color: 人类方颜色 ('red' 或 'black')
    """

    def __init__(self, model_path=None, num_simulations=200, human_color='red',
                 screenshot_dir=None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.num_simulations = num_simulations
        self.human_color = human_color
        self.screenshot_dir = screenshot_dir

        self.game = ChessGame()
        self.game.reset()

        self.model = None
        self.mcts = None

        self.selected = None      # 选中的棋子坐标 (x, y)
        self.legal_targets = []   # 选中棋子的合法目标
        self.last_move = None     # 上一步走法
        self.ai_thinking = False  # AI是否在思考
        self.status_text = ""     # 状态栏文字
        self.game_over = False
        self.has_chinese_font = False  # 是否有中文字体
        self.font = None
        self.small_font = None
        self.piece_font = None

    def load_model(self):
        """加载AI模型"""
        self.model = ChessModel()
        if os.path.exists(self.model_path):
            self.model.load(self.model_path)
            self.status_text = "Model loaded"
        else:
            self.model.build()
            self.status_text = "No model found, using random model"
        self.mcts = MCTS(self.model, num_simulations=self.num_simulations)

    def run(self):
        """启动游戏界面"""
        try:
            import pygame
        except ImportError:
            print("错误: 需要安装pygame库")
            print("请运行: pip install pygame")
            sys.exit(1)

        pygame.init()

        # 初始化字体
        self.piece_font, self.has_chinese_font = _try_load_chinese_font(pygame, 28)
        if self.has_chinese_font:
            self.font, _ = _try_load_chinese_font(pygame, 20)
            self.small_font, _ = _try_load_chinese_font(pygame, 16)
        else:
            self.font = pygame.font.Font(None, 22)
            self.small_font = pygame.font.Font(None, 18)

        board_width = MARGIN * 2 + (BOARD_WIDTH - 1) * CELL_SIZE
        board_height = MARGIN * 2 + (BOARD_HEIGHT - 1) * CELL_SIZE
        window_width = board_width + INFO_WIDTH
        window_height = board_height

        screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Chinese Chess AI")

        # 加载模型
        self.status_text = "Loading model..."
        self.load_model()

        clock = pygame.time.Clock()
        running = True

        # 如果人类执黑，AI先走
        if self.human_color == 'black':
            self._start_ai_move()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self._reset_game()
                    elif event.key == pygame.K_s:
                        # 保存棋盘截图
                        out = save_board_screenshot(screen, save_dir=self.screenshot_dir)
                        if out:
                            self.status_text = f"Screenshot saved: {os.path.basename(out)}"
                            print(f"棋盘截图已保存: {out}")
                        else:
                            self.status_text = "Screenshot failed"

            self._draw(screen, pygame)
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    def _handle_click(self, pos):
        """处理鼠标点击"""
        if self.ai_thinking or self.game_over:
            # 游戏结束时点击重置
            if self.game_over:
                self._reset_game()
            return

        # 检查是否点击了重新开始按钮
        board_width = MARGIN * 2 + (BOARD_WIDTH - 1) * CELL_SIZE
        btn_x = board_width + 20
        btn_y = MARGIN * 2 + (BOARD_HEIGHT - 1) * CELL_SIZE - 50
        btn_w = INFO_WIDTH - 40
        btn_h = 40
        if btn_x <= pos[0] <= btn_x + btn_w and btn_y <= pos[1] <= btn_y + btn_h:
            self._reset_game()
            return

        bx, by = pixel_to_board(pos[0], pos[1])
        if bx is None:
            self.selected = None
            self.legal_targets = []
            return

        is_human_turn = (self.game.red_to_move and self.human_color == 'red') or \
                        (not self.game.red_to_move and self.human_color == 'black')

        if not is_human_turn:
            return

        piece = self.game.board[by][bx]

        if self.selected is not None:
            # 已选中棋子，尝试移动
            sx, sy = self.selected
            move = f"{sx}{sy}{bx}{by}"
            if (bx, by) in self.legal_targets:
                self._make_human_move(move)
                return
            elif piece is not None and self.game.is_own_piece(piece):
                # 选择另一个己方棋子
                self.selected = (bx, by)
                self._update_legal_targets()
                return

            self.selected = None
            self.legal_targets = []
        else:
            # 选择棋子
            if piece is not None and self.game.is_own_piece(piece):
                self.selected = (bx, by)
                self._update_legal_targets()

    def _update_legal_targets(self):
        """更新选中棋子的合法目标"""
        if self.selected is None:
            self.legal_targets = []
            return
        sx, sy = self.selected
        all_moves = self.game.get_legal_moves()
        self.legal_targets = []
        for move in all_moves:
            x0, y0, x1, y1 = int(move[0]), int(move[1]), int(move[2]), int(move[3])
            if x0 == sx and y0 == sy:
                self.legal_targets.append((x1, y1))

    def _make_human_move(self, move):
        """执行人类走法"""
        self.game.step(move)
        self.last_move = move
        self.selected = None
        self.legal_targets = []

        if self.game.done:
            self._handle_game_over()
        else:
            self._start_ai_move()

    def _start_ai_move(self):
        """启动AI思考（在后台线程）"""
        self.ai_thinking = True
        self.status_text = "AI thinking..."
        thread = threading.Thread(target=self._ai_move_thread, daemon=True)
        thread.start()

    def _ai_move_thread(self):
        """AI走法线程"""
        try:
            actions, probs = self.mcts.get_action_probs(
                self.game, temperature=0.1
            )

            if not actions:
                self.ai_thinking = False
                self.status_text = "AI has no legal moves"
                self._handle_game_over()
                return

            # 选择最佳走法
            best_idx = np.argmax(probs)
            action = actions[best_idx]

            # 执行走法
            if not self.game.red_to_move:
                actual_action = flip_move(action)
            else:
                actual_action = action

            self.game.step(actual_action)
            self.mcts.update_with_move(action)
            self.last_move = actual_action

            if self.game.done:
                self._handle_game_over()
            else:
                self.status_text = f"AI moved: {actual_action}"
        except Exception as e:
            self.status_text = f"AI error: {e}"
        finally:
            self.ai_thinking = False

    def _handle_game_over(self):
        """处理游戏结束"""
        self.game_over = True
        if self.game.winner == 'red':
            self.status_text = "Red wins! Click or press R to restart"
        elif self.game.winner == 'black':
            self.status_text = "Black wins! Click or press R to restart"
        else:
            self.status_text = "Draw! Click or press R to restart"

    def _reset_game(self):
        """重置游戏"""
        self.game = ChessGame()
        self.game.reset()
        self.mcts = MCTS(self.model, num_simulations=self.num_simulations)
        self.selected = None
        self.legal_targets = []
        self.last_move = None
        self.game_over = False
        self.ai_thinking = False
        self.status_text = "New game started"

        if self.human_color == 'black':
            self._start_ai_move()

    def _draw(self, screen, pygame):
        """绘制整个界面"""
        board_width = MARGIN * 2 + (BOARD_WIDTH - 1) * CELL_SIZE
        board_height = MARGIN * 2 + (BOARD_HEIGHT - 1) * CELL_SIZE

        # 背景
        screen.fill(COLOR_BG)

        # 绘制棋盘
        self._draw_board(screen, pygame)

        # 绘制标记（上一步、选中、合法目标）
        self._draw_markers(screen, pygame)

        # 绘制棋子
        self._draw_pieces(screen, pygame)

        # 绘制右侧信息栏
        self._draw_info_panel(screen, pygame, board_width, board_height)

    def _draw_board(self, screen, pygame):
        """绘制棋盘"""
        # 绘制河界背景
        river_top = MARGIN + (BOARD_HEIGHT - 1 - 5) * CELL_SIZE
        river_bottom = MARGIN + (BOARD_HEIGHT - 1 - 4) * CELL_SIZE
        river_rect = pygame.Rect(
            MARGIN - CELL_SIZE // 4, river_top,
            (BOARD_WIDTH - 1) * CELL_SIZE + CELL_SIZE // 2,
            river_bottom - river_top
        )
        pygame.draw.rect(screen, COLOR_RIVER, river_rect)

        # 横线
        for i in range(BOARD_HEIGHT):
            y = MARGIN + i * CELL_SIZE
            pygame.draw.line(screen, COLOR_LINE,
                             (MARGIN, y),
                             (MARGIN + (BOARD_WIDTH - 1) * CELL_SIZE, y), 1)

        # 竖线（上下半部分分开画）
        for i in range(BOARD_WIDTH):
            x = MARGIN + i * CELL_SIZE
            # 上半部分
            pygame.draw.line(screen, COLOR_LINE,
                             (x, MARGIN),
                             (x, MARGIN + 4 * CELL_SIZE), 1)
            # 下半部分
            pygame.draw.line(screen, COLOR_LINE,
                             (x, MARGIN + 5 * CELL_SIZE),
                             (x, MARGIN + 9 * CELL_SIZE), 1)

        # 边框竖线连接
        pygame.draw.line(screen, COLOR_LINE,
                         (MARGIN, MARGIN + 4 * CELL_SIZE),
                         (MARGIN, MARGIN + 5 * CELL_SIZE), 1)
        pygame.draw.line(screen, COLOR_LINE,
                         (MARGIN + 8 * CELL_SIZE, MARGIN + 4 * CELL_SIZE),
                         (MARGIN + 8 * CELL_SIZE, MARGIN + 5 * CELL_SIZE), 1)

        # 九宫斜线（红方 - 下方）
        px1, py1 = board_to_pixel(3, 0)
        px2, py2 = board_to_pixel(5, 2)
        pygame.draw.line(screen, COLOR_LINE, (px1, py1), (px2, py2), 1)
        px1, py1 = board_to_pixel(5, 0)
        px2, py2 = board_to_pixel(3, 2)
        pygame.draw.line(screen, COLOR_LINE, (px1, py1), (px2, py2), 1)

        # 九宫斜线（黑方 - 上方）
        px1, py1 = board_to_pixel(3, 7)
        px2, py2 = board_to_pixel(5, 9)
        pygame.draw.line(screen, COLOR_LINE, (px1, py1), (px2, py2), 1)
        px1, py1 = board_to_pixel(5, 7)
        px2, py2 = board_to_pixel(3, 9)
        pygame.draw.line(screen, COLOR_LINE, (px1, py1), (px2, py2), 1)

        # 河界文字
        river_y = MARGIN + int(4.5 * CELL_SIZE)
        if self.has_chinese_font:
            text_left = self.piece_font.render("楚  河", True, COLOR_TEXT)
            text_right = self.piece_font.render("漢  界", True, COLOR_TEXT)
        else:
            text_left = self.piece_font.render("Chu He", True, COLOR_TEXT)
            text_right = self.piece_font.render("Han Jie", True, COLOR_TEXT)

        screen.blit(text_left, (MARGIN + CELL_SIZE, river_y - 14))
        screen.blit(text_right, (MARGIN + 5 * CELL_SIZE, river_y - 14))

    def _draw_markers(self, screen, pygame):
        """绘制标记"""
        # 上一步标记
        if self.last_move:
            x0, y0 = int(self.last_move[0]), int(self.last_move[1])
            x1, y1 = int(self.last_move[2]), int(self.last_move[3])
            for bx, by in [(x0, y0), (x1, y1)]:
                px, py = board_to_pixel(bx, by)
                pygame.draw.circle(screen, COLOR_LAST_MOVE, (px, py),
                                   PIECE_RADIUS + 4, 3)

        # 选中标记
        if self.selected:
            px, py = board_to_pixel(*self.selected)
            pygame.draw.circle(screen, COLOR_SELECT, (px, py),
                               PIECE_RADIUS + 4, 3)

        # 合法目标
        for tx, ty in self.legal_targets:
            px, py = board_to_pixel(tx, ty)
            if self.game.board[ty][tx] is not None:
                # 有棋子的位置画圆环
                pygame.draw.circle(screen, COLOR_HINT, (px, py),
                                   PIECE_RADIUS + 2, 2)
            else:
                # 空位画小圆点
                pygame.draw.circle(screen, COLOR_HINT, (px, py), 6)

    def _draw_pieces(self, screen, pygame):
        """绘制棋子"""
        piece_names = PIECE_NAMES_CN if self.has_chinese_font else PIECE_NAMES_EN

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                piece = self.game.board[y][x]
                if piece is None:
                    continue

                px, py = board_to_pixel(x, y)
                is_red = piece.isupper()

                # 棋子背景圆
                pygame.draw.circle(screen, COLOR_PIECE_BG, (px, py), PIECE_RADIUS)
                # 棋子边框
                border_color = COLOR_RED if is_red else COLOR_BLACK
                pygame.draw.circle(screen, border_color, (px, py), PIECE_RADIUS, 2)
                pygame.draw.circle(screen, border_color, (px, py), PIECE_RADIUS - 3, 1)

                # 棋子文字
                name = piece_names.get(piece, piece)
                text_color = COLOR_RED if is_red else COLOR_BLACK
                text_surface = self.piece_font.render(name, True, text_color)
                text_rect = text_surface.get_rect(center=(px, py))
                screen.blit(text_surface, text_rect)

    def _draw_info_panel(self, screen, pygame, board_width, board_height):
        """绘制右侧信息栏"""
        panel_x = board_width + 10
        panel_width = INFO_WIDTH - 20
        cn = self.has_chinese_font

        y = MARGIN

        # 标题
        title_text = "中国象棋AI" if cn else "Chinese Chess AI"
        title = self.font.render(title_text, True, COLOR_TEXT)
        screen.blit(title, (panel_x, y))
        y += 35

        # 当前走棋方
        if cn:
            turn_text = "红方走" if self.game.red_to_move else "黑方走"
        else:
            turn_text = "Red's turn" if self.game.red_to_move else "Black's turn"
        turn_color = COLOR_RED if self.game.red_to_move else COLOR_BLACK
        turn_surface = self.font.render(turn_text, True, turn_color)
        screen.blit(turn_surface, (panel_x, y))
        y += 30

        # 步数
        label = "Moves" if not cn else "\u6b65\u6570"
        moves_text = self.small_font.render(f"{label}: {self.game.num_moves}", True, COLOR_TEXT)
        screen.blit(moves_text, (panel_x, y))
        y += 25

        # AI思考状态
        if self.ai_thinking:
            think_str = "AI\u601d\u8003\u4e2d..." if cn else "AI thinking..."
            think_text = self.font.render(think_str, True, (0, 100, 200))
            screen.blit(think_text, (panel_x, y))
        y += 35

        # 操作提示
        y = board_height - 180
        if cn:
            hint_strs = ["\u64cd\u4f5c\u8bf4\u660e:", "\u70b9\u51fb\u9009\u62e9\u68cb\u5b50",
                         "\u70b9\u51fb\u76ee\u6807\u4f4d\u7f6e\u8d70\u68cb",
                         "\u6309R\u952e\u91cd\u65b0\u5f00\u59cb",
                         "\u6309S\u952e\u4fdd\u5b58\u622a\u56fe"]
        else:
            hint_strs = ["Instructions:", "Click to select piece",
                         "Click target to move", "Press R to restart",
                         "Press S to screenshot"]
        for s in hint_strs:
            screen.blit(self.small_font.render(s, True, COLOR_TEXT), (panel_x, y))
            y += 22

        # 重新开始按钮
        y = board_height - 60
        btn_rect = pygame.Rect(panel_x, y, panel_width, 40)
        pygame.draw.rect(screen, COLOR_BUTTON, btn_rect, border_radius=5)
        btn_label = "\u91cd\u65b0\u5f00\u59cb" if cn else "Restart"
        btn_text = self.font.render(btn_label, True, COLOR_BUTTON_TEXT)
        btn_text_rect = btn_text.get_rect(center=btn_rect.center)
        screen.blit(btn_text, btn_text_rect)

        # 状态栏（底部叠加）
        status_y = board_height - 25
        status_rect = pygame.Rect(0, status_y, board_width + INFO_WIDTH, 25)
        pygame.draw.rect(screen, COLOR_STATUS_BG, status_rect)
        if self.status_text:
            status_surface = self.small_font.render(self.status_text, True, COLOR_STATUS_TEXT)
            screen.blit(status_surface, (10, status_y + 4))


def main():
    parser = argparse.ArgumentParser(description='简化中国象棋AI - 图形界面')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型文件路径')
    parser.add_argument('--num_simulations', type=int, default=200,
                        help='AI搜索模拟次数 (默认: 200)')
    parser.add_argument('--human_color', type=str, default='red',
                        choices=['red', 'black'],
                        help='人类执哪方 (默认: red)')
    parser.add_argument('--screenshot_dir', type=str, default=None,
                        help='截图保存目录（默认: simple_chess_ai/runs/screenshots/）')

    args = parser.parse_args()
    gui = ChessGUI(
        model_path=args.model_path,
        num_simulations=args.num_simulations,
        human_color=args.human_color,
        screenshot_dir=args.screenshot_dir,
    )
    gui.run()


if __name__ == '__main__':
    main()
