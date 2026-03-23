"""
简化中国象棋游戏逻辑

实现完整的中国象棋规则，包括：
- 棋盘表示（9x10）
- 所有棋子的移动规则
- 合法走法生成
- 胜负判断

坐标系统：
- x: 列 (0-8)
- y: 行 (0-9)
- 红方在下方 (y=0-4)，黑方在上方 (y=5-9)

棋子编码（FEN风格）：
- 大写=红方: R(车) N(马) B(象) A(仕) K(帅) C(炮) P(兵)
- 小写=黑方: r(车) n(马) b(象) a(仕) k(将) c(炮) p(卒)
"""

import numpy as np
import copy
import random

# 棋子类型索引映射（用于神经网络输入平面）
PIECE_TO_INDEX = {
    'P': 0, 'p': 0,  # 兵/卒
    'C': 1, 'c': 1,  # 炮
    'R': 2, 'r': 2,  # 车
    'N': 3, 'n': 3,  # 马
    'B': 4, 'b': 4,  # 象
    'A': 5, 'a': 5,  # 仕
    'K': 6, 'k': 6,  # 帅/将
}

# 初始棋盘 FEN
INIT_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"

BOARD_HEIGHT = 10
BOARD_WIDTH = 9


def create_action_labels():
    """
    创建所有可能走法的标签列表。
    走法格式: "x0y0x1y1"，其中x为列(0-8)，y为行(0-9)。
    """
    labels = []
    for y1 in range(BOARD_HEIGHT):
        for x1 in range(BOARD_WIDTH):
            # 直线走法（车/炮/兵/将等）
            dests = [(y1, x) for x in range(BOARD_WIDTH)] + \
                    [(y, x1) for y in range(BOARD_HEIGHT)]
            # 马的走法
            dests += [(y1 + dy, x1 + dx) for dy, dx in
                      [(-2, -1), (-1, -2), (-2, 1), (1, -2),
                       (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for y2, x2 in dests:
                if (y1, x1) != (y2, x2) and 0 <= y2 < BOARD_HEIGHT and 0 <= x2 < BOARD_WIDTH:
                    labels.append(f"{x1}{y1}{x2}{y2}")

    # 仕的斜线走法（红方）
    for move in ['3041', '5041', '3241', '5241',
                 '4130', '4150', '4132', '4152']:
        if move not in labels:
            labels.append(move)
    # 仕的斜线走法（黑方）
    for move in ['3948', '5948', '3748', '5748',
                 '4839', '4859', '4837', '4857']:
        if move not in labels:
            labels.append(move)

    # 象的斜线走法（红方）
    for move in ['2002', '2042', '6042', '6082',
                 '2402', '2442', '6442', '6482',
                 '0220', '4220', '4260', '8260',
                 '0224', '4224', '4264', '8264']:
        if move not in labels:
            labels.append(move)
    # 象的斜线走法（黑方）
    for move in ['2907', '2947', '6947', '6987',
                 '2507', '2547', '6547', '6587',
                 '0729', '4729', '4769', '8769',
                 '0725', '4725', '4765', '8765']:
        if move not in labels:
            labels.append(move)

    return labels


ACTION_LABELS = create_action_labels()
LABEL_TO_INDEX = {label: i for i, label in enumerate(ACTION_LABELS)}
NUM_ACTIONS = len(ACTION_LABELS)

_rng = random.Random(42)
# 14 piece types × 90 squares Zobrist table
_ZOBRIST_PIECES = [[_rng.getrandbits(64) for _ in range(90)] for _ in range(14)]
_ZOBRIST_SIDE = _rng.getrandbits(64)  # side to move

PIECE_TO_ZOBRIST_IDX = {
    'P': 0, 'C': 1, 'R': 2, 'N': 3, 'B': 4, 'A': 5, 'K': 6,
    'p': 7, 'c': 8, 'r': 9, 'n': 10, 'b': 11, 'a': 12, 'k': 13,
}


def flip_move(move):
    """翻转走法（用于黑方视角）"""
    x0, y0, x1, y1 = int(move[0]), int(move[1]), int(move[2]), int(move[3])
    return f"{8-x0}{9-y0}{8-x1}{9-y1}"


def flip_policy(policy):
    """翻转策略向量（红方视角 <-> 黑方视角）"""
    flipped = np.zeros_like(policy)
    for i, label in enumerate(ACTION_LABELS):
        flipped_label = flip_move(label)
        if flipped_label in LABEL_TO_INDEX:
            flipped[LABEL_TO_INDEX[flipped_label]] = policy[i]
    return flipped


class ChessGame:
    """
    中国象棋游戏类

    维护棋盘状态，提供走法生成、走子、胜负判断等功能。
    """

    def __init__(self, repetition_draw_threshold: int = 3):
        """
        Args:
            repetition_draw_threshold: 重复局面判和的阈值（≥ 3）。默认 3，即第三次出现
                同一局面时（若无长将/长捉）判和。训练时可调大（如 5 或 6）以减少短局
                重复判和，获得更丰富的训练信号。

        Raises:
            ValueError: 当 repetition_draw_threshold < 3 时抛出。
        """
        if repetition_draw_threshold < 3:
            raise ValueError(
                f"repetition_draw_threshold 必须 >= 3，得到 {repetition_draw_threshold}"
            )
        self.repetition_draw_threshold = repetition_draw_threshold
        self.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self.red_to_move = True
        self.winner = None  # 'red', 'black', 'draw', or None
        self.num_moves = 0
        self.pos_hash = 0
        self.pos_history = []
        self.move_history = []
        self.check_history = []
        self.chase_history = []
        self.terminate_reason = None

    def reset(self, fen=None):
        """重置棋盘到初始状态或指定FEN"""
        if fen is None:
            fen = INIT_FEN
        self.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self._load_fen(fen)
        self.red_to_move = True
        self.winner = None
        self.num_moves = 0
        self._init_hash()
        return self

    def _load_fen(self, fen):
        """从FEN字符串加载棋盘
        
        FEN的第一行对应棋盘最上方（y=9，黑方底线），
        最后一行对应棋盘最下方（y=0，红方底线）。
        """
        rows = fen.split('/')
        for i in range(BOARD_HEIGHT):
            y = BOARD_HEIGHT - 1 - i  # FEN行i -> 棋盘行y
            x = 0
            for ch in rows[i]:
                if ch.isdigit():
                    x += int(ch)
                else:
                    self.board[y][x] = ch
                    x += 1

    def get_fen(self):
        """获取当前棋盘的FEN字符串"""
        rows = []
        for i in range(BOARD_HEIGHT):
            y = BOARD_HEIGHT - 1 - i  # 从上到下
            row_str = ''
            empty = 0
            for x in range(BOARD_WIDTH):
                piece = self.board[y][x]
                if piece is None:
                    empty += 1
                else:
                    if empty > 0:
                        row_str += str(empty)
                        empty = 0
                    row_str += piece
            if empty > 0:
                row_str += str(empty)
            rows.append(row_str)
        return '/'.join(rows)

    def get_observation(self):
        """获取当前玩家视角的观察状态"""
        if self.red_to_move:
            return self.get_fen()
        else:
            return self._flip_fen(self.get_fen())

    def _flip_fen(self, fen):
        """翻转FEN（180度旋转 + 大小写互换）"""
        rows = fen.split('/')
        flipped_rows = []
        for row in reversed(rows):
            flipped_row = ''
            for ch in reversed(row):
                if ch.isalpha():
                    flipped_row += ch.swapcase()
                else:
                    flipped_row += ch
            flipped_rows.append(flipped_row)
        return '/'.join(flipped_rows)

    def is_red_piece(self, piece):
        """判断是否为红方棋子"""
        return piece is not None and piece.isupper()

    def is_black_piece(self, piece):
        """判断是否为黑方棋子"""
        return piece is not None and piece.islower()

    def is_own_piece(self, piece):
        """判断是否为当前方棋子"""
        if piece is None:
            return False
        if self.red_to_move:
            return piece.isupper()
        return piece.islower()

    def is_enemy_piece(self, piece):
        """判断是否为对方棋子"""
        if piece is None:
            return False
        if self.red_to_move:
            return piece.islower()
        return piece.isupper()

    def copy(self):
        """深拷贝当前游戏状态"""
        return copy.deepcopy(self)

    def get_legal_moves(self, side=None):
        """
        获取合法走法列表。

        Args:
            side: 'red'|'black'|None。None表示当前走子方。
        """
        if side is not None:
            old_red = self.red_to_move
            self.red_to_move = (side == 'red')
            moves = self._get_all_legal_moves()
            self.red_to_move = old_red
            return moves
        return self._get_all_legal_moves()

    def _get_all_legal_moves(self):
        """获取当前方所有合法走法（内部方法）"""
        moves = []
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                piece = self.board[y][x]
                if self.is_own_piece(piece):
                    piece_moves = self._get_piece_moves(x, y, piece)
                    for move in piece_moves:
                        if not self._move_leaves_king_in_check(move):
                            moves.append(move)
        return moves

    def _init_hash(self):
        """初始化Zobrist哈希和历史"""
        self.pos_hash = self._compute_hash()
        self.pos_history = [self.pos_hash]
        self.move_history = []
        self.check_history = []
        self.chase_history = []
        self.terminate_reason = None

    def reset_history(self):
        """重置历史（手动设置棋盘后调用）"""
        self._init_hash()

    def _compute_hash(self):
        """从当前棋盘状态计算完整的Zobrist哈希"""
        h = 0
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                piece = self.board[y][x]
                if piece is not None and piece in PIECE_TO_ZOBRIST_IDX:
                    sq = y * BOARD_WIDTH + x
                    pidx = PIECE_TO_ZOBRIST_IDX[piece]
                    h ^= _ZOBRIST_PIECES[pidx][sq]
        if not self.red_to_move:
            h ^= _ZOBRIST_SIDE
        return h

    def _zobrist_piece(self, piece, x, y):
        """计算一个棋子对哈希的贡献"""
        if piece is None or piece not in PIECE_TO_ZOBRIST_IDX:
            return 0
        sq = y * BOARD_WIDTH + x
        return _ZOBRIST_PIECES[PIECE_TO_ZOBRIST_IDX[piece]][sq]

    def is_in_check(self, for_red=None):
        """
        检查指定方是否处于被将状态。

        Args:
            for_red: True=检查红方, False=检查黑方, None=检查当前走子方

        Returns:
            bool
        """
        if for_red is None:
            for_red = self.red_to_move
        return self._is_in_check(for_red)

    def _detect_perpetual_chase(self):
        """
        长捉判负规则。

        在三次重复局面的循环内，若一方始终在捉对方的某一个（或多个）固定棋子，
        而另一方未始终如此，则捉子方判负。

        "捉"的定义（简化）：走棋后己方棋子直接攻击对方非将棋子。

        Returns:
            'red_loses' | 'black_loses' | None
        """
        current_hash = self.pos_hash
        history = self.pos_history

        # 找循环起始位置（与 _detect_perpetual_check 相同逻辑）
        cycle_start = None
        for i in range(len(history) - 1, -1, -1):
            if history[i] == current_hash:
                cycle_start = i
                break

        if cycle_start is None or not self.chase_history:
            return None

        cycle_chases = self.chase_history[cycle_start:]
        if not cycle_chases:
            return None

        # 确定循环起始时谁在走棋
        cycle_start_ply = cycle_start
        red_to_move_at_cycle_start = (cycle_start_ply % 2 == 0)

        # 按走棋方分类捉子集合
        red_chases = []
        black_chases = []
        for i, chased in enumerate(cycle_chases):
            mover_is_red = (i % 2 == 0) == red_to_move_at_cycle_start
            if mover_is_red:
                red_chases.append(chased)
            else:
                black_chases.append(chased)

        if not red_chases or not black_chases:
            return None

        # 红方是否始终在捉某个固定棋子（循环内所有红方走棋后均攻击同一棋子）
        red_common = red_chases[0]
        for c in red_chases[1:]:
            red_common = red_common & c
        red_perpetual = len(red_common) > 0

        # 黑方是否始终在捉某个固定棋子
        black_common = black_chases[0]
        for c in black_chases[1:]:
            black_common = black_common & c
        black_perpetual = len(black_common) > 0

        if red_perpetual and not black_perpetual:
            return 'red_loses'
        elif black_perpetual and not red_perpetual:
            return 'black_loses'
        else:
            return None

    def _get_chased_pieces(self, attacker_is_red):
        """
        返回被 attacker_is_red 方"捉"的对方非将棋子的坐标集合。

        "捉"的定义（简化）：攻击方可通过下一步直接吃掉对方非将棋子。
        通过枚举攻击方各棋子的伪合法走法来判断，避免误用飞将规则。

        Args:
            attacker_is_red: True 表示检查红方捉黑方棋子；False 反之。

        Returns:
            frozenset of (x, y) tuples
        """
        chased = set()
        # 临时切换走子方，以便正确生成攻击方棋子的走法
        old_red = self.red_to_move
        self.red_to_move = attacker_is_red
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                piece = self.board[y][x]
                if piece is None:
                    continue
                if piece.isupper() != attacker_is_red:
                    continue  # 非攻击方棋子
                for move in self._get_piece_moves(x, y, piece):
                    tx, ty = int(move[2]), int(move[3])
                    target = self.board[ty][tx]
                    # 吃子走法且目标不是将/帅
                    if target is not None and target.upper() != 'K':
                        chased.add((tx, ty))
        self.red_to_move = old_red
        return frozenset(chased)

    def _detect_perpetual_check(self):
        """
        检测长将。

        规则说明（简化版）：
        采用规则：重复三次局面，若一方连续将军维持循环则判该方负；
        双方均将或无法判定则和。简化：连续将军指循环内该方每次行棋后对方均处于被将状态。

        Returns:
            'red_loses' | 'black_loses' | 'draw' | None
        """
        current_hash = self.pos_hash
        history = self.pos_history
        if len(history) < 2:
            return None

        # 找最近一次出现相同局面的位置
        cycle_start = None
        for i in range(len(history) - 1, -1, -1):
            if history[i] == current_hash:
                cycle_start = i
                break

        if cycle_start is None:
            return None

        # 获取循环内的将军记录
        cycle_checks = self.check_history[cycle_start:]
        if not cycle_checks:
            return None

        # 确定循环起始时谁在走棋
        # 游戏从红方开始，偶数步（0,2,4...）红方走
        cycle_start_ply = cycle_start
        red_to_move_at_cycle_start = (cycle_start_ply % 2 == 0)

        # 按走棋方分类将军记录
        red_checks = []
        black_checks = []
        for i, check in enumerate(cycle_checks):
            mover_is_red = (i % 2 == 0) == red_to_move_at_cycle_start
            if mover_is_red:
                red_checks.append(check)
            else:
                black_checks.append(check)

        if not red_checks or not black_checks:
            return None

        red_perpetual = all(red_checks)
        black_perpetual = all(black_checks)

        if red_perpetual and not black_perpetual:
            return 'red_loses'
        elif black_perpetual and not red_perpetual:
            return 'black_loses'
        else:
            return 'draw'

    def _find_king(self, for_red):
        """查找将/帅的位置，返回 (x, y) 或 None"""
        king = 'K' if for_red else 'k'
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.board[y][x] == king:
                    return (x, y)
        return None

    def _is_attacked(self, x, y, for_red):
        """
        检查位置 (x, y) 是否被对方棋子攻击。

        Args:
            x, y: 要检查的位置
            for_red: True 表示检查红方的将/帅是否被黑方攻击
        """
        enemy_is_red = not for_red

        # 车攻击：直线，无遮挡
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            while 0 <= nx < BOARD_WIDTH and 0 <= ny < BOARD_HEIGHT:
                piece = self.board[ny][nx]
                if piece is not None:
                    if (enemy_is_red and piece == 'R') or (not enemy_is_red and piece == 'r'):
                        return True
                    break
                nx += dx
                ny += dy

        # 炮攻击：直线，隔一子
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            jumped = False
            while 0 <= nx < BOARD_WIDTH and 0 <= ny < BOARD_HEIGHT:
                piece = self.board[ny][nx]
                if not jumped:
                    if piece is not None:
                        jumped = True
                else:
                    if piece is not None:
                        if (enemy_is_red and piece == 'C') or (not enemy_is_red and piece == 'c'):
                            return True
                        break
                nx += dx
                ny += dy

        # 马攻击：反向推导哪些马能攻击 (x, y)
        for dx, dy, bx, by in [
            (-1, -2, 0, -1), (1, -2, 0, -1),
            (-2, -1, -1, 0), (-2, 1, -1, 0),
            (-1, 2, 0, 1), (1, 2, 0, 1),
            (2, -1, 1, 0), (2, 1, 1, 0),
        ]:
            nx, ny = x - dx, y - dy  # 马所在位置
            block_x, block_y = nx + bx, ny + by  # 蹩马腿位置
            if not (0 <= nx < BOARD_WIDTH and 0 <= ny < BOARD_HEIGHT):
                continue
            if not (0 <= block_x < BOARD_WIDTH and 0 <= block_y < BOARD_HEIGHT):
                continue
            if self.board[block_y][block_x] is not None:
                continue  # 蹩马腿，无法攻击
            piece = self.board[ny][nx]
            if (enemy_is_red and piece == 'N') or (not enemy_is_red and piece == 'n'):
                return True

        # 兵/卒攻击
        if for_red:
            # 黑卒攻击红方：黑卒向下走（y减小），过河后可横走
            # 黑卒在 (x, y+1) 可向下攻击 (x, y)（无论是否过河）
            # 黑卒在 (x±1, y) 且已过河（ny <= 4）可横向攻击 (x, y)
            for check_x, check_y in [(x, y + 1), (x + 1, y), (x - 1, y)]:
                if 0 <= check_x < BOARD_WIDTH and 0 <= check_y < BOARD_HEIGHT:
                    piece = self.board[check_y][check_x]
                    if piece == 'p':
                        if check_y == y + 1:  # 黑卒在上方，向下攻击
                            return True
                        elif check_y == y and check_y <= 4:  # 黑卒过河后横向攻击
                            return True
        else:
            # 红兵攻击黑方：红兵向上走（y增大），过河后可横走
            for check_x, check_y in [(x, y - 1), (x + 1, y), (x - 1, y)]:
                if 0 <= check_x < BOARD_WIDTH and 0 <= check_y < BOARD_HEIGHT:
                    piece = self.board[check_y][check_x]
                    if piece == 'P':
                        if check_y == y - 1:  # 红兵在下方，向上攻击
                            return True
                        elif check_y == y and check_y >= 5:  # 红兵过河后横向攻击
                            return True

        # 飞将：将帅同列且中间无棋子
        enemy_king = 'k' if for_red else 'K'
        for dy in [1, -1]:
            ny = y + dy
            while 0 <= ny < BOARD_HEIGHT:
                piece = self.board[ny][x]
                if piece is not None:
                    if piece == enemy_king:
                        return True
                    break
                ny += dy

        return False

    def _is_in_check(self, for_red):
        """检查指定方的将/帅是否处于被将状态"""
        king_pos = self._find_king(for_red)
        if king_pos is None:
            return True  # 将/帅不存在，视为被将
        return self._is_attacked(king_pos[0], king_pos[1], for_red)

    def _move_leaves_king_in_check(self, move):
        """
        检查执行走法后己方将/帅是否处于被将状态。
        通过临时修改棋盘并还原来避免深拷贝，提高性能。
        """
        x0, y0 = int(move[0]), int(move[1])
        x1, y1 = int(move[2]), int(move[3])

        # 临时执行走法
        captured = self.board[y1][x1]
        self.board[y1][x1] = self.board[y0][x0]
        self.board[y0][x0] = None

        # 检查己方是否被将
        in_check = self._is_in_check(self.red_to_move)

        # 撤销走法
        self.board[y0][x0] = self.board[y1][x1]
        self.board[y1][x1] = captured

        return in_check

    def _get_piece_moves(self, x, y, piece):
        """获取指定棋子的所有合法走法"""
        piece_type = piece.upper()
        if piece_type == 'R':
            return self._rook_moves(x, y)
        elif piece_type == 'N':
            return self._knight_moves(x, y)
        elif piece_type == 'B':
            return self._bishop_moves(x, y)
        elif piece_type == 'A':
            return self._advisor_moves(x, y)
        elif piece_type == 'K':
            return self._king_moves(x, y)
        elif piece_type == 'C':
            return self._cannon_moves(x, y)
        elif piece_type == 'P':
            return self._pawn_moves(x, y)
        return []

    def _make_move_str(self, x0, y0, x1, y1):
        """生成走法字符串"""
        return f"{x0}{y0}{x1}{y1}"

    def _can_move_to(self, x, y):
        """检查目标位置是否可以走（空位或对方棋子）"""
        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
            return False
        return not self.is_own_piece(self.board[y][x])

    def _rook_moves(self, x, y):
        """车的走法：直线移动，遇到棋子停止（可吃对方）"""
        moves = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            while 0 <= nx < BOARD_WIDTH and 0 <= ny < BOARD_HEIGHT:
                target = self.board[ny][nx]
                if target is None:
                    moves.append(self._make_move_str(x, y, nx, ny))
                elif self.is_enemy_piece(target):
                    moves.append(self._make_move_str(x, y, nx, ny))
                    break
                else:
                    break
                nx += dx
                ny += dy
        return moves

    def _knight_moves(self, x, y):
        """马的走法：日字形移动，有蹩马腿规则"""
        moves = []
        for dx, dy, bx, by in [
            (-1, -2, 0, -1), (1, -2, 0, -1),
            (-2, -1, -1, 0), (-2, 1, -1, 0),
            (-1, 2, 0, 1), (1, 2, 0, 1),
            (2, -1, 1, 0), (2, 1, 1, 0),
        ]:
            nx, ny = x + dx, y + dy
            block_x, block_y = x + bx, y + by
            if not (0 <= nx < BOARD_WIDTH and 0 <= ny < BOARD_HEIGHT):
                continue
            if not (0 <= block_x < BOARD_WIDTH and 0 <= block_y < BOARD_HEIGHT):
                continue
            if self.board[block_y][block_x] is not None:
                continue  # 蹩马腿
            if self._can_move_to(nx, ny):
                moves.append(self._make_move_str(x, y, nx, ny))
        return moves

    def _bishop_moves(self, x, y):
        """象的走法：斜走两格，不能过河，有塞象眼规则"""
        moves = []
        is_red = self.is_red_piece(self.board[y][x])
        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            nx, ny = x + dx, y + dy
            bx, by = x + dx // 2, y + dy // 2
            if not (0 <= nx < BOARD_WIDTH and 0 <= ny < BOARD_HEIGHT):
                continue
            # 红方象不能过河（y <= 4），黑方象不能过河（y >= 5）
            if is_red and ny > 4:
                continue
            if not is_red and ny < 5:
                continue
            if self.board[by][bx] is not None:
                continue  # 塞象眼
            if self._can_move_to(nx, ny):
                moves.append(self._make_move_str(x, y, nx, ny))
        return moves

    def _advisor_moves(self, x, y):
        """仕的走法：斜走一格，只能在九宫内"""
        moves = []
        is_red = self.is_red_piece(self.board[y][x])
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if not (3 <= nx <= 5):
                continue
            if is_red and not (0 <= ny <= 2):
                continue
            if not is_red and not (7 <= ny <= 9):
                continue
            if self._can_move_to(nx, ny):
                moves.append(self._make_move_str(x, y, nx, ny))
        return moves

    def _king_moves(self, x, y):
        """将/帅的走法：直走一格，只能在九宫内"""
        moves = []
        is_red = self.is_red_piece(self.board[y][x])
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if not (3 <= nx <= 5):
                continue
            if is_red and not (0 <= ny <= 2):
                continue
            if not is_red and not (7 <= ny <= 9):
                continue
            if self._can_move_to(nx, ny):
                moves.append(self._make_move_str(x, y, nx, ny))
        return moves

    def _cannon_moves(self, x, y):
        """炮的走法：直线移动（不吃子时不跳），隔一个棋子吃子"""
        moves = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            jumped = False
            while 0 <= nx < BOARD_WIDTH and 0 <= ny < BOARD_HEIGHT:
                target = self.board[ny][nx]
                if not jumped:
                    if target is None:
                        moves.append(self._make_move_str(x, y, nx, ny))
                    else:
                        jumped = True  # 找到炮架
                else:
                    if target is not None:
                        if self.is_enemy_piece(target):
                            moves.append(self._make_move_str(x, y, nx, ny))
                        break
                nx += dx
                ny += dy
        return moves

    def _pawn_moves(self, x, y):
        """兵/卒的走法：未过河只能前进，过河后可左右"""
        moves = []
        is_red = self.is_red_piece(self.board[y][x])

        if is_red:
            # 红兵向上走（y增大方向为黑方）
            forward = (0, 1)
            crossed_river = y >= 5
        else:
            # 黑卒向下走
            forward = (0, -1)
            crossed_river = y <= 4

        # 前进
        nx, ny = x + forward[0], y + forward[1]
        if 0 <= ny < BOARD_HEIGHT and self._can_move_to(nx, ny):
            moves.append(self._make_move_str(x, y, nx, ny))

        # 过河后可以横走
        if crossed_river:
            for dx in [-1, 1]:
                nx = x + dx
                if 0 <= nx < BOARD_WIDTH and self._can_move_to(nx, y):
                    moves.append(self._make_move_str(x, y, nx, y))

        return moves

    def step(self, action):
        """
        执行走法

        Args:
            action: 走法字符串 "x0y0x1y1"

        Returns:
            tuple: (obs, reward, done, info)
            - obs: 当前玩家视角的观察
            - reward: 走子方的即时奖励 (+1/-1/0)
            - done: 游戏是否结束
            - info: {'reason': terminate_reason, 'winner': winner}
        """
        x0, y0 = int(action[0]), int(action[1])
        x1, y1 = int(action[2]), int(action[3])

        # 增量更新哈希：移除起始位置棋子
        piece_moving = self.board[y0][x0]
        captured = self.board[y1][x1]

        if piece_moving is not None:
            self.pos_hash ^= self._zobrist_piece(piece_moving, x0, y0)
        if captured is not None:
            self.pos_hash ^= self._zobrist_piece(captured, x1, y1)

        # 执行走法
        self.board[y1][x1] = piece_moving
        self.board[y0][x0] = None
        self.num_moves += 1

        # 增量更新哈希：添加目标位置棋子
        if piece_moving is not None:
            self.pos_hash ^= self._zobrist_piece(piece_moving, x1, y1)

        # 记录是否将军（走棋后对方是否被将）
        gave_check = self._is_in_check(not self.red_to_move)
        self.check_history.append(gave_check)
        # 记录是否捉子（走棋后己方直接攻击对方非将棋子）
        gave_chase = self._get_chased_pieces(self.red_to_move)
        self.chase_history.append(gave_chase)
        self.move_history.append(action)

        # 检查是否吃掉了对方的将/帅
        if captured is not None and captured.upper() == 'K':
            self.winner = 'red' if self.red_to_move else 'black'
            self.terminate_reason = 'king_captured'

        # 检查将帅是否面对面
        if self.winner is None:
            self._check_king_face()

        # 切换走子方（更新哈希中的走子方标志）
        self.pos_hash ^= _ZOBRIST_SIDE
        self.red_to_move = not self.red_to_move

        # 检测重复局面（count 为当前局面在历史中已出现的次数；
        # threshold=3 时第3次出现触发，即 count >= threshold-1）
        if self.winner is None:
            count = self.pos_history.count(self.pos_hash)
            if count >= (self.repetition_draw_threshold - 1):  # 将成为第 threshold 次出现
                perp_result = self._detect_perpetual_check()
                if perp_result == 'red_loses':
                    self.winner = 'black'
                    self.terminate_reason = 'perpetual_check'
                elif perp_result == 'black_loses':
                    self.winner = 'red'
                    self.terminate_reason = 'perpetual_check'
                else:
                    chase_result = self._detect_perpetual_chase()
                    if chase_result == 'red_loses':
                        self.winner = 'black'
                        self.terminate_reason = 'perpetual_chase'
                    elif chase_result == 'black_loses':
                        self.winner = 'red'
                        self.terminate_reason = 'perpetual_chase'
                    else:
                        self.winner = 'draw'
                        self.terminate_reason = 'repetition'

        self.pos_history.append(self.pos_hash)

        # 检查对方是否无子可走（将死或困毙）
        if self.winner is None:
            legal = self.get_legal_moves()
            if len(legal) == 0:
                if self._is_in_check(self.red_to_move):
                    # 被将死
                    self.winner = 'black' if self.red_to_move else 'red'
                    self.terminate_reason = 'checkmate'
                else:
                    # 困毙（无子可走但未被将）
                    self.winner = 'black' if self.red_to_move else 'red'
                    self.terminate_reason = 'stalemate'

        # 计算奖励（从走子方视角）
        if self.done:
            if self.winner == 'draw':
                reward = 0.0
            else:
                # red_to_move已切换，刚走棋的是 not self.red_to_move
                just_moved_red = not self.red_to_move
                if (self.winner == 'red' and just_moved_red) or \
                   (self.winner == 'black' and not just_moved_red):
                    reward = 1.0
                else:
                    reward = -1.0
        else:
            reward = 0.0

        obs = self.get_observation()
        info = {'reason': self.terminate_reason, 'winner': self.winner}
        return obs, reward, self.done, info

    def _check_king_face(self):
        """检查将帅是否面对面（飞将）"""
        red_king = None
        black_king = None
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.board[y][x] == 'K':
                    red_king = (x, y)
                elif self.board[y][x] == 'k':
                    black_king = (x, y)

        if red_king is None or black_king is None:
            return

        if red_king[0] != black_king[0]:
            return

        # 检查两王之间是否有棋子
        x = red_king[0]
        min_y = min(red_king[1], black_king[1])
        max_y = max(red_king[1], black_king[1])
        for y in range(min_y + 1, max_y):
            if self.board[y][x] is not None:
                return

        # 将帅面对面，当前走子方输
        self.winner = 'black' if self.red_to_move else 'red'

    @property
    def done(self):
        return self.winner is not None

    def to_planes(self):
        """
        将棋盘转换为16通道特征平面
        
        通道结构:
        - 0-13: 棋子位置 (原有)
        - 14: 走子方标记 (红方=1, 黑方=0)
        - 15: 步数标记 (归一化, 用于判断开局/残局)
        
        Returns:
            numpy array, shape (16, 10, 9)
        """
        # 1. 获取当前视角的 FEN (如果是黑方，get_observation 已经翻转了棋盘)
        fen = self.get_observation()
        
        # 2. 初始化 16 通道平面
        planes = np.zeros((16, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        
        # 3. 填充前 14 通道 (棋子位置)
        rows = fen.split('/')
        for y in range(len(rows)):
            x = 0
            for ch in rows[y]:
                if ch.isdigit():
                    x += int(ch)
                elif ch.isalpha():
                    # 计算 Piece Index
                    idx = PIECE_TO_INDEX[ch] + (7 if ch.islower() else 0)
                    planes[idx][y][x] = 1
                    x += 1
        
        # 4. 填充第 15 通道 (走子方)
        # 因为 get_observation 已经翻转了视角，模型看到的永远是"红方(自己)在下方"。
        # 但我们需要告诉模型：实际上现在是谁走？
        # 如果当前是红方走，填 1；黑方走，填 0。
        if self.red_to_move:
            planes[14, :, :] = 1.0
            
        # 5. 填充第 16 通道 (步数/回合数)
        # 归一化：除以 200.0 (假设一局棋通常 200 步以内)
        # 这有助于模型区分开局(0.0)、中局(0.5)和残局(1.0+)
        planes[15, :, :] = self.num_moves / 200.0

        return planes

    def print_board(self):
        """打印棋盘到控制台"""
        print("  0 1 2 3 4 5 6 7 8")
        for y in range(BOARD_HEIGHT - 1, -1, -1):
            row = f"{y} "
            for x in range(BOARD_WIDTH):
                piece = self.board[y][x]
                row += (piece if piece else '.') + ' '
            print(row)
        print()


def fen_to_planes(fen, red_to_move=True, num_moves=0):
    """
    将FEN字符串转换为16通道特征平面 (外部调用接口)
    """
    planes = np.zeros((16, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    rows = fen.split('/')
    for y in range(len(rows)):
        x = 0
        for ch in rows[y]:
            if ch.isdigit():
                x += int(ch)
            elif ch.isalpha():
                idx = PIECE_TO_INDEX[ch] + (7 if ch.islower() else 0)
                planes[idx][y][x] = 1
                x += 1
    
    # 填充扩展通道
    if red_to_move:
        planes[14, :, :] = 1.0
    planes[15, :, :] = num_moves / 200.0
    
    return planes