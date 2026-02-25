"""
动作编码模块

基于"起始格+走法类型"的结构化编码。
59个走法平面 × 90个格子 = 5310个动作索引。

走法平面：
- 0-8: 右移1..9步
- 9-17: 左移1..9步
- 18-26: 上移1..9步
- 27-35: 下移1..9步
- 36-43: 8个马跳方向
- 44-47: 4个象斜跳（×2步）
- 48-51: 4个仕斜走
- 52-55: 4个将直走（冗余，已被0-35覆盖，保留用于向后兼容）
- 56-58: 兵走法（前/左/右，几何编码中已被0-35覆盖）
"""

NUM_PLANES = 59
NUM_ACTION_ENCODING = 90 * NUM_PLANES  # 5310

# 马的8个跳法 (dx, dy)
KNIGHT_MOVES = [(-1, -2), (1, -2), (-2, -1), (-2, 1),
                (-1, 2), (1, 2), (2, -1), (2, 1)]

# 象的4个斜跳 (dx, dy)
BISHOP_MOVES = [(-2, -2), (-2, 2), (2, -2), (2, 2)]

# 仕的4个斜走 (dx, dy)
ADVISOR_MOVES = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# 将的4个直走 (dx, dy)
KING_MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def action_index(from_sq, plane):
    """计算动作索引。from_sq = y*9+x"""
    return from_sq * NUM_PLANES + plane


def encode_move(move_str, is_red=True):
    """
    将走法字符串"x0y0x1y1"编码为动作索引。

    优先级（几何编码，不依赖棋子类型）：
    1. 直线滑动（车/炮/兵直走/将直走）: 平面0-35
    2. 马跳: 平面36-43
    3. 象斜跳（步长2）: 平面44-47
    4. 仕/将斜走（步长1）: 平面48-51

    Returns:
        int or None: 动作索引，无法编码时返回None
    """
    if len(move_str) != 4:
        return None
    x0, y0, x1, y1 = int(move_str[0]), int(move_str[1]), int(move_str[2]), int(move_str[3])
    dx = x1 - x0
    dy = y1 - y0

    if dx == 0 and dy == 0:
        return None

    from_sq = y0 * 9 + x0

    # 直线滑动
    if dx == 0 and dy > 0:
        plane = 18 + (dy - 1)  # 上移
        return action_index(from_sq, plane)
    elif dx == 0 and dy < 0:
        plane = 27 + (-dy - 1)  # 下移
        return action_index(from_sq, plane)
    elif dy == 0 and dx > 0:
        plane = 0 + (dx - 1)  # 右移
        return action_index(from_sq, plane)
    elif dy == 0 and dx < 0:
        plane = 9 + (-dx - 1)  # 左移
        return action_index(from_sq, plane)

    # 马跳
    for i, (kx, ky) in enumerate(KNIGHT_MOVES):
        if dx == kx and dy == ky:
            return action_index(from_sq, 36 + i)

    # 象斜跳
    if abs(dx) == 2 and abs(dy) == 2:
        for i, (bx, by) in enumerate(BISHOP_MOVES):
            if dx == bx and dy == by:
                return action_index(from_sq, 44 + i)

    # 仕斜走（步长1，非直线）
    if abs(dx) == 1 and abs(dy) == 1:
        for i, (ax, ay) in enumerate(ADVISOR_MOVES):
            if dx == ax and dy == ay:
                return action_index(from_sq, 48 + i)

    return None


def decode_action(action_idx):
    """
    将动作索引解码为走法坐标 (from_x, from_y, to_x, to_y)。

    Returns:
        tuple (from_x, from_y, to_x, to_y) 或 None（越界时）
    """
    if action_idx < 0 or action_idx >= NUM_ACTION_ENCODING:
        return None

    from_sq = action_idx // NUM_PLANES
    plane = action_idx % NUM_PLANES

    from_x = from_sq % 9
    from_y = from_sq // 9

    if from_x < 0 or from_x >= 9 or from_y < 0 or from_y >= 10:
        return None

    # 计算 (dx, dy)
    if 0 <= plane <= 8:
        dx, dy = plane + 1, 0  # 右移
    elif 9 <= plane <= 17:
        dx, dy = -(plane - 9 + 1), 0  # 左移
    elif 18 <= plane <= 26:
        dx, dy = 0, plane - 18 + 1  # 上移
    elif 27 <= plane <= 35:
        dx, dy = 0, -(plane - 27 + 1)  # 下移
    elif 36 <= plane <= 43:
        dx, dy = KNIGHT_MOVES[plane - 36]
    elif 44 <= plane <= 47:
        dx, dy = BISHOP_MOVES[plane - 44]
    elif 48 <= plane <= 51:
        dx, dy = ADVISOR_MOVES[plane - 48]
    elif 52 <= plane <= 55:
        dx, dy = KING_MOVES[plane - 52]
    elif plane == 56:
        # 兵前进：无法确定方向（依赖棋子颜色），返回None
        return None
    elif plane == 57:
        dx, dy = -1, 0  # 兵左
    elif plane == 58:
        dx, dy = 1, 0   # 兵右
    else:
        return None

    to_x = from_x + dx
    to_y = from_y + dy

    if not (0 <= to_x < 9 and 0 <= to_y < 10):
        return None

    return (from_x, from_y, to_x, to_y)


def legal_action_indices(game):
    """
    获取当前局面的所有合法走法对应的动作索引列表。

    Args:
        game: ChessGame实例

    Returns:
        List[int]: 合法走法的动作索引
    """
    moves = game.get_legal_moves()
    indices = []
    for move in moves:
        idx = encode_move(move)
        if idx is not None:
            indices.append(idx)
    return indices
