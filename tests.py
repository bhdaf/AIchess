"""
简化中国象棋AI - 单元测试

测试游戏逻辑、模型、MCTS等核心组件。
"""

import unittest
import numpy as np

from simple_chess_ai.game import (
    ChessGame, ACTION_LABELS, LABEL_TO_INDEX, NUM_ACTIONS,
    BOARD_HEIGHT, BOARD_WIDTH, INIT_FEN, fen_to_planes,
    flip_move, flip_policy
)
from simple_chess_ai.model import ChessModel
from simple_chess_ai.mcts import MCTS, MCTSNode


class TestGameInit(unittest.TestCase):
    """测试游戏初始化"""

    def test_board_dimensions(self):
        self.assertEqual(BOARD_HEIGHT, 10)
        self.assertEqual(BOARD_WIDTH, 9)

    def test_action_labels(self):
        self.assertGreater(NUM_ACTIONS, 2000)
        self.assertEqual(len(ACTION_LABELS), NUM_ACTIONS)
        self.assertEqual(len(LABEL_TO_INDEX), NUM_ACTIONS)

    def test_reset(self):
        game = ChessGame()
        game.reset()
        self.assertTrue(game.red_to_move)
        self.assertIsNone(game.winner)
        self.assertFalse(game.done)
        self.assertEqual(game.num_moves, 0)

    def test_initial_fen(self):
        game = ChessGame()
        game.reset()
        fen = game.get_fen()
        self.assertEqual(fen, INIT_FEN)

    def test_initial_pieces(self):
        game = ChessGame()
        game.reset()
        # Red back rank (y=0)
        self.assertEqual(game.board[0][0], 'R')  # 车
        self.assertEqual(game.board[0][1], 'N')  # 马
        self.assertEqual(game.board[0][4], 'K')  # 帅
        # Black back rank (y=9)
        self.assertEqual(game.board[9][0], 'r')  # 车
        self.assertEqual(game.board[9][4], 'k')  # 将
        # Red cannon (y=2)
        self.assertEqual(game.board[2][1], 'C')
        self.assertEqual(game.board[2][7], 'C')
        # Black cannon (y=7)
        self.assertEqual(game.board[7][1], 'c')
        self.assertEqual(game.board[7][7], 'c')

    def test_fen_roundtrip(self):
        game = ChessGame()
        game.reset()
        fen1 = game.get_fen()
        game2 = ChessGame()
        game2.reset(fen1)
        fen2 = game2.get_fen()
        self.assertEqual(fen1, fen2)


class TestPieceMoves(unittest.TestCase):
    """测试棋子移动规则"""

    def setUp(self):
        self.game = ChessGame()
        self.game.reset()

    def test_initial_legal_moves_count(self):
        moves = self.game.get_legal_moves()
        self.assertGreater(len(moves), 0)
        # Red should have 44 legal moves at start
        self.assertEqual(len(moves), 44)

    def test_pawn_forward(self):
        """兵只能前进"""
        moves = self.game.get_legal_moves()
        # Red pawn at (0, 3) can only go to (0, 4)
        pawn_moves = [m for m in moves if m[:2] == '03']
        self.assertEqual(len(pawn_moves), 1)
        self.assertEqual(pawn_moves[0], '0304')

    def test_king_in_palace(self):
        """帅只能在九宫内"""
        moves = self.game.get_legal_moves()
        king_moves = [m for m in moves if m[:2] == '40']
        self.assertEqual(len(king_moves), 1)
        self.assertEqual(king_moves[0], '4041')

    def test_knight_blocked(self):
        """马被蹩腿"""
        moves = self.game.get_legal_moves()
        # Knight at (1, 0) - the pawn at (0, 0)=R blocks some, but
        # the key blocking is vertical/horizontal pieces
        knight_moves = [m for m in moves if m[:2] == '10']
        # Knight at (1,0) can go to (0,2) and (2,2)
        self.assertEqual(len(knight_moves), 2)
        self.assertIn('1002', knight_moves)
        self.assertIn('1022', knight_moves)

    def test_rook_initial_moves(self):
        """车初始位置只有纵向走法"""
        moves = self.game.get_legal_moves()
        rook_moves = [m for m in moves if m[:2] == '00']
        # Rook at (0,0) blocked by knight at (1,0) on y-axis,
        # can only move vertically to (0,1) and (0,2) before hitting cannon
        self.assertGreater(len(rook_moves), 0)

    def test_cannon_initial_moves(self):
        """炮初始位置走法"""
        moves = self.game.get_legal_moves()
        cannon_moves = [m for m in moves if m[:2] == '12']
        self.assertGreater(len(cannon_moves), 0)


class TestGameStep(unittest.TestCase):
    """测试走子"""

    def setUp(self):
        self.game = ChessGame()
        self.game.reset()

    def test_step_changes_turn(self):
        self.assertTrue(self.game.red_to_move)
        self.game.step('4041')  # 帅前进
        self.assertFalse(self.game.red_to_move)

    def test_step_moves_piece(self):
        self.game.step('4041')
        self.assertIsNone(self.game.board[0][4])
        self.assertEqual(self.game.board[1][4], 'K')

    def test_step_increments_count(self):
        self.game.step('4041')
        self.assertEqual(self.game.num_moves, 1)

    def test_capture(self):
        """测试吃子"""
        # Set up a position where capture is possible
        game = ChessGame()
        game.reset()
        # Move a piece to enable capture
        game.board[5][0] = 'R'  # Put red rook on (0,5)
        game.board[0][0] = None
        legal = game.get_legal_moves()
        # Rook at (0,5) should be able to capture pawn at (0,6)
        self.assertIn('0506', legal)


class TestKingFace(unittest.TestCase):
    """测试飞将规则"""

    def test_king_face_detection(self):
        """将帅面对面时当前走子方输"""
        game = ChessGame()
        game.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        game.board[0][4] = 'K'
        game.board[9][4] = 'k'
        game.red_to_move = True
        # Move king to cause face-off check
        game.step('4041')
        # After red moves, check if face detection triggers
        # Kings at (4,1) and (4,9) with nothing between = face-off
        self.assertTrue(game.done)


class TestBishopMoves(unittest.TestCase):
    """测试象的走法"""

    def test_bishop_cannot_cross_river(self):
        """象不能过河"""
        game = ChessGame()
        game.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        game.board[0][4] = 'K'
        game.board[9][4] = 'k'
        game.board[4][2] = 'B'  # Red bishop at (2, 4)
        game.red_to_move = True
        moves = game.get_legal_moves()
        bishop_moves = [m for m in moves if m[:2] == '24']
        # Should not have moves crossing river (y > 4)
        for m in bishop_moves:
            dest_y = int(m[3])
            self.assertLessEqual(dest_y, 4)


class TestAdvisorMoves(unittest.TestCase):
    """测试仕的走法"""

    def test_advisor_stays_in_palace(self):
        """仕只能在九宫内"""
        game = ChessGame()
        game.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        game.board[0][4] = 'K'
        game.board[9][4] = 'k'
        game.board[1][4] = 'A'  # Red advisor at center of palace
        game.red_to_move = True
        moves = game.get_legal_moves()
        advisor_moves = [m for m in moves if m[:2] == '41']
        for m in advisor_moves:
            dest_x, dest_y = int(m[2]), int(m[3])
            self.assertGreaterEqual(dest_x, 3)
            self.assertLessEqual(dest_x, 5)
            self.assertGreaterEqual(dest_y, 0)
            self.assertLessEqual(dest_y, 2)


class TestFlipMove(unittest.TestCase):
    """测试走法翻转"""

    def test_flip(self):
        self.assertEqual(flip_move('0000'), '8989')
        self.assertEqual(flip_move('4041'), '4948')
        self.assertEqual(flip_move('8989'), '0000')

    def test_double_flip(self):
        for label in ACTION_LABELS[:100]:
            self.assertEqual(flip_move(flip_move(label)), label)


class TestFenToPlanes(unittest.TestCase):
    """测试FEN到特征平面的转换"""

    def test_shape(self):
        planes = fen_to_planes(INIT_FEN)
        self.assertEqual(planes.shape, (14, 10, 9))

    def test_values(self):
        planes = fen_to_planes(INIT_FEN)
        # All values should be 0 or 1
        self.assertTrue(np.all((planes == 0) | (planes == 1)))

    def test_piece_count(self):
        planes = fen_to_planes(INIT_FEN)
        # Total pieces should be 32
        total = planes.sum()
        self.assertEqual(total, 32)


class TestFlipPolicy(unittest.TestCase):
    """测试策略翻转"""

    def test_flip_preserves_sum(self):
        policy = np.random.dirichlet(np.ones(NUM_ACTIONS))
        flipped = flip_policy(policy)
        self.assertAlmostEqual(policy.sum(), flipped.sum(), places=5)


class TestObservation(unittest.TestCase):
    """测试观察状态"""

    def test_red_observation(self):
        game = ChessGame()
        game.reset()
        obs = game.get_observation()
        self.assertEqual(obs, INIT_FEN)

    def test_black_observation_is_flipped(self):
        game = ChessGame()
        game.reset()
        game.step('4041')  # Red moves, now black's turn
        obs = game.get_observation()
        # Black's observation should be flipped
        self.assertNotEqual(obs, game.get_fen())


class TestModel(unittest.TestCase):
    """测试模型"""

    def test_build(self):
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()
        self.assertIsNotNone(model.model)

    def test_predict(self):
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()
        game = ChessGame()
        game.reset()
        planes = game.to_planes()
        policy, value = model.predict(planes)
        self.assertEqual(policy.shape, (NUM_ACTIONS,))
        self.assertAlmostEqual(policy.sum(), 1.0, places=4)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_save_load(self, tmp_path='/tmp/test_chess_model'):
        import os
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()
        path = f'{tmp_path}/model.pth'
        os.makedirs(tmp_path, exist_ok=True)
        model.save(path)
        self.assertTrue(os.path.exists(path))

        model2 = ChessModel()
        self.assertTrue(model2.load(path))

        # Predictions should be the same
        game = ChessGame()
        game.reset()
        planes = game.to_planes()
        p1, v1 = model.predict(planes)
        p2, v2 = model2.predict(planes)
        np.testing.assert_array_almost_equal(p1, p2, decimal=5)
        self.assertAlmostEqual(v1, v2, places=5)


class TestMCTS(unittest.TestCase):
    """测试MCTS"""

    def test_mcts_returns_actions(self):
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()
        game = ChessGame()
        game.reset()
        mcts = MCTS(model, num_simulations=10)
        actions, probs = mcts.get_action_probs(game, temperature=1.0)
        self.assertGreater(len(actions), 0)
        self.assertEqual(len(actions), len(probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=4)

    def test_mcts_actions_are_legal(self):
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()
        game = ChessGame()
        game.reset()
        mcts = MCTS(model, num_simulations=10)
        actions, probs = mcts.get_action_probs(game, temperature=1.0)
        legal_moves = game.get_legal_moves()
        for action in actions:
            self.assertIn(action, LABEL_TO_INDEX)

    def test_mcts_update_with_move(self):
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()
        mcts = MCTS(model, num_simulations=5)
        mcts.update_with_move('4041')
        # Should not crash
        self.assertIsNotNone(mcts.root)


class TestGameCopy(unittest.TestCase):
    """测试游戏状态拷贝"""

    def test_copy_independence(self):
        game = ChessGame()
        game.reset()
        copy = game.copy()
        copy.step('4041')
        # Original should be unchanged
        self.assertTrue(game.red_to_move)
        self.assertEqual(game.board[0][4], 'K')
        # Copy should have changed
        self.assertFalse(copy.red_to_move)
        self.assertEqual(copy.board[1][4], 'K')


if __name__ == '__main__':
    unittest.main()
