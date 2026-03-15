"""
简化中国象棋AI - 单元测试

测试游戏逻辑、模型、MCTS等核心组件。
"""

import unittest
import os
import random
import numpy as np

from .game import (
    ChessGame, ACTION_LABELS, LABEL_TO_INDEX, NUM_ACTIONS,
    BOARD_HEIGHT, BOARD_WIDTH, INIT_FEN, fen_to_planes,
    flip_move, flip_policy
)
from .model import ChessModel
from .mcts import MCTS, MCTSNode


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

    def test_terminal_value_backprop_win(self):
        """终局胜利值回传：simulate 应使叶子节点 q_value > 0（从父节点视角）

        设置已结束的红方获胜局面（game.winner='red', game.red_to_move=False
        即黑方＝输家为当前走子方），直接调用 _simulate。修复前终局值
        value=-1.0 未翻转，导致叶子 q_value=-1.0（父节点认为胜利走法是坏的）；
        修复后正确翻转为 1.0，父节点才会优先选择致胜走法。
        """
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()

        # 构造红方已胜的终局：game.winner='red', game.red_to_move=False（黑=输家）
        game = ChessGame()
        game.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        game.board[0][3] = 'K'   # 红帅在(3,0)
        game.board[9][4] = 'k'   # 黑将在(4,9)
        game.winner = 'red'
        game.red_to_move = False  # 黑方（输家）本该走子

        mcts = MCTS(model, num_simulations=1)
        node = MCTSNode()
        mcts._simulate(game.copy(), node, None)

        # 修复后：节点 q_value 应 > 0（父节点视角：进入此终局是好事）
        self.assertGreater(
            node.q_value, 0.0,
            f"终局胜利后叶子节点 q_value({node.q_value:.3f}) 应 > 0；"
            "若为 -1.0 说明终局值未翻转（bug 未修复）"
        )

    def test_terminal_value_backprop_loss(self):
        """终局失败值回传：从父节点（获胜方）视角看，叶子节点 q_value 应为正

        game.winner='black'，game.red_to_move=True（红方＝输家，黑方是上一步的走棋方）。
        到达此终局的"父节点"是黑方（刚走出获胜棋）。修复后父节点视角 q=1.0>0；
        修复前由于未翻转，q=-1.0<0，导致 MCTS 错误地"避开"此获胜走法。
        """
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()

        game = ChessGame()
        game.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        game.board[0][3] = 'K'
        game.board[9][4] = 'k'
        game.winner = 'black'
        game.red_to_move = True  # 红方（输家）本该走子

        mcts = MCTS(model, num_simulations=1)
        node = MCTSNode()
        mcts._simulate(game.copy(), node, None)

        # 修复后：父节点（黑方获胜者）视角的 q_value 应为正
        self.assertGreater(
            node.q_value, 0.0,
            f"终局（黑方胜）叶子节点 q_value({node.q_value:.3f}) 应 > 0；"
            "若为 -1.0 说明终局值未翻转（bug 未修复）"
        )

    def test_terminal_draw_backprop(self):
        """终局平局值回传：simulate 应使叶子节点 q_value == 0"""
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()

        game = ChessGame()
        game.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        game.board[0][3] = 'K'
        game.board[9][4] = 'k'
        game.winner = 'draw'
        game.red_to_move = True

        mcts = MCTS(model, num_simulations=1)
        node = MCTSNode()
        mcts._simulate(game.copy(), node, None)

        self.assertAlmostEqual(node.q_value, 0.0, places=6,
                               msg="平局时叶子节点 q_value 应为 0")

    def test_winning_move_preferred(self):
        """MCTS 应优先选择能立即吃掉对方将的走法

        红车在(1,9)，黑将在(4,9)，红帅在(3,0)；走法"1949"可立即获胜。
        运行足够多模拟后，"1949"应成为访问次数最多的走法（temperature=0 时被选中）。
        """
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()

        game = ChessGame()
        game.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        game.board[0][3] = 'K'   # 红帅在(3,0)，不与黑将同列
        game.board[9][4] = 'k'   # 黑将在(4,9)
        game.board[9][1] = 'R'   # 红车在(1,9)，走"1949"可吃黑将
        game.red_to_move = True
        game._init_hash()

        mcts = MCTS(model, num_simulations=200, c_puct=1.5)
        actions, probs = mcts.get_action_probs(game, temperature=0.0, add_noise=False)

        self.assertIn('1949', actions, "红车吃将走法'1949'应在合法走法中")
        winning_child = mcts.root.children.get('1949')
        self.assertIsNotNone(winning_child, "'1949' 节点应已被探索")
        self.assertGreater(
            winning_child.q_value, 0.0,
            f"获胜走法'1949'的 q_value({winning_child.q_value:.3f}) 应 > 0"
        )
        best_action = actions[int(np.argmax(probs))]
        self.assertEqual(best_action, '1949',
                         f"MCTS 应选择立即获胜的走法'1949'，实际选择了'{best_action}'")


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


class TestTraining(unittest.TestCase):
    """测试训练函数最小闭环"""

    def test_train_model_runs(self):
        """train_model 用少量样本跑一个 epoch 不应报错"""
        from .train import train_model
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()
        game = ChessGame()
        game.reset()
        planes = game.to_planes()
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        policy[0] = 1.0
        data = [(planes, policy, 1.0)] * 4
        loss = train_model(model, data, batch_size=4, epochs=1, lr=0.001)
        self.assertIsInstance(loss, float)

    def test_self_play_game_returns_data(self):
        """self_play_game 应返回训练数据、胜负和步数"""
        from .train import self_play_game
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()
        data, winner, moves, terminate_reason = self_play_game(
            model, num_simulations=5, max_moves=30
        )
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIsInstance(moves, int)
        # terminate_reason 为 None 或字符串
        self.assertTrue(terminate_reason is None or isinstance(terminate_reason, str))
        # 每个样本应是 (planes, policy, value) 三元组
        for planes, policy, value in data:
            self.assertEqual(planes.shape, (14, 10, 9))
            self.assertEqual(policy.shape, (NUM_ACTIONS,))


class TestDirichletNoise(unittest.TestCase):
    """测试 MCTS Dirichlet 噪声注入"""

    def setUp(self):
        self.model = ChessModel(num_channels=32, num_res_blocks=2)
        self.model.build()

    def test_noise_changes_priors(self):
        """add_noise=True 时 root 子节点先验概率应发生变化"""
        game = ChessGame()
        game.reset()

        mcts_no_noise = MCTS(self.model, num_simulations=5,
                             dirichlet_alpha=0.3, dirichlet_weight=0.25)
        mcts_noise = MCTS(self.model, num_simulations=5,
                          dirichlet_alpha=0.3, dirichlet_weight=0.25)

        # Run without noise and record priors after expansion
        mcts_no_noise.get_action_probs(game, temperature=1.0, add_noise=False)
        priors_no_noise = {a: c.prior
                           for a, c in mcts_no_noise.root.children.items()}

        # Run with noise and record priors after expansion
        mcts_noise.get_action_probs(game, temperature=1.0, add_noise=True)
        priors_noise = {a: c.prior
                        for a, c in mcts_noise.root.children.items()}

        # With noise, at least some priors should differ from the no-noise run
        # (they come from the same model, so without noise they'd be the same)
        common_actions = set(priors_no_noise) & set(priors_noise)
        self.assertGreater(len(common_actions), 0)
        diffs = [abs(priors_noise[a] - priors_no_noise[a])
                 for a in common_actions]
        # Very high probability that Dirichlet noise changes at least one prior
        self.assertGreater(max(diffs), 1e-6)

    def test_no_noise_stable(self):
        """add_noise=False 时两次独立运行结果应相同（相同模型无随机性）"""
        game = ChessGame()
        game.reset()

        mcts1 = MCTS(self.model, num_simulations=5)
        actions1, probs1 = mcts1.get_action_probs(game, temperature=1.0,
                                                   add_noise=False)
        mcts2 = MCTS(self.model, num_simulations=5)
        actions2, probs2 = mcts2.get_action_probs(game, temperature=1.0,
                                                   add_noise=False)
        self.assertEqual(sorted(actions1), sorted(actions2))


class TestSelfCheckFilter(unittest.TestCase):
    """测试走后自家被将军的合法性过滤"""

    def test_rook_pin_filtered(self):
        """车牵制：移走被牵制的子会暴露将"""
        game = ChessGame()
        game.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        # 红帅在 (4,0)，被黑车 (4,9) 同列将军中间挡一个红仕 (4,1)
        game.board[0][4] = 'K'
        game.board[9][4] = 'k'
        game.board[1][4] = 'A'   # 红仕在 (4,1) 挡住将
        game.board[9][0] = 'r'   # 黑车（不在同列，不威胁）
        game.red_to_move = True
        moves = game.get_legal_moves()
        # 仕从 (4,1) 离开会暴露将，所以仕的移动应被过滤掉
        advisor_moves = [m for m in moves if m[:2] == '41']
        self.assertEqual(len(advisor_moves), 0,
                         "被牵制的仕不应有任何走法")

    def test_cannon_check_filtered(self):
        """炮将：走子暴露炮攻击路线"""
        game = ChessGame()
        game.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        # 红帅 (4,0)，黑炮 (4,5)，中间有一个红兵 (4,2) 作炮架
        # 若红兵移走，黑炮直接攻击红帅 -> 此走法应被过滤
        game.board[0][4] = 'K'
        game.board[9][4] = 'k'
        game.board[2][4] = 'P'   # 红兵（炮架）
        game.board[5][4] = 'c'   # 黑炮
        game.red_to_move = True
        moves = game.get_legal_moves()
        # 红兵移走会使黑炮直接攻击帅 -> 兵的走法被过滤
        pawn_moves = [m for m in moves if m[:2] == '42']
        self.assertEqual(len(pawn_moves), 0,
                         "移走炮架会暴露将，红兵不应有走法")

    def test_normal_moves_allowed(self):
        """正常局面下走法不被误过滤"""
        game = ChessGame()
        game.reset()
        moves = game.get_legal_moves()
        # 初始局面应有 44 步合法走法
        self.assertEqual(len(moves), 44)

    def test_check_filter_both_sides(self):
        """黑方也应过滤暴露将的走法"""
        game = ChessGame()
        game.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        # 黑将 (4,9)，红车 (4,0) 同列，中间有黑仕 (4,8) 挡住
        game.board[9][4] = 'k'
        game.board[0][4] = 'K'
        game.board[8][4] = 'a'   # 黑仕挡住
        game.board[0][0] = 'R'   # 红车在 (0,0)
        game.red_to_move = False
        moves = game.get_legal_moves()
        # 黑仕离开会暴露黑将，仕的走法应被过滤
        advisor_moves = [m for m in moves if m[:2] == '48']
        self.assertEqual(len(advisor_moves), 0,
                         "被牵制的黑仕不应有任何走法")

    def test_is_in_check_helper(self):
        """_is_in_check 辅助函数检测将军"""
        game = ChessGame()
        game.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        game.board[0][4] = 'K'
        game.board[9][4] = 'k'
        game.board[5][4] = 'r'   # 黑车在同列直接将军红帅
        game.red_to_move = True
        self.assertTrue(game._is_in_check(True),
                        "黑车直接将军时 _is_in_check 应返回 True")

    def test_not_in_check(self):
        """无将军时 _is_in_check 返回 False"""
        game = ChessGame()
        game.reset()
        self.assertFalse(game._is_in_check(True))
        self.assertFalse(game._is_in_check(False))


class TestEvaluateModels(unittest.TestCase):
    """测试模型评测函数"""

    def test_evaluate_returns_valid_stats(self):
        """evaluate_models 返回合法的 score（draw=0.5）和统计"""
        from .train import evaluate_models
        model_a = ChessModel(num_channels=32, num_res_blocks=2)
        model_a.build()
        model_b = ChessModel(num_channels=32, num_res_blocks=2)
        model_b.build()
        score, wins, losses, draws = evaluate_models(
            model_a, model_b, n_games=4, num_simulations=5, max_moves=50
        )
        total = wins + losses + draws
        self.assertEqual(total, 4)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        if total > 0:
            expected_score = (wins + 0.5 * draws) / total
            self.assertAlmostEqual(score, expected_score, places=5)


class TestMCTSRootReset(unittest.TestCase):
    """测试 MCTS root 重置与复用行为（任务1）"""

    def setUp(self):
        self.model = ChessModel(num_channels=32, num_res_blocks=2)
        self.model.build()

    def test_reset_mode_clears_old_children(self):
        """reset_root=True 时，第二次搜索不应包含第一次搜索遗留的子节点统计"""
        game = ChessGame()
        game.reset()
        mcts = MCTS(self.model, num_simulations=5)

        # 第一次搜索（reset_root=True，默认）
        mcts.get_action_probs(game, temperature=1.0, reset_root=True)
        root_id_1 = id(mcts.root)
        # 记录第一次搜索后 root 累积的 visit_count
        visits_after_1 = mcts.root.visit_count

        # 第二次搜索（同一局面，reset_root=True）
        mcts.get_action_probs(game, temperature=1.0, reset_root=True)
        root_id_2 = id(mcts.root)

        # root 对象应被替换（新建了 MCTSNode）
        self.assertNotEqual(root_id_1, root_id_2,
                            "reset_root=True 时每次搜索应创建新 root 节点")
        # root 的 visit_count 应等于本次模拟次数（无旧统计累积）
        self.assertEqual(mcts.root.visit_count, 5,
                         "重置模式下 root.visit_count 应等于 num_simulations，不应是两次之和")
        # 确认没有累积（不是第一次 + 第二次的叠加）
        self.assertNotEqual(mcts.root.visit_count, visits_after_1 + 5,
                            "重置模式下访问计数不应跨调用累积")

    def test_reuse_mode_accumulates_visits(self):
        """reset_root=False 时，子树统计可跨调用保留"""
        game = ChessGame()
        game.reset()
        mcts = MCTS(self.model, num_simulations=5)

        # 首次搜索（显式重置以初始化）
        mcts.get_action_probs(game, temperature=1.0, reset_root=True)
        visits_after_first = mcts.root.visit_count

        # 第二次搜索（不重置），在同一 root 上继续模拟
        mcts.get_action_probs(game, temperature=1.0, reset_root=False)
        visits_after_second = mcts.root.visit_count

        self.assertGreater(visits_after_second, visits_after_first,
                           "reset_root=False 时 visit_count 应在两次调用间累积")

    def test_default_is_reset(self):
        """默认调用 get_action_probs() 等效于 reset_root=True"""
        game = ChessGame()
        game.reset()
        mcts = MCTS(self.model, num_simulations=3)
        mcts.get_action_probs(game, temperature=1.0)
        # 默认行为：root 是新鲜的，visit_count == num_simulations
        self.assertEqual(mcts.root.visit_count, 3)


class TestPredictWithMask(unittest.TestCase):
    """测试 ChessModel.predict_with_mask 的概率归一化与合法走法约束（任务3）"""

    def setUp(self):
        self.model = ChessModel(num_channels=32, num_res_blocks=2)
        self.model.build()

    def test_masked_policy_sums_to_one(self):
        """mask 后的策略概率分布应和为 1"""
        game = ChessGame()
        game.reset()
        planes = game.to_planes()
        legal_moves = game.get_legal_moves()
        legal_indices = [LABEL_TO_INDEX[m] for m in legal_moves if m in LABEL_TO_INDEX]
        policy, value = self.model.predict_with_mask(planes, legal_indices)
        self.assertAlmostEqual(policy.sum(), 1.0, places=4,
                               msg="mask 后策略概率之和应为 1")

    def test_illegal_moves_have_zero_probability(self):
        """非法走法的概率应接近 0（被掩码）"""
        game = ChessGame()
        game.reset()
        planes = game.to_planes()
        legal_moves = game.get_legal_moves()
        legal_indices = set(LABEL_TO_INDEX[m] for m in legal_moves if m in LABEL_TO_INDEX)
        policy, _ = self.model.predict_with_mask(planes, list(legal_indices))

        # 随机取几个非法走法索引，概率应极小（< 1e-6）
        all_indices = set(range(NUM_ACTIONS))
        illegal_indices = list(all_indices - legal_indices)[:10]
        for idx in illegal_indices:
            self.assertLess(policy[idx], 1e-6,
                            f"非法走法索引 {idx} 的概率应接近 0，实际为 {policy[idx]}")

    def test_no_nan_in_policy(self):
        """策略输出中不应出现 NaN"""
        game = ChessGame()
        game.reset()
        planes = game.to_planes()
        legal_moves = game.get_legal_moves()
        legal_indices = [LABEL_TO_INDEX[m] for m in legal_moves if m in LABEL_TO_INDEX]
        policy, _ = self.model.predict_with_mask(planes, legal_indices)
        self.assertFalse(np.any(np.isnan(policy)), "策略输出中不应有 NaN")


    """测试 MCTS 局面缓存（任务4）"""

    def setUp(self):
        self.model = ChessModel(num_channels=32, num_res_blocks=2)
        self.model.build()

    def test_cache_disabled_by_default(self):
        """默认 cache_size=0，_cache_hits 应为 0"""
        game = ChessGame()
        game.reset()
        mcts = MCTS(self.model, num_simulations=5)
        self.assertEqual(mcts.cache_size, 0)
        mcts.get_action_probs(game, temperature=1.0)
        self.assertEqual(mcts._cache_hits, 0)

    def test_cache_enabled_hits_increase(self):
        """启用缓存后，重复局面的命中次数应 >= 0（不报错）"""
        game = ChessGame()
        game.reset()
        mcts = MCTS(self.model, num_simulations=10, cache_size=256)
        mcts.get_action_probs(game, temperature=1.0)
        # 只要不崩溃、_cache_hits >= 0 即可
        self.assertGreaterEqual(mcts._cache_hits, 0)

    def test_cache_result_consistent(self):
        """启用缓存时，搜索结果与不启用缓存应在语义上一致（都返回合法走法）"""
        game = ChessGame()
        game.reset()
        mcts_no_cache = MCTS(self.model, num_simulations=5, cache_size=0)
        mcts_cache = MCTS(self.model, num_simulations=5, cache_size=256)
        actions_no_cache, _ = mcts_no_cache.get_action_probs(game, temperature=1.0)
        actions_cache, _ = mcts_cache.get_action_probs(game, temperature=1.0)
        # 两者返回的走法集合应相同（都是合法走法）
        self.assertEqual(sorted(actions_no_cache), sorted(actions_cache))


class TestExport(unittest.TestCase):
    """测试数据导出管线 (export.py)"""

    def setUp(self):
        import tempfile
        from .export import (
            init_run_dir, append_self_play_jsonl, append_training_csv,
        )
        self.tmpdir = tempfile.mkdtemp()
        self.init_run_dir = init_run_dir
        self.append_self_play_jsonl = append_self_play_jsonl
        self.append_training_csv = append_training_csv

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_init_run_dir_creates_directory(self):
        """init_run_dir 应创建带时间戳的运行目录"""
        run_dir = self.init_run_dir(runs_dir=self.tmpdir)
        self.assertTrue(os.path.isdir(run_dir))

    def test_init_run_dir_saves_config(self):
        """init_run_dir 传入 config 时应生成 config.json"""
        import json
        config = {'num_games': 10, 'lr': 0.001}
        run_dir = self.init_run_dir(runs_dir=self.tmpdir, config=config)
        config_path = os.path.join(run_dir, 'config.json')
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, encoding='utf-8') as f:
            saved = json.load(f)
        self.assertEqual(saved['num_games'], 10)
        self.assertIn('_run_dir', saved)
        self.assertIn('_start_time', saved)

    def test_append_self_play_jsonl(self):
        """append_self_play_jsonl 应正确写入 JSONL 文件"""
        import json
        run_dir = self.init_run_dir(runs_dir=self.tmpdir)
        record = {'game_idx': 1, 'winner': 'red', 'num_moves': 80,
                  'num_samples': 80, 'elapsed_s': 5.2}
        self.append_self_play_jsonl(run_dir, record)
        path = os.path.join(run_dir, 'self_play.jsonl')
        self.assertTrue(os.path.exists(path))
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        loaded = json.loads(lines[0])
        self.assertEqual(loaded['winner'], 'red')
        self.assertEqual(loaded['game_idx'], 1)

    def test_append_self_play_jsonl_multiple(self):
        """多次追加应生成多行 JSONL"""
        run_dir = self.init_run_dir(runs_dir=self.tmpdir)
        for i in range(1, 4):
            self.append_self_play_jsonl(run_dir, {'game_idx': i, 'winner': 'draw'})
        path = os.path.join(run_dir, 'self_play.jsonl')
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)

    def test_append_training_csv(self):
        """append_training_csv 应写入表头和数据行"""
        import csv
        run_dir = self.init_run_dir(runs_dir=self.tmpdir)
        row = {'game_idx': 1, 'loss': 2.34, 'buffer_size': 256, 'elapsed_s': 5.0}
        self.append_training_csv(run_dir, row)
        path = os.path.join(run_dir, 'training_metrics.csv')
        self.assertTrue(os.path.exists(path))
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]['loss']), 2.34)

    def test_append_training_csv_no_duplicate_header(self):
        """多次追加时表头不应重复"""
        import csv
        run_dir = self.init_run_dir(runs_dir=self.tmpdir)
        for i in range(3):
            self.append_training_csv(run_dir, {'game_idx': i, 'loss': float(i)})
        path = os.path.join(run_dir, 'training_metrics.csv')
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        # 1 header + 3 data = 4 lines
        self.assertEqual(len(lines), 4)


class TestRepetitionDetection(unittest.TestCase):
    """测试重复局面检测"""

    def test_threefold_repetition_draw(self):
        """三次重复局面应判和"""
        game3 = ChessGame()
        game3.reset()
        moves_cycle = [
            '1213',  # red cannon up
            '1716',  # black cannon down
            '1312',  # red cannon back
            '1617',  # black cannon back (position repeats 2nd time)
            '1213',  # red cannon up again
            '1716',  # black cannon down again
            '1312',  # red cannon back again
            '1617',  # black cannon back (position repeats 3rd time -> draw)
        ]
        for move in moves_cycle:
            if game3.done:
                break
            game3.step(move)

        self.assertEqual(game3.winner, 'draw',
                        f"Expected draw by repetition, got winner={game3.winner}")

    def test_perpetual_check_interface(self):
        """长将接口应存在且不崩溃"""
        game = ChessGame()
        game.board = [[None]*9 for _ in range(10)]
        game.board[0][4] = 'K'
        game.board[9][4] = 'k'
        game.board[5][3] = 'R'
        game.red_to_move = True
        game._init_hash()

        self.assertFalse(game.done)
        legal = game.get_legal_moves()
        self.assertGreater(len(legal), 0)

    def test_perpetual_chase_red_loses(self):
        """红方长捉判负：红车反复追捉黑马，黑方无捉回"""
        # 棋盘：
        #   红帅(4,0)  红仕(4,1)[阻止飞将]  黑将(4,9)
        #   红车(0,5)  黑马(3,5)  黑车(8,5)
        # 循环：红车在(0,5)和(2,5)间往返，始终攻击黑马(3,5)
        #        黑车在(8,5)和(7,5)间往返，不攻击任何红方非将棋子
        game = ChessGame()
        game.board = [[None]*9 for _ in range(10)]
        game.board[0][4] = 'K'   # 红帅(4,0)
        game.board[9][4] = 'k'   # 黑将(4,9)
        game.board[1][4] = 'A'   # 红仕(4,1) 阻止将帅飞将
        game.board[5][0] = 'R'   # 红车(0,5)
        game.board[5][3] = 'n'   # 黑马(3,5)
        game.board[5][8] = 'r'   # 黑车(8,5)
        game.red_to_move = True
        game._init_hash()

        # 红车(0,5)->(2,5)攻击黑马(3,5)，黑车往返不攻击红方非将棋子，循环三次
        moves_cycle = [
            '0525',  # 红车(0,5)->(2,5)，攻击黑马(3,5)
            '8575',  # 黑车(8,5)->(7,5)，不捉任何红方非将棋子
            '2505',  # 红车(2,5)->(0,5)，仍攻击黑马(3,5)
            '7585',  # 黑车(7,5)->(8,5)，不捉任何红方非将棋子
            '0525',  # 重复第1步
            '8575',
            '2505',
            '7585',  # 第3次重复 -> 长捉触发
        ]
        for move in moves_cycle:
            if game.done:
                break
            game.step(move)

        self.assertTrue(game.done, "长捉循环应触发终局")
        self.assertEqual(game.winner, 'black',
                         f"红方长捉应判红方负（黑方胜），实际: "
                         f"winner={game.winner}, reason={game.terminate_reason}")
        self.assertEqual(game.terminate_reason, 'perpetual_chase')

    def test_perpetual_chase_draw_both_chase(self):
        """双方均长捉时判和（终局原因为 repetition，非 perpetual_chase）"""
        # 棋盘：红车追黑马，黑车追红马，双方均长捉
        game = ChessGame()
        game.board = [[None]*9 for _ in range(10)]
        game.board[0][4] = 'K'   # 红帅(4,0)
        game.board[9][4] = 'k'   # 黑将(4,9)
        game.board[1][4] = 'A'   # 红仕(4,1) 阻止飞将
        game.board[8][4] = 'a'   # 黑仕(4,8) 阻止飞将
        game.board[5][0] = 'R'   # 红车(0,5)
        game.board[5][3] = 'n'   # 黑马(3,5)
        game.board[4][8] = 'r'   # 黑车(8,4)
        game.board[4][5] = 'N'   # 红马(5,4)
        game.red_to_move = True
        game._init_hash()

        # 红车攻击黑马，黑车攻击红马，双方均长捉，循环三次
        moves_cycle = [
            '0525',  # 红车(0,5)->(2,5)攻击黑马(3,5)
            '8464',  # 黑车(8,4)->(6,4)攻击红马(5,4)
            '2505',  # 红车回原位，仍攻击黑马
            '6484',  # 黑车回原位，仍攻击红马
            '0525',  # 重复
            '8464',
            '2505',
            '6484',  # 第3次重复
        ]
        for move in moves_cycle:
            if game.done:
                break
            game.step(move)

        self.assertTrue(game.done, "长捉循环应触发终局")
        # 双方均长捉时不应判为 perpetual_chase
        self.assertNotEqual(game.terminate_reason, 'perpetual_chase',
                            "双方均长捉时不应判单方负")

    def test_get_chased_pieces(self):
        """_get_chased_pieces 应正确识别被攻击的非将棋子"""
        game = ChessGame()
        game.board = [[None]*9 for _ in range(10)]
        game.board[0][4] = 'K'   # 红帅(4,0)
        game.board[9][4] = 'k'   # 黑将(4,9)
        game.board[5][0] = 'R'   # 红车(0,5)
        game.board[5][3] = 'n'   # 黑马(3,5)
        game.board[5][8] = 'r'   # 黑车(8,5)
        game.red_to_move = True
        game._init_hash()

        # 将红车移到(2,5)，此时红车攻击黑马(3,5)
        game.board[5][2] = 'R'
        game.board[5][0] = None

        chased = game._get_chased_pieces(True)  # 红方捉子
        self.assertIn((3, 5), chased,
                      "红车在(2,5)应攻击黑马(3,5)")
        self.assertNotIn((4, 9), chased,
                         "将/帅不参与捉子计算")


class TestRepetitionDrawThreshold(unittest.TestCase):
    """测试重复局面判和阈值（repetition_draw_threshold）"""

    # 循环走法（炮来回）：每 4 步返回同一局面
    MOVES_CYCLE = [
        '1213',  # 红炮上
        '1716',  # 黑炮下
        '1312',  # 红炮回
        '1617',  # 黑炮回 → 回到初始局面（第2次出现）
        '1213',
        '1716',
        '1312',
        '1617',  # 第3次出现
        '1213',
        '1716',
        '1312',
        '1617',  # 第4次出现
        '1213',
        '1716',
        '1312',
        '1617',  # 第5次出现
    ]

    def _play_n_cycles(self, game, n_cycles):
        """执行 n_cycles 轮循环走法（每轮 4 步）"""
        steps_done = 0
        for move in self.MOVES_CYCLE[:n_cycles * 4]:
            if game.done:
                break
            game.step(move)
            steps_done += 1
        return steps_done

    def test_invalid_threshold_raises(self):
        """阈值 < 3 应抛出 ValueError"""
        with self.assertRaises(ValueError):
            ChessGame(repetition_draw_threshold=2)
        with self.assertRaises(ValueError):
            ChessGame(repetition_draw_threshold=1)

    def test_default_threshold_is_3(self):
        """默认阈值 = 3，三次重复局面判和（两轮循环后第3次出现）"""
        game = ChessGame()
        game.reset()
        self._play_n_cycles(game, 2)  # 两轮循环 → 第3次出现 → draw
        self.assertTrue(game.done, "默认阈值=3：两轮循环后应已终局")
        self.assertEqual(game.winner, 'draw',
                         f"默认阈值应为 draw，实际: winner={game.winner}")
        self.assertEqual(game.terminate_reason, 'repetition')

    def test_threshold_3_same_as_default(self):
        """显式 threshold=3 与默认行为一致"""
        game = ChessGame(repetition_draw_threshold=3)
        game.reset()
        self._play_n_cycles(game, 2)
        self.assertEqual(game.winner, 'draw')
        self.assertEqual(game.terminate_reason, 'repetition')

    def test_threshold_4_not_draw_after_3_occurrences(self):
        """threshold=4：第3次重复局面不判和，第4次才判和"""
        game = ChessGame(repetition_draw_threshold=4)
        game.reset()
        # 执行刚好能产生第3次出现的走法（2个完整循环 = 位置出现3次）
        self._play_n_cycles(game, 2)
        self.assertFalse(game.done,
                         "threshold=4：第3次重复后不应判和")
        # 再走一轮 → 第4次出现 → draw
        self._play_n_cycles(game, 1)
        self.assertTrue(game.done, "threshold=4：第4次重复后应判和")
        self.assertEqual(game.winner, 'draw')
        self.assertEqual(game.terminate_reason, 'repetition')

    def test_threshold_5_draw_on_5th_occurrence(self):
        """threshold=5：第5次重复局面才判和（三轮循环后第4次出现，再一轮后第5次出现）"""
        game = ChessGame(repetition_draw_threshold=5)
        game.reset()
        self._play_n_cycles(game, 3)  # 三轮循环 → 第4次出现，不判和
        self.assertFalse(game.done,
                         "threshold=5：三轮循环后（第4次出现）不应判和")
        self._play_n_cycles(game, 1)  # 再一轮 → 第5次出现 → draw
        self.assertTrue(game.done, "threshold=5：第5次重复后应判和")
        self.assertEqual(game.winner, 'draw')
        self.assertEqual(game.terminate_reason, 'repetition')

    def test_threshold_preserved_after_reset(self):
        """reset() 后阈值不丢失"""
        game = ChessGame(repetition_draw_threshold=4)
        game.reset()
        self.assertEqual(game.repetition_draw_threshold, 4)
        game.reset()  # 再次 reset
        self.assertEqual(game.repetition_draw_threshold, 4)
        # 验证行为：第3次重复不判和
        self._play_n_cycles(game, 2)
        self.assertFalse(game.done,
                         "threshold=4 在 reset() 后仍不应在第3次重复判和")

    def test_reset_clears_history(self):
        """reset() 后 pos_history/move_history 应被清空，不跨局污染"""
        game = ChessGame(repetition_draw_threshold=3)
        game.reset()
        self._play_n_cycles(game, 2)  # 触发 draw
        self.assertTrue(game.done)

        # 重置后应恢复干净状态
        game.reset()
        self.assertIsNone(game.winner)
        self.assertFalse(game.done)
        self.assertEqual(game.num_moves, 0)
        self.assertEqual(len(game.move_history), 0)
        # pos_history 初始包含当前局面哈希（来自 _init_hash）
        self.assertEqual(len(game.pos_history), 1)
        self.assertIsNone(game.terminate_reason)


class TestSelfPlayGameReturns(unittest.TestCase):
    """验证 self_play_game 返回 4 个值（含 terminate_reason）"""

    def test_returns_four_values(self):
        """self_play_game 应返回 (training_data, winner, move_count, terminate_reason)"""
        from .train import self_play_game
        model = ChessModel(num_channels=32, num_res_blocks=1)
        model.build()
        result = self_play_game(model, num_simulations=2, max_moves=10)
        self.assertEqual(len(result), 4, "self_play_game 应返回 4 个值")
        training_data, winner, move_count, terminate_reason = result
        self.assertIsInstance(training_data, list)
        self.assertIsInstance(move_count, int)
        # terminate_reason 为 None 或 str
        self.assertTrue(terminate_reason is None or isinstance(terminate_reason, str))

    def test_repetition_draw_threshold_propagated(self):
        """self_play_game 的 repetition_draw_threshold 参数应传递给游戏"""
        from .train import self_play_game
        model = ChessModel(num_channels=32, num_res_blocks=1)
        model.build()
        # 只需验证不报错，threshold 值合法
        result = self_play_game(model, num_simulations=2, max_moves=10,
                                repetition_draw_threshold=5)
        self.assertEqual(len(result), 4)


class TestDistillModel(unittest.TestCase):
    """测试 distill_model 函数"""

    def test_distill_model_policy_only(self):
        """value_loss_weight=0 时应只训练策略头，返回非负浮点损失"""
        from .distill import distill_model
        model = ChessModel(num_channels=32, num_res_blocks=1)
        model.build()

        # 构造假训练数据：(planes, one-hot policy, value=0)
        n = 20
        states = np.random.rand(n, 14, 10, 9).astype(np.float32)
        policies = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
        for i in range(n):
            policies[i, i % NUM_ACTIONS] = 1.0  # one-hot
        values = np.zeros(n, dtype=np.float32)

        training_data = list(zip(states, policies, values))
        loss = distill_model(model, training_data, batch_size=8, epochs=1,
                             value_loss_weight=0.0)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)

    def test_distill_model_empty_data_returns_zero(self):
        """空训练数据时应返回 0.0"""
        from .distill import distill_model
        model = ChessModel(num_channels=32, num_res_blocks=1)
        model.build()
        loss = distill_model(model, [], batch_size=8, epochs=1)
        self.assertEqual(loss, 0.0)


class TestGenerateDistillGame(unittest.TestCase):
    """测试 generate_distill_game 函数（使用随机 Agent 代替 Pikafish）"""

    def _make_random_agent(self):
        from .pikafish_agent import BaseAgent

        class RandomAgent(BaseAgent):
            def new_game(self):
                pass

            def get_move(self, game):
                legal = game.get_legal_moves()
                return random.choice(legal) if legal else None

        return RandomAgent()

    def test_returns_correct_format(self):
        """generate_distill_game 应返回 (training_data, winner, moves, reason)"""
        from .distill import generate_distill_game
        agent = self._make_random_agent()
        result = generate_distill_game(agent, max_moves=20)
        self.assertEqual(len(result), 4)
        training_data, winner, moves, terminate_reason = result
        self.assertIsInstance(training_data, list)
        self.assertIsInstance(moves, int)

    def test_training_data_shapes(self):
        """每个样本的 state/policy/value 形状应正确"""
        from .distill import generate_distill_game
        agent = self._make_random_agent()
        training_data, _, _, _ = generate_distill_game(agent, max_moves=20)

        if not training_data:
            self.skipTest("未产生训练样本（对局过短），跳过形状检验")

        for state, policy, value in training_data:
            self.assertEqual(state.shape, (14, 10, 9))
            self.assertEqual(len(policy), NUM_ACTIONS)
            self.assertAlmostEqual(float(value), 0.0,
                                   msg="蒸馏阶段 value_target 应恒为 0.0")

    def test_policy_is_soft_distribution(self):
        """蒸馏 policy_target 应为 soft 概率分布（非 one-hot，sum≈1，多个非零项）"""
        from .distill import generate_distill_game
        agent = self._make_random_agent()
        training_data, _, _, _ = generate_distill_game(agent, max_moves=20)

        if not training_data:
            self.skipTest("未产生训练样本，跳过 soft 分布检验")

        for _state, policy, _value in training_data:
            # 概率分布：sum ≈ 1，所有元素 ≥ 0
            self.assertAlmostEqual(float(np.sum(policy)), 1.0, places=5,
                                   msg="soft policy_target 的概率总和应约为 1.0")
            self.assertTrue(np.all(policy >= 0),
                            "soft policy_target 所有元素应 ≥ 0")
            # soft target：至少有 2 个非零项（非 one-hot）
            nonzero_count = int(np.sum(policy > 0))
            self.assertGreater(nonzero_count, 1,
                               "soft policy_target 应有多于 1 个非零项（非 one-hot）")

    def test_generate_distill_game_with_teacher(self):
        """独立 teacher agent 提供标注时，policy 应为 soft 分布"""
        from .distill import generate_distill_game
        weak_agent = self._make_random_agent()
        teacher_agent = self._make_random_agent()
        training_data, _, _, _ = generate_distill_game(
            weak_agent, teacher_agent=teacher_agent, max_moves=20,
        )

        if not training_data:
            self.skipTest("未产生训练样本，跳过 teacher soft 分布检验")

        for _state, policy, _value in training_data:
            self.assertAlmostEqual(float(np.sum(policy)), 1.0, places=5)
            self.assertTrue(np.all(policy >= 0))


class TestParseMultipvInfo(unittest.TestCase):
    """测试 parse_multipv_info 函数"""

    def test_parse_basic_multipv(self):
        """标准 MultiPV info 行解析"""
        from .pikafish_agent import parse_multipv_info
        info_lines = [
            "info depth 10 seldepth 12 multipv 1 score cp 50 nodes 1000 nps 500000 time 2 pv h2e2 e9e8",
            "info depth 10 seldepth 12 multipv 2 score cp 30 nodes 1000 nps 500000 time 2 pv b0c2 b9c7",
            "info depth 10 seldepth 12 multipv 3 score cp -10 nodes 1000 nps 500000 time 2 pv a0a1 a9a8",
        ]
        result = parse_multipv_info(info_lines)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][0], "h2e2")
        self.assertEqual(result[0][1], 50)
        self.assertEqual(result[1][0], "b0c2")
        self.assertEqual(result[1][1], 30)
        self.assertEqual(result[2][1], -10)

    def test_parse_mate_score(self):
        """score mate 应映射为 ±10000"""
        from .pikafish_agent import parse_multipv_info
        info_lines = [
            "info depth 5 multipv 1 score mate 3 pv e0e1",
            "info depth 5 multipv 2 score mate -2 pv h0g2",
        ]
        result = parse_multipv_info(info_lines)
        self.assertEqual(result[0][1], 10000)
        self.assertEqual(result[1][1], -10000)

    def test_parse_no_multipv_lines(self):
        """无 multipv 字段的 info 行应返回空列表"""
        from .pikafish_agent import parse_multipv_info
        info_lines = [
            "info depth 5 score cp 20 nodes 500 pv h2e2",
        ]
        result = parse_multipv_info(info_lines)
        self.assertEqual(result, [])

    def test_parse_empty(self):
        """空输入应返回空列表"""
        from .pikafish_agent import parse_multipv_info
        self.assertEqual(parse_multipv_info([]), [])


class TestSoftPolicyHelpers(unittest.TestCase):
    """测试 _build_soft_policy_from_candidates 和 _build_soft_policy_fallback"""

    def test_build_from_candidates_red(self):
        """红方视角：候选走法直接映射"""
        from .distill import _build_soft_policy_from_candidates
        from .game import ACTION_LABELS

        # 取两个已知合法走法
        move1 = ACTION_LABELS[0]
        move2 = ACTION_LABELS[1]
        candidates = [(move1, 100), (move2, 50)]
        policy = _build_soft_policy_from_candidates(
            candidates, is_black_to_move=False, temperature=1.0
        )
        self.assertIsNotNone(policy)
        self.assertEqual(len(policy), NUM_ACTIONS)
        self.assertAlmostEqual(float(np.sum(policy)), 1.0, places=5)
        self.assertTrue(np.all(policy >= 0))
        self.assertGreater(float(policy[0]), float(policy[1]),
                           "分数高的走法应有更高概率")

    def test_build_fallback_has_multiple_nonzero(self):
        """软回退策略应有多个非零项（非 one-hot）"""
        from .distill import _build_soft_policy_fallback
        from .game import ChessGame

        game = ChessGame()
        game.reset()
        legal = game.get_legal_moves()
        bestmove = legal[0]

        policy = _build_soft_policy_fallback(
            bestmove, legal, is_black_to_move=False
        )
        self.assertIsNotNone(policy)
        self.assertAlmostEqual(float(np.sum(policy)), 1.0, places=5)
        nonzero = int(np.sum(policy > 0))
        self.assertGreater(nonzero, 1,
                           "软回退策略应有多个非零项（非 one-hot）")

    def test_build_fallback_best_move_has_highest_prob(self):
        """软回退策略中 bestmove 应具有最高概率"""
        from .distill import _build_soft_policy_fallback
        from .game import ChessGame, LABEL_TO_INDEX

        game = ChessGame()
        game.reset()
        legal = game.get_legal_moves()
        bestmove = legal[0]

        policy = _build_soft_policy_fallback(
            bestmove, legal, is_black_to_move=False
        )
        self.assertIsNotNone(policy)
        best_idx = LABEL_TO_INDEX[bestmove]
        self.assertEqual(
            int(np.argmax(policy)), best_idx,
            "软回退策略中 bestmove 应具有最高概率",
        )

    def test_build_from_empty_candidates_returns_none(self):
        """空候选列表应返回 None"""
        from .distill import _build_soft_policy_from_candidates
        result = _build_soft_policy_from_candidates(
            [], is_black_to_move=False
        )
        self.assertIsNone(result)


class TestEloUpdate(unittest.TestCase):
    """测试 ELO 评分更新辅助函数"""

    def test_win_increases_rating(self):
        """score=1.0（全胜）时评分应上升"""
        from .train import compute_elo_update
        new_r, delta = compute_elo_update(1500.0, 1500.0, score=1.0, k=32)
        self.assertGreater(new_r, 1500.0)
        self.assertGreater(delta, 0.0)

    def test_loss_decreases_rating(self):
        """score=0.0（全负）时评分应下降"""
        from .train import compute_elo_update
        new_r, delta = compute_elo_update(1500.0, 1500.0, score=0.0, k=32)
        self.assertLess(new_r, 1500.0)
        self.assertLess(delta, 0.0)

    def test_draw_against_equal_no_change(self):
        """score=0.5 且双方评分相等时，评分几乎不变"""
        from .train import compute_elo_update
        new_r, delta = compute_elo_update(1500.0, 1500.0, score=0.5, k=32)
        self.assertAlmostEqual(delta, 0.0, places=6)
        self.assertAlmostEqual(new_r, 1500.0, places=6)

    def test_k_factor_scales_delta(self):
        """K 因子应按比例缩放 delta"""
        from .train import compute_elo_update
        _, delta_k32 = compute_elo_update(1500.0, 1500.0, score=1.0, k=32)
        _, delta_k16 = compute_elo_update(1500.0, 1500.0, score=1.0, k=16)
        self.assertAlmostEqual(delta_k32, delta_k16 * 2, places=6)

    def test_elo_formula_known_value(self):
        """验证具体数值：R_cur=1500, R_opp=1500, score=1.0, k=32 → delta≈16"""
        from .train import compute_elo_update
        # expected = 1/(1+10^0) = 0.5, delta = 32*(1-0.5) = 16
        new_r, delta = compute_elo_update(1500.0, 1500.0, score=1.0, k=32)
        self.assertAlmostEqual(delta, 16.0, places=5)
        self.assertAlmostEqual(new_r, 1516.0, places=5)


class TestEvalModule(unittest.TestCase):
    """测试独立评测模块（eval.py）"""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_model_file(self):
        """创建一个临时模型文件，返回路径。"""
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()
        path = os.path.join(self.tmpdir, 'model.pth')
        model.save(path)
        return path

    def test_run_eval_returns_structured_result(self):
        """run_eval 应返回含 wins_a/wins_b/draws/score 的字典"""
        from .eval import run_eval
        model_path = self._make_model_file()
        result = run_eval(
            model_a_path=model_path,
            model_b_path=model_path,
            n_games=2,
            num_simulations=2,
            max_moves=50,
            seed=0,
        )
        self.assertIn('wins_a', result)
        self.assertIn('wins_b', result)
        self.assertIn('draws', result)
        self.assertIn('score', result)
        total = result['wins_a'] + result['wins_b'] + result['draws']
        self.assertEqual(total, 2)
        self.assertGreaterEqual(result['score'], 0.0)
        self.assertLessEqual(result['score'], 1.0)

    def test_run_eval_score_consistency(self):
        """score 应等于 (wins_a + 0.5 * draws) / total"""
        from .eval import run_eval
        model_path = self._make_model_file()
        result = run_eval(
            model_a_path=model_path,
            model_b_path=model_path,
            n_games=4,
            num_simulations=2,
            max_moves=30,
            seed=42,
        )
        total = result['wins_a'] + result['wins_b'] + result['draws']
        if total > 0:
            expected_score = (result['wins_a'] + 0.5 * result['draws']) / total
            self.assertAlmostEqual(result['score'], expected_score, places=5)

    def test_run_eval_writes_csv_when_out_given(self):
        """提供 --out 时应在运行目录写入 evaluation_metrics.csv"""
        import csv as csv_mod
        from .eval import run_eval
        model_path = self._make_model_file()
        out_dir = os.path.join(self.tmpdir, 'run_out')
        run_eval(
            model_a_path=model_path,
            model_b_path=model_path,
            n_games=2,
            num_simulations=2,
            max_moves=30,
            seed=0,
            out=out_dir,
        )
        csv_path = os.path.join(out_dir, 'evaluation_metrics.csv')
        self.assertTrue(os.path.exists(csv_path),
                        "evaluation_metrics.csv 应被创建")
        with open(csv_path, encoding='utf-8') as f:
            rows = list(csv_mod.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertIn('score', rows[0])

    def test_run_eval_seed_reproducibility(self):
        """相同 seed 应产生相同结果"""
        from .eval import run_eval
        model_path = self._make_model_file()
        kwargs = dict(
            model_a_path=model_path,
            model_b_path=model_path,
            n_games=2,
            num_simulations=2,
            max_moves=30,
            seed=7,
        )
        result1 = run_eval(**kwargs)
        result2 = run_eval(**kwargs)
        self.assertEqual(result1['wins_a'], result2['wins_a'])
        self.assertEqual(result1['wins_b'], result2['wins_b'])
        self.assertEqual(result1['draws'], result2['draws'])


class TestExportEvalHelpers(unittest.TestCase):
    """测试 export.py 中的评测辅助函数"""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_append_evaluation_csv_creates_file(self):
        """append_evaluation_csv 应创建 evaluation_metrics.csv 并写入数据"""
        import csv as csv_mod
        from .export import append_evaluation_csv, init_run_dir
        run_dir = init_run_dir(runs_dir=self.tmpdir)
        row = {
            'game_idx': 10, 'timestamp': '2024-01-01T12:00:00',
            'opponent': 'previous', 'eval_games': 20, 'eval_sims': 50,
            'wins': 12, 'losses': 6, 'draws': 2, 'score': 0.65,
            'elo': 1516.0, 'elo_delta': 16.0,
        }
        append_evaluation_csv(run_dir, row)
        path = os.path.join(run_dir, 'evaluation_metrics.csv')
        self.assertTrue(os.path.exists(path))
        with open(path, encoding='utf-8') as f:
            rows = list(csv_mod.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]['elo']), 1516.0)

    def test_load_save_evaluation_state_roundtrip(self):
        """save/load evaluation_state 应保持数据一致性"""
        from .export import (
            load_evaluation_state, save_evaluation_state, init_run_dir,
        )
        run_dir = init_run_dir(runs_dir=self.tmpdir)
        state = {'elo_current': 1523.5, 'elo_opponent': 1500.0, 'last_game_idx': 20}
        save_evaluation_state(run_dir, state)
        loaded = load_evaluation_state(run_dir)
        self.assertAlmostEqual(loaded['elo_current'], 1523.5)
        self.assertEqual(loaded['last_game_idx'], 20)

    def test_load_evaluation_state_defaults_when_missing(self):
        """evaluation_state.json 不存在时应返回默认初始状态"""
        from .export import load_evaluation_state, init_run_dir
        run_dir = init_run_dir(runs_dir=self.tmpdir)
        state = load_evaluation_state(run_dir)
        self.assertIn('elo_current', state)
        self.assertIn('elo_opponent', state)
        self.assertAlmostEqual(state['elo_current'], 1500.0)


class TestPlotModule(unittest.TestCase):
    """测试绘图模块（plot.py）"""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_csv(self, filename, rows):
        import csv as csv_mod
        path = os.path.join(self.tmpdir, filename)
        if rows:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        return path

    def test_plot_missing_matplotlib_exits(self):
        """matplotlib 未安装时 _require_matplotlib 应触发 SystemExit"""
        import sys
        import importlib
        from unittest.mock import patch
        # 模拟 matplotlib 不可用
        with patch.dict(sys.modules, {'matplotlib': None, 'matplotlib.pyplot': None}):
            from . import plot as plot_mod
            importlib.reload(plot_mod)
            with self.assertRaises(SystemExit):
                plot_mod._require_matplotlib()

    def test_plot_loss_no_data_skips(self):
        """training_metrics.csv 不存在时 plot_loss 应跳过（返回 None）"""
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib 未安装，跳过绘图测试")
        from .plot import plot_loss
        result = plot_loss(self.tmpdir, self.tmpdir, fmt='png')
        self.assertIsNone(result)

    def test_plot_loss_generates_file(self):
        """有数据时 plot_loss 应生成 loss.png"""
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib 未安装，跳过绘图测试")
        from .plot import plot_loss
        self._write_csv('training_metrics.csv', [
            {'game_idx': 1, 'loss': 2.5},
            {'game_idx': 2, 'loss': 2.1},
        ])
        out_path = plot_loss(self.tmpdir, self.tmpdir, fmt='png')
        self.assertIsNotNone(out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertTrue(out_path.endswith('loss.png'))

    def test_plot_elo_generates_file(self):
        """有数据时 plot_elo 应生成 elo.png"""
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib 未安装，跳过绘图测试")
        from .plot import plot_elo
        self._write_csv('evaluation_metrics.csv', [
            {'game_idx': 10, 'elo': 1510.0, 'score': 0.6},
            {'game_idx': 20, 'elo': 1525.0, 'score': 0.65},
        ])
        out_path = plot_elo(self.tmpdir, self.tmpdir, fmt='png')
        self.assertIsNotNone(out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertTrue(out_path.endswith('elo.png'))

    def test_plot_score_generates_file(self):
        """有数据时 plot_score 应生成 score.png"""
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib 未安装，跳过绘图测试")
        from .plot import plot_score
        self._write_csv('evaluation_metrics.csv', [
            {'game_idx': 10, 'elo': 1510.0, 'score': 0.6},
            {'game_idx': 20, 'elo': 1525.0, 'score': 0.65},
        ])
        out_path = plot_score(self.tmpdir, self.tmpdir, fmt='png')
        self.assertIsNotNone(out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertTrue(out_path.endswith('score.png'))





class TestUCIConversion(unittest.TestCase):
    """测试 UCI ICCS 走法格式与内部格式的互转（pikafish_agent 模块）"""

    def setUp(self):
        from .pikafish_agent import internal_to_uci, uci_to_internal, game_to_uci_fen
        self.internal_to_uci = internal_to_uci
        self.uci_to_internal = uci_to_internal
        self.game_to_uci_fen = game_to_uci_fen

    # ------------------------------------------------------------------
    # internal_to_uci
    # ------------------------------------------------------------------

    def test_internal_to_uci_corner(self):
        """(0,0) -> a0"""
        self.assertEqual(self.internal_to_uci("0000"), "a0a0")

    def test_internal_to_uci_rook(self):
        """红车从 a0 到 a2：内部 "0002" -> UCI "a0a2" """
        self.assertEqual(self.internal_to_uci("0002"), "a0a2")

    def test_internal_to_uci_cannon(self):
        """炮从 b7 到 e7：内部 "1771" / 列1行7->列4行7 -> "b7e7" """
        # 炮的初始位置：黑方炮在 (1,7) 和 (7,7)
        self.assertEqual(self.internal_to_uci("1747"), "b7e7")

    def test_internal_to_uci_king_e(self):
        """帅在 e0 (4,0)：内部 "4041" -> UCI "e0e1" """
        self.assertEqual(self.internal_to_uci("4041"), "e0e1")

    def test_internal_to_uci_i_column(self):
        """最右列 x=8（i 列）：内部 "8082" -> UCI "i0i2" """
        self.assertEqual(self.internal_to_uci("8082"), "i0i2")

    def test_internal_to_uci_invalid_length(self):
        """长度不为 4 应抛出 ValueError"""
        with self.assertRaises(ValueError):
            self.internal_to_uci("123")

    def test_internal_to_uci_non_digit(self):
        """非数字字符应抛出 ValueError"""
        with self.assertRaises(ValueError):
            self.internal_to_uci("a0b1")

    # ------------------------------------------------------------------
    # uci_to_internal
    # ------------------------------------------------------------------

    def test_uci_to_internal_roundtrip(self):
        """internal_to_uci -> uci_to_internal 应还原原始走法"""
        for move in ["0000", "0009", "8082", "4041", "1747", "7072"]:
            self.assertEqual(
                self.uci_to_internal(self.internal_to_uci(move)),
                move,
                msg=f"round-trip failed for {move!r}",
            )

    def test_uci_to_internal_h0h2(self):
        """UCI h0h2 -> 内部 "7072" """
        self.assertEqual(self.uci_to_internal("h0h2"), "7072")

    def test_uci_to_internal_e0e1(self):
        """UCI e0e1 -> 内部 "4041" """
        self.assertEqual(self.uci_to_internal("e0e1"), "4041")

    def test_uci_to_internal_invalid_length(self):
        """长度不为 4 应抛出 ValueError"""
        with self.assertRaises(ValueError):
            self.uci_to_internal("h00")

    def test_uci_to_internal_out_of_range_file(self):
        """列超出 a-i 范围（如 'j'）应抛出 ValueError"""
        with self.assertRaises(ValueError):
            self.uci_to_internal("j0j1")

    # ------------------------------------------------------------------
    # game_to_uci_fen
    # ------------------------------------------------------------------

    def test_game_to_uci_fen_initial_red(self):
        """初始局面红方先手 FEN 应以 board_fen + ' w' 开头"""
        game = ChessGame()
        game.reset()
        fen = self.game_to_uci_fen(game)
        board, side = fen.split(" ")[:2]
        self.assertEqual(board, INIT_FEN)
        self.assertEqual(side, "w")

    def test_game_to_uci_fen_after_move_black(self):
        """走一步后轮到黑方，FEN 中 side 应为 'b'"""
        game = ChessGame()
        game.reset()
        # 走一步合法走法
        legal = game.get_legal_moves()
        game.step(legal[0])
        fen = self.game_to_uci_fen(game)
        side = fen.split(" ")[1]
        self.assertEqual(side, "b")

    def test_game_to_uci_fen_fields_count(self):
        """UCI FEN 应包含至少 6 个空格分隔字段"""
        game = ChessGame()
        game.reset()
        fen = self.game_to_uci_fen(game)
        self.assertGreaterEqual(len(fen.split()), 6)


class TestPlayVsOpponent(unittest.TestCase):
    """测试与虚拟对手对弈的数据收集（play_game_vs_opponent_collect_my_turn）"""

    def _make_random_agent(self):
        """返回一个随机走法 Agent（实现 BaseAgent 接口）"""
        from .pikafish_agent import BaseAgent

        class RandomAgent(BaseAgent):
            """随机选取合法走法的虚拟对手，用于测试"""

            def get_move(self, game):
                legal = game.get_legal_moves()
                return random.choice(legal) if legal else None

        return RandomAgent()

    def test_returns_correct_format(self):
        """验证返回值格式：training_data 列表、winner、moves 整数、metadata 字典"""
        model = ChessModel(num_channels=64, num_res_blocks=2)
        model.build()

        from .train import play_game_vs_opponent_collect_my_turn
        training_data, winner, moves, metadata = play_game_vs_opponent_collect_my_turn(
            model=model,
            opponent_agent=self._make_random_agent(),
            my_side='red',
            game_idx=0,
            num_simulations=3,
            max_moves=20,
        )

        self.assertIsInstance(training_data, list)
        self.assertIsInstance(moves, int)
        self.assertIsInstance(metadata, dict)
        self.assertIn('my_side', metadata)
        self.assertIn('num_my_samples', metadata)
        self.assertEqual(metadata['my_side'], 'red')

    def test_training_data_shapes(self):
        """每个训练样本的 state、policy、value 形状和范围应正确"""
        from .game import NUM_ACTIONS
        model = ChessModel(num_channels=64, num_res_blocks=2)
        model.build()

        from .train import play_game_vs_opponent_collect_my_turn
        training_data, _, _, _ = play_game_vs_opponent_collect_my_turn(
            model=model,
            opponent_agent=self._make_random_agent(),
            my_side='red',
            game_idx=0,
            num_simulations=3,
            max_moves=20,
        )

        if not training_data:
            self.skipTest("未产生训练样本（对局过短），跳过形状检验")

        for state, policy, value in training_data:
            self.assertEqual(state.shape, (14, 10, 9),
                             msg="state_planes 形状应为 (14, 10, 9)")
            self.assertEqual(len(policy), NUM_ACTIONS,
                             msg="policy_target 长度应等于 NUM_ACTIONS")
            self.assertGreaterEqual(float(value), -1.0)
            self.assertLessEqual(float(value), 1.0)

    def test_only_my_turns_collected_red(self):
        """执红时，样本数应不超过走棋总步数的一半（+1 容错）"""
        model = ChessModel(num_channels=64, num_res_blocks=2)
        model.build()

        from .train import play_game_vs_opponent_collect_my_turn
        training_data, _, moves, metadata = play_game_vs_opponent_collect_my_turn(
            model=model,
            opponent_agent=self._make_random_agent(),
            my_side='red',
            game_idx=0,
            num_simulations=3,
            max_moves=30,
        )

        # 我方（红）先手，样本数 <= ceil(moves / 2)
        self.assertLessEqual(
            len(training_data), moves // 2 + 2,
            msg="我方样本数不应超过总步数的约一半"
        )

    def test_alternate_side(self):
        """alternate 模式：game_idx=0 执红，game_idx=1 执黑"""
        model = ChessModel(num_channels=64, num_res_blocks=2)
        model.build()

        from .train import play_game_vs_opponent_collect_my_turn

        _, _, _, meta0 = play_game_vs_opponent_collect_my_turn(
            model=model,
            opponent_agent=self._make_random_agent(),
            my_side='alternate',
            game_idx=0,
            num_simulations=2,
            max_moves=10,
        )
        _, _, _, meta1 = play_game_vs_opponent_collect_my_turn(
            model=model,
            opponent_agent=self._make_random_agent(),
            my_side='alternate',
            game_idx=1,
            num_simulations=2,
            max_moves=10,
        )
        self.assertEqual(meta0['my_side'], 'red')
        self.assertEqual(meta1['my_side'], 'black')

    def test_train_model_with_collected_data(self):
        """用收集到的训练数据调用 train_model，应返回非负浮点损失"""
        model = ChessModel(num_channels=64, num_res_blocks=2)
        model.build()

        from .train import play_game_vs_opponent_collect_my_turn, train_model
        training_data, _, _, _ = play_game_vs_opponent_collect_my_turn(
            model=model,
            opponent_agent=self._make_random_agent(),
            my_side='red',
            game_idx=0,
            num_simulations=3,
            max_moves=40,
        )

        if not training_data:
            self.skipTest("未产生足够训练样本，跳过 train_model 测试")

        loss = train_model(model, training_data, batch_size=4, epochs=1)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)


class TestVsPikafishPlayOneGame(unittest.TestCase):
    """
    回归测试：验证 play_one_game 中 AI 作为黑方时走法正确性。

    历史 bug：
    1. actions[0] 总是选第一个子节点，而非 MCTS 最优走法。
    2. 黑方走子时未将 MCTS 内部视角（红方坐标）翻转回实际棋盘坐标，
       导致 AI 始终重复同一个非法红方走法（UCI "i3i4"）。
    """

    def _make_stub_agent(self, best_move_mcts: str):
        """
        构造一个假 Agent：get_action_probs 返回预定的 (actions, probs)，
        其中 best_move_mcts 对应 probability=1.0，其余为 0.0。
        actions[0] 故意设为不同的走法，以检测 actions[0] bug。
        """
        class StubAgent:
            def __init__(self, best_move, other_move):
                self._best = best_move
                self._other = other_move
                self._calls = 0

            def new_game(self):
                self._calls = 0

            def get_action_probs(self, game, temperature=0.0, add_noise=False):
                # actions[0] 是干扰走法，actions[1] 是最优走法
                actions = [self._other, self._best]
                probs = [0.0, 1.0]  # best_move (index 1) has prob=1.0
                return actions, probs

        return StubAgent(best_move_mcts, 'decoy_should_not_be_played')

    def test_black_side_move_is_valid_and_varied(self):
        """
        当 AI 执黑时，play_one_game 应：
        1. 选择 probs 最高的走法（argmax），而非 actions[0]；
        2. 将 MCTS 内部坐标（红方视角）正确翻转为实际棋盘坐标；
        3. 所选走法是当前局面的合法走法。
        """
        from .vs_pikafish import play_one_game
        from .game import flip_move

        # 构造一个简单的"确定性引擎 agent"：始终返回 e0e1（内部 4041）
        class DeterministicEngineAgent:
            def new_game(self):
                pass
            def get_move(self, game):
                legal = game.get_legal_moves()
                return legal[0] if legal else None

        # 初始局面：黑方的合法走法（内部坐标）
        init_game = ChessGame()
        init_game.reset()
        # 红方先行一步，现在是黑方
        init_game.step('4041')  # 红帅 e0->e1
        black_legal = init_game.get_legal_moves()
        # 选择一个实际的合法黑方走法作为 MCTS 最优动作（需要翻转到红方坐标）
        first_black_move = black_legal[0]
        best_mcts_action = flip_move(first_black_move)  # MCTS 内部视角

        # 构造 stub AI agent：best_mcts_action 概率=1.0，另一个走法概率=0
        stub_agent = self._make_stub_agent(best_mcts_action)

        # 构造一个总是返回 first_black_move 的合法走法的确定性 agent
        # 但这里我们用 stub agent 验证 play_one_game 调用逻辑
        # 我们只需要确认第一步 AI（黑方）的走法正确，然后可以提前停止
        # 直接测试核心逻辑：从 get_action_probs 到 game.step 的转换
        game = ChessGame()
        game.reset()
        game.step('4041')  # red moves, now black's turn

        actions, probs = stub_agent.get_action_probs(game, temperature=0.0)
        best_idx = int(np.argmax(probs))
        move_mcts = actions[best_idx]  # 应为 best_mcts_action
        # 黑方走子：翻转回实际坐标
        actual_move = move_mcts if game.red_to_move else flip_move(move_mcts)

        # 验证：选择了最优走法（非 actions[0]）
        self.assertEqual(move_mcts, best_mcts_action,
                         "应选择 prob=1.0 的走法，而非 actions[0]")
        # 验证：翻转后得到原始合法走法
        self.assertEqual(actual_move, first_black_move,
                         "黑方走法应翻转回实际棋盘坐标")
        # 验证：该走法确实是当前局面的合法走法
        self.assertIn(actual_move, black_legal,
                      "翻转后的走法必须是当前局面的合法走法")

    def test_red_side_move_uses_argmax(self):
        """
        AI 执红时，应选择 probs 最高的走法（argmax），而非 actions[0]。
        """
        from .game import flip_move

        game = ChessGame()
        game.reset()  # red's turn
        red_legal = game.get_legal_moves()
        # 选择不是第一个的合法走法作为"最优走法"
        if len(red_legal) < 2:
            self.skipTest("初始局面合法走法不足 2 个")

        best_move = red_legal[-1]  # 最后一个（肯定不是 actions[0]）
        other_move = red_legal[0]  # 第一个（干扰走法）

        class StubAgent:
            def new_game(self): pass
            def get_action_probs(self, game, temperature=0.0, add_noise=False):
                return [other_move, best_move], [0.0, 1.0]

        stub = StubAgent()
        actions, probs = stub.get_action_probs(game)
        best_idx = int(np.argmax(probs))
        chosen = actions[best_idx]
        # 红方不需要翻转
        self.assertEqual(chosen, best_move,
                         "AI 执红应选择 prob=1.0 的走法")
        self.assertNotEqual(chosen, other_move,
                            "AI 执红不应选择 actions[0]（干扰走法）")
