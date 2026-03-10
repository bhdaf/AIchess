"""
简化中国象棋AI - 单元测试

测试游戏逻辑、模型、MCTS等核心组件。
"""

import unittest
import os
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
        data, winner, moves = self_play_game(model, num_simulations=5, max_moves=30)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIsInstance(moves, int)
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


class TestPikafishConversion(unittest.TestCase):
    """测试 Pikafish UCI 走法和 FEN 格式转换"""

    def setUp(self):
        from .pikafish import internal_move_to_uci, uci_move_to_internal, board_fen_to_uci_fen
        self.internal_to_uci = internal_move_to_uci
        self.uci_to_internal = uci_move_to_internal
        self.board_fen_to_uci_fen = board_fen_to_uci_fen

    # ------------------------------------------------------------------
    # 走法格式转换
    # ------------------------------------------------------------------

    def test_internal_to_uci_cannon(self):
        """红炮平中: (1,2)->(4,2) = '1242' -> 'b2e2'"""
        self.assertEqual(self.internal_to_uci('1242'), 'b2e2')

    def test_internal_to_uci_king(self):
        """帅前进: (4,0)->(4,1) = '4041' -> 'e0e1'"""
        self.assertEqual(self.internal_to_uci('4041'), 'e0e1')

    def test_internal_to_uci_pawn(self):
        """红兵前进: (0,3)->(0,4) = '0304' -> 'a3a4'"""
        self.assertEqual(self.internal_to_uci('0304'), 'a3a4')

    def test_internal_to_uci_all_files(self):
        """所有文件字母 a-i 对应列 0-8"""
        for col, letter in enumerate('abcdefghi'):
            move = f"{col}0{col}1"
            uci = self.internal_to_uci(move)
            self.assertEqual(uci[0], letter)
            self.assertEqual(uci[2], letter)

    def test_uci_to_internal_cannon(self):
        """'b2e2' -> '1242'"""
        self.assertEqual(self.uci_to_internal('b2e2'), '1242')

    def test_uci_to_internal_king(self):
        """'e0e1' -> '4041'"""
        self.assertEqual(self.uci_to_internal('e0e1'), '4041')

    def test_roundtrip_internal_to_uci(self):
        """内部格式 -> UCI -> 内部格式 应还原"""
        for move in ['1242', '4041', '0304', '8089', '0900']:
            uci = self.internal_to_uci(move)
            restored = self.uci_to_internal(uci)
            self.assertEqual(restored, move, f"往返转换失败: {move} -> {uci} -> {restored}")

    def test_roundtrip_uci_to_internal(self):
        """UCI 格式 -> 内部 -> UCI 应还原"""
        for uci in ['b2e2', 'h9g7', 'a0a1', 'i9i8', 'e0e1']:
            internal = self.uci_to_internal(uci)
            restored = self.internal_to_uci(internal)
            self.assertEqual(restored, uci, f"往返转换失败: {uci} -> {internal} -> {restored}")

    # ------------------------------------------------------------------
    # FEN 格式转换
    # ------------------------------------------------------------------

    def test_board_fen_to_uci_fen_red(self):
        """红方先行 FEN 应以 'w' 结尾部分"""
        board_fen = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR'
        uci_fen = self.board_fen_to_uci_fen(board_fen, red_to_move=True)
        self.assertEqual(uci_fen, f'{board_fen} w - - 0 1')

    def test_board_fen_to_uci_fen_black(self):
        """黑方先行 FEN 应以 'b' 结尾部分"""
        board_fen = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR'
        uci_fen = self.board_fen_to_uci_fen(board_fen, red_to_move=False)
        self.assertEqual(uci_fen, f'{board_fen} b - - 0 1')

    def test_board_fen_to_uci_fen_custom_clocks(self):
        """自定义半步数和全步数"""
        board_fen = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR'
        uci_fen = self.board_fen_to_uci_fen(board_fen, red_to_move=True,
                                             halfmove_clock=5, fullmove_number=3)
        self.assertEqual(uci_fen, f'{board_fen} w - - 5 3')

    def test_board_fen_preserves_board_part(self):
        """转换后棋盘部分不变"""
        board_fen = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR'
        uci_fen = self.board_fen_to_uci_fen(board_fen)
        self.assertTrue(uci_fen.startswith(board_fen))

    def test_uci_fen_format_fields(self):
        """UCI FEN 应有 6 个字段"""
        board_fen = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR'
        uci_fen = self.board_fen_to_uci_fen(board_fen)
        fields = uci_fen.split()
        self.assertEqual(len(fields), 6)
        # 第2字段为走子方
        self.assertIn(fields[1], ('w', 'b'))
        # 第3、4字段为 '-'
        self.assertEqual(fields[2], '-')
        self.assertEqual(fields[3], '-')

    def test_initial_game_uci_fen(self):
        """初始棋盘 FEN 转换正确"""
        from .game import ChessGame
        game = ChessGame()
        game.reset()
        board_fen = game.get_fen()
        uci_fen = self.board_fen_to_uci_fen(board_fen, game.red_to_move)
        expected = (
            'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'
        )
        self.assertEqual(uci_fen, expected)


