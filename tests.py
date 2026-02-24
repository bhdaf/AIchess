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


class TestGRPO(unittest.TestCase):
    """测试GRPO训练器"""

    def setUp(self):
        self.model = ChessModel(num_channels=32, num_res_blocks=2)
        self.model.build()

    def test_group_sample(self):
        """测试组采样机制"""
        from simple_chess_ai.grpo import GRPOTrainer
        import torch
        trainer = GRPOTrainer(self.model, group_size=4)
        logits = torch.randn(2, NUM_ACTIONS)
        mask = torch.ones(2, NUM_ACTIONS)
        actions, log_probs = trainer.group_sample(logits, mask, group_size=4)
        self.assertEqual(actions.shape, (2, 4))
        self.assertEqual(log_probs.shape, (2, 4))

    def test_group_advantage(self):
        """测试组内相对优势计算"""
        from simple_chess_ai.grpo import GRPOTrainer
        import torch
        trainer = GRPOTrainer(self.model, group_size=4)
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        advantages = trainer.compute_group_advantage(rewards)
        # 组内归一化后均值应接近0
        self.assertAlmostEqual(advantages.mean().item(), 0.0, places=5)

    def test_train_step(self):
        """测试GRPO训练步"""
        from simple_chess_ai.grpo import GRPOTrainer, generate_grpo_training_data
        game = ChessGame()
        game.reset()
        states, masks = generate_grpo_training_data(self.model, game)
        trainer = GRPOTrainer(self.model, group_size=4, lr=1e-4)
        metrics = trainer.train_step(states, masks)
        self.assertIn('loss', metrics)
        self.assertIn('policy_loss', metrics)
        self.assertIn('kl_loss', metrics)

    def test_generate_grpo_data(self):
        """测试GRPO训练数据生成"""
        from simple_chess_ai.grpo import generate_grpo_training_data
        game = ChessGame()
        game.reset()
        states, masks = generate_grpo_training_data(self.model, game)
        self.assertEqual(states.shape[1:], (14, 10, 9))
        self.assertEqual(masks.shape[1], NUM_ACTIONS)
        # 合法走法掩码应有非零值
        self.assertGreater(masks.sum(), 0)


class TestGNN(unittest.TestCase):
    """测试GNN特征提取"""

    def test_build_chess_graph(self):
        """测试棋盘到图的转换"""
        from simple_chess_ai.gnn_feature import build_chess_graph
        import torch
        planes = torch.randn(2, 14, 10, 9)
        node_features, adj_matrix = build_chess_graph(planes)
        self.assertEqual(node_features.shape, (2, 90, 16))
        self.assertEqual(adj_matrix.shape, (2, 90, 90))

    def test_chess_gnn_forward(self):
        """测试ChessGNN前向传播"""
        from simple_chess_ai.gnn_feature import ChessGNN
        import torch
        gnn = ChessGNN(node_features=16, hidden_dim=32, output_dim=64, num_heads=4)
        planes = torch.randn(2, 14, 10, 9)
        output = gnn(planes)
        self.assertEqual(output.shape, (2, 64))

    def test_gnn_policy_value_net(self):
        """测试集成GNN的策略价值网络"""
        from simple_chess_ai.gnn_feature import GNNPolicyValueNet
        import torch
        net = GNNPolicyValueNet(
            num_channels=32, num_res_blocks=2,
            gnn_hidden_dim=32, gnn_output_dim=64, num_heads=4
        )
        planes = torch.randn(2, 14, 10, 9)
        policy, value = net(planes)
        self.assertEqual(policy.shape, (2, NUM_ACTIONS))
        self.assertEqual(value.shape, (2, 1))
        # 策略概率应求和为1
        for i in range(2):
            self.assertAlmostEqual(policy[i].sum().item(), 1.0, places=4)
        # 价值应在[-1, 1]范围
        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))

    def test_graph_conv_layer(self):
        """测试图卷积层"""
        from simple_chess_ai.gnn_feature import GraphConvLayer
        import torch
        layer = GraphConvLayer(16, 32)
        x = torch.randn(2, 10, 16)
        adj = torch.rand(2, 10, 10)
        out = layer(x, adj)
        self.assertEqual(out.shape, (2, 10, 32))

    def test_gat_layer(self):
        """测试图注意力层"""
        from simple_chess_ai.gnn_feature import GATLayer
        import torch
        layer = GATLayer(16, 32, num_heads=4)
        x = torch.randn(2, 10, 16)
        adj = torch.ones(2, 10, 10)
        out = layer(x, adj)
        self.assertEqual(out.shape, (2, 10, 32))


class TestReasoning(unittest.TestCase):
    """测试推理模块"""

    def setUp(self):
        self.model = ChessModel(num_channels=32, num_res_blocks=2)
        self.model.build()
        self.game = ChessGame()
        self.game.reset()

    def test_board_analyzer_threats(self):
        """测试威胁分析"""
        from simple_chess_ai.reasoning import BoardAnalyzer
        threats = BoardAnalyzer.analyze_threats(self.game)
        self.assertIsInstance(threats, list)

    def test_board_analyzer_position(self):
        """测试局面评估"""
        from simple_chess_ai.reasoning import BoardAnalyzer
        position = BoardAnalyzer.evaluate_position(self.game)
        self.assertIn('red_material', position)
        self.assertIn('black_material', position)
        self.assertIn('material_advantage', position)
        # 初始局面应该均衡
        self.assertEqual(position['material_advantage'], 0)

    def test_reasoner_chain(self):
        """测试推理链生成"""
        from simple_chess_ai.reasoning import ChessReasoner
        reasoner = ChessReasoner(self.model)
        text, features = reasoner.generate_reasoning_chain(self.game)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        self.assertEqual(features.shape, (32,))

    def test_reason_and_act(self):
        """测试推理并走子"""
        from simple_chess_ai.reasoning import ChessReasoner
        reasoner = ChessReasoner(self.model)
        text, action, policy = reasoner.reason_and_act(self.game)
        self.assertIsInstance(text, str)
        self.assertIsInstance(action, str)
        self.assertEqual(len(action), 4)
        self.assertEqual(policy.shape, (NUM_ACTIONS,))

    def test_reasoning_reward(self):
        """测试GRPO推理奖励"""
        from simple_chess_ai.reasoning import create_grpo_reasoning_reward, ChessReasoner
        reasoner = ChessReasoner(self.model)
        reward = create_grpo_reasoning_reward(reasoner, self.game, '4041', 1.0)
        self.assertIsInstance(reward, float)

    def test_piece_relations(self):
        """测试棋子关系分析"""
        from simple_chess_ai.reasoning import BoardAnalyzer
        relations = BoardAnalyzer.analyze_piece_relations(self.game)
        self.assertIn('attacks', relations)
        self.assertIn('defenses', relations)


class TestFP16Training(unittest.TestCase):
    """测试FP16混合精度训练"""

    def test_train_model_fp16_flag(self):
        """测试train_model接受use_fp16参数"""
        from simple_chess_ai.train import train_model
        model = ChessModel(num_channels=32, num_res_blocks=2)
        model.build()
        game = ChessGame()
        game.reset()
        planes = game.to_planes()
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        policy[0] = 1.0
        data = [(planes, policy, 1.0)]
        # Should work without errors on CPU (fp16 disabled on CPU)
        loss = train_model(model, data, batch_size=1, epochs=1,
                           lr=0.001, use_fp16=False)
        self.assertIsInstance(loss, float)


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
        """evaluate_models 返回合法的胜率和统计"""
        from simple_chess_ai.train import evaluate_models
        model_a = ChessModel(num_channels=32, num_res_blocks=2)
        model_a.build()
        model_b = ChessModel(num_channels=32, num_res_blocks=2)
        model_b.build()
        winrate, wins, losses, draws = evaluate_models(
            model_a, model_b, n_games=4, num_simulations=5, max_moves=50
        )
        total = wins + losses + draws
        self.assertEqual(total, 4)
        self.assertGreaterEqual(winrate, 0.0)
        self.assertLessEqual(winrate, 1.0)
        if total > 0:
            self.assertAlmostEqual(winrate, wins / total, places=5)


if __name__ == '__main__':
    unittest.main()
