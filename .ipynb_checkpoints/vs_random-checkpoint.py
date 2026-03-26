"""
蒸馏模型 VS 纯随机模型 评测脚本

用法:
    python -m AIchess.vs_random --model_path saved_model/model_distill.pth --n_games 20 --save_dir logs/vs_random
"""

import argparse
import random
import os
import datetime
import numpy as np

from .game import ChessGame, flip_move
from .model import ChessModel
from .mcts import MCTS

def evaluate_vs_random(model_path, n_games=20, num_simulations=50, max_moves=200, save_dir=None):
    print(f"正在加载模型: {model_path} ...")
    model = ChessModel()
    model.load(model_path)
    
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0
    
    # 如果指定了保存目录，则创建该目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"对局详细记录将保存在目录: {save_dir}")
        
    print(f"开始评测: 共 {n_games} 局，AI 思考模拟 {num_simulations} 次。")
    print("-" * 40)
    
    for game_idx in range(n_games):
        game = ChessGame()
        game.reset()
        
        # 记录本局的走法历史
        move_history = []
        
        # 交替执红（偶数局AI执红，奇数局AI执黑）
        ai_plays_red = (game_idx % 2 == 0)
        
        mcts = MCTS(model, num_simulations=num_simulations, debug_mcts=False)
        move_count = 0
        
        while not game.done and move_count < max_moves:
            is_red_turn = game.red_to_move
            
            if is_red_turn == ai_plays_red:
                # -------------------------
                # AI 的回合 (MCTS)
                # -------------------------
                actions, probs = mcts.get_action_probs(game, temperature=0.0, add_noise=False)
                if not actions:
                    break
                
                # 贪心策略：选择概率最高的走法
                best_idx = int(np.argmax(probs))
                chosen_action = actions[best_idx]  # MCTS内部视角的走法
                
                # 转换到实际物理棋盘的走法
                actual_action = chosen_action if is_red_turn else flip_move(chosen_action)
                
                game.step(actual_action)
                mcts.update_with_move(chosen_action)
                
            else:
                # -------------------------
                # 纯随机模型的回合
                # -------------------------
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    break
                    
                # 完全随机地选择一步合法走法
                actual_action = random.choice(legal_moves)
                
                # 为了让 AI 的 MCTS 树能继续复用，需要告诉 MCTS 对手走了什么
                mcts_action = actual_action if is_red_turn else flip_move(actual_action)
                
                game.step(actual_action)
                mcts.update_with_move(mcts_action)
            
            # 记录真实走法坐标
            current_player = "AI" if is_red_turn == ai_plays_red else "Random"
            move_history.append(f"{current_player}({actual_action})")
            
            move_count += 1
            
        winner = game.winner
        total_moves += move_count
        
        # 统计胜负
        if winner == 'draw' or winner is None:
            draws += 1
            res_text = "和棋"
        elif (winner == 'red' and ai_plays_red) or (winner == 'black' and not ai_plays_red):
            wins += 1
            res_text = "AI 胜"
        else:
            losses += 1
            res_text = "随机 胜"
            
        ai_color = "红" if ai_plays_red else "黑"
        print(f"[第 {game_idx + 1:2d}/{n_games} 局] 结果: {res_text:^5} | AI执{ai_color} | 步数: {move_count}")
        
        # --- 保存对局记录到文件 ---
        if save_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"game_{game_idx + 1:03d}_{res_text.replace(' ', '')}.txt"
            filepath = os.path.join(save_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"=== 对局详情 ===\n")
                f.write(f"时间: {timestamp}\n")
                f.write(f"AI 执色: {'红方' if ai_plays_red else '黑方'}\n")
                f.write(f"最终结果: {res_text}\n")
                f.write(f"总步数: {move_count}\n")
                f.write(f"终止原因: {game.terminate_reason if game.terminate_reason else '达到最大步数或无合法走法'}\n")
                f.write("-" * 30 + "\n")
                f.write("走法记录:\n")
                # 每10步换一行，方便查看
                for i in range(0, len(move_history), 10):
                    f.write(" ".join(move_history[i:i+10]) + "\n")
                f.write("-" * 30 + "\n")
        
    # 输出最终统计
    print("-" * 40)
    print("【最终评测结果】")
    print(f"AI 胜:   {wins} 局")
    print(f"随机 胜: {losses} 局")
    print(f"和 棋:   {draws} 局")
    print(f"AI 胜率: {wins / max(1, n_games) * 100:.1f}% (未算和棋)")
    score = (wins + 0.5 * draws) / max(1, n_games) * 100
    print(f"AI 得分率: {score:.1f}% (胜=1, 和=0.5)")
    print(f"平均步数: {total_moves / max(1, n_games):.1f} 步")


def main():
    parser = argparse.ArgumentParser(description="让蒸馏模型对战纯随机模型的评测脚本")
    parser.add_argument('--model_path', type=str, required=True, 
                        help="蒸馏出来的模型权重路径 (例如 saved_model/model_distill.pth)")
    parser.add_argument('--n_games', type=int, default=20, 
                        help="对局总数 (默认: 20)")
    parser.add_argument('--num_simulations', type=int, default=100, 
                        help="MCTS 模拟次数 (默认: 50，对付随机模型不需要太高)")
    parser.add_argument('--max_moves', type=int, default=300, 
                        help="一局的最大步数限制 (默认: 200)")
    parser.add_argument('--save_dir', type=str, default=None,
                        help="（可选）保存详细对局过程的文件夹路径，例如 logs/vs_random")
    
    args = parser.parse_args()
    evaluate_vs_random(
        model_path=args.model_path, 
        n_games=args.n_games, 
        num_simulations=args.num_simulations,
        max_moves=args.max_moves,
        save_dir=args.save_dir
    )

if __name__ == '__main__':
    main()