import chess
from game.chess_env import ChessEnv, encode_board
from mcts.mcts import MCTS


def play_game(model1, model2):
    env = ChessEnv()
    mcts1 = MCTS(model1, simulations=15)
    mcts2 = MCTS(model2, simulations=15)

    while not env.is_game_over():
        if env.board.turn == chess.WHITE:
            move, _ = mcts1.search(env, encode_board)
        else:
            move, _ = mcts2.search(env, encode_board)

        env.push(move)

    return env.result()


def evaluate(new_model, old_model, games=6):
    wins = 0

    for _ in range(games):
        result = play_game(new_model, old_model)
        if result == 1:
            wins += 1

    return wins / games
