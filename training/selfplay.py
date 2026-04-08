from game.chess_env import ChessEnv, encode_board
from mcts.mcts import MCTS


def self_play(model, games=2):
    data = []

    for _ in range(games):
        env = ChessEnv()
        mcts = MCTS(model, simulations=10)

        game_data = []

        while not env.is_game_over():
            move, policy = mcts.search(env, encode_board)
            state = encode_board(env.board)

            game_data.append((state, policy))
            env.push(move)

        result = env.result()

        for state, policy in game_data:
            data.append((state, policy, result))

    return data