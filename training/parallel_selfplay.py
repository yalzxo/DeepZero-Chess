import torch
import copy
from multiprocessing import Pool, cpu_count

from game.chess_env import ChessEnv, encode_board
from mcts.mcts import MCTS


def play_single_game(model_state_dict):
    from network.model import AlphaZeroNet
    from game.move_encoding import ACTION_SIZE

    # Rebuild model inside process
    model = AlphaZeroNet(ACTION_SIZE)
    model.load_state_dict(model_state_dict)
    model.eval()

    env = ChessEnv()
    mcts = MCTS(model, simulations=15)

    game_data = []

    while not env.is_game_over():
        move, policy = mcts.search(env, encode_board)
        state = encode_board(env.board)

        game_data.append((state, policy))
        env.push(move)

    result = env.result()

    final_data = []
    for state, policy in game_data:
        final_data.append((state, policy, result))

    return final_data


def parallel_self_play(model, games=8):
    print("Running parallel self-play on", cpu_count(), "CPUs")

    model_state_dict = copy.deepcopy(model.state_dict())

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(
            play_single_game,
            [model_state_dict] * games
        )

    # Flatten list
    data = []
    for game_data in results:
        data.extend(game_data)

    return data
