import chess
from game.chess_env import ChessEnv, encode_board
from mcts.mcts import MCTS
from network.model import AlphaZeroNet
from game.move_encoding import ACTION_SIZE


def print_board(board):
    print("\nCurrent board:")
    print(board)
    print()


def play_vs_ai(model):
    env = ChessEnv()
    mcts = MCTS(model, simulations=10)  # slightly stronger search

    print("You are WHITE. Enter moves in UCI format (example: e2e4)\n")

    while not env.is_game_over():
        print_board(env.board)

        # ---- Human move ----
        if env.board.turn == chess.WHITE:
            move_str = input("Your move: ")

            try:
                move = chess.Move.from_uci(move_str)
                if move not in env.board.legal_moves:
                    print("Illegal move — try again.")
                    continue
                env.push(move)

            except:
                print("Invalid format. Example: e2e4")
                continue

        # ---- AI move ----
        else:
            print("AI thinking...")
            move, _ = mcts.search(env, encode_board)
            env.push(move)
            print("AI played:", move)

    print_board(env.board)
    print("Game result:", env.board.result())


if __name__ == "__main__":
    model = AlphaZeroNet(ACTION_SIZE)

    # Optional: load trained weights here later
    # model.load_state_dict(torch.load("model.pth"))

    play_vs_ai(model)