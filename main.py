from training.parallel_selfplay import parallel_self_play
from training.replay_buffer import ReplayBuffer
from training.train import train
from network.model import AlphaZeroNet
from game.move_encoding import ACTION_SIZE
                                                                                                                                                    

if __name__ == "__main__":
    model = AlphaZeroNet(ACTION_SIZE)
    replay_buffer = ReplayBuffer(max_size=5000)

    for iteration in range(10):
        print("Iteration:", iteration)

        data = parallel_self_play(model, games=8)

        replay_buffer.add(data)

        print("Replay buffer size:", len(replay_buffer))

        train(model, replay_buffer, epochs=1)