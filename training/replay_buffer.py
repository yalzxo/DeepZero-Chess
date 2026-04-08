import random
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size=5000):
        self.buffer = deque(maxlen=max_size)

    def add(self, data):
        """
        data = list of (state, policy, value)
        """
        self.buffer.extend(data)

    def sample(self, batch_size=64):
        return random.sample(
            self.buffer,
            min(len(self.buffer), batch_size)
        )

    def __len__(self):
        return len(self.buffer)