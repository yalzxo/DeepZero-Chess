import numpy as np
import torch

from game.move_encoding import encode_move, ACTION_SIZE


class Node:
    def __init__(self, board, parent=None, prior=0):
        self.board = board.copy()
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = prior

    @property
    def q(self):
        return self.value_sum / self.visits if self.visits > 0 else 0


class MCTS:
    def __init__(self, model, simulations=5, c_puct=1.0):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct

    def search(self, env, encode_fn):
        root = Node(env.board)

        for _ in range(self.simulations):
            self.simulate(root, encode_fn)

        policy = np.zeros(ACTION_SIZE)

        for move, child in root.children.items():
            idx = encode_move(move)
            if idx is not None:
                policy[idx] = child.visits

        total = np.sum(policy)
        if total > 0:
            policy /= total
        else:
            policy = np.ones_like(policy) / ACTION_SIZE

        best_move = max(
            root.children.items(),
            key=lambda x: x[1].visits
        )[0]
        # Add exploration noise at root
        alpha = 0.3
        epsilon = 0.25

        moves = list(root.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves))

        for i, move in enumerate(moves):
            root.children[move].prior = (
                (1 - epsilon) * root.children[move].prior
                + epsilon * noise[i]
            )

        return best_move, policy

    def simulate(self, node, encode_fn):
        # Terminal node
        if node.board.is_game_over():
            result = node.board.result()
            if result == "1-0":
                return 1
            elif result == "0-1":
                return -1
            return 0

        # Leaf node
        if not node.children:
            state = encode_fn(node.board)
            state = torch.tensor(state).unsqueeze(0).float()

            with torch.no_grad():
                logits, value = self.model(state)

            probs = torch.softmax(logits, dim=1).numpy()[0]

            for move in node.board.legal_moves:
                idx = encode_move(move)
                if idx is None:
                    continue

                child_board = node.board.copy()
                child_board.push(move)

                node.children[move] = Node(
                    child_board,
                    node,
                    probs[idx]
                )

            return value.item()

        # Selection (PUCT)
        best_score = -1e9
        best_child = None

        for move, child in node.children.items():
            u = self.c_puct * child.prior * \
                np.sqrt(node.visits + 1) / (child.visits + 1)

            score = child.q + u

            if score > best_score:
                best_score = score
                best_child = child

        v = self.simulate(best_child, encode_fn)

        # Backpropagation
        node.visits += 1
        node.value_sum += v

        return -v  # IMPORTANT: flip perspective
