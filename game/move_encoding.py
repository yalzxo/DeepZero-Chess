import chess
import numpy as np

BOARD_SIZE = 8
ACTION_PLANES = 73


# Direction vectors
DIRECTIONS = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]

KNIGHT_DIRS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]


def square_to_coord(square):
    return divmod(square, 8)


def encode_move(move):
    """Convert chess.Move to AlphaZero-style action index."""
    fr, fc = square_to_coord(move.from_square)
    tr, tc = square_to_coord(move.to_square)

    dr = tr - fr
    dc = tc - fc

    # Sliding moves
    for dir_idx, (r, c) in enumerate(DIRECTIONS):
        for dist in range(1, 8):
            if dr == r * dist and dc == c * dist:
                plane = dir_idx * 7 + (dist - 1)
                return plane * 64 + fr * 8 + fc

    # Knight moves
    for k, (r, c) in enumerate(KNIGHT_DIRS):
        if dr == r and dc == c:
            plane = 56 + k
            return plane * 64 + fr * 8 + fc

    # Promotions (simplified)
    if move.promotion:
        plane = 64
        return plane * 64 + fr * 8 + fc

    return None


ACTION_SIZE = ACTION_PLANES * 64