from typing import Optional, Iterator, Tuple
from matplotlib import pyplot as plt
import tensorflow as tf
import math
# We utilize a representation of a diamond/triangle as a two dimensional array.
# A diamond is shifted such that:
#     A
#    B C               A C F
#   D E F      =>      B E H
#    G H               D G I
#     I

# A position on the board can either be empty or occupied by one of the players
CellState = int
EMPTY: CellState = 0
PLAYER1: CellState = 1
PLAYER2: CellState = 2

# The position of a cell in the actual representation (i.e. array indices)
Position = Tuple[int, int]
# A direction is stored as a simple integer, with `offset` providing the direction it corresponds to
Direction = Tuple[int, int]
# Constants for the directions
DIRECTIONS = SE, NW, NE, SW, E, W = [
    (+1, 0), (-1, 0), (0, +1), (0, -1), (+1, -1), (-1, +1)
]

# A move is identified by the position of a peg, and what direction we will move it in
Action = Tuple[Position, Direction]

# Reward is a simple number
Reward = float


class Hex:
    def __init__(self, size: int, initial_player: CellState = PLAYER1):
        """Initialize a PegSolitare with the provided grid"""
        self.size = size
        self.current_player = initial_player
        self.grid = [[EMPTY for _ in range(size)] for _ in range(size)]

    def index(self, x: int, y: int) -> Optional[CellState]:
        """Index the value at the specific index, defaulting to `None` if out of bounds"""
        if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[y]):
            return self.grid[y][x]
        return None

    def set_cell(self, x: int, y: int, state: CellState):
        """Set the cell at position `(x, y)` to `state`. Does nothing if out of bounds"""
        if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[y]):
            self.grid[y][x] = state

    def copy(self) -> 'Hex':
        """Constructs a copy of `self`"""
        copy = Hex(self.size, self.current_player)

        for y in range(len(self.grid)):
            for x in range(len(self.grid[y])):
                copy.grid[y][x] = self.grid[y][x]

        return copy

    def neighbours(self, x: int, y: int) -> Iterator[Position]:
        """Returns the neighbours of the position `(x, y)`"""
        deltas = [(+1, 0), (-1, 0), (0, +1), (0, -1), (+1, -1), (-1, +1)]
        return ((x + dx, y + dy) for (dx, dy) in deltas if self.index(x + dx, y + dy) is not None)

    def nodes(self):
        """Returns all the positions of the board"""
        return [(y, x) for y in range(len(self.grid)) for x in range(len(self.grid[y]))]

    def is_final(self) -> bool:
        """Returns true iff the game is finished."""
        no_moves_available = next(self.available_actions(), None) is None
        somebody_won = self.winner() is not None
        return no_moves_available or somebody_won

    def winner(self) -> Optional[int]:
        # To coincide with the OHT, we have Player 1 running from northeast to southwest, and Player 2 the other way
        players = [PLAYER1, PLAYER2]
        goals = [
            lambda t: t[1] == len(self.grid) - 1,
            lambda t: t[0] == len(self.grid[0]) - 1
        ]
        initial = [
            [(x, 0) for x in range(len(self.grid[0]))
             if self.index(x, 0) == PLAYER1],
            [(0, y) for y in range(len(self.grid)) if self.index(0, y) == PLAYER2],
        ]

        for (player, goal, stack) in zip(players, goals, initial):
            seen = set(stack)

            while stack:
                node = stack.pop()
                if goal(node):
                    return player

                for n in self.neighbours(*node):
                    if self.index(*n) == player and n not in seen:
                        seen.add(n)
                        stack.append(n)

        return None

    def available_actions(self) -> Iterator[Action]:
        """Returns an iterator over the available actions given the current board state."""
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if self.index(x, y) == EMPTY:
                    yield (x, y)

    def next_state(self, action: Action) -> 'Hex':
        """Performs the action `action` if it is valid, otherwise leaves the board as-is"""
        (x, y) = action
        next_player = PLAYER2 if self.current_player == PLAYER1 else PLAYER1

        if self.index(x, y) != EMPTY:
            return self

        copy = self.copy()
        copy.set_cell(x, y, self.current_player)
        copy.current_player = next_player

        return copy

    def _fig(self, ax=None, close=True):
        nodes = self.nodes()
        edges = set(
            [tuple(sorted((p, q))) for p in nodes for q in self.neighbours(*p)]
        )

        if ax is None:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        colors = {
            EMPTY: 'white',
            PLAYER1: 'red',
            PLAYER2: 'blue',
        }

        def drawing_coords(x, y):
            ANGLE = math.pi / 6
            dxx, dyx = math.cos(-ANGLE), math.sin(-ANGLE)
            dxy, dyy = math.cos(math.pi+ANGLE), math.sin(math.pi+ANGLE)
            tx = dxx * x + dxy * y
            ty = dyx * x + dyy * y
            return (tx / 100, ty / 100)

        for (p, q) in edges:
            (x0, y0) = drawing_coords(*p)
            (x1, y1) = drawing_coords(*q)

            if p[1] == q[1] == 0 or p[1] == q[1] == self.size - 1:
                c = colors[PLAYER1]
            elif p[0] == q[0] == 0 or p[0] == q[0] == self.size - 1:
                c = colors[PLAYER2]
            else:
                c = 'grey'

            ax.plot([x0, x1], [y0, y1], '-', c=c)

        for (x, y) in nodes:
            ax.scatter(
                *drawing_coords(x, y), s=64**2 * (4 / self.size)**2,
                c=colors[self.index(x, y)], edgecolor='black', zorder=100, marker='H',
            )

        ax.set_aspect('equal')
        ax.axis(False)

        if close:
            plt.close(fig)

        return fig

    def _repr_svg_(self):
        fig = self._fig(close=True)
        return display(fig)

    def __hash__(self):
        return hash(tuple(map(tuple, self.grid)))

    def __eq__(self, other):
        return tuple(self.grid) == tuple(other.grid)


def normalized_encoder(hex: Hex):
    """Constructs a board that is rotated such that both players experience the same 'perspective' when fed into a neural network"""
    # We will utilize two feature layers: "self" and "opponent"
    player, opponent = (PLAYER1, PLAYER2) if hex.current_player == PLAYER1 else (
        PLAYER2, PLAYER1)

    mirror = player == PLAYER2

    N = len(hex.grid)
    grid = hex.grid if not mirror else [
        [hex.grid[x][y] for x in range(N)] for y in range(N)]

    player_plane = [[1.0 if x == player else 0.0 for x in row]
                    for row in grid]
    opponent_plane = [[1.0 if x == opponent else 0.0 for x in row]
                      for row in grid]

    return tf.stack([
        tf.convert_to_tensor(player_plane),
        tf.convert_to_tensor(opponent_plane),
    ], axis=2)


def current_player_encoder(hex: Hex):
    grid = hex.grid
    p1 = [[1.0 if x == PLAYER1 else 0.0 for x in r] for r in grid]
    p2 = [[1.0 if x == PLAYER2 else 0.0 for x in r] for r in grid]
    p1_to_play = [[hex.current_player == PLAYER1 for x in r] for r in grid]
    p2_to_play = [[hex.current_player == PLAYER2 for x in r] for r in grid]

    return tf.stack([
        p1,
        p2,
        tf.convert_to_tensor(p1_to_play, dtype=tf.float32),
        tf.convert_to_tensor(p2_to_play, dtype=tf.float32),
    ], axis=2)
