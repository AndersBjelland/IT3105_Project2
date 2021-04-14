from src import actor
from src import oht
from src import games
import numpy as np


def from_oht_state(state):
    """Converts from the OHT state representation to ours. Note that while the OHT representation allows
    player 2 as the starting player, we DO NOT. Thus, we will always set player 1 as the starting player"""
    player, *board = state
    size = int(len(board)**0.5)

    assert len(board) == size * size

    # If we see an even number of pieces it means we started, if we see an odd number we were second
    current_started = sum(1 for x in board if x != 0) % 2 == 0
    flip = player == 1 and not current_started or player == 2 and current_started

    # Numpy makes it all simpler
    board = np.array(board).reshape((size, size))

    # Okay, so if we are player one, and we started, then we're all set.

    # If we're player one but did not start, then we're effectively player 2 in our representation
    # As such, we need to replace 1s with 2s and transpose the board.
    # By the same logic, we can find that we must do the same if we started as player 2, since
    # is really player 1 in our representation
    if flip:
        board = board.T
        ones = board == 1
        twos = board == 2
        board[ones] = 2
        board[twos] = 1

    game = games.Hex(size=size)
    game.grid = [[x for x in row] for row in board]
    game.current_player = 1 if current_started else 2

    return (not flip, game)


if __name__ == '__main__':
    agent = actor.SFPredictionAgent(
        path='/Users/akselborgen/Downloads/oht6x6resnet128-v44',
        size=6,
        policy_kind='greedy',
        name='v44',
    )

    print(f'current agent: {agent}')

    bsa = oht.BasicClientActor(agent, verbose=False, conversion=from_oht_state)
    bsa.connect_to_server()
