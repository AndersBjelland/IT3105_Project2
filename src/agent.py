from .actor import Actor
from .monte_carlo_ts import MTCS
from .hex import Hex


class Agent:

    def __init__(self, actor: Actor, mcts: MTCS):
        self.actor = actor
        self.mcts = mcts
