from .actor import Actor
from .monte_carlo_ts import MCTS
from .hex import Hex
import numpy as np


class Agent:

    def __init__(self, actor: Actor, mcts: MCTS):
        self.actor = actor
        self.mcts = mcts


    def run_episode(self, env: Hex):
        replay_buffer = []

        while env.get_winner() == 0:
            distribution = self.mcts.search(n_simulations=100)
            # Choose action with probability proportional to the traverse count

            action = list(distribution.keys())[np.random.choice([_ for _ in range(len(distribution.keys()))], p=list(distribution.values()))]
            replay_buffer.append((env, distribution))
            
            env.make_action(action)
            env.display_board()
            
            self.mcts.set_new_root(action)
        print("winner:", env.get_winner())
        return replay_buffer, env

    



