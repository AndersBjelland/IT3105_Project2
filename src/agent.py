from .actor import Actor
from .monte_carlo_ts import MCTS
from .hex import Hex
import numpy as np
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, actor: Actor, mcts: MCTS):
        self.actor = actor
        self.mcts = mcts


    def run_episode(self, env: Hex):
        replay_buffer = []

        while env.get_winner() == 0:
            distribution = self.mcts.search(n_simulations=1000)
            self.plot_distribution(distribution)
            # Choose action with probability proportional to the traverse count
            
            action = list(distribution.keys())[np.random.choice([_ for _ in range(len(distribution.keys()))], p=list(distribution.values()))]
            print("Chosen action is {}".format(action))
            replay_buffer.append((env, distribution))
            
            env.make_action(action)
            env.display_board()
            
            self.mcts.set_new_root(action)
        print("winner:", env.get_winner())
        return replay_buffer, env

    def plot_distribution(self, distribution):
        print(distribution)
        plt.figure(figsize=(10,5))
        plt.bar(x=[_ + 1 for _ in range(len(distribution))], height=distribution.values(), tick_label=[str(_) for _ in distribution.keys()])
        plt.show()



