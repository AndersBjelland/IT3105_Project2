from .actor import Actor
from .monte_carlo_ts import MCTS
from .hex import Hex
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Agent:

    def __init__(self, actor: Actor):
        self.actor = actor
        


    def run_episode(self, env: Hex, n_simulations: int):
        
        mcts = MCTS(self.actor, env=env)

        replay_buffer = []

        while env.get_winner() == 0:
            distribution = mcts.search(n_simulations=n_simulations)
            
            # Choose action with probability proportional to the traverse count
            action = list(distribution.keys())[np.random.choice([_ for _ in range(len(distribution.keys()))], p=list(distribution.values()))]
            replay_buffer.append((env, distribution.values()))
            
            env.make_action(action)
            
            mcts.set_new_root(action)
        
        env.reset()
        
        return replay_buffer

    def train_agent(self, env: Hex, n_episodes:int, n_simulations: int):
        #replay_buffer = []

        for _ in tqdm(range(n_episodes)):
            replay_buffer = self.run_episode(env=env, n_simulations=n_simulations)
            self.actor.end_of_episode(replay_buffer)
            





    

    def plot_distribution(self, distribution):
        print(distribution)
        plt.figure(figsize=(10,5))
        plt.bar(x=[_ + 1 for _ in range(len(distribution))], height=distribution.values(), tick_label=[str(_) for _ in distribution.keys()])
        plt.show()



