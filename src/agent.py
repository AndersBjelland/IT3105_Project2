from .actor import Actor
from .monte_carlo_ts import MCTS
from .hex import Hex
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import math


class Agent:

    def __init__(self, actor: Actor):
        self.actor = actor
        
    def run_episode(self, env: Hex, n_simulations: int):
        
        mcts = MCTS(self.actor, env=env)

        replay_buffer = []

        while env.get_winner() == 0:
            distribution = mcts.search(n_simulations=n_simulations)
            
            # Choose action greedily
            action = max(distribution, key=distribution.get)
            
            replay_buffer.append((env.copy(), distribution))
            
            env.make_action(action)
            
            mcts.set_new_root(action)
        
        env.reset()
        
        return replay_buffer

    def train_agent(self, env: Hex, n_episodes:int, n_simulations: int, epochs=1):
        replay_buffer = []
        

        for _ in tqdm(range(n_episodes)):
            # Add new training examples to the replay buffer
            replay_buffer = self.run_episode(env=env, n_simulations=n_simulations) + replay_buffer
            # Only keep the last 5000 steps
            replay_buffer = replay_buffer[:5000]
            self.actor.end_of_episode(replay_buffer, epochs=epochs)

            if _ == math.floor(n_episodes/2):
                self.actor.model.save('model1.h5')
                
        self.actor.model.save('model2.h5')

        

            





    

    def plot_distribution(self, distribution):
        print(distribution)
        plt.figure(figsize=(10,5))
        plt.bar(x=[_ + 1 for _ in range(len(distribution))], height=distribution.values(), tick_label=[str(_) for _ in distribution.keys()])
        plt.show()



