from .actor import Actor
from .critic import Critic
from .monte_carlo_ts import MCTS
from .hex import Hex
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import random

import math


class Agent:

    def __init__(self, actor: Actor, critic: Critic):
        self.actor = actor
        self.critic = critic
        
    def run_episode(self, env: Hex, n_simulations: int, rollout_prob:float, action_strategy='probabilistic'):
        
        mcts = MCTS(self.actor, env=env)
        replay_buffer = []

        while env.get_winner() == 0:
            distribution = mcts.search(n_simulations=n_simulations, rollout_prob=rollout_prob)
            
            if action_strategy == 'greedy':
                # Choose action greedily
                action = max(distribution, key=distribution.get)
            elif action_strategy == 'probabilistic':
                action = random.choices(list(distribution.keys()), weights=list(distribution.values()), k=1)[0]
            
            replay_buffer.append((env.copy(), distribution))
            
            env.make_action(action)
            
            mcts.set_new_root(action)
        # include a label in each element of the replay buffer with the outcome for critic training
        winner = env.get_winner()
        for entry in replay_buffer:
            entry += (winner,)
        
        # reset env
        env.reset()
        
        return replay_buffer

    def train_agent(self, env: Hex, n_episodes:int, n_simulations: int, rollout_start_prob:float, rollout_end_prob:float, epochs=1, M=1, file_path=''):

        replay_buffer = []
        epsilon_decay_factor = (self.actor.epsilon - self.actor.end_epsilon)/n_episodes
        save_model_interval = math.floor(n_episodes/M)
        rollout_prob = rollout_start_prob
        reduction_factor = rollout_start_prob/n_episodes + rollout_end_prob
        for i in tqdm(range(n_episodes)):
            # Add new training examples to the replay buffer
            replay_buffer = self.run_episode(env=env, n_simulations=n_simulations, rollout_prob=rollout_prob) + replay_buffer
            # Only keep the last 5000 steps
            replay_buffer = replay_buffer[:300]
            # Train network for actor and critic
            print("--------------Actor training--------------")
            self.actor.end_of_episode(replay_buffer, epochs=epochs)
            print("--------------Critic training-------------")
            self.critic.end_of_episode(replay_buffer, epochs=epochs)
            # Update epsilon
            self.actor.epsilon = self.actor.epsilon - epsilon_decay_factor*(i+1)
            # Reduce prob of rollout
            rollout_prob -= reduction_factor

            # Save model to file
            if (i+1) % save_model_interval == 0:
                self.actor.model.save(file_path+str(i+1)+'.h5')
                self.critic.model.save(file_path+'_critic_'+str(i+1)+'.h5')


    def plot_distribution(self, distribution):
        print(distribution)
        plt.figure(figsize=(10,5))
        plt.bar(x=[_ + 1 for _ in range(len(distribution))], height=distribution.values(), tick_label=[str(_) for _ in distribution.keys()])
        plt.show()



