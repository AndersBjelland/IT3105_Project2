from .actor import Actor
from .critic import Critic
from .monte_carlo_ts import MCTS
from .hex import Hex
from .topp import Topp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import random
from typing import Dict
from tensorflow import keras as KER

import math


class Agent:

    def __init__(self, actor: Actor, critic: Critic):
        self.actor = actor
        self.critic = critic
        
    def run_episode(self, env: Hex, n_simulations: int, rollout_prob: float, action_strategy='probabilistic'):
        
        mcts = MCTS(self.actor, env=env, critic=self.critic)
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
        winner = env.get_winner()
        
        replay_buffer = [entry + (1,) if entry[0].get_current_player() == winner else entry + (-1,) for entry in replay_buffer]
        
        env.reset()
        
        return replay_buffer

    def train_agent(self, env: Hex, n_episodes:int, n_simulations: int, start_rollout_prob: float, end_rollout_prob:float, epochs=1, M=1, file_path='', compete_rate=10, threshold=0.55):

        replay_buffer = []
        epsilon_decay_factor = (self.actor.epsilon - self.actor.end_epsilon)/n_episodes
        save_model_interval = math.floor(n_episodes/M)
        
        rollout_prob = start_rollout_prob
        decreasing_step = (start_rollout_prob - end_rollout_prob)/n_simulations

        # Checkpoint critic and actor
        self.actor.model.save('checkpoints/actor0.h5')
        self.critic.model.save('checkpoints/critic0.h5')

        for i in tqdm(range(n_episodes)):
            # Add new training examples to the replay buffer
            replay_buffer = self.run_episode(env=env, n_simulations=n_simulations, rollout_prob=rollout_prob) + replay_buffer
            # Only keep the last 50000 steps
            replay_buffer = replay_buffer[:50000]
            # Train network
            print("--------------actor-------------")
            self.actor.end_of_episode(replay_buffer, epochs=epochs)
            print("--------------critic------------")
            self.critic.end_of_episode(replay_buffer, epochs=epochs)

            # let the new network compete against current best network
            if n_episodes % compete_rate == 0:
                current_best_actor_nn = KER.models.load_model('checkpoints/actor' + i + '.h5')
                current_best_critic_nn = KER.models.load_model('checkpoints/critic' + i + '.h5')
                arena = Arena(self.actor, Actor(encoder=env.encoder, load_from='checkpoints/actor' + i + '.h5'), env, num_games=50)
                dist = arena.play_games()
                if dist[self.actor] < threshold:
                    print("------------new model did not beat current best-----------")
                    self.actor.model = current_best_actor_nn
                    self.critic.model = current_best_critic_nn
                else:
                    print("------------new model beat current best--------------")
                    # checkpoint new best actor and critic
                    self.actor.model.save('checkpoints/actor' + i + '.h5')
                    self.critic.model.save('checkpoints/critic' + i + '.h5')

            # Update epsilon
            self.actor.epsilon = self.actor.epsilon - epsilon_decay_factor*(i+1)

            # Save model to file
            if (i+1) % save_model_interval == 0:
                self.actor.model.save(file_path+str(i+1)+'.h5')
                self.critic.model.save(file_path+'_critic_'+str(i+1)+'.h5')

            # update rollout_prob
            rollout_prob -= decreasing_step

    
    def plot_distribution(self, distribution):
        print(distribution)
        plt.figure(figsize=(10,5))
        plt.bar(x=[_ + 1 for _ in range(len(distribution))], height=distribution.values(), tick_label=[str(_) for _ in distribution.keys()])
        plt.show()


class Arena:

    def __init__(self, actor1, actor2, env, num_games=50):
        self.actor1 = actor1
        self.actor2 = actor2
        self.env = env.copy()
        self.num_games = num_games

    def play_game(self, player1, player2):
        while self.env.get_winner() == 0:
            if self.env.current_player == 1:
                action = player1.get_action(self.env)
            else:
                action = player2.get_action(self.env)
            
            self.env.make_action(action)
        winner = self.env.get_winner()
        self.env.reset()
        return winner

    def play_games(self):
        winner_scores = {self.actor1:0, self.actor2:0}

        for _ in range(self.num_games):
            # randomly shuffle to vary who starts
            players = [self.actor1, self.actor2]
            random.shuffle(players)
            player1 = players[0]
            player2 = players[1]
            winner = self.play_game(player1, player2)
            if winner == 1:
                winner_scores[player1] += 1
            else:
                winner_scores[player2] += 1
        
        # normalize winner scores
        winner_scores[player1] /= self.num_games
        winner_scores[player2] /= self.num_games

        return winner_scores



