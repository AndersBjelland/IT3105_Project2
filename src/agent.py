from .actor import Actor
from .critic import Critic
from .monte_carlo_ts import MCTS
from .hex import Hex

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import random
from typing import Dict
from tensorflow import keras as KER
import pickle

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

        replay_buffer = [entry + (1,) if entry[0].get_current_player() == winner else entry + (0,) for entry in replay_buffer]
        
        env.reset()
        
        return replay_buffer

    def train_agent(self, env: Hex, n_episodes:int, n_simulations: int, start_rollout_prob: float, end_rollout_prob:float, epochs=1, M=1, file_path='', compete=False, compete_rate=10, threshold=0.55, compete_num_games=50, old_replay=None, search_in_comp=True, simulations_in_comp=100):

        replay_buffer = [] if old_replay is None else old_replay
        epsilon_decay_factor = (self.actor.epsilon - self.actor.end_epsilon)/n_episodes
        save_model_interval = math.floor(n_episodes/(M-1))
        
        rollout_prob = start_rollout_prob
        decreasing_step = (start_rollout_prob - end_rollout_prob)/n_simulations

        # run env through the critic and actor network to set weights
        #self.actor.get_action(env)
        #self.critic.get_value(env)
        shape = [None] + env.encoder.get_encoding().shape[1:]
        self.actor.model.build(shape)
        self.critic.model.build(shape)

        # checkpoint initial model as best
        self.actor.model.save(file_path + 'checkpointActor0.h5')
        self.critic.model.save(file_path + 'checkpointCritic0.h5')
        best_index = 0

        for i in tqdm(range(n_episodes)):

            # Save model to file
            if (i) % save_model_interval == 0:
                self.actor.model.save(file_path+str(i)+'.h5')
                self.critic.model.save(file_path+'_critic_'+str(i)+'.h5')

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
            if (i+1) % compete_rate == 0 and compete:
                current_best_actor_nn = KER.models.load_model(file_path + 'checkpointActor' + str(best_index) + '.h5')
                current_best_critic_nn = KER.models.load_model(file_path + 'checkpointCritic' + str(best_index) + '.h5')
                current_best_actor = Actor(encoder=env.encoder, load_from=file_path + 'checkpointActor' + str(best_index) + '.h5')
                current_best_critic = Critic(learning_rate=0.001, nn_loss='mse', encoder=env.encoder, load_from=file_path + 'checkpointCritic' + str(best_index) + '.h5')
                current_best_agent = Agent(current_best_actor, current_best_critic)
                arena = Arena(self, current_best_agent, env, num_games=compete_num_games)
                print("---------------competing-----------")
                dist = arena.play_games(search_in_comp,simulations_in_comp)
                print("new_model won " + str(100*dist[self.actor]) + "%")
                print("current best won " + str(100*dist[current_best_actor]) + "%")
                print(dist)
                if dist[self.actor] < threshold: 
                    print("------------new model did not beat current best-----------")
                    self.actor.model = current_best_actor_nn
                    self.critic.model = current_best_critic_nn
                else:
                    print("------------new model beat current best--------------")
                    # checkpoint new best actor and critic
                    self.actor.model.save(file_path + 'checkpointActor' + str(i) + '.h5')
                    self.critic.model.save(file_path + 'checkpointCritic' + str(i) + '.h5')
                    best_index = i

            # Update epsilon
            self.actor.epsilon = self.actor.epsilon - epsilon_decay_factor*(i+1)

            # update rollout_prob
            rollout_prob -= decreasing_step

        # save last model to file
        if (n_episodes) % save_model_interval == 0:
            self.actor.model.save(file_path+str(i+1)+'.h5')
            self.critic.model.save(file_path+'_critic_'+str(i+1)+'.h5')


        # save last replay buffer
        with open(file_path + 'replay_buffer_6_6.pkl', 'wb') as f:
            pickle.dump(replay_buffer, f)




    def plot_distribution(self, distribution):
        print(distribution)
        plt.figure(figsize=(10,5))
        plt.bar(x=[_ + 1 for _ in range(len(distribution))], height=distribution.values(), tick_label=[str(_) for _ in distribution.keys()])
        plt.show()


class Arena:

    def __init__(self, agent1, agent2, env, num_games=50):
        self.agent1 = agent1
        self.agent2 = agent2
        self.env = env.copy()
        self.num_games = num_games

    def play_game(self, player1, player2, search, n_simulations):
        if search:
            mcts1 = MCTS(player1.actor, self.env, player1.critic)
            mcts2 = MCTS(player2.actor, self.env, player2.critic)

        while self.env.get_winner() == 0:
            if self.env.current_player == 1:
                if search:
                    dist = mcts1.search(n_simulations, 0)
                    action = max(dist, key=dist.get)
                    mcts1.set_new_root(action)
                else:
                    action = player1.actor.get_action(self.env)
            else:
                if search:
                    dist = mcts2.search(n_simulations, 0)
                    action = max(dist, key=dist.get)
                    mcts2.set_new_root(action)
                else:
                    action = player2.actor.get_action(self.env)
            
            self.env.make_action(action)
        winner = self.env.get_winner()
        if player1 == self.agent1:
            print('agent1 is player1')
        else:
            print('agent2 is player2')
        print("winner: ", winner)
        self.env.reset()
        return winner

    def play_games(self, search, n_simulations):
        winner_scores = {self.agent1:0, self.agent2:0}

        for _ in range(self.num_games):
            
            # Let player 1 and 2 be the starting player for the same number of games
            players = [self.agent1, self.agent2] if self.num_games/2 <= _ else [self.agent2, self.agent1]
            
            player1 = players[0]
            player2 = players[1]
            winner = self.play_game(player1, player2, search, n_simulations)
            if winner == 1:
                winner_scores[player1] += 1
            else:
                winner_scores[player2] += 1
        
        # normalize winner scores
        winner_scores[player1] /= self.num_games
        winner_scores[player2] /= self.num_games

        return winner_scores



