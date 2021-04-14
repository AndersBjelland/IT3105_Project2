from .agent import Agent
from .hex import Hex

import random


class Topp():

    def __init__(self, agents, env, number_of_games=10):
        self.agents = agents
        self.env = env
        self.number_of_games = number_of_games

    def play(self, agent1, agent2, env, stochastic):
        states = []
        while env.get_winner() == 0:
            states.append(env.copy())
            if env.current_player == 1:
                action = agent1.actor.get_action(env, stochastic=stochastic)
            
            else:
                action = agent2.actor.get_action(env, stochastic=stochastic)
            
            env.make_action(action)
        winner = env.get_winner()
        states.append(env.copy())
        env.reset()
        return winner, states

    def play_games(self, agent1, agent2, env, stochastic):
        winner_scores = {agent1:0, agent2:0}
        games = {}
        for _ in range(self.number_of_games):
            # randomly shuffle to vary who starts
            
            players = [agent1, agent2] if self.number_of_games/2 <= _ else [agent2, agent1]
            player1 = players[0]
            player2 = players[1]
            winner, states = self.play(player1, player2, env, stochastic)
            if (player1, player2) not in games:
                 games[(player1, player2)] = []
       
            games[(player1, player2)].append(states)
            
            if winner == 1:
                winner_scores[player1] += 1
            else:
                winner_scores[player2] += 1
        
        return winner_scores, games

    def topp(self, stochastic=True):
        winner_scores = {agent:0 for agent in self.agents}
        games_dict = {}
        for player1 in range(len(self.agents)):
            for player2 in range(player1+1, len(self.agents)):
                agent1 = self.agents[player1]
                agent2 = self.agents[player2]
                result, games = self.play_games(agent1, agent2, self.env, stochastic=stochastic)
                games_dict = {**games_dict, **games}

                # update winner scores
                winner_scores[agent1] += result[agent1]
                winner_scores[agent2] += result[agent2]


        return winner_scores, games_dict



    
