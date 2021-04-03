from .agent import Agent
from .hex import Hex

import random


class Topp():

    def __init__(self, agents, env, number_of_games=10):
        self.agents = agents
        self.env = env
        self.number_of_games = number_of_games

    def play(self, agent1, agent2, env):
        while env.get_winner() == 0:
            if env.current_player == 1:
                action = agent1.actor.get_action(env)
            
            else:
                action = agent2.actor.get_action(env)
            
            env.make_action(action)
        winner = env.get_winner()
        env.reset()
        return winner

    def play_games(self, agent1, agent2, env):
        winner_scores = {agent1:0, agent2:0}

        for _ in range(self.number_of_games):
            # randomly shuffle to vary who starts
            
            players = [agent1, agent2] if self.number_of_games/2 <= _ else [agent2, agent1]
            player1 = players[0]
            player2 = players[1]
            winner = self.play(player1, player2, env)
            if winner == 1:
                winner_scores[player1] += 1
            else:
                winner_scores[player2] += 1
        
        return winner_scores

    def topp(self):
        winner_scores = {agent:0 for agent in self.agents}
        for player1 in range(len(self.agents)):
            for player2 in range(player1+1, len(self.agents)):
                agent1 = self.agents[player1]
                agent2 = self.agents[player2]
                result = self.play_games(agent1, agent2, self.env)

                # update winner scores
                winner_scores[agent1] += result[agent1]
                winner_scores[agent2] += result[agent2]


        return winner_scores



    
