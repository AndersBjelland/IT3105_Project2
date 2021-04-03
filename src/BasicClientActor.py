import math
import numpy as np
from .BasicClientActorAbs import BasicClientActorAbs
from .monte_carlo_ts import MCTS
from .critic import Critic
from .actor import Actor
from .agent import Agent
from .encoder import HexEncoder
from .hex import Hex

class BasicClientActor(BasicClientActorAbs):

    def __init__(self, IP_address=None, verbose=True, actor_load_from='', critic_load_from=''):
        self.actor = Actor(learning_rate=0.01,
                            epsilon=0,
                            end_epsilon=0,
                            encoder=HexEncoder(padding=2),
                            load_from=actor_load_from
                            )
        self.critic = critic = Critic(learning_rate=0.01,
                            nn_loss="mse",
                            encoder=HexEncoder(padding=2),
                            load_from=critic_load_from
                            )
        self.prev_env = None
        self.mcts = None
        self.series_id = -1
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)

    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """

        # Flip the state and board if starting player is 2 (red)
        flip = False if self.starting_player == 1 else True
        
        env = Hex.create_env_from_state(state, flip, encoder=HexEncoder(padding=2))
        
        
        if self.mcts is None:
            self.mcts = MCTS(self.actor, env, self.critic)
        else:
            # get the action performed by the opponent
            similarity = env.board == self.prev_env.board
            action = np.where(similarity==False)
            action = (action[0][0], action[1][0])
            self.mcts.set_new_root(action)


        dist = self.mcts.search(n_simulations=1000, rollout_prob=1, c=1)
        action = max(dist, key=dist.get)
        env.make_action(action)
        self.mcts.set_new_root(action)

        self.prev_env = env

        # flip back if flip
        return action if not flip else (action[1], action[0])

    



    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return

        """
        self.series_id = series_id
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        ##############################
        print("Game over, these are the stats:")
        print('Winner: ' + str(winner))
        print('End state: ' + str(end_state))

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Series ended, these are the stats:")
        print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))


if __name__ == '__main__':
    bsa = BasicClientActor(verbose=False, actor_load_from='models/third6_6140.h5', critic_load_from='models/first6_6__critic_300')
    bsa.connect_to_server()
