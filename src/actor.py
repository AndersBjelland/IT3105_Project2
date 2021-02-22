from .hexagonal_grid import Cell
from .hex import Hex
from .encoder import Encoder

from tensorflow import keras as KER
import tensorflow as tf
import numpy as np
import random

from typing import Tuple, Callable, List, Dict

"""
An actor using a convolutional network.
"""

class Actor:

    def __init__(self, learning_rate: float, epsilon: float, end_epsilon: float, nn_shape: Tuple, filters: Tuple, kernel_sizes: Tuple[Tuple[int,int]],  
            nn_opt: "optimizer", nn_activation: "ActivationFunc", nn_last_activation: "ActivationFunc", nn_loss: "LossFunc", encoder: Encoder):

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.end_epsilon = end_epsilon
        self.encoder = encoder
        self.model = KER.models.Sequential()

        opt = eval('KER.optimizers.'+nn_opt)
        loss = eval('KER.losses.' + nn_loss) if type(nn_loss) == str else nn_loss

        # Now we build the neural network. If filters and kernel_sizes are not empty we build a CNN, else a dense network.
        if len(filters) != 0 and len(kernel_sizes) != 0:
            for i in range(len(filters)):
                self.model.add(KER.layers.Conv2D(filters=filters[i], kernel_size=kernel_sizes[i], activation=nn_activation))
            
            self.model.add(KER.layers.Flatten())
        
        for i in range(len(nn_shape)):
            if i != len(nn_shape)-1:
                self.model.add(KER.layers.Dense(nn_shape[i],activation=nn_activation))
            else:
                self.model.add(KER.layers.Dense(nn_shape[i], activation=nn_last_activation))

        self.model.compile(optimizer=opt(lr=self.learning_rate), loss=loss, metrics=['accuracy'])

    def get_action(self, env):
        feature_maps = self.encoder.encode(env)

        prob_dist = self.model(feature_maps).numpy().reshape((-1,))

        # legal moves is a list with tuples like [ (1, (1,2)), (0,(0,0)) ]
        legal_moves = env.available_actions_binary()
        
        # Combine probability dist from the network and legal moves to dict on the form {action:probability, ...}
        prob_dist = {legal_moves[i][1]:prob_dist[i] for i in range(len(prob_dist)) if legal_moves[i][0]}
        
        # Return an action in a epsilon greedy manner
        if random.random() <= self.epsilon:
            return random.choice(list(prob_dist.keys()))
        
        return max(prob_dist, key=prob_dist.get)
        
        

    def end_of_episode(self, replay_buffer, epochs=1, batch_size=128):
        x,y = self.convert_to_network_input(replay_buffer)
        return self.model.fit(x,y, epochs=epochs, batch_size=batch_size)


    def convert_to_network_input(self, replay_buffer):
        x = tf.concat([self.encoder.encode(entry[0]) for entry in replay_buffer], 0)        
        y = tf.convert_to_tensor([self.to_full_dist(entry) for entry in replay_buffer])
        return x,y

    def to_full_dist(self, state_dist_pair: Tuple[Hex, Dict[Tuple[int, int], float]]):
        """
        Takes in a probability distribution over available actions in env

        Returns:
        Transfered to a vector with lenght equal to the maximal number of availablre actions in env, i.e. when env is empty.
        Probability for actions not availablre are set to zero.
        """
        env, dist = state_dist_pair[0], state_dist_pair[1]
       
        moves = [action[1] for action in env.available_actions_binary()]

        
        full_dist = [dist[move] if move in dist else 0 for move in moves]
        return full_dist
