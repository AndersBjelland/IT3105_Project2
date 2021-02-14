from .hex import Hex

from tensorflow import keras as KER
import numpy as np

from typing import Tuple

"""
An actor using a convolutional network.
"""

class Actor:

    def __init__(self, learning_rate: float, epsilon: float, end_epsilon: float, nn_shape: Tuple, filters: Tuple, kernel_sizes: Tuple[Tuple[int,int]],  
            nn_opt: "optimizer", nn_activation: "ActivationFunc", nn_last_activation: "ActivationFunc", nn_loss: "LossFunc"):

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.end_epsilon = end_epsilon
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

        self.model.compile(optimizer=opt(lr=self.learning_rate), loss=loss, metrics=[KER.metrics.categorical_accuracy])

        self.model.build((None, 13, 13, 1))


    def encode(self, state):
        raise NotImplementedError("Not implemented")


    def get_action(self, env):
        feature_maps = self.encode(env.get_state())
        prob_dist = self.model(feature_maps)
        # legal moves is a list with tuples like [ (1, (1,2)), (0,(0,0)) ]
        legal_moves = env.available_actions_binary()

        prob_dist = prob_dist*np.array([entry[0] for entry in legal_moves])
        return legal_moves[np.argmax(prob_dist)][1]

    





actor = Actor(learning_rate=0.1,
                epsilon=0.4,
                end_epsilon=0.1,
                nn_shape=(20,25,10),
                filters=(8,7,9),
                kernel_sizes=(5,3,3),
                nn_opt='Adam',
                nn_activation='relu',
                nn_last_activation='softmax',
                nn_loss='mse')
                
actor.model.summary()