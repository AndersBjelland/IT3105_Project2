from .hex import Hex
from .encoder import Encoder
from .helpers import copy_model

from tensorflow import keras as KER
import tensorflow as tf
import numpy as np
import random

from typing import Tuple, Callable, List, Dict

"""
An actor using a convolutional network.
"""

class Critic:

    def __init__(self, learning_rate: float, nn_loss:float, encoder: Encoder, load_from='', **kwargs):

        self.nn_loss = nn_loss
        self.learning_rate = learning_rate
        self.encoder =  encoder
        
        # Load model from file
        if load_from:
            self.model = KER.models.load_model(load_from)
        
        # Build model
        else:
            self.model = KER.models.Sequential()

            nn_shape = kwargs.get('nn_shape')
            filters = kwargs.get('filters')
            kernel_sizes = kwargs.get('kernel_sizes')
            nn_opt = kwargs.get('nn_opt')
            nn_activation = kwargs.get('nn_activation')
            nn_last_activation = kwargs.get('nn_last_activation')
            #nn_loss = kwargs.get('nn_loss')

            opt = eval('KER.optimizers.'+nn_opt)
            loss = eval('KER.losses.' + self.nn_loss) if type(self.nn_loss) == str else self.nn_loss

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

    def get_value(self, env):
        feature_maps = env.encoder.get_encoding()

        output = self.model(feature_maps).numpy()[0,0]
        return -1 + 2*output

    def end_of_episode(self, replay_buffer, epochs=1, batch_size=128):
        """ try:
            self.model = copy_model(self.model, loss=self.nn_loss)
        except ValueError:
            pass """

        x,y = self.convert_to_network_input(replay_buffer)
        return self.model.fit(x,y, epochs=epochs, batch_size=batch_size)

    def convert_to_network_input(self, replay_buffer):
        x = tf.concat([entry[0].encoder.get_encoding() for entry in replay_buffer], 0)  
        print("first replay: ", len(replay_buffer[0]))      
        y = tf.convert_to_tensor([entry[2] for entry in replay_buffer])
        return x,y