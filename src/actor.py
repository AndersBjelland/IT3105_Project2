from .hex import Hex

from tensorflow import keras as KER
import tensorflow as tf
import numpy as np

from typing import Tuple, Callable

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

    def get_action(self, env):
        feature_maps = encode(env)
        prob_dist = self.model(feature_maps)
        # legal moves is a list with tuples like [ (1, (1,2)), (0,(0,0)) ]
        legal_moves = env.available_actions_binary()

        prob_dist = prob_dist*np.array([entry[0] for entry in legal_moves])
        return legal_moves[np.argmax(prob_dist)][1]


def encode(state):
    """
    Function for creating planes
    """
    raise NotImplementedError("Not implemented")

def player_stones_encoding(env: Hex, piece_owner: int) -> "tensor":
    """
    piece_owner is an int indicating which plane to return, with respect to EMPTY, BLUE, RED
    """
    state = env.get_state()

    # Slice state because current player is first element of state
    feat = [1 if piece==piece_owner else 0 for piece in state[1:]]
    feat = np.array(feat).reshape(env.get_size())
    feat = tf.convert_to_tensor(feat)
    return feat

def form_bridge_encoding(env: Hex,  piece_owner: int, plane_type: str) -> 'tensor':
    feat = [[0 for _ in range(env.size[1])] for _ in range(env.size[0])]
    for row in range(env.size[0]):
        for column in range(env.size[1]):

            piece = env.get_board().get_cell(row, column).get_piece()
            if piece != piece_owner:
                continue

            end_points = get_bridge_end_points(row, column)

            for bridge_row, bridge_column in end_points:
                bridge_cell = env.get_board().get_cell(bridge_row, bridge_column)
                
                if bridge_cell is None:
                    continue
                
                if plane_type == 'save':

                    carrier_point1, carrier_point2 = get_carrier_points((row, column), (bridge_row, bridge_column))
                    
                    carrier_point1_piece = env.get_board().get_cell(carrier_point1[0], carrier_point1[1]) 
                    carrier_point2_piece = env.get_board().get_cell(carrier_point2[0], carrier_point2[1]) 

                    if carrier_point2_piece != 0 and carrier_point2_piece != piece_owner:
                        feat[carrier_point1[0]][carrier_point1[1]] = 1
                    else:
                        feat[carrier_point1[0]][carrier_point1[1]] += 0

                    if carrier_point1_piece != 0 and carrier_point1_piece != piece_owner:
                        feat[carrier_point2[0]][carrier_point2[1]] = 1
                    else:
                        feat[carrier_point2[0]][carrier_point2[1]] += 0
                    
                elif plane_type == 'form':
                    feat[bridge_row][bridge_column] = 1 if bridge_cell.get_piece() == 0 else 0

                elif plane_type == 'endpoints':
                    feat[bridge_row][bridge_column] = 1 if bridge_cell.get_piece() == piece_owner else 0

                else:
                    raise ValueError("plane_type must be one of 'save', 'form', 'endpoints' (got {})".format(plane_type))
                

    return tf.convert_to_tensor(np.array(feat))

def to_play_endocing(env: Hex, player: int) -> 'tensor':
    feat = [[1 if env.current_player == player else 0 for _ in range(env.size[1])] for _ in range(env.size[0])]
    return tf.convert_to_tensor(np.array(feat))

def get_carrier_points(current_pos, end_point):
    
    row_diff, column_diff = current_pos[0] - end_point[0], current_pos[1] - end_point[1]

    if abs(row_diff) == 2:
        row1 = current_pos[0] - 1*np.sign(row_diff)
        row2 = current_pos[0] - 1*np.sign(row_diff)
    elif abs(row_diff) == 1:
        row1 = current_pos[0] - 1*np.sign(row_diff)
        row2 = current_pos[0]
    elif abs(row_diff) == 0:
        row1 = current_pos[0]
        row2 = current_pos[0]
    
    if abs(column_diff) == 2:
        col1 = current_pos[1] - 1*np.sign(column_diff)
        col2 = current_pos[1] - 1*np.sign(column_diff)
    elif abs(column_diff) == 1:
        col1 = current_pos[1]
        col2 = current_pos[1] - 1*np.sign(column_diff)
    elif abs(column_diff) == 0:
        col1 = current_pos[1]
        col2 = current_pos[1]
    
    return [(row1,col1), (row2, col2)]
"""
    flag = False
    while True:
        rows.append(current_pos[0] - 1*np.sign(row_diff))
        row_diff -= 1*row_diff_sign
        if flag:
            break
        if row_diff == 0:
            flag = True
    flag = False
    count = 0
    while abs(column_diff) >= 0:
        columns.append(current_pos[1] - 1*np.sign(column_diff)*count)
        column_diff -= 1*column_diff_sign
        if flag:
            break
        if column_diff == 0:
            flag = True
    
    return list(zip(rows, columns))

"""

def get_bridge_carrier_points(row, column):
    return [(row-1, column), (row-1,column+1),
                (row, column-1), (row, column+1), 
                (row+1, column-1), (row+1, column)]

def get_bridge_end_points(row, column):
    return [(row-2, column+1), (row-1, column+2), 
            (row+1, column+1), (row+2, column-1), 
            (row+1, column-2), (row-1, column-1)]



hex = Hex((5,5))
hex.make_action((0,0))
hex.make_action((0,1))
hex.make_action((3,3))
print(player_stones_encoding(hex, 0))



"""
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
"""

