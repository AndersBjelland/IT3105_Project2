from .hex import Hex
import abc
import numpy as np
import tensorflow as tf

from typing import List, Tuple

class Encoder(metaclass = abc.ABCMeta):

    @abc.abstractclassmethod
    def encode(self, piece_owner):
        pass


class HexEncoder(Encoder):

    def __init__(self, padding = 0):
        self.padding = padding


    def encode(self, env: Hex) -> 'tensor':
        """
        Returns a tensor that will serve as one feature/encoding of a Hex board. 
        The dimension of the tensor will be (1, row_size, column_size, n_planes)

        The padding parrameters specifies how much the planes will be padded.
        
        """
        padding = self.padding
        # Alter the environment according to the padding parameter.
        # We add 'padding' layers of cells outside the current board and set the owner of those cells
        # according to the player that owns that side.
        env = self.create_padded_env(env=env, padding=padding)

        blue = self.player_stones_encoding(1, env=env)
        red = self.player_stones_encoding(2, env=env)
        empty = self.player_stones_encoding(0, env=env)

        blue_bridge = self.bridge_encoding(1, plane_type='endpoints', env=env)
        red_bridge = self.bridge_encoding(2, plane_type='endpoints', env=env)

        to_play_blue = self.to_play_endocing(1, env=env)
        to_play_red = self.to_play_endocing(2, env=env)

        to_play_save_bridge = self.bridge_encoding(env.current_player, plane_type='save', env=env)
        to_play_form_bridge = self.bridge_encoding(env.current_player, plane_type='form', env=env)

        planes = [blue, red, empty, blue_bridge, red_bridge, to_play_blue, to_play_red, to_play_save_bridge, to_play_form_bridge]
        # Create new axis in all planes
        for i in range(len(planes)):
            planes[i] = np.expand_dims(planes[i], axis=2)

        feat = np.concatenate(planes, axis=2).astype(float)
        feat = np.expand_dims(feat, axis=0)
        # Return as tensor
        return tf.convert_to_tensor(feat)

    def create_padded_env(self, env: Hex, padding:int) -> 'env':
        if padding < 1:
            return env
        current_env_size = env.size
        new_size = (current_env_size[0] + padding*2, current_env_size[1] + padding*2)
        coordinate_scaler = lambda x: (x[0] + padding, x[1] + padding) # coresponding coordinate in new env

        new_env = Hex(new_size, env.start_player)

        # Iterate of cells in old env and set corresponding cells in new
        for cell in env.get_board().get_cells():
            coordinate = (cell.get_row(), cell.get_column())
            coordinate = coordinate_scaler(coordinate)
            new_env.set_piece(coordinate, cell.get_piece())

        # Place stones in the added layers
        for layer in range(padding):
            NW, SE = new_env.get_NW_coordinates(layer=layer), new_env.get_SE_coordinates(layer=layer)
            NE, SW = new_env.get_NE_coordinates(layer=layer), new_env.get_SW_coordinates(layer=layer)

            for coordinate in NE + SW:
                new_env.set_piece(coordinate, 2)

            for coordinate in NW + SE:
                new_env.set_piece(coordinate, 1)

        return new_env  

    def get_bridge_end_points(self, row: int, column: int) -> List[Tuple[int, int]]:
        return [(row-2, column+1), (row-1, column+2), 
                (row+1, column+1), (row+2, column-1), 
                (row+1, column-2), (row-1, column-1)]

    def get_carrier_points(self, current_pos: Tuple[int, int], end_point: Tuple[int, int]) -> List[Tuple[int, int]]:

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

    def player_stones_encoding(self, piece_owner: int, env: Hex) -> np.ndarray:
        """
        piece_owner is an int indicating which plane to return, with respect to EMPTY, BLUE, RED
        Returns a plane of the board with entry 1 if piece owner owns that piece, 0 otherwise

        Example: If there is a 3x3 board where blue has played (1,1), the feature map for blue will look like
                            [
                            [0,0,0],
                            [0,1,0],
                            [0,0,0]
                            ]               
        
        """
        state = env.get_state()

        # Slice state because current player is first element of state
        feat = [1 if piece==piece_owner else 0 for piece in state[1:]]
        feat = np.array(feat).reshape(env.get_size())
        return feat

    def to_play_endocing(self, player: int, env: Hex) -> np.ndarray:
        """
        Returns a plane with all 1 if the current player and the player argument are equal, all 0 otherwise.
        """

        feat = [[1 if env.current_player == player else 0 for _ in range(env.size[1])] for _ in range(env.size[0])]
        return np.array(feat)


    def bridge_encoding(self, piece_owner: int, plane_type: str, env: Hex) -> np.ndarray:
        """
        Bridge encoding takes a plane_type argument that cause the method to return the following planes.

        'save': An encoding that sets 1 if that cell is a save bridge cell.
        'form': An encoding that sets 1 if that cell can be used to create a new bridge.
        'endpoints': An encoding taht sets 1 if that cell forms a bridge.
        """

        feat = [[0 for _ in range(env.size[1])] for _ in range(env.size[0])]
        for row in range(env.size[0]):
            for column in range(env.size[1]):

                piece = env.get_board().get_cell(row, column).get_piece()
                
                # if the piece is not owned by the piece_owner we move to next
                if piece != piece_owner:
                    continue
                
                # bridge endpoints around the current row and column
                end_points = self.get_bridge_end_points(row, column)
                
                # Iterate over the bridge endpoints
                for bridge_row, bridge_column in end_points:
                    
                    # Cell of the current endpoint
                    bridge_cell = env.get_board().get_cell(bridge_row, bridge_column)
                    
                    # If the bridge cell is None we know we are 'outside' the board
                    if bridge_cell is None:
                        continue

                    # Check scenarios
                    # Process as a save point
                    if plane_type == 'save':
                        # Check if the endpoint forms a bridge together with the current cell
                        if bridge_cell.get_piece() != piece_owner:
                            continue
                        feat = self._process_save_endpoint(feat, bridge_row, bridge_column, row, column, piece_owner, env)

                    # Process as a form point
                    elif plane_type == 'form':
                        feat[bridge_row][bridge_column] = 1 if bridge_cell.get_piece() == 0 else 0

                    # Process as an endpoint point
                    elif plane_type == 'endpoints':
                        feat[bridge_row][bridge_column] = 1 if bridge_cell.get_piece() == piece_owner else 0

                    else:
                        raise ValueError("plane_type must be one of 'save', 'form', 'endpoints' (got {})".format(plane_type))

        return np.array(feat)       

    def _process_save_endpoint(self, feat, bridge_row: int, bridge_column: int, row: int, column: int, piece_owner: int, env: Hex) -> List[List[int]]:
        carrier_point1, carrier_point2 = self.get_carrier_points((row, column), (bridge_row, bridge_column))
        if env == None:
            print("hssdfsdf")   
        carrier_point1_piece = env.get_board().get_cell(carrier_point1[0], carrier_point1[1]).get_piece() 
        carrier_point2_piece = env.get_board().get_cell(carrier_point2[0], carrier_point2[1]).get_piece()

        
        if carrier_point2_piece != 0 and carrier_point2_piece != piece_owner and carrier_point1_piece==0:
            feat[carrier_point1[0]][carrier_point1[1]] = 1
        else:
            feat[carrier_point1[0]][carrier_point1[1]] += 0

        if carrier_point1_piece != 0 and carrier_point1_piece != piece_owner and carrier_point1_piece==0:
            feat[carrier_point2[0]][carrier_point2[1]] = 1
        else:
            feat[carrier_point2[0]][carrier_point2[1]] += 0

        return feat
    
    
    

    