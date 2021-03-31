from .hex import Hex
import abc
import numpy as np
import tensorflow as tf

from typing import List, Tuple

class Encoder(metaclass = abc.ABCMeta):

    def __init__(self, padding = 0):
        self.padding = padding
        self.encoding = None
    
    @abc.abstractclassmethod
    def encode(self, env: Hex):
        pass

    @abc.abstractclassmethod
    def update_encoding(self, coordinate:Tuple[int, int], env:Hex):
        pass

    def coordinate_scaler(self, x):
        return x[0] + self.padding, x[1] + self.padding

    def get_encoding(self):
        if self.encoding is not None:
            return self.encoding
        raise TypeError("An encoding has not been initialized")


    def create_padded_env(self, env: Hex, padding:int) -> 'env':
        #if padding < 1:
        #    return env.copy()
        current_env_size = env.size
        new_size = (current_env_size[0] + padding*2, current_env_size[1] + padding*2)
        coordinate_scaler = lambda x: (x[0] + padding, x[1] + padding) # coresponding coordinate in new env

        new_env = Hex(new_size, env.start_player)

        # Iterate of cells in old env and set corresponding cells in new
        for coordinate in env.get_coordinates():
            padded_coordinate = coordinate_scaler(coordinate)
            new_env.set_piece(padded_coordinate, env.value_of(coordinate))

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

    def to_play_encoding(self, player: int, env:Hex, size:Tuple[int,int]) -> np.ndarray:
        """
        Returns a plane with all 1 if the current player and the player argument are equal, all 0 otherwise.
        """

        feat = [[1 if env.current_player == player else 0 for _ in range(size[1])] for _ in range(size[0])]
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
                feat = self._coordinate_bridge_encoding(piece_owner, row, column, env, plane_type, feat)

        return np.array(feat)     

    def _coordinate_bridge_encoding(self, piece_owner: int, row: int, column: int, env: Hex, plane_type: 'str', feat) -> np.ndarray:
        
        value = env.value_of((row, column))
                
        # if the piece is not owned by the piece_owner we move to next
        if value != piece_owner:
            return feat
        
        # bridge endpoints around the current row and column
        end_points = self.get_bridge_end_points(row, column)
        
        # Iterate over the bridge endpoints
        for bridge_row, bridge_column in end_points:
            
            # Value of the current endpoint
            bridge_value = env.value_of((bridge_row, bridge_column))
            
            # If the bridge value is None we know we are 'outside' the board
            if bridge_value is None:
                continue

            # Check scenarios
            # Process as a save point
            if plane_type == 'save':
                # Check if the endpoint forms a bridge together with the current coordinate
                if bridge_value != piece_owner:
                    continue
                feat = self._process_save_endpoint(feat, bridge_row, bridge_column, row, column, piece_owner, env)

            # Process as a form point
            elif plane_type == 'form':
                feat[bridge_row][bridge_column] = 1 if bridge_value == 0 else 0

            # Process as an endpoint point
            elif plane_type == 'endpoints':
                if bridge_value == piece_owner:
                    feat[bridge_row][bridge_column] = 1
                    feat[row][column] = 1
            
            else:
                raise ValueError("plane_type must be one of 'save', 'form', 'endpoints' (got {})".format(plane_type))
        return feat  

    def _process_save_endpoint(self, feat, bridge_row: int, bridge_column: int, row: int, column: int, piece_owner: int, env: Hex) -> List[List[int]]:
        carrier_point1, carrier_point2 = self.get_carrier_points((row, column), (bridge_row, bridge_column))
        
        carrier_point1_value = env.value_of(carrier_point1)
        carrier_point2_value = env.value_of(carrier_point2) 

        
        if carrier_point2_value != 0 and carrier_point2_value != piece_owner and carrier_point1_value==0:
            feat[carrier_point1[0]][carrier_point1[1]] = 1
        else:
            feat[carrier_point1[0]][carrier_point1[1]] += 0

        if carrier_point1_value != 0 and carrier_point1_value != piece_owner and carrier_point2_value==0:
            feat[carrier_point2[0]][carrier_point2[1]] = 1
        else:
            feat[carrier_point2[0]][carrier_point2[1]] += 0

        return feat

    def convert_planes_to_tensor(self, planes) -> np.ndarray:
        # Create new axis in all planes
        for i in range(len(planes)):
            planes[i] = np.expand_dims(planes[i], axis=2)

        feat = np.concatenate(planes, axis=2).astype(float)
        feat = np.expand_dims(feat, axis=0)
        # convert to tensor
        return tf.convert_to_tensor(feat)
    


class SimpleHexEncoder(Encoder):
    
    def encode(self, env: Hex) -> 'tensor':
        """
        Sets the encoding to a tensor that will serve as one feature/encoding of a Hex board. 
        The dimension of the tensor will be (1, row_size, column_size, n_planes)

        The padding parrameters specifies how much the planes will be padded.
        
        """

        padding = self.padding

        env = self.create_padded_env(env=env, padding=padding)

        blue = self.player_stones_encoding(1, env=env)
        red = self.player_stones_encoding(2, env=env)
        empty = self.player_stones_encoding(0, env=env)

        to_play_blue = self.to_play_encoding(1, env=env, size=env.size)
        to_play_red = self.to_play_encoding(2, env=env, size=env.size)

        planes = [blue, red, empty, to_play_blue, to_play_red]

        # Create new axis in all planes
        for i in range(len(planes)):
            planes[i] = np.expand_dims(planes[i], axis=2)

        feat = np.concatenate(planes, axis=2).astype(float)
        feat = np.expand_dims(feat, axis=0)
        # convert to tensor
        self.encoding = tf.convert_to_tensor(feat)
        return self.encoding

    

class HexEncoder(Encoder):

    def __init__(self, padding):
        super().__init__(padding)
        self.planes = None
        self.padded_env = None

    def copy(self):
        enc = HexEncoder(padding=self.padding)
        enc.planes = {key:np.copy(value) for key, value in self.planes.items()}
        enc.padded_env = self.padded_env.copy_without_encoder()
        enc.encoding = tf.identity(self.encoding)
        return enc

    def encode(self, env: Hex) -> 'tensor':
        """
        Sets the encoding to a tensor that will serve as one feature/encoding of a Hex board. 
        The dimension of the tensor will be (1, row_size, column_size, n_planes)

        The padding parrameters specifies how much the planes will be padded.
        
        """
        padding = self.padding
        # Alter the environment according to the padding parameter.
        # We add 'padding' layers of cells outside the current board and set the owner of those cells
        # according to the player that owns that side.
        env = self.create_padded_env(env=env, padding=padding)
        self.padded_env = env

        blue = self.player_stones_encoding(1, env=env)
        red = self.player_stones_encoding(2, env=env)
        empty = self.player_stones_encoding(0, env=env)

        blue_bridge = self.bridge_encoding(1, plane_type='endpoints', env=env)
        red_bridge = self.bridge_encoding(2, plane_type='endpoints', env=env)

        to_play_blue = self.to_play_encoding(1, env=env, size=env.size)

        to_play_red = self.to_play_encoding(2, env=env, size=env.size)

        save_bridge_blue = self.bridge_encoding(1, plane_type='save', env=env)
        save_bridge_red = self.bridge_encoding(2, plane_type='save', env=env)

        form_bridge_blue = self.bridge_encoding(1, plane_type='form', env=env)
        form_bridge_red = self.bridge_encoding(2, plane_type='form', env=env)

        # We're only using save and form bridge planes for the current player. Both versions are created to ease the updating up planes
        to_play_save_bridge = save_bridge_blue if env.current_player == 1 else save_bridge_red
        to_play_form_bridge = form_bridge_blue if env.current_player == 1 else form_bridge_red

        planes = [blue, red, empty, blue_bridge, red_bridge, to_play_blue, to_play_red, to_play_save_bridge, to_play_form_bridge]
        self.planes = {'blue':blue, 'red':red, 'empty':empty, 'blue_bridge':blue_bridge, 'red_bridge':red_bridge, 'to_play_blue':to_play_blue, 
                        'to_play_red':to_play_red, 'save_bridge_blue':save_bridge_blue, 'save_bridge_red':save_bridge_red,
                        'form_bridge_blue':form_bridge_blue, 'form_bridge_red':form_bridge_red}

        
        self.encoding = self.convert_planes_to_tensor(planes)
        return self.encoding


    def update_encoding(self, coordinate: Tuple[int, int], env: Hex):
        # Update the padded env
        coordinate = self.coordinate_scaler(coordinate)
        # Since this method is called after the acton is made in the env, the current player when updating the padded env is the opposite of the env's current player
        current = 1 if env.get_current_player()==2 else 2
        self.padded_env.set_piece(coordinate, current)

        blue = self.update_player_stones(1, plane=self.planes['blue'], coordinate=coordinate, env=self.padded_env)
        red = self.update_player_stones(2, plane=self.planes['red'], coordinate=coordinate, env=self.padded_env)
        empty = self.update_player_stones(0, plane=self.planes['empty'], coordinate=coordinate, env=self.padded_env)

        blue_bridge = self.update_bridge(1, plane_type='endpoints', plane=self.planes['blue_bridge'], coordinate=coordinate, env=self.padded_env)
        red_bridge = self.update_bridge(2, plane_type='endpoints', plane=self.planes['red_bridge'], coordinate=coordinate, env=self.padded_env)

        to_play_blue = self.to_play_encoding(1, env=env, size=self.planes['to_play_blue'].shape)
        to_play_red = self.to_play_encoding(2, env=env, size=self.planes['to_play_red'].shape)

        save_bridge_blue = self.update_bridge(1, plane_type='save', plane=self.planes['save_bridge_blue'], coordinate=coordinate, env=self.padded_env)
        save_bridge_red = self.update_bridge(2, plane_type='save', plane=self.planes['save_bridge_red'], coordinate=coordinate, env=self.padded_env)

        form_bridge_blue = self.update_bridge(1, plane_type='form', plane=self.planes['form_bridge_blue'], coordinate=coordinate, env=self.padded_env)
        form_bridge_red = self.update_bridge(2, plane_type='form', plane=self.planes['form_bridge_red'], coordinate=coordinate, env=self.padded_env)

        # We're only using save and form bridge planes for the current player. Both versions are created to ease the updating up planes
        to_play_save_bridge = save_bridge_blue if env.current_player == 1 else save_bridge_red
        to_play_form_bridge = form_bridge_blue if env.current_player == 1 else form_bridge_red
        
        planes = [blue, red, empty, blue_bridge, red_bridge, to_play_blue, to_play_red, to_play_save_bridge, to_play_form_bridge]
        self.planes = {'blue':blue, 'red':red, 'empty':empty, 'blue_bridge':blue_bridge, 'red_bridge':red_bridge, 'to_play_blue':to_play_blue, 
                        'to_play_red':to_play_red, 'save_bridge_blue':save_bridge_blue, 'save_bridge_red':save_bridge_red,
                        'form_bridge_blue':form_bridge_blue, 'form_bridge_red':form_bridge_red}
        
        self.encoding = self.convert_planes_to_tensor(planes)

    def update_player_stones(self, piece_owner:int, plane:np.array, coordinate:Tuple[int,int], env:Hex):
        piece_value = env.value_of(coordinate)
        if piece_value == piece_owner:
            plane[coordinate] = 1
        else:
            plane[coordinate] = 0

        return plane

    def update_bridge(self, piece_owner:int, plane_type:'str', plane:np.ndarray, coordinate:Tuple[int, int], env:Hex):
        # Update such that the now occupied cell is not considered a possible bridge form point if is was before
        if plane_type=='form':
            plane[coordinate] = 0
        return self._coordinate_bridge_encoding(piece_owner, coordinate[0], coordinate[1], env, plane_type, plane)



class DemoEncoder(Encoder):

    def encode(self, env: Hex):
        player = env.get_current_player()
        encoding = []
        for coordinate in env.get_coordinates():
            value = env.value_of(coordinate)
            if value == player:
                one_hot = [1,1]
            elif value == 0:
                one_hot = [0,0]
            else:
                one_hot = [1,0]
            encoding += one_hot

        encoding = np.array(encoding)
        encoding = np.expand_dims(encoding, axis=0)
        self.encoding =  tf.convert_to_tensor(encoding)
        return self.encoding

    def update_encoding(self, coordinate, env):
        self.encode(env)