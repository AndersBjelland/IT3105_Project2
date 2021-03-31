
from .helpers import rotate

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from typing import Tuple, List, Set
from itertools import product


import abc

class Environment(metaclass = abc.ABCMeta):

    @abc.abstractclassmethod
    def copy(self) -> 'Environment':
        pass

    @abc.abstractclassmethod
    def get_winner(self) -> int:
        pass

    @abc.abstractclassmethod
    def available_actions(self):
        pass

    @abc.abstractclassmethod
    def make_action(self, action):
        pass

    @abc.abstractclassmethod
    def get_current_player(self):
        pass



"""
Class representing the game of HEX.
There are two players, BLUE and RED.
BLUE owns the northwest and southeast side.
RED owns the northeast and southwest side.

We rotate the board such that 
cell [0,0] is the top 
cell [n,n] is the bottom
cell [n,0] is the left (west)
cell [0,n] is the right (east)
"""


EMPTY, BLUE, RED = 0, 1, 2

class Hex(Environment):
    def __init__(self, size: Tuple[int, int], start_player=BLUE):
        """
        size: size of grid (n,n)
        empty_cells: [(row, column)...]
        """
        if start_player not in [1, 2]:
            raise ValueError("start_player must be {} or {}, not {}".format(BLUE, RED, start_player))
        self.current_player = start_player
        self.start_player = start_player

        self.size = size
        self.board = np.zeros(size)
        self.coordinates = list(product([_ for _ in range(size[0])], [_ for _ in range(size[0])]))
        self.neighbours = self._generate_neighbours()
        
        self.encoder = None


    def _generate_neighbours(self):
        neighbours = {}
        for coordinate in self.get_coordinates():
            row, column = coordinate[0], coordinate[1]
            neighbour_coordinates = self.get_neighbouring_indecies(row, column)
            for n_row, n_column in neighbour_coordinates:
                if 0 <= n_row < self.size[0] and 0 <= n_column < self.size[1]:
                    neighbours[(row, column)] = neighbours[(row, column)] + [(n_row, n_column)] if (row, column) in neighbours else [(n_row, n_column)]
        return neighbours

    def get_neighbouring_indecies(self, row: int, column: int) -> List[Tuple[int,int]]:
        return [(row-1, column), (row-1,column+1),
                (row, column-1), (row, column+1), 
                (row+1, column-1), (row+1, column)]


    # We include an encoder to efficiently update the encoding of the game state when moves are made
    def set_encoder(self, encoder):
        self.encoder = encoder
        if encoder is not None:
            encoder.encode(self)

    def reset(self):
        self.board = np.zeros(self.get_size())
        
        self.current_player = self.start_player
        self.encoder.encode(self)

    def copy(self) -> 'Hex':
        new_hex = Hex(self.size, self.current_player)
        new_hex.board = np.copy(self.get_board())
        new_hex.encoder = self.encoder.copy()
        
        return new_hex

    def copy_without_encoder(self) -> 'Hex':
        new_hex = Hex(self.size, self.current_player)
        new_hex.board = np.copy(self.get_board())
        return new_hex

    def get_current_player(self) -> int:
        return self.current_player

    def get_board(self) -> np.ndarray:
        return self.board 

    def get_size(self) -> Tuple[int, int]:
        return self.size

    def make_action(self, coordinate: Tuple):
        if self.value_of(coordinate) != EMPTY:
            print("tried this action {} on this board".format(coordinate))
            self.display_board()
            raise ValueError("Action cannot be made because {} is not empty".format(coordinate))
        self.set_piece(coordinate, owner=self.current_player)
        self.current_player = BLUE if self.current_player == RED else RED
        self.encoder.update_encoding(coordinate, self)
    

    def get_NW_coordinates(self, layer=0) -> Tuple[Tuple[int,int]]:
        coordinates = []
        for row in range(self.size[0]-2*layer):
            row += layer
            coordinates.append((row, layer))
        return tuple(coordinates)

    def get_NE_coordinates(self, layer=0) -> Tuple[Tuple[int,int]]:
        coordinates = []
        for column in range(self.size[1] - 2*layer):
            column += layer
            coordinates.append((layer, column))
        return tuple(coordinates)
    
    def get_SW_coordinates(self, layer=0) -> Tuple[Tuple[int,int]]:
        coordinates = []
        for column in range(self.size[1] - 2*layer):
            column += layer
            coordinates.append(((self.size[0]-1)-layer, column))
        return tuple(coordinates)

    def get_SE_coordinates(self, layer=0) -> Tuple[Tuple[int,int]]:
        coordinates = []
        for row in range(self.size[0] - 2*layer):
            row += layer
            coordinates.append((row, (self.size[1]-1) - layer))
        return tuple(coordinates)


    def _search_from(self, coordinate: Tuple[int,int], player: int) -> List[Tuple[int, int]]:
        """
        Performs a DFS search with respect to the player and return all leaf cells.
        """
        visited = []
        to_visit = []
        
        if self.value_of(coordinate) == player:
            to_visit.append(coordinate)
        current_coordinate = coordinate
        while len(to_visit) != 0:
            # Add current cell to the visited list
            visited.append(current_coordinate)

            # Get the neighbours that are owned by the player and are not already visited
            neighbour_coordinates = [coordinate for coordinate in self.get_neighbours(current_coordinate) if self.value_of(coordinate) == player and coordinate not in visited]
            # Add neighbours to the to_visit list 
            to_visit += neighbour_coordinates

            # Set current cell to the next we will update and remove that for to_visit
            current_coordinate = to_visit.pop()
        
        return [coordinate for coordinate in visited]


    def get_winner(self) -> int:
        NW, SE = self.get_NW_coordinates(), self.get_SE_coordinates()
        NE, SW = self.get_NE_coordinates(), self.get_SW_coordinates()
        side_pairs = [(NW, SE), (NE, SW)]

        for i in range(len(side_pairs)):
            player = i+1
            side_pair = side_pairs[i]
            for coordinate in side_pair[0]:
                path = set(self._search_from(coordinate, player=player))
                if len(path & set(side_pair[1])) > 0:
                    return BLUE if side_pair[0] == NW else RED
        
        return 0

    def available_actions(self) -> List[Tuple[int,int]]:
        return [coordinate for coordinate in self.get_coordinates() if self.value_of(coordinate) == EMPTY]


    def available_actions_binary(self) -> List[Tuple[int, Tuple[int, int]]]:
        """
        Returns a list with 0 or 1 indicating if an action is legal or not, combined with the corresponding action
        """
        return [(1, coordinate) if self.value_of(coordinate) == EMPTY else (0, coordinate) for coordinate in self.get_coordinates()]

    def get_state(self) -> List[int]:
        """
        A cell is represented as 0 - EMPTY, 1 - BLUE, 2 - RED.
        We also include the current player in the first element of the state.
        """
        state = list(self.get_board().reshape(-1))
        player = [self.current_player]
        
        return player + state

    def one_hot_encode(self, state: List[int]) -> List[str]:
        encoded = [np.binary_repr(state,2)]
        return encoded

    def set_piece(self, coordinate: Tuple[int, int], owner: int):
        """
        Place a stone on the board without changing current player.
        This method is only used when modifying the board outside a game.
        It also overwrites if there is a stone in a cell already
        """
        self.get_board()[coordinate] = owner
        
    def equals(self, hex) -> bool:
        "Checks if two hex games are equal"
        return (self.board == hex.get_board()).all() and self.get_current_player() == hex.get_current_player()

    def value_of(self, coordinate):
        try:
            return self.board[coordinate]
        except IndexError:
            return None

    def get_coordinates(self):
        return self.coordinates

    def get_neighbours(self, coordinate):
        return self.neighbours[coordinate]




    ######### Visualization methods ##############

    def get_node_colors(self, G: "Graph", distribution=None) -> "ColorMap":
        color_map = plt.cm.get_cmap('YlGn')
        c_map = []
        
        if distribution:
            node_color = {location:distribution[location] if location in distribution else 0 for location in G.nodes()}
            vmax = max(node_color.values())*1.2
            vmin = min(list(distribution.values()))/2
        
        for location in G.nodes():
            piece = self.value_of(location)
            if piece == EMPTY:
                color = 'gray' if not distribution else color_map((node_color[location] - vmin)/vmax)
                c_map.append(color)
            elif piece == RED:
                c_map.append('red')
            elif piece == BLUE:
                c_map.append('blue')
        
        return c_map

    def get_edge_colors(self, G: "Graph") -> "ColorMap":
        c_map = []
        NW, SE = self.get_NW_coordinates(), self.get_SE_coordinates()
        NE, SW = self.get_NE_coordinates(), self.get_SW_coordinates()
        for u, v in G.edges():
            if u in NW and v in NW or u in SE and v in SE:
                c_map.append('blue')
                G[u][v]['weight'] = 2
            elif u in NE and v in NE or u in SW and v in SW:
                c_map.append('red')
                G[u][v]['weight'] = 2
            else:
                c_map.append('black')
        return c_map


    def get_networkx_graph(self) -> "Graph":
        coordinates = self.get_coordinates()
        visited = []
        G = nx.Graph()
        for coordinate in coordinates:
            node1 = (coordinate)
            for neighbour in self.get_neighbours(coordinate):
                if neighbour in visited:
                    continue
                node2 = (neighbour)
                G.add_edge(node1, node2, weight=1)
            visited.append(coordinate)
        return G

    def get_angle(self):
        return 5*np.pi/4

    def display_board(self, pace = 0.1, ax=None, distribution=None):
        angle = 5*np.pi/4
        G = self.get_networkx_graph()
        # Rotate the position of the nodes
        pos = rotate(G, angle)
        # create color map according to whether the cells are empty or not
        c_map = self.get_node_colors(G, distribution=distribution)
        edge_color = self.get_edge_colors(G)
        edge_weights = [G[u][v]['weight'] for u,v in G.edges()]
        # Display the board
        nx.draw(G, ax=ax, with_labels=True, node_size=1000, pos=pos, node_color=c_map, edge_color=edge_color, width=edge_weights, node_shape='h')
        plt.pause(pace)