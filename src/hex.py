from .hexagonal_grid import Diamond, Cell
from .helpers import rotate

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from typing import Tuple, List, Set


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
        self.board = Diamond(size)
        
        # Initialize the board as empty
        for cell in self.get_board().get_cells():
            cell.set_piece(EMPTY)

    def reset(self):
        for cell in self.get_board().get_cells():
            cell.set_piece(EMPTY)

    def copy(self) -> 'Hex':
        new_hex = Hex(self.size, self.current_player)
        for cell in self.get_board().get_cells():
            equivalent_cell = new_hex.get_board().get_cell(cell.get_row(), cell.get_column())
            equivalent_cell.set_piece(cell.get_piece())
        return new_hex

    def get_current_player(self) -> int:
        return self.current_player


    def get_board(self) -> "HexagonalGrid":
        return self.board 

    def get_size(self) -> Tuple[int, int]:
        return self.size

    def make_action(self, coordinate: Tuple):
        cell = self.get_board().get_cell(coordinate[0], coordinate[1])
        if cell.get_piece() != EMPTY:
            print("tried this action {} on this board".format(coordinate))
            self.display_board()
            raise Exception()
        cell.set_piece(self.current_player)
        self.current_player = BLUE if self.current_player == RED else RED
    

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
        current_cell = self.get_board().get_cell(coordinate[0], coordinate[1])

        if current_cell.get_piece() == player:
            to_visit.append(current_cell)

        while len(to_visit) != 0:
            # Add current cell to the visited list
            visited.append(current_cell)

            # Get the neighbours that are owned by the player and are not already visited
            neighbour_cells = [cell  for cell in current_cell.get_neighbours() if cell.get_piece() == player and cell not in visited]

            # Add neighbours to the to_visit list 
            to_visit += neighbour_cells

            # Set current cell to the next we will update and remove that for to_visit
            current_cell = to_visit.pop()
        
        return [(cell.get_row(), cell.get_column()) for cell in visited]


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
        return [(cell.get_row(), cell.get_column()) for cell in self.get_board().get_cells() if cell.get_piece() == 0]

    def available_actions_binary(self) -> List[Tuple[int, Tuple[int, int]]]:
        """
        Returns a list with 0 or 1 indicating if an action is legal or not, combined with the corresponding action
        """
        return [(1, (cell.get_row(), cell.get_column())) if cell.get_piece() == 0 else (0, (cell.get_row(), cell.get_column())) for cell in self.get_board().get_cells()]

    def get_state(self) -> List[int]:
        """
        A cell is represented as 0 - EMPTY, 1 - BLUE, 2 - RED.
        We also include the current player in the first element of the state.
        """
        state = [cell.piece for cell in self.get_board().get_cells()]
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
        cell = self.get_board().get_cell(coordinate[0], coordinate[1])
        cell.set_piece(owner)
        
    def equals(self, hex) -> bool:
        "Checks if two hex games are equal"
        if self.size != hex.size:
            return False

        for row in range(self.size[0]):
            for column in range(self.size[1]):
                if self.get_board().get_cell(row, column).get_piece() != hex.get_board().get_cell(row, column).get_piece():
                    return False
        if self.current_player != hex.current_player:
            return False
        return True




    ######### Visualization methods ##############

    def get_node_colors(self, G: "Graph") -> "ColorMap":
        c_map = []
        for location in G.nodes():
            piece = self.get_board().get_cell(location[0], location[1]).get_piece()
            if piece == EMPTY:
                c_map.append('gray')
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
        cells = self.get_board().get_cells()
        visited = []
        G = nx.Graph()
        for cell in cells:
            node1 = (cell.get_row(), cell.get_column())
            for neighbour in cell.get_neighbours():
                if neighbour in visited:
                    continue
                node2 = (neighbour.get_row(), neighbour.get_column())
                G.add_edge(node1, node2, weight=1)
            visited.append(cell)
        return G

    def display_board(self, pace = 0.3):
        angle = 5*np.pi/4
        G = self.get_networkx_graph()
        # Rotate the position of the nodes
        pos = rotate(G, angle)
        # create color map according to whether the cells are empty or not
        c_map = self.get_node_colors(G)
        edge_color = self.get_edge_colors(G)
        edge_weights = [G[u][v]['weight'] for u,v in G.edges()]
        # Display the board
        nx.draw(G, with_labels=True, node_size=1000, pos=pos, node_color=c_map, edge_color=edge_color, width=edge_weights)
        plt.pause(pace)
