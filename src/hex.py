from hexagonal_grid import Diamond, Cell
from helpers import rotate

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from typing import Tuple, List, Set

"""
Class representing the game of HEX.
There are two players, BLACK and RED.
BLACK owns the northwest and southeast side.
RED owns the northeast and southwest side.

We rotate the board such that 
cell [0,0] is the top 
cell [n,n] is the bottom
cell [n,0] is the left (west)
cell [0,n] is the right (east)
"""


EMPTY, BLACK, RED = 0, 1, 2

class Hex:
    def __init__(self, size: Tuple[int, int], start_player=BLACK):
        """
        size: size of grid (n,n)
        empty_cells: [(row, column)...]
        """
        if start_player not in [1, 2]:
            raise Exception("start_player must be {} or {}, not {}".format(BLACK, RED, start_player))
        self.current = start_player

        self.size = size
        self.board = Diamond(size)
        
        # Initialize the board as empty
        for cell in self.get_board().get_cells():
            cell.set_piece(EMPTY)

    def reset(self):
        for cell in self.get_board().get_cells():
            cell.set_piece(EMPTY)

    def get_board(self) -> "HexagonalGrid":
        return self.board 

    def get_size(self) -> Tuple[int, int]:
        return self.size

    def make_action(self, coordinate: Tuple):
        cell = self.get_board().get_cell(coordinate[0], coordinate[1])
        if cell.get_piece() != EMPTY:
            return 
        cell.set_piece(self.current)
        self.current = BLACK if self.current == RED else RED
    

    def _get_NW_coordinates(self) -> Set[Tuple[int,int]]:
        coordinates = []
        for row in range(self.size[0]):
            coordinates.append((row, 0))
        return set(coordinates)

    def _get_NE_coordinates(self) -> Set[Tuple[int,int]]:
        coordinates = []
        for column in range(self.size[1]):
            coordinates.append((0, column))
        return set(coordinates)
    
    def _get_SW_coordinates(self) -> Set[Tuple[int,int]]:
        coordinates = []
        for column in range(self.size[1]):
            coordinates.append((self.size[0]-1, column))
        return set(coordinates)

    def _get_SE_coordinates(self) -> Set[Tuple[int,int]]:
        coordinates = []
        for row in range(self.size[0]):
            coordinates.append((row, self.size[1]-1))
        return set(coordinates)


    def _search_from(self, coordinate: Tuple[int,int]) -> bool:
        """
        Performs a DFS search and return all leaf cells.
        """
        visited = []
        to_visit = []
        leaf_cells = []
        current_cell = self.get_board().get_cell(coordinate[0], coordinate[1])
        
        player = BLACK if current_cell.get_piece() == BLACK else RED

        to_visit.append(current_cell)

        while len(to_visit) != 0:
            # Add current cell to the visited list
            visited.append(current_cell)

            # Get the neighbours that are owned by the player and are not already visited
            neighbour_cells = [cell  for cell in current_cell.get_neighbours() if cell.get_piece() == player and cell not in visited]

            # Check if current cell is leaf cell
            if len(neighbour_cells) == 0:
                leaf_cells.append((current_cell.get_row(), current_cell.get_column()))

            # Add neighbours to the to_visit list 
            to_visit += neighbour_cells

            # Set current cell to the next we will update and remove that for to_visit
            current_cell = to_visit.pop()
        
        return leaf_cells


    def check_victory(self) -> Tuple[bool, int]:
        NW, SE = self._get_NW_coordinates(), self._get_SE_coordinates()
        NE, SW = self._get_NE_coordinates(), self._get_SW_coordinates()
        side_pairs = [(NW, SE), (NE, SW)]

        for side_pair in side_pairs:
            for coordinate in side_pair[0]:
                leaf_coordinates = set(self._search_from(coordinate))
                if len(leaf_coordinates & side_pair[1]) > 0:
                    return (True, BLACK if side_pair[0] == NW else RED)
        
        return (False, None)


    def available_actions(self) -> List[Tuple[int,int]]:
        return [(cell.get_row(), cell.get_column()) for cell in self.get_board().get_cells() if cell.get_piece() == 0]

    def get_state(self) -> List[int]:
        """
        A cell is represented as 0 - EMPTY, 1 - BLACK, 2 - RED.
        We also include the current player in the first element of the state.
        """
        state = [cell for cell in self.get_board().get_cells()]
        player = [self.current]
        
        return player + state

    def one_hot_encode(self, state: List[int]) -> List[str]:
        encoded = [np.binary_repr(state,2)]
        return encoded
        


    ######### Visualization methods ##############

    def get_node_colors(self, G: "Graph") -> "ColorMap":
        c_map = []
        for location in G.nodes():
            piece = self.get_board().get_cell(location[0], location[1]).get_piece()
            if piece == EMPTY:
                c_map.append('gray')
            elif piece == RED:
                c_map.append('red')
            elif piece == BLACK:
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

    def display_board(self, edge_color=None, pace = 0.3):
        angle = 5*np.pi/4
        G = self.get_networkx_graph()
        # Rotate the position of the nodes
        pos = rotate(G, angle)
        # create color map according to whether the cells are empty or not
        c_map = self.get_node_colors(G)
        # Display the board
        nx.draw(G, with_labels=True, node_size=1000, pos=pos, node_color=c_map, edge_color=edge_color)
        plt.pause(pace)
            






            





        

    




    