import abc
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from typing import Tuple, List

class Cell:
    def __init__(self, row: int, column: int):
        self.neighbours = []
        self.row  = row
        self.column = column
        self.edge_cell = None
        self.piece = None
    
    def add_neighbour(self, cell: "Cell"):
        self.neighbours.append(cell)
        
    def set_edge_cell(self, boolean: bool):
        self.edge_cell = boolean
    
    def is_edge(self) -> bool:
        return self.edge_cell
    
    def get_row(self) -> int:
        return self.row
    
    def get_column(self) -> int:
        return self.column
    
    def get_neighbours(self) -> List["Cell"]:
        return self.neighbours
    
    def set_piece(self, piece: int):
        self.piece = piece

    def get_piece(self) -> int:
        return self.piece

    def __hash__(self):
        return (self.row, self.column)


class HexagonalGrid(metaclass=abc.ABCMeta):

    def __init__(self, size: Tuple[int,int], cells=None, neighbours=None):
        self.size = size
        self.cells = np.zeros(size) if cells is None else np.copy(cells)
        self.coordinates = list(product([_ for _ in range(size[0])], [_ for _ in range(size[0])]))
        self.neighbours = {} if neighbours is None else neighbours # Dictionary on the format {(row, column): [neighbour coordinates...]}
    
    def generate_neighbours(self):
        for coordinate in self.get_cells():
            row, column = coordinate[0], coordinate[1]
            neighbour_coordinates = self.get_neighbouring_indecies(row, column)
            for n_row, n_column in neighbour_coordinates:
                if 0 <= n_row < self.size[0] and 0 <= n_column < self.size[1]:
                    self.neighbours[(row, column)] = self.neighbours[(row, column)] + [(n_row, n_column)] if (row, column) in self.neighbours else [(n_row, n_column)]
            
      
    def get_cell(self, row: int, column: int) -> int:
        try:
            return self.cells[row,column]
        except IndexError:
            return None
        """
        cells = self.get_cells()
        for cell in cells:
            if cell.get_row() == row and cell.get_column() == column:
                return cell
        """
    def get_cells(self) -> np.array:
        return self.coordinates

    def get_neighbours(self, coordinate: Tuple[int, int]) -> List[Tuple[int, int]]:
        return self.neighbours[coordinate] if coordinate in self.neighbours else None

    def set_value(self, coordinate: Tuple[int, int], value: int):
        self.cells[coordinate] = value

    def compare_grids(self, hexagonal_grid: 'HexagonalGrid'):
        """
        Returns true if all cells in the grids are equal
        """
        return (self.cells == hexagonal_grid.cells).all()

    @abc.abstractclassmethod
    def get_neighbouring_indecies(self, row: int, column: int):
        pass
        

class Triangle(HexagonalGrid):
    # Row 0 will have 1 cell, row 2 will have 1 and so on..
    def __init__(self, size: Tuple[int,int]):
        super().__init__(size)
        base = size[0]
        for row in range(base):
            for column in range(row+1):
                cell = Cell(row,column)
                self.cells[cell.__hash__()] = cell
        self.generate_neighbours()
    
    def get_neighbouring_indecies(self, row: int, column: int) -> List[Tuple[int,int]]:
        return [(row-1,column-1), (row-1, column), 
                (row, column-1), (row, column+1), 
                (row+1, column), (row+1, column+1)]

class Diamond(HexagonalGrid):
    def __init__(self, size: Tuple[int,int], cells=None, neighbours=None):
        if cells is not None:
            super().__init__(size, cells, neighbours)
        else:
            super().__init__(size)
            self.generate_neighbours()
    
    def get_neighbouring_indecies(self, row: int, column: int) -> List[Tuple[int,int]]:
        return [(row-1, column), (row-1,column+1),
                (row, column-1), (row, column+1), 
                (row+1, column-1), (row+1, column)]
    
    def copy(self) -> 'Diamond':
        return Diamond(size=self.size, cells=self.cells, neighbours=self.neighbours)
