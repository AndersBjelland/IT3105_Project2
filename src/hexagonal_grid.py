import abc
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
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
        return hash((self.row, self.column))


class HexagonalGrid(metaclass=abc.ABCMeta):

    def __init__(self, size: Tuple[int,int]):
        self.cells = {}
    
    def generate_neighbours(self):
        cells = self.get_cells().values()
        for cell in cells:
            neighbour_coordinates = self.get_neighbouring_indecies(cell.get_row(), cell.get_column())
            for coordinate in neighbour_coordinates:
                neighbour = self.get_cell(coordinate[0], coordinate[1])
                # The get_cell method returns None if a cell with the specified coordinates doesn't exist. 
                # This means that the cell currently evaluated is at an edge.
                if neighbour == None:
                    continue
                if neighbour not in cell.get_neighbours():
                    cell.add_neighbour(neighbour)
                    neighbour.add_neighbour(cell)     
      
    def get_cell(self, row: int, column: int) -> Cell:
        try:
            return self.cells[hash((row,column))]
        except KeyError:
            return None
        """
        cells = self.get_cells()
        for cell in cells:
            if cell.get_row() == row and cell.get_column() == column:
                return cell
        """
    def get_cells(self) -> List[Cell]:
        return self.cells

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
                self.cells.append(Cell(row,column))
        self.generate_neighbours()
    
    def get_neighbouring_indecies(self, row: int, column: int) -> List[Tuple[int,int]]:
        return [(row-1,column-1), (row-1, column), 
                (row, column-1), (row, column+1), 
                (row+1, column), (row+1, column+1)]

class Diamond(HexagonalGrid):
    def __init__(self, size: Tuple[int,int]):
        super().__init__(size)
        n_rows = size[0]
        n_columns = size[1]
        for row in range(n_rows):
            for column in range(n_columns):
                cell = Cell(row, column)
                self.cells[cell.__hash__()] = cell
        self.generate_neighbours()
    
    def get_neighbouring_indecies(self, row: int, column: int) -> List[Tuple[int,int]]:
        return [(row-1, column), (row-1,column+1),
                (row, column-1), (row, column+1), 
                (row+1, column-1), (row+1, column)]