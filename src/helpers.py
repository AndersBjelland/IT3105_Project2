import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List

def rotate(G: "Graph", angle=np.pi/4) -> Dict[Tuple[int,int], Tuple[float,float]]:
    keys = G.nodes()
    X = np.array([n for n in G.nodes()])
    rot = np.array([[np.cos(angle), - np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = tuple(rot.dot(X.T).T.tolist())
    a = list((zip(keys, list(transformed))))
    return dict((tuple(n[0]), tuple(n[1])) for n in a)

def rotate_coordinate(coordinate, angle):
    X = np.array(coordinate)
    rot = np.array([[np.cos(angle), - np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = tuple(rot.dot(X.T).T.tolist())
    return transformed




