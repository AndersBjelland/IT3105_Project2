import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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


def copy_model(model, loss):
    new_model = tf.keras.models.clone_model(model)
    opt = model.optimizer
    new_model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    new_model.set_weights(model.get_weights())
    return new_model

