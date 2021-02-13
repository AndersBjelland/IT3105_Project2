from hex import Hex
from node import Node

from typing import Callable

"""
This class contains code to perform a Monte Carlo Tree Search (MCTS).
It takes as input a tree policy that is used to traverse the tree and a default/target/bahaviour policy
for managing rollouts.
We will in this project use the actor network as the target policy.
"""

class MTCS():

    def __init__(self, tree_policy: Callable, target_policy, environment: Hex):
        self.tree_policy = tree_policy
        self.target_policy = target_policy
        self.environment = environment
        self.root = Node(environment=environment)

    def search_to_leaf(self) -> Node:
        current = self.root
        while not self._is_leaf_node(current):
            # Use the policy to get the next node and set it to current
            current = self.tree_policy(current)
        return current

    def expand(self):
        raise NotImplementedError("Not implemented")

    def rollout(self):
        raise NotImplementedError("Not implemented")

    def back_prop(self):
        raise NotImplementedError("Not implemented")

    def _is_leaf_node(self, node) -> bool:
        """
        A node is defined as a leaf node if it has a potential
        child from which no simulation (rollout) has yet been initiated
        """
        available_actions = node.environment.available_actions()

        # If the number of children equals the number of available actions we can conclude that it is not a leaf node
        if len(available_actions) == len(node.get_children()):
            return False

        return True
        

                    
    

    


        





