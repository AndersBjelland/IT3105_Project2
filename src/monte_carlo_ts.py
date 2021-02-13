from hex import Hex
from node import Node

from typing import Callable, List, Tuple
import numpy as np

"""
This class contains code to perform a Monte Carlo Tree Search (MCTS).
It takes as input a tree policy that is used to traverse the tree and a default/target/bahaviour policy
for managing rollouts.
We will in this project use the actor network as the target policy.
"""

class MTCS():

    def __init__(self, target_policy, environment: Hex, exploration_bonus='utc', c=1):
        if exploration_bonus=='uct':
            self.exploration_bonus = lambda s,a: self._utc(s,a,c)
        else:
            raise ValueError("exploration_bonus must be one of 'utc' (got {})".format(exploration_bonus))
        self.target_policy = target_policy
        self.environment = environment
        self.root = Node(environment=environment)

    def traverse_to_leaf(self) -> List[Node]:
        current = self.root
        path = []
        while not self._is_leaf_node(current):
            path.append(current)
            # Use the policy to get the next node and set it to current
            current = self.tree_policy(current)
        # Add the leaf node to path
        path.append(current)
        return path

    def expand(self, node: Node) -> Node: 
        # Check if the node represents a final state
        if node.environment.get_winner():
            return node

        env = node.environment.copy()
        actions = [action for action in env.available_actions() if action not in node.actions]
        action = np.random.choice(actions)
        env.make_action(action)

        return Node(environment=env, parent=node, action=action)

    def rollout(self, node: Node) -> int:
        env = node.environment.copy()
        while not env.get_winner():
            action = self.target_policy.get_action(env.get_state())
            env.make_action(action)
        
        winner = env.get_winner()
        return 1 if self.root.environment.current == winner else -1
            

    def back_prop(self, node: Node, value: float):
        "starts by updating q-value of the edge between the given node and the parent node - check if correct"
        current = node

        while current != self.root:
            current.state_value += value
            parent = current.parent
            current.traverse_count += 1

            parent.q_values[current.action] = value if current.action not in parent.q_values else (parent.state_value + value)/current.traverse_count
            
            current = node.parent


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

    def _utc(self, node: Node, action: Tuple[int,int], c: float) -> float:
        visits_to_node = node.traverse_count
        child = node.get_children(action)
        performed_action_count = child.traverse_count

        return c*np.sqrt(np.log(visits_to_node)/(1+performed_action_count))

    def tree_policy(self, node: Node) -> Node:
        """
        Returns the next node in the search
        """
        available_actions = node.environment.available_actions()
        q_and_u_values = {action : (node.q_values[action], self.exploration_bonus(node, action)) for action in available_actions}
        if node.environment.current() == self.root.environment.current():
            # return argmax q+u
            return node.get_child(max(q_and_u_values.keys(), key=lambda x : sum(q_and_u_values[x])))

        # return argmin q-u
        return node.get_child(min(q_and_u_values.keys(), key=lambda x : q_and_u_values[0] - q_and_u_values[1]))
        



    
        

                    
    

    


        





