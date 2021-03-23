from .hex import Hex
from .nim import Nim
from .node import Node

from typing import Callable, List, Tuple, Dict
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

"""
This class contains code to perform a Monte Carlo Tree Search (MCTS).
It takes as input a tree policy that is used to traverse the tree and a default/target/bahaviour policy
for managing rollouts.
We will in this project use the actor network as the target policy.

The one search simiulation is split in four stages. 

Step 1 - Traverse to leaf:
    Traverse to a leaf node given the tree policy. A leaf node is defined as a node where all actions are not yet been explored.

Step 2 - Expand leaf node:
    Expand the leaf node and create a new node representing one of the actions still not explored from the leaf node.

Step 3 - Rollout:
    Use the target policy to perform a rollout to a final state. Instead of recording the result as +1 or -1 
    depending on the current player at the root node is the final winner or not, we record +1 or -1 depending on 
    the winner is the current player at the node we run the rollout from. 

Step 4 - Backpropagation:
    Update the traverse count in the path chosen from root node to the expanded node. In addition we update the q-values in each node as follows.
    
    We also update the state value in all nodes using the value recorded from rollout, V, in the following way, 
    new_state_value = old_state_value + V if node.player == expanded_node.player else old_state_value - V. This updating rule results in the value in each state being
    with respect to the current player in that state.

    The Q-values are calculated and updated by negating the state value of the next state and divide it by the traverse count.
    
    Q_values at each node is updated to be new_state_value divided by the traverse count.

By alternating the sign of V when backpropagating up the path we allow the tree policy to always choose the best node by maximizing, instead of 
alternating maximizing and minimizing depending on which player that plays at each node. This makes it easier when we later prune the tree and move the root. 
"""

class MCTS():

    def __init__(self, target_policy:'Actor', env: Hex):
        self.exploration_bonus = None
        self.target_policy = target_policy
        self.env = env.copy()
        self.org_env = env.copy()
        self.root = Node(current_player=self.env.get_current_player(), env=self.env, parent=None, action=None)
        

    def _traverse_to_leaf(self) -> Node:
        current = self.root
        while not self._is_leaf_node(current):
            
            # Use the policy to get the next node and set it to current
            current = self.tree_policy(current)
            #self.env.make_action(current.action)
        
        return current

    def _expand(self, node: Node) -> Node: 
        # Check if the node represents a final state
        if node.env.get_winner():
            return node

        # Get actions that are not yet taken from the node we expand
        actions = [action for action in node.env.available_actions() if node.get_child(action) is None]
        action = random.choice(actions)
        env = node.env.copy()
        env.make_action(action)

        return Node(current_player = env.get_current_player(), env=env, parent=node, action=action)

    def _rollout(self, node: Node) -> int:
        player = node.env.get_current_player()
        env_copy = node.env.copy()
        while not env_copy.get_winner():
            action = self.target_policy.get_action(env_copy)
            env_copy.make_action(action)
        
        winner = env_copy.get_winner()
        return 1 if player == winner else -1
            

    def _back_prop(self, node: Node, value: float):
        "starts by updating q-value of the edge between the given node and the parent node - check if correct"
        current = node

        while current != self.root:
            # Update state value according to which player plays at current node compared to the node we expanded from
            update_value = value if current.current_player == node.current_player else - value
            current.state_value += update_value
            parent = current.parent
            current.traverse_count += 1
                
            parent.q_values[current.action] = -update_value if current.action not in parent.q_values else -(current.state_value)/current.traverse_count
            
            current = current.parent

        current.traverse_count += 1

    def _is_leaf_node(self, node) -> bool:
        """
        A node is defined as a leaf node if it has a potential
        child from which no simulation (rollout) has yet been initiated
        We know that for a k x k board there are k x k - d available moves for a node at depth d.
        We therefore check if there are k x k - d children, if it is not we are at a leaf node.
        """
        available_actions = node.env.available_actions()

        # If there are available actions and the number of children equals the number of available actions we can conclude that it is not a leaf node
        if len(available_actions) > 0 and len(available_actions) == len(node.get_children()):
            return False

        return True

    def _utc(self, node: Node, action: Tuple[int,int], c: float) -> float:
        visits_to_node = node.traverse_count
        child = node.get_child(action)
        performed_action_count = child.traverse_count

        return c*np.sqrt(np.log(visits_to_node)/(1+performed_action_count))

    def tree_policy(self, node: Node) -> Node:
        """
        Returns the next node in the search
        """
        available_actions = [child.action for child in node.get_children()]
        q_and_u_values = {action : (node.q_values[action], self.exploration_bonus(node, action)) for action in available_actions}
        # return argmax q+u
        return node.get_child(max(q_and_u_values.keys(), key=lambda x : sum(q_and_u_values[x])))
        
    
    def set_new_root(self, action):
        child = self.root.get_child(action)
        if child:
            self.root = child
            #self.env.make_action(action)
            #self.org_env = self.env.copy()
        else:
            env_copy = self.root.env.copy()
            env_copy.make_action(action)
            current_player = env_copy.get_current_player()
            
            self.root = Node(current_player=current_player, env=env_copy, parent=self.root, action=action)

    def perform_simulation(self):

        leaf_node = self._traverse_to_leaf()
        expanded_node = self._expand(leaf_node)
        value = self._rollout(expanded_node)
        self._back_prop(expanded_node, value)
        

    def search(self, n_simulations: int, exploration_bonus='uct', c=1, update_rate = 10, ax=None, plotting=False) -> Dict['action','prob']:
        if exploration_bonus=='uct':
            self.exploration_bonus = lambda s,a: self._utc(s,a,c)
        else:
            raise ValueError("exploration_bonus must be one of 'utc' (got {})".format(exploration_bonus))

        for _ in range(n_simulations):
            self.perform_simulation()

            if (_+1) % update_rate == 0 and plotting:
                ax.clear()
                
                distribution = {child.action : child.traverse_count for child in self.root.get_children()}
                factor = 1/sum(distribution.values())
                distribution = {action : v*factor for action, v in distribution.items()}
                self.root.env.display_board(ax=ax, distribution=distribution)
                
                plt.draw()
                plt.pause(0.5)
        
        distribution = {child.action : child.traverse_count for child in self.root.get_children()}
        factor = 1/sum(distribution.values())
        distribution = {action : v*factor for action, v in distribution.items()}
        
        return distribution


    def visualize_tree(self, root_color):
        edge_labels = {}
        G = nx.DiGraph()
        to_visit = [self.root]
        visited = []
        while len(to_visit) > 0:
            current = to_visit.pop()
            visited.append(current)
            parent_node = (current.action, id(current))
            G.add_node(parent_node, color = root_color)
            root_color = 'blue' if root_color != 'blue' else 'red'
            
            for child in current.get_children():
                child_node = (child.action, id(child))
                G.add_node(child_node, color = root_color)
                G.add_edge(parent_node, child_node)
                edge_labels[(parent_node, child_node)] = (child.traverse_count, current.q_values[child.action])
            
            root_color = 'blue' if root_color != 'blue' else 'red'
            to_visit += current.get_children()
            
        return G, edge_labels

    def visualize_tree_nim(self):
            edge_labels = {}
            G = nx.DiGraph()
            to_visit = [self.root]
            visited = []
            remaining = self.env.n
            while len(to_visit) > 0:
                current = to_visit.pop()
                remaining = remaining - current.action if current.action != None else remaining
                
                visited.append(current)
                parent_node = (current.action, remaining, current.state_value, id(current))
                G.add_node(parent_node)
                
                
                for child in current.get_children():
                    child_node = (child.action, remaining-child.action, child.state_value, id(child))
                    
                    G.add_edge(parent_node, child_node)
                    edge_labels[(parent_node, child_node)] = (child.traverse_count, round(current.q_values[child.action],4))
            
                to_visit += current.get_children()
                
            return G, edge_labels




    



        
    



    
        

                    
    

    


        





