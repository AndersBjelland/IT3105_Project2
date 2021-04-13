
class Node():
    def __init__(self, env: "env", current_player: int, parent=None, action=None):
        """
        action is the action taken by the parent to get to this node
        """
        self.env = env.copy()
        self.current_player = current_player
        self.action = action
        self.set_parent(parent)
        self.children = {} # {action:node}
        self.q_values = {} # {action:q_value}
        self.set_depth()
        self.prior = None #{action:priorProb}
        
        # Number of times the edge from parent to this node has been traversed, 
        # for the root this can be interpreted as number of times traversed into the tree
        self.traverse_count = 0

        
        self.state_value = 0
        self.visited = 0

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children.values()

    def set_parent(self, parent):
        if parent is not None:
            parent.add_child(self)
        self.parent = parent

    def set_depth(self):
        if self.parent is not None:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    def add_child(self, child):
        self.children[child.action] = child

    def get_child(self, action) -> 'Node':
        try:
            return self.children[action]
        except KeyError:
            return None
        

    
        