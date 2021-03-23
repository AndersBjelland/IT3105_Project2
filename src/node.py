
class Node():
    def __init__(self, current_player: int, env:'env', parent=None, action=None):
        """
        action is the action taken by the parent to get to this node
        """
        self.current_player = current_player
        self.set_parent(parent)
        self.children = []
        self.q_values = {} # {action:q_value}
        self.set_depth()
        self.env = env
        
        # Number of times the edge from parent to this node has been traversed, 
        # for the root this can be interpreted as number of times traversed into the tree
        self.traverse_count = 0

        self.action = action
        self.state_value = 0
        self.visited = 0

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

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
        self.children.append(child)

    def get_child(self, action) -> 'Node':
        for child in self.get_children():
            if child.action == action:
                return child
        return None

    
        