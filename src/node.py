
class Node():
    def __init__(self, environment=None, parent=None, action=None):
        """
        action is the action taken by the parent to get to this node
        """
        self.environment = environment
        self.set_parent(parent)
        self.children = []
        self.q_values = {} # {action:q_value}
        self.traverse_count = 0 # Number of the edge from parent to this node has been traversed
        self.action = action
        self.state_value = 0

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def set_parent(self, parent):
        if parent is not None:
            parent.add_child(self)
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)

    def get_child(self, action) -> 'Node':
        for child in self.get_children():
            if child.action == action:
                return child
        return None

    
        