
class Node():
    def __init__(self, environment=None, parent=None):
        self.environment = environment
        self.set_parent(parent)
        self.children = []
        self.q_values = []
        self.traverse_count = 0

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
    
        