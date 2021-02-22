from .hex import Environment

class Nim(Environment):
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.remaining = n
        self.current_player = 1

    def available_actions(self):
        return [i+1 for i in range(self.k) if i+1 <= self.remaining]
    
    def get_current_player(self):
        return self.current_player

    def get_winner(self):
        if self.remaining == 0:
            return self.get_current_player()
        return 0

    def make_action(self, action: int):
        self.remaining -= action
        self.current_player = 2 if self.get_current_player() == 1 else 1

    def copy(self):
        nim = Nim(n=self.n, k = self.k)
        nim.remaining = self.remaining
        nim.current_player = self.current_player
        return nim



