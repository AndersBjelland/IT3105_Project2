from .hex import *

from typing import Iterator, TypeVar, Generic

Action = TypeVar('Action')


class Game(Generic[Action]):
    """An interface which a game is expected to follow"""

    def available_actions(self) -> Iterator[Action]:
        raise NotImplementedError()

    def next_state(self, action: Action) -> 'Game':
        raise NotImplementedError()
