from typing import Tuple, Protocol
from dataclasses import dataclass
from itertools import product
import numpy as np

@dataclass
class GameState:
    opponent: Tuple[int, int]
    me: Tuple[int, int]

@dataclass
class AttackMove:
    from_pos: int
    to_pos: int

@dataclass
class SplitMove:
    new_lhs: int
    new_rhs: int

Move = AttackMove | SplitMove

_PossibleMoves = [AttackMove(i,j) for i,j in product(range(2), repeat=2)] + [SplitMove(i,j) for i,j in product(range(1,5), repeat=2)]
assert len(_PossibleMoves) == 20


class Agent(Protocol):
    def act(s: GameState, )
    

def play_game():
    pass

# game
# agent
# play()
# actions list

def main():
    pass

if __name__=="__main__":
    main()