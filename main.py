from typing import Tuple, Protocol, List
from dataclasses import dataclass
from itertools import product
import numpy as np

@dataclass
class GameState:
    p0: Tuple[int, int]
    p1: Tuple[int, int]
    player_to_move: int
    turn: int = 0

@dataclass
class OrientedGameState:
    me: Tuple[int, int]
    opp: Tuple[int, int]

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
    def start_game(self): ...
    def end_game(self): ...
    def act(self, s: OrientedGameState, mask: np.ndarray) -> int: ...
    def observe_final_state(self, s: OrientedGameState, r: float): ...


class HumanStdinAgent:
    def __init__(self):
        pass

    def start_game(self):
        pass

    def end_game(self):
        pass

    def act(self, s: OrientedGameState, mask: np.ndarray) -> int:
        print(f"Game state: you: {s.me}  opponent: {s.opp}")
        move_str:str = input("> ")
        move_tokens = move_str.strip().split()
        assert len(move_tokens) == 3
        assert move_tokens[0] in ["a", "s"]
        move = AttackMove(int(move_tokens[1]), int(move_tokens[2])) if move_tokens[0] == "a" else SplitMove(int(move_tokens[1]), int(move_tokens[2]))
        move_index = _PossibleMoves.index(move)
        assert mask[move_index]
        return move_index
    
    def observe_final_state(self, s: OrientedGameState, r: float):
        print(f"Final state: you: {s.me}  opponent: {s.opp}  reward: {r}")

class GameEnv:
    def __init__(self, max_turns: int = 100):
        self.max_turns = max_turns
        self.s: GameState = None

    def _is_terminal(self):
        return (
            (self.s.p0[0] == 0 and self.s.p0[1] == 0) or 
            (self.s.p1[0] == 0 and self.s.p1[1] == 0) or
            (self.s.turn >= self.max_turns)
        )
    
    def _winner(self):
        if self.s.p0[0] == 0 and self.s.p0[1] == 0:
            return 1
        elif self.s.p1[0] == 0 and self.s.p1[1] == 0:
            return 0
        else:
            return None

    def _is_legal_move(self, s: GameState, m: Move) -> bool:
        assert isinstance(m, (AttackMove, SplitMove))
        if isinstance(m, AttackMove):
            # from hand and to hand must not be dead
            from_hand = s.p0[m.from_pos] if s.player_to_move == 0 else s.p1[m.from_pos]
            to_hand   = s.p1[m.to_pos]   if s.player_to_move == 0 else s.p0[m.to_pos]
            from_pos_not_dead = (from_hand != 0)
            to_pos_not_dead = (to_hand != 0)
            return from_pos_not_dead and to_pos_not_dead
        elif isinstance(m, SplitMove):
            # basic version: must be able to split total fingers evenly
            total_fingers = (s.p0[0] + s.p0[1]) if s.player_to_move == 0 else (s.p1[0] + s.p1[1])
            can_split_evenly = (total_fingers % 2 == 0)
            return can_split_evenly

    def new_game(self):
        self.s = GameState((1,1), (1,1), 0, 0)

    def observe(self):
        r: float = 0.0
        mask: np.ndarray
        done: bool            
        me, opp = (self.s.p0, self.s.p1) if self.s.player_to_move == 0 else (self.s.p1, self.s.p0)
        if self._is_terminal():
            winner = self._winner()
            if winner is not None:
                r = 1.0 if winner == self.s.player_to_move else -1.0
            mask = None
            done = True
        else:
            r = 0.0
            mask = np.zeros(len(_PossibleMoves), dtype=np.int8)
            for i,m in enumerate(_PossibleMoves):
                mask[i] = 1 if self._is_legal_move(self.s, m) else 0
            done = False

        return OrientedGameState(me, opp), r, mask, done

    def make_move(self, move_index: int):
        move = _PossibleMoves[move_index]
        assert self._is_legal_move(self.s, move)
        if self.s.player_to_move == 0:
            p_self = list(self.s.p0)
            p_opp  = list(self.s.p1)
        else:
            p_self = list(self.s.p1)
            p_opp  = list(self.s.p0)

        if isinstance(move, AttackMove):
            p_opp[move.to_pos] = (p_opp[move.to_pos] + p_self[move.from_pos]) % 5
        elif isinstance(move, SplitMove):
            total_fingers = sum(p_self)
            p_self[0] = move.new_lhs
            p_self[1] = move.new_rhs
            assert sum(p_self) == total_fingers

        if self.s.player_to_move == 0:
            self.s = GameState(tuple(p_self), tuple(p_opp), 1, self.s.turn + 1)
        else:
            self.s = GameState(tuple(p_opp), tuple(p_self), 0, self.s.turn + 1)

    def make_end_of_game_null_move(self):
        # advance the game state without making a real move
        self.s = GameState(self.s.p0, self.s.p1, 1 - self.s.player_to_move, self.s.turn + 1)
    
@dataclass
class Transition:
    s: OrientedGameState
    a: int
    r: float
    s2: OrientedGameState
    mask: np.ndarray
    mask2: np.ndarray
    done: bool

def play_game(env: GameEnv, agents: List[Agent]) -> Tuple[List[float], List[List[Transition]]]:
    env.new_game()
    assert len(agents) == 2
    for p in agents:
        p.start_game()
    pending_state_action_mask_tuples = [None, None]
    transitions = [[], []]
    total_rewards = [0.0, 0.0]
    has_observed_final_state = [False, False]

    while True:
        s: OrientedGameState
        r: float
        mask: np.ndarray
        done: bool
        s, r, mask, done = env.observe()

        current_agent = agents[env.s.player_to_move]
        previous_state_action_mask = pending_state_action_mask_tuples[env.s.player_to_move]

        # Record transition
        if previous_state_action_mask is not None:
            prev_s, prev_a, prev_mask = previous_state_action_mask
            transition = Transition(prev_s, prev_a, r, s, prev_mask, mask, done)
            transitions[env.s.player_to_move].append(transition)

        if done:
            if has_observed_final_state[env.s.player_to_move]:
                break
            total_rewards[env.s.player_to_move] += r
            current_agent.observe_final_state(s, r)
            has_observed_final_state[env.s.player_to_move] = True
            pending_state_action_mask_tuples[env.s.player_to_move] = None
            env.make_end_of_game_null_move()
        else:
            total_rewards[env.s.player_to_move] += r
            move_index = current_agent.act(s, mask)
            pending_state_action_mask_tuples[env.s.player_to_move] = (s, move_index, mask)
            env.make_move(move_index)

    for p in agents:
        p.end_game()

    return total_rewards, transitions


def main():
    env = GameEnv(max_turns=100)
    agents = [HumanStdinAgent(), HumanStdinAgent()]
    while(True):
        total_rewards, transitions = play_game(env, agents)
        print(f"Game over! Total rewards: {total_rewards}")

if __name__=="__main__":
    main()