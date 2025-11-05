from typing import Tuple, Protocol, List, Callable
from dataclasses import dataclass
from itertools import product
import numpy as np
import torch.nn as nn
import torch

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

def encode_state(s: OrientedGameState) -> np.ndarray:
    encoding = np.zeros(20, dtype=np.float32)
    encoding[0 + s.me[0]] = 1.0
    encoding[5 + s.me[1]] = 1.0
    encoding[10 + s.opp[0]] = 1.0
    encoding[15 + s.opp[1]] = 1.0
    return encoding

_StateEncodings = {
    (a,b,c,d): encode_state(OrientedGameState((a,b),(c,d))) 
    for a,b,c,d in product(range(5), repeat=4)
}

class Agent(Protocol):
    def start_game(self): ...
    def end_game(self): ...
    def act(self, s: OrientedGameState, mask: np.ndarray) -> int: ...
    def observe_final_state(self, s: OrientedGameState, r: float): ...


class QNetwork(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    @torch.no_grad()
    def act(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: ...

class FFNQNetwork(nn.Module):
    def __init__(self, encoding_length: int = 20):
        super().__init__()
        self.m = encoding_length
        self.n = len(_PossibleMoves)
        self.net = nn.Sequential(
            nn.Linear(self.m, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    @torch.no_grad()
    def act(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q_values = self.forward(x)
        q_values_masked = q_values.masked_fill(mask == 0, float('-inf'))
        actions = torch.argmax(q_values_masked, dim=-1)
        return actions

class QAgent(Agent):
    def __init__(self, qnetwork: nn.Module):
        self.q = qnetwork

    def start_game(self):
        pass

    def end_game(self):
        pass

    def act(self, s: OrientedGameState, mask: np.ndarray) -> int:
        state_encoding = _StateEncodings[(*s.me, *s.opp)]
        state_tensor = torch.from_numpy(state_encoding).unsqueeze(0)  # shape (1, 20)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # shape (1, 20)
        action_tensor = self.q.act(state_tensor, mask_tensor)  # shape (1,)
        return int(action_tensor.item())
    
    def observe_final_state(self, s: OrientedGameState, r: float):
        pass

class RandomAgent(Agent):
    def __init__(self):
        pass

    def start_game(self):
        pass

    def end_game(self):
        pass

    def act(self, s: OrientedGameState, mask: np.ndarray) -> int:
        legal_moves_indices = np.nonzero(mask)[0]
        chosen_move_index = np.random.choice(legal_moves_indices)
        return int(chosen_move_index)
    
    def observe_final_state(self, s: OrientedGameState, r: float):
        pass


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
            new_total_fingers_matches_old = (m.new_lhs + m.new_rhs) == total_fingers
            return can_split_evenly and new_total_fingers_matches_old

    def new_game(self):
        self.s = GameState((1,1), (1,1), 0, 0)

    def observe(self):
        r: float = 0.0
        mask: np.ndarray
        term: bool
        trunc: bool            
        mask = np.zeros(len(_PossibleMoves), dtype=np.int8)

        me, opp = (self.s.p0, self.s.p1) if self.s.player_to_move == 0 else (self.s.p1, self.s.p0)
        if self._is_terminal():
            winner = self._winner()
            if winner is None:
                trunc, term = (True, False)
            else:
                trunc, term = (False, True)
                r = 1.0 if winner == self.s.player_to_move else -1.0
        else:
            r = 0.0
            for i,m in enumerate(_PossibleMoves):
                mask[i] = 1 if self._is_legal_move(self.s, m) else 0
            trunc, term = (False, False)

        return OrientedGameState(me, opp), r, mask, trunc, term

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
            assert sum(p_self) == total_fingers, f"invalid split move: original total fingers {total_fingers}, new total fingers {sum(p_self)}"

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
    trunc: bool
    term: bool


class TransitionReplayBuffer:
    def __init__(self, 
                 capacity: int,
                 encoding_length: int = 20):
        self.capacity = capacity
        self.position = 0
        self.size = 0

        self.s = np.ndarray((capacity, encoding_length), dtype=np.float32)
        self.a = np.ndarray((capacity,), dtype=np.int8)
        self.r = np.ndarray((capacity,), dtype=np.float32)
        self.s2 = np.ndarray((capacity, encoding_length), dtype=np.float32)
        self.mask = np.ndarray((capacity, len(_PossibleMoves)), dtype=np.int8)
        self.mask2 = np.ndarray((capacity, len(_PossibleMoves)), dtype=np.int8)
        self.term = np.ndarray((capacity,), dtype=bool)
        self.trunc = np.ndarray((capacity,), dtype=bool)
    
    def add_transition(self, transition: Transition):
        idx = self.position
        self.s[idx] = _StateEncodings[(*transition.s.me, *transition.s.opp)]
        self.a[idx] = transition.a
        self.r[idx] = transition.r
        self.s2[idx] = _StateEncodings[(*transition.s2.me, *transition.s2.opp)]
        self.mask[idx] = transition.mask
        self.mask2[idx] = transition.mask2
        self.trunc[idx] = transition.trunc
        self.term[idx] = transition.term

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size: int):
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = {
            's': torch.from_numpy(self.s[indices]),
            'a': torch.from_numpy(self.a[indices]),
            'r': torch.from_numpy(self.r[indices]),
            's2': torch.from_numpy(self.s2[indices]),
            'mask': torch.from_numpy(self.mask[indices]),
            'mask2': torch.from_numpy(self.mask2[indices]),
            'trunc': torch.from_numpy(self.trunc[indices]),
            'term': torch.from_numpy(self.term[indices]),
        }
        return batch

class DQNTrainer:
    def __init__(self,
                 q: QNetwork,
                 target_q: QNetwork,
                 replay_buffer: TransitionReplayBuffer,
                 min_buffer_size_to_train: int,
                 batch_size: int,
                 loss_fn: nn.modules.loss._Loss,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 train_every: int = 1,
                 update_target_every: int = 1000,
                 max_grad_norm:float = 10.0,
                 gamma: float = 0.9,
                 device: str = 'cpu'):
        self.q = q
        self.target_q = target_q
        self.replay_buffer = replay_buffer
        self.min_buffer_size_to_train = min_buffer_size_to_train
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.gamma = gamma
        self.train_every = train_every
        self.update_target_every = update_target_every
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.step_counter = 0

    def add_episode(self, episode: List[Transition]):
        for transition in episode:
            self.replay_buffer.add_transition(transition)
            self.step()

    def step(self):
        self.step_counter += 1
        if self.step_counter % self.train_every:
            self.train_step()
        if self.step_counter % self.update_target_every:
            self.target_q.load_state_dict(self.q.state_dict())

    def train_step(self):
        if self.replay_buffer.size < self.min_buffer_size_to_train:
            return
        
        batch = self.replay_buffer.sample_batch(self.batch_size)

        s = batch['s'].to(self.device)
        a = batch['a'].to(self.device).long()
        r = batch['r'].to(self.device)
        s2 = batch['s2'].to(self.device)
        #mask = batch['mask'].to(self.device)
        mask2 = batch['mask2'].to(self.device)
        #trunc = batch['trunc'].to(self.device)
        term = batch['term'].to(self.device)

        # Compute Q-values for current states
        q = self.q(s)        # B, N
        q_a = torch.gather(q, 1, a.unsqueeze(1)).squeeze(1)

        # Sentinels/constants
        neginf = torch.tensor(torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype)
        #zero   = torch.zeros((), device=q.device, dtype=q.dtype)  # scalar 0
        B = s.size(0)

        # Compute target values
        with torch.no_grad():
            q2 = self.q(s2)            # B, N
            tq2 = self.target_q(s2)    # B, N
            q2_masked = q2.masked_fill(~mask2, neginf)        # B, N
            tq2_masked = tq2.masked_fill(~mask2, neginf)       # B, N

            max_tq2 = tq2.new_zeros(B)
            nt = ~term

            if nt.any():
                assert torch.all(mask2[nt].any(dim=1)), "there is a state with all actions masked"
                a_star = q2_masked[nt].argmax(dim=1)  # B_nt,
                max_tq2_nt = torch.gather(tq2_masked[nt], 1, a_star.unsqueeze(1)).squeeze(1)
                max_tq2[nt] = max_tq2_nt

            y = r + self.gamma * max_tq2 # B,
        
        if not torch.isfinite(y).all(): raise RuntimeError("NaN/Inf in targets")

        loss = self.loss_fn(q_a, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()



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
        s, r, mask, trunc, term = env.observe()

        current_agent = agents[env.s.player_to_move]
        previous_state_action_mask = pending_state_action_mask_tuples[env.s.player_to_move]

        # Record transition
        if previous_state_action_mask is not None:
            prev_s, prev_a, prev_mask = previous_state_action_mask
            transition = Transition(prev_s, prev_a, r, s, prev_mask, mask, trunc, term)
            transitions[env.s.player_to_move].append(transition)

        if trunc or term:
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
            #print (f"Player={env.s.player_to_move}  State={s}   Mask={mask}   MoveIndex={move_index}  Move={_PossibleMoves[move_index]}")
            pending_state_action_mask_tuples[env.s.player_to_move] = (s, move_index, mask)
            env.make_move(move_index)

    for p in agents:
        p.end_game()

    return total_rewards, transitions


def make_trainer(qnet: FFNQNetwork):
    target_qnet = FFNQNetwork()
    target_qnet.load_state_dict(qnet.state_dict())
    replay_buffer = TransitionReplayBuffer(10000)
    min_buffer_size_to_train = 1000
    batch_size = 128
    loss_fn = nn.modules.loss.HuberLoss(reduction="mean")
    lr = 0.01
    opt = torch.optim.AdamW(params=qnet.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: 1.0)
    train_every = 1
    update_target_every = 1000
    max_grad_norm = 1.0
    gamma = 0.9
    device = 'cuda'

    trainer = DQNTrainer(qnet,
                         target_q=target_qnet,
                         replay_buffer=replay_buffer,
                         min_buffer_size_to_train=min_buffer_size_to_train,
                         batch_size=batch_size,
                         loss_fn=loss_fn,
                         optimizer=opt,
                         lr_scheduler=lr_scheduler,
                         train_every=train_every,
                         update_target_every=update_target_every,
                         max_grad_norm=max_grad_norm,
                         gamma=gamma,
                         device=device)
    return trainer

class RecentValuesRingBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, value: float):
        if len(self.buffer) < self.capacity:
            self.buffer.append(value)
        else:
            self.buffer[self.position] = value
        self.position = (self.position + 1) % self.capacity

    def average(self) -> float:
        if not self.buffer:
            return 0.0
        return sum(self.buffer) / len(self.buffer)

def main():
    qnet = FFNQNetwork()
    trainer = make_trainer(qnet)
    env = GameEnv(max_turns=100)
    dqn_agent = QAgent(qnet)
    opponent = RandomAgent()
    dqn_rewards_buffer = RecentValuesRingBuffer(10000)
    time_step = 0
    while(True):
        random_bit = np.random.randint(0,2)
        if random_bit == 0:
            agents = [dqn_agent, opponent]
        else:
            agents = [opponent, dqn_agent]
        total_rewards, transitions = play_game(env, agents)
        trainer.add_episode(transitions[random_bit])
        dqn_rewards_buffer.add(total_rewards[random_bit])
        if time_step % 100 == 0:
            avg_reward = dqn_rewards_buffer.average()
            print(f"Time step: {time_step}  DQN agent average reward (last {dqn_rewards_buffer.capacity} games): {avg_reward:.3f}")
        time_step += 1
        #print(f"Game over! Total rewards: {total_rewards}")

if __name__=="__main__":
    main()