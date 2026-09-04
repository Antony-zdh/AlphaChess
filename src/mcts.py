"""AlphaZero-style batched MCTS for the AlphaChess policy/value network.

Runs one independent search tree per board, and at each simulation step advances
every board's selection by one leaf, then evaluates all newly-reached leaves in a
single batched forward pass. This fits eval/competition.play_games, which calls
opponent.select_moves_batch(batch_boards) with up to batch_size boards that share
the same side to move.

Value convention (negamax): the network value is always from the side-to-move's
perspective. During backprop the side to move alternates each ply, so the sign
flips at every level. We store node.W = node's-own-perspective value. Therefore
the PUCT Q at a parent, for the action leading to a child, is -(child.W/child.N)
(child's perspective negated back to parent's perspective).

Note on performance: best_child_action iterates only the ~30 legal actions
(Python dict loop) — for a ~30-element set this beats a 4096-wide tensor
argmax. GPU util stays modest because each simulation is serial tree traversal
punctuated by one batched forward; cross-board batched selection (a larger
rewrite) is the way to push util higher.
"""
import math

import numpy as np
import torch
import chess

try:  # works both as `python src/mcts.py` and `from src.mcts import ...`
    from src.model import AlphaChess, BoardEncoder
except ImportError:
    from model import AlphaChess, BoardEncoder


def move_to_action(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def action_to_move(board: chess.Board, action: int) -> chess.Move:
    """Decode a 4096 action index to a legal move. Promotions collapse to one
    index; resolve by preferring queen promotion (matches AlphaChessOpponent)."""
    move_from = action // 64
    move_to = action % 64
    legal = list(board.legal_moves)
    cands = [m for m in legal if m.from_square == move_from and m.to_square == move_to]
    if cands:
        queen_promo = [m for m in cands if m.promotion == chess.QUEEN]
        return queen_promo[0] if queen_promo else cands[0]
    return legal[0]  # safety fallback (should not happen for a legal action)


class MCTSNode:
    __slots__ = ("board", "parent", "children", "prior", "N", "W",
                "expanded", "is_terminal")

    def __init__(self, board: chess.Board, parent=None):
        self.board = board
        self.parent = parent
        self.children = {}   # action -> MCTSNode, created lazily on first visit
        self.prior = {}      # action -> float, set when node is expanded
        self.N = 0
        self.W = 0.0         # this node's own-perspective value sum
        self.expanded = False
        self.is_terminal = board.is_game_over()

    def best_child_action(self, c_puct: float):
        """PUCT: argmax_a  Q_parent(a) + c_puct*P(a)*sqrt(N)/(1+N_child).
        Q_parent(a) = -(child.W/child.N)  (negamax sign flip).
        Iterates only legal actions in self.prior (~30)."""
        best_a, best_score = None, -float("inf")
        sqrtN = math.sqrt(max(self.N, 1))
        for a, p in self.prior.items():
            child = self.children.get(a)
            if child is None or child.N == 0:
                q = 0.0
                nc = 0
            else:
                q = -child.W / child.N   # child's perspective -> parent's
                nc = child.N
            u = c_puct * p * sqrtN / (1 + nc)
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a


class BatchMCTS:
    def __init__(self, model, encoder, device,
                 n_simulations=400, c_puct=4.0,
                 dirichlet_eps=0.25, dirichlet_alpha=0.3,
                 temperature=0.0):
        self.model = model
        self.encoder = encoder
        self.device = device
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_eps = dirichlet_eps
        self.dirichlet_alpha = dirichlet_alpha
        self.temperature = temperature

    @torch.no_grad()
    def _batch_eval(self, boards):
        """Returns policy logits (B,4096) and value (B,) from side-to-move view."""
        states = torch.stack([self.encoder.encode(b) for b in boards]).to(self.device)
        self.model.eval()
        policy, value = self.model(states)
        return policy, value.squeeze(-1)

    def _expand(self, node: MCTSNode, policy_logits, add_dirichlet=False):
        legal = list(node.board.legal_moves)
        if not legal:
            node.expanded = True
            return
        actions = [move_to_action(m) for m in legal]
        mask = torch.zeros(4096, device=self.device)
        for a in actions:
            mask[a] = 1.0
        masked = policy_logits.masked_fill(mask == 0, float("-inf"))
        probs = torch.softmax(masked, dim=0)
        for a in actions:
            node.prior[a] = probs[a].item()
        if add_dirichlet and self.dirichlet_eps > 0 and len(actions) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            for a, n in zip(actions, noise):
                node.prior[a] = ((1 - self.dirichlet_eps) * node.prior[a]
                                 + self.dirichlet_eps * float(n))
        node.expanded = True

    def _select(self, root: MCTSNode):
        """Traverse from root to a leaf. Lazily creates the child on the chosen
        edge if it doesn't exist yet (that fresh child is the leaf to evaluate)."""
        node = root
        path = []  # list of (node, action)
        while node.expanded and not node.is_terminal:
            a = node.best_child_action(self.c_puct)
            if a is None:
                break
            if a not in node.children:
                child_board = node.board.copy()
                child_board.push(action_to_move(node.board, a))
                child = MCTSNode(child_board, parent=node)
                node.children[a] = child
                path.append((node, a))
                node = child
                break  # freshly created child is a leaf
            path.append((node, a))
            node = node.children[a]
        return node, path

    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == board.turn else -1.0

    @staticmethod
    def _backprop(leaf: MCTSNode, value: float):
        node = leaf
        val = value
        while node is not None:
            node.N += 1
            node.W += val
            val = -val
            node = node.parent

    def _run_simulations(self, roots):
        """Run root eval + expand + n_simulations on the given root nodes (in place)."""
        active = [i for i, r in enumerate(roots) if not r.is_terminal]
        if active:
            rb = [roots[i].board for i in active]
            policy, value = self._batch_eval(rb)
            for j, i in enumerate(active):
                self._expand(roots[i], policy[j], add_dirichlet=True)
                self._backprop(roots[i], value[j].item())
        for _ in range(self.n_simulations):
            leaves = [None] * len(roots)
            to_eval = []
            for i in active:
                leaf, _ = self._select(roots[i])
                leaves[i] = leaf
                if not leaf.is_terminal:
                    to_eval.append(i)
            if to_eval:
                lb = [leaves[i].board for i in to_eval]
                policy, value = self._batch_eval(lb)
                for j, i in enumerate(to_eval):
                    self._expand(leaves[i], policy[j], add_dirichlet=False)
                    self._backprop(leaves[i], value[j].item())
            for i in active:
                if leaves[i] is not None and leaves[i].is_terminal:
                    self._backprop(leaves[i], self._terminal_value(leaves[i].board))
        return roots

    def _choose_action(self, root: MCTSNode):
        if root.is_terminal or not root.children:
            return None
        actions = list(root.children.keys())
        visits = np.array([root.children[a].N for a in actions], dtype=np.float64)
        if self.temperature and self.temperature > 0:
            probs = visits ** (1.0 / self.temperature)
            s = probs.sum()
            probs = probs / s if s > 0 else np.ones_like(probs) / len(probs)
            return int(np.random.choice(actions, p=probs))
        return int(actions[int(np.argmax(visits))])

    def search(self, boards):
        """Run MCTS for each board in parallel; return list of chosen actions
        (None for terminal boards)."""
        roots = self._run_simulations([MCTSNode(b.copy()) for b in boards])
        return [self._choose_action(r) for r in roots]

    def search_with_pi(self, boards):
        """Run MCTS and return (actions, pi). pi is a (len(boards), 4096) float
        tensor of the root visit-count distribution (legal actions sum to 1,
        illegal = 0) — the policy training target."""
        roots = self._run_simulations([MCTSNode(b.copy()) for b in boards])
        actions = []
        pis = torch.zeros(len(roots), 4096, dtype=torch.float32)
        for i, r in enumerate(roots):
            if r.is_terminal or not r.children:
                actions.append(None)
                continue
            acts = list(r.children.keys())
            visits = np.array([r.children[a].N for a in acts], dtype=np.float64)
            total = visits.sum()
            if total <= 0:
                actions.append(int(acts[0]))
                continue
            for a, p in zip(acts, visits / total):
                pis[i, a] = float(p)
            actions.append(self._choose_action(r))
        return actions, pis

    def root_visit_counts(self, boards):
        """Run a search and return per-board {action: visit_count} dicts (smoke/inspect)."""
        roots = self._run_simulations([MCTSNode(b.copy()) for b in boards])
        return [{a: r.children[a].N for a in r.children} for r in roots]


if __name__ == "__main__":
    # Smoke test: MCTS on the start position. Expect visits concentrated on
    # e2e4/d2d4 (the strongest opening moves).
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/sft_v2/alpha_chess_best.pth")
    parser.add_argument("--n_simulations", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaChess().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    encoder = BoardEncoder()

    mcts = BatchMCTS(model, encoder, device, n_simulations=args.n_simulations,
                     temperature=0.0)
    board = chess.Board()
    counts = mcts.root_visit_counts([board])[0]
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    print(f"Startpos MCTS visit counts (n_sims={args.n_simulations}):")
    for a, n in ranked[:8]:
        mv = action_to_move(board, a)
        print(f"  {mv.uci():>5}  visits={n}")
