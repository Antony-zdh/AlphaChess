# PPO Training Flaws

Diagnosed from `ppo_eval_results.json` and code review (2025-05-20).
The PPO peaks at ~iteration 300 then regresses — by iteration 1000 it loses to the SFT baseline (9.4% win, 14.1% loss). `ppo_val` drifts negative while `sft_val` trends positive, indicating value head divergence.

---

## Bug 1 — GAE `next_value` sign is wrong for two-player chess

**File:** `src/ppo.py:283`

`next_value` is assigned `trajectory.steps[i+1]["value"]` — the model's value estimate two plies forward (same player's *next* turn). But the model always evaluates from the *current player's* perspective. The opponent's intermediate ply must be negated. The correct Bellman delta is:

```
δ_t = r_t + γ·(-V_opponent(s')) - V(s_t)
```

The current code uses `γ·V(s_{t+2})` which conflates the opponent's evaluation with the current player's, corrupting all advantage estimates and causing the value head to drift.

---

## Bug 2 — Only one epoch per collected batch (non-standard PPO)

**File:** `src/ppo.py:356`

The 4 "mini-batches" are sequential chunks of the 32 trajectories — each step is trained on exactly once before being discarded. Standard PPO runs K=3–10 epochs over the same collected data. Underutilizing each batch severely slows policy improvement.

---

## Bug 3 — No gradient clipping

**File:** `src/ppo.py:390`

`torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)` is absent. Without it, a large update following a decisive game (rare, given the high draw rate) can destabilize the value head, which then corrupts subsequent advantage estimates. This matches the observed collapse after iteration 300.

---

## Flaw 4 — Self-play draws starve the training signal

**Draw rate: 70–82%.** All games start from the initial position; both sides have equal strength → repeated draws by 50-move rule or threefold repetition. Most trajectories carry `reward=0`, providing almost no gradient.

Fixes:
- Randomize opening book (6–10 random plies) to reach diverse mid-game positions.
- Small draw penalty (−0.05) to incentivize decisive play.

---

## Flaw 5 — No opponent pool (self-play only)

Pure self-play against a single evolving policy is unstable. The model learns to counter its current self, then adapts back → cycling without net progress. Mixing 20–30% of games against frozen historical checkpoints anchors training and provides consistent signal.
