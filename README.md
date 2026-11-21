# Sokoban solver — push-optimal, per-room decomposition

## Idea

Generated levels are a hallway plus 1–4 rooms (3–4 boxes/goals each, one
door). A box pushed out of a room can never return, so rooms are
independent and P* is the sum of the rooms' optima. The solver:

1. Parses the XSB level and splits it into rooms by cutting the hallway
   row's dead squares (floor where a box reaches no goal) out of the floor.
2. Solves each room independently with A* over **pushes**: a macro-state is
   the sorted box squares plus the player's reachable region (named by its
   lowest square). Other rooms' boxes act as walls.
3. Joins the rooms with plain BFS walks through the hallway (walks cost
   0 pushes, so they never affect the score).

## Optimality

- Heuristic: optimal box→goal assignment (brute-force permutations,
  3–4 boxes) over BFS distances that ignore boxes. Admissible (one push
  moves one box one square) and consistent, so the first solved state
  popped is push-optimal.
- Deadlock pruning: dead squares (reverse pull-BFS from the goals) and
  frozen 2×2 blocks. Both prune only unsolvable states.
- Fallbacks per room: A* → plain push-BFS (h = 0) → greedy best-first
  (h only), so a room is almost always solved even if a budget is hit.
  Whole-level search, then an empty valid answer, as the last resorts —
  every instance is always answered in the stated format.

## Self-verification

Before writing each `solution.txt`, the joined move string is replayed on
the initial level with the Sokoban rules (walk into a wall/box and push
into a wall/box are illegal; every box must end on a goal). A solution
that fails replay is never written — the instance gets an empty
`solution.txt` instead, and the failure is logged to stderr.

## Determinism & budgets

Budgets are counted in expansions/states, never wall-clock; no randomness,
no clock reads, no network, no files outside /input, /output, /tmp.
Tie-breaking is by insertion order, so two runs are byte-identical.

## Run

```
MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=<output> node main.ts
```

Node 22, plain JavaScript in `main.ts` (erasable syntax only), node
builtins only. One file, no dependencies.
