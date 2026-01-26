# Decaying Stones Solver (Tic-Tac-Toe / Connect-4 Variants)

A small game engine + solver for grid-based games like **Tic-Tac-Toe** and **Connect-4**, with optional **stone decay** and optional **gravity**.

It builds a game graph up to a configurable depth (ply limit), then runs a **retrograde analysis** to compute:
- the **optimal result** from each reachable state (`X`, `O`, `draw`, or `indeterminate`)
- a **policy** (best move) for winning or best loss-delay
- **remoteness** (plies to the terminal result under perfect play) for forced wins/losses

---

## Features

- ✅ Supports **gravity** mode (Connect-4 style column drops)
- ✅ Supports **non-gravity** mode (Tic-Tac-Toe style free placement)
- ✅ Supports **stone decay** using TTL (*Time To Live*)
- ✅ Retrograde solve with **remoteness** (distance-to-end in plies)
- ✅ Optional **policy rollout** with **loop detection**
- ✅ Trace events for graph + solve diagnostics

---

## Terminology / Acronyms

- **TTL**: *Time To Live* — a stone’s remaining “life” on the board.  
  - `TTL == 0` means empty.
  - `TTL > 0` means occupied.
- **Decay**: reducing TTL values for the **current player’s stones** after each move.
- **ply**: one half-move by a single player (X moves = 1 ply, O moves = 1 ply).
- **BFS**: *Breadth-First Search* — graph expansion using a queue.
- **successors / predecessors**
  - successors (`succ`): edges from state → child states
  - predecessors (`pred`): reverse edges child → parent states
- **remoteness**: number of plies to reach the terminal outcome under perfect play.

---

## Game Rules Implemented

Each move follows this pipeline:

1. **Place** a stone for `player_to_move` with:
   - `TTL = stone_life + stone_decay`
2. **Decay** all stones belonging to `player_to_move` by:
   - `stone_decay`
   - stones reaching `TTL <= 0` disappear
3. If **gravity** is enabled and `stone_decay > 0`, re-apply **gravity** to pack stones downward.
4. Swap `player_to_move`.

### Gravity vs Non-Gravity

- `gravity = True`: a move is a **column index** (like Connect-4)
- `gravity = False`: a move is a **(row, col)** pair (like Tic-Tac-Toe)

---

## Output Labels

The solver assigns each state one of:

- `X`: X has a forced win
- `O`: O has a forced win
- `draw`: neither can force a win (when full state space is explored)
- `indeterminate`: unresolved due to ply limit / incomplete exploration

---

## Installation

This is a single-file Python script, no external dependencies.

Requirements:
- Python **3.10+** (uses modern typing and `|` unions)

---

## Running

From the project directory:

```bash
python main_<version>.py
```

---

## Configuration

In `__main__`, choose a preset:

```python
selected_game_name = "tic-tac-toe-decay"
selected_game = GAME_PRESETS[selected_game_name]
```

### Included Presets

- `tic-tac-toe`
- `tic-tac-toe-decay`
- `connect-4`
- `connect-4-decay`

Each preset defines:

- `ROWS`: board height
- `COLS`: board width
- `WIN_LENGTH`: length needed to win
- `GRAVITY_ENABLED`: whether stones fall
- `STONE_LIFE`: base TTL
- `STONE_DECAY`: per-move decay amount

### Solver Controls

- `ply_limit`: how deep to expand the graph from the start state
- `max_rollout_plies`: how far to simulate the policy line
- `trace_first_n`: how many trace events to print

---

## Example: Custom Starting State

Create an empty board:

```python
start_state = create_empty_state(rows, cols, player_to_move="X")
```

Then apply moves:

- Non-gravity:
  ```python
  start_state = apply_move(
      start_state, (1, 1),
      rows, cols, gravity_enabled=False,
      stone_life_amount=3, stone_decay_amount=1
  )
  ```

- Gravity:
  ```python
  start_state = apply_move(
      start_state, 3,
      rows, cols, gravity_enabled=True,
      stone_life_amount=14, stone_decay_amount=1
  )
  ```

---

## How It Works (High Level)

### 1) Graph Build (BFS)

The solver explores all reachable states from the start state (up to `ply_limit`), storing:

- successors (state → children)
- predecessors (child → parents)
- terminal labels (X/O/draw/None)
- depth (plies from start)

### 2) Retrograde Solve (Backpropagation)

Starting from known terminal states:
- A state is a **win** for the side to move if **any** child is a win for that side.
- A state is a **loss** if **all** children are wins for the opponent.
- For wins: choose **minimum remoteness**
- For losses: choose the move that **maximizes remoteness** (delay loss)

If exploration is complete, any remaining unsolved states become `draw`.  
If exploration is incomplete (hit ply limit), remaining become `indeterminate`.

### 3) Policy Rollout

Simulates play from the start using the computed policy, with:
- step-by-step printed boards
- optional loop detection (repeated state on the played line)

---

## Performance Notes

State explosion can happen quickly, especially for:
- larger boards (e.g., Connect-4)
- higher `STONE_LIFE` (more TTL variability)
- higher `ply_limit`

Tips:
- Start with smaller `ply_limit`
- Use Tic-Tac-Toe presets to validate logic
- Increase gradually while monitoring node counts

---

## Project Structure

Typical single-file layout uses IDE foldable regions:

- Type Aliases and Data Models
- Basic Helpers
- Display Helpers
- Gravity Mechanics
- Move Generation and Application
- Win Checking
- Graph Building (BFS)
- Retrograde Solve
- Policy Rollout and Loop Detection
- Printing and Formatting
- Example Main

---

## Troubleshooting

### `indeterminate` results
This usually means the solver hit the `ply_limit` frontier before proving outcomes.

Fix:
- increase `ply_limit`
- or accept partial solving (useful for analysis / heuristics)

### Illegal move exceptions
- In gravity mode: column is full
- In non-gravity mode: cell is already occupied
