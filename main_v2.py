from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

# region Type Aliases and Data Models

Player = str
"""
Player marker. Expected values are:
- "X"
- "O"
"""

Move = Union[int, Tuple[int, int]]
"""
A move is:
- int column index when gravity=True (Connect-4 style)
- (row, col) tuple when gravity=False (Tic-Tac-Toe style)
"""

BoardVector = List[int]
"""
A flattened board vector holding integer TTL (time-to-live) values per cell.
Value meanings:
- 0: empty
- >0: occupied (stone exists) with that remaining TTL
"""

State = Tuple[BoardVector, BoardVector, Player]
"""
Mutable state representation: (x_ttl_by_cell, o_ttl_by_cell, player_to_move)

The board is represented by two parallel vectors:
- x_ttl_by_cell[i] is TTL of X stone at index i (0 if none)
- o_ttl_by_cell[i] is TTL of O stone at index i (0 if none)
"""

Key = Tuple[Tuple[int, ...], Tuple[int, ...], Player]
"""
Immutable, hashable state key representation:
(x_ttl_tuple, o_ttl_tuple, player_to_move)
"""


@dataclass(frozen=True)
class Edge:
    """A directed move edge from a state to a child key."""
    move: Move
    child_key: Key


@dataclass
class SolveStats:
    """Operational statistics for graph build + solve."""
    nodes: int = 0
    terminals: int = 0
    proven: int = 0
    indeterminate: int = 0
    explored_complete: bool = True


@dataclass(frozen=True)
class Outcome:
    """
    Solved outcome for a state.

    result:
      - "X" / "O": a player has a forced win
      - "draw": no player can force a win (with complete exploration)
      - "indeterminate": not fully solved (e.g., ply-limit frontier)

    remoteness:
      - For X/O wins: number of plies to end under best play (distance-to-end)
      - None for draw/indeterminate
    """
    result: str
    remoteness: Optional[int]


@dataclass(frozen=True)
class PlayedStep:
    """A single step in a policy rollout."""
    ply_index: int
    move: Move
    state_before: State
    state_after: State
    note: Optional[str] = None


@dataclass(frozen=True)
class LoopWitness:
    """Loop witness for on-policy repetition."""
    start_step_index: int
    end_step_index: int
    cycle_moves: List[Move]
    cycle_states: List[State]

# endregion


# region Basic Helpers (Players, Indexing, State Conversion)

def get_other_player(player: Player) -> Player:
    """Return the opponent of the given player."""
    return "O" if player == "X" else "X"


def flatten_index(row: int, col: int, total_cols: int) -> int:
    """Convert (row, col) into flat index for vectors of length rows*cols."""
    return row * total_cols + col


def create_empty_state(rows: int, cols: int, player_to_move: Player = "X") -> State:
    """Create an empty board state."""
    cell_count = rows * cols
    x_ttl_by_cell = [0] * cell_count
    o_ttl_by_cell = [0] * cell_count
    return (x_ttl_by_cell, o_ttl_by_cell, player_to_move)


def to_key(state: State) -> Key:
    """Convert a mutable State into an immutable Key."""
    x_ttl_by_cell, o_ttl_by_cell, player_to_move = state
    return (tuple(x_ttl_by_cell), tuple(o_ttl_by_cell), player_to_move)


def count_occupied_cells(state: State) -> int:
    """Count the number of occupied cells (either X or O has TTL > 0)."""
    x_ttl_by_cell, o_ttl_by_cell, _ = state
    occupied_count = 0
    for index in range(len(x_ttl_by_cell)):
        if x_ttl_by_cell[index] > 0 or o_ttl_by_cell[index] > 0:
            occupied_count += 1
    return occupied_count

# endregion


# region Display Helpers

def _format_cell(ttl_value: int, player_char: str, stone_decay_amount: int) -> str:
    """
    Format a cell for display.

    If stone_decay_amount > 0, stones near expiration (ttl <= stone_decay_amount)
    are displayed in lowercase to visually indicate imminent decay.
    """
    if ttl_value <= 0:
        return " "
    if stone_decay_amount > 0 and ttl_value <= stone_decay_amount:
        return player_char.lower()
    return player_char


def draw_board(state: State, rows: int, cols: int, stone_decay_amount: int) -> None:
    """Print the board to stdout."""
    x_ttl_by_cell, o_ttl_by_cell, player_to_move = state
    print()
    print(f"(to_play={player_to_move})")
    for row in range(rows):
        rendered_row: List[str] = []
        for col in range(cols):
            index = flatten_index(row, col, cols)
            if x_ttl_by_cell[index] > 0:
                rendered_row.append(_format_cell(x_ttl_by_cell[index], "X", stone_decay_amount))
            elif o_ttl_by_cell[index] > 0:
                rendered_row.append(_format_cell(o_ttl_by_cell[index], "O", stone_decay_amount))
            else:
                rendered_row.append(" ")
        print(f'║{"│".join(rendered_row)}║')

# endregion


# region Gravity Mechanics

def apply_gravity_to_board(
    x_ttl_by_cell: BoardVector,
    o_ttl_by_cell: BoardVector,
    rows: int,
    cols: int,
) -> Tuple[BoardVector, BoardVector]:
    """
    Apply gravity column-by-column: stones fall to the bottom of each column,
    preserving their relative order (bottom-most stones remain bottom-most).
    """
    total_cells = rows * cols
    new_x_ttl_by_cell = [0] * total_cells
    new_o_ttl_by_cell = [0] * total_cells

    for col in range(cols):
        # Collect stones bottom-to-top for this column.
        column_stack: List[Tuple[Player, int]] = []
        for row in range(rows - 1, -1, -1):
            index = flatten_index(row, col, cols)
            if x_ttl_by_cell[index] > 0:
                column_stack.append(("X", x_ttl_by_cell[index]))
            elif o_ttl_by_cell[index] > 0:
                column_stack.append(("O", o_ttl_by_cell[index]))

        # Re-pack to bottom.
        fill_row = rows - 1
        for owner, ttl_value in column_stack:
            target_index = flatten_index(fill_row, col, cols)
            if owner == "X":
                new_x_ttl_by_cell[target_index] = ttl_value
            else:
                new_o_ttl_by_cell[target_index] = ttl_value
            fill_row -= 1

    return new_x_ttl_by_cell, new_o_ttl_by_cell

# endregion


# region Move Generation and Application

def list_legal_moves(state: State, rows: int, cols: int, gravity_enabled: bool) -> List[Move]:
    """
    Generate legal moves for a given state.

    Gravity mode:
      - A move is a column index whose top cell is empty.

    Non-gravity mode:
      - A move is any (row, col) that is empty.
    """
    x_ttl_by_cell, o_ttl_by_cell, _ = state
    legal: List[Move] = []

    if gravity_enabled:
        for col in range(cols):
            top_index = flatten_index(0, col, cols)
            if x_ttl_by_cell[top_index] == 0 and o_ttl_by_cell[top_index] == 0:
                legal.append(col)
        return legal

    for row in range(rows):
        for col in range(cols):
            index = flatten_index(row, col, cols)
            if x_ttl_by_cell[index] == 0 and o_ttl_by_cell[index] == 0:
                legal.append((row, col))
    return legal


def _decay_ttl_in_place(ttl_by_cell: BoardVector, decay_amount: int) -> None:
    """Reduce TTL values in-place, clamping at zero."""
    if decay_amount <= 0:
        return

    for index in range(len(ttl_by_cell)):
        if ttl_by_cell[index] > 0:
            ttl_by_cell[index] -= decay_amount
            if ttl_by_cell[index] <= 0:
                ttl_by_cell[index] = 0


def _place_stone_with_gravity(
    current_player_ttl: BoardVector,
    opponent_player_ttl: BoardVector,
    col: int,
    rows: int,
    cols: int,
    placed_ttl_value: int,
) -> None:
    """Place a stone into a column using gravity rules (lowest available cell)."""
    for row in range(rows - 1, -1, -1):
        index = flatten_index(row, col, cols)
        if current_player_ttl[index] == 0 and opponent_player_ttl[index] == 0:
            current_player_ttl[index] = placed_ttl_value
            return
    raise ValueError(f"Illegal move: column {col} is full")


def _place_stone_without_gravity(
    current_player_ttl: BoardVector,
    opponent_player_ttl: BoardVector,
    row: int,
    col: int,
    cols: int,
    placed_ttl_value: int,
) -> None:
    """Place a stone into an explicitly chosen empty cell."""
    index = flatten_index(row, col, cols)
    if current_player_ttl[index] != 0 or opponent_player_ttl[index] != 0:
        raise ValueError(f"Illegal move: cell {(row, col)} is occupied")
    current_player_ttl[index] = placed_ttl_value


def apply_move(
    state: State,
    move: Move,
    rows: int,
    cols: int,
    gravity_enabled: bool,
    stone_life_amount: int,
    stone_decay_amount: int,
) -> State:
    """
    Apply a move to the state, returning a new state.

    Rules summary:
      1) Place a stone for player_to_move with TTL = stone_life_amount + stone_decay_amount
      2) Then decay ALL stones belonging to player_to_move by stone_decay_amount
         (including the newly placed stone).
      3) If gravity_enabled and stone_decay_amount > 0: apply gravity after decay
      4) Flip player_to_move
    """
    x_ttl_by_cell, o_ttl_by_cell, player_to_move = state
    new_x_ttl_by_cell = x_ttl_by_cell.copy()
    new_o_ttl_by_cell = o_ttl_by_cell.copy()

    placed_ttl_value = stone_life_amount + stone_decay_amount

    current_player_ttl = new_x_ttl_by_cell if player_to_move == "X" else new_o_ttl_by_cell
    opponent_player_ttl = new_o_ttl_by_cell if player_to_move == "X" else new_x_ttl_by_cell

    # Placement
    if gravity_enabled:
        column_index = int(move)
        _place_stone_with_gravity(
            current_player_ttl=current_player_ttl,
            opponent_player_ttl=opponent_player_ttl,
            col=column_index,
            rows=rows,
            cols=cols,
            placed_ttl_value=placed_ttl_value,
        )
    else:
        row_index, col_index = move  # type: ignore[assignment]
        _place_stone_without_gravity(
            current_player_ttl=current_player_ttl,
            opponent_player_ttl=opponent_player_ttl,
            row=int(row_index),
            col=int(col_index),
            cols=cols,
            placed_ttl_value=placed_ttl_value,
        )

    # Decay (only current player's stones)
    _decay_ttl_in_place(current_player_ttl, stone_decay_amount)

    # Gravity after decay (only if decay can create holes)
    if gravity_enabled and stone_decay_amount > 0:
        new_x_ttl_by_cell, new_o_ttl_by_cell = apply_gravity_to_board(
            new_x_ttl_by_cell, new_o_ttl_by_cell, rows, cols
        )

    next_player_to_move = get_other_player(player_to_move)
    return (new_x_ttl_by_cell, new_o_ttl_by_cell, next_player_to_move)

# endregion


# region Win Checking

def generate_winning_lines(rows: int, cols: int, win_length: int) -> List[List[int]]:
    """Precompute all lines of length win_length that count as wins."""
    lines: List[List[int]] = []

    # Horizontal
    for row in range(rows):
        for col in range(cols - win_length + 1):
            lines.append([flatten_index(row, col + offset, cols) for offset in range(win_length)])

    # Vertical
    for row in range(rows - win_length + 1):
        for col in range(cols):
            lines.append([flatten_index(row + offset, col, cols) for offset in range(win_length)])

    # Diagonal down-right
    for row in range(rows - win_length + 1):
        for col in range(cols - win_length + 1):
            lines.append([flatten_index(row + offset, col + offset, cols) for offset in range(win_length)])

    # Diagonal down-left
    for row in range(rows - win_length + 1):
        for col in range(win_length - 1, cols):
            lines.append([flatten_index(row + offset, col - offset, cols) for offset in range(win_length)])

    return lines


def check_winner(state: State, rows: int, cols: int, win_length: int) -> Optional[str]:
    """
    Determine if the position is terminal:
      - "X" or "O" if that player has a line of win_length
      - "draw" if board is full
      - None if not terminal
    """
    x_ttl_by_cell, o_ttl_by_cell, _ = state
    winning_lines = generate_winning_lines(rows, cols, win_length)

    for line in winning_lines:
        if all(x_ttl_by_cell[index] > 0 for index in line):
            return "X"
        if all(o_ttl_by_cell[index] > 0 for index in line):
            return "O"

    total_cells = rows * cols
    if all((x_ttl_by_cell[i] > 0 or o_ttl_by_cell[i] > 0) for i in range(total_cells)):
        return "draw"

    return None

# endregion


# region Graph Building (BFS up to ply limit)

def build_state_graph(
    start_state: State,
    rows: int,
    cols: int,
    win_length: int,
    gravity_enabled: bool,
    stone_life_amount: int,
    stone_decay_amount: int,
    ply_limit: int,
    trace_events: List[Dict[str, Any]],
) -> Tuple[
    Dict[Key, List[Edge]],        # successors
    Dict[Key, List[Key]],         # predecessors
    Dict[Key, Optional[str]],     # terminal label per node
    Dict[Key, int],               # depth (plies from start)
    SolveStats,
]:
    """
    Build a directed graph of reachable states using BFS (breadth-first search),
    limited to ply_limit plies from the start_state.

    Frontier nodes at ply_limit are not expanded and are treated as indeterminate
    if explored_complete ends up False.
    """
    successors_by_key: Dict[Key, List[Edge]] = {}
    predecessors_by_key: Dict[Key, List[Key]] = {}
    terminal_label_by_key: Dict[Key, Optional[str]] = {}
    depth_by_key: Dict[Key, int] = {}

    start_key = to_key(start_state)
    queue: Deque[State] = deque([start_state])
    visited_keys: Set[Key] = {start_key}
    depth_by_key[start_key] = 0

    stats = SolveStats()

    while queue:
        current_state = queue.popleft()
        current_key = to_key(current_state)
        current_depth = depth_by_key[current_key]

        stats.nodes += 1

        terminal_label = check_winner(current_state, rows, cols, win_length)
        terminal_label_by_key[current_key] = terminal_label

        if terminal_label is not None:
            stats.terminals += 1
            successors_by_key[current_key] = []
            continue

        if current_depth >= ply_limit:
            stats.explored_complete = False
            successors_by_key[current_key] = []
            continue

        edges: List[Edge] = []
        for move in list_legal_moves(current_state, rows, cols, gravity_enabled):
            child_state = apply_move(
                current_state,
                move,
                rows,
                cols,
                gravity_enabled,
                stone_life_amount,
                stone_decay_amount,
            )
            child_key = to_key(child_state)

            edges.append(Edge(move=move, child_key=child_key))
            predecessors_by_key.setdefault(child_key, []).append(current_key)

            if child_key not in visited_keys:
                visited_keys.add(child_key)
                depth_by_key[child_key] = current_depth + 1
                queue.append(child_state)

        successors_by_key[current_key] = edges

    trace_events.append(
        {
            "event": "graph_built",
            "nodes": stats.nodes,
            "terminals": stats.terminals,
            "explored_complete": stats.explored_complete,
            "ply_limit": ply_limit,
        }
    )
    return successors_by_key, predecessors_by_key, terminal_label_by_key, depth_by_key, stats

# endregion


# region Retrograde Solve (with Remoteness)

def _get_edge_move_to_child(successors: Dict[Key, List[Edge]], parent: Key, child: Key) -> Move:
    """Find the move that takes parent -> child (assumes it exists)."""
    for edge in successors[parent]:
        if edge.child_key == child:
            return edge.move
    raise KeyError("Child key not found among parent's successors")


def solve_graph_with_remoteness(
    successors_by_key: Dict[Key, List[Edge]],
    predecessors_by_key: Dict[Key, List[Key]],
    terminal_label_by_key: Dict[Key, Optional[str]],
    explored_complete: bool,
    trace_events: List[Dict[str, Any]],
) -> Tuple[Dict[Key, Outcome], Dict[Key, Move], SolveStats]:
    """
    Retrograde analysis:
      - If any child is a win for the side to move => current is a win (min remoteness)
      - If all children are wins for the opponent => current is a loss (max remoteness to delay)
      - Remaining nodes: draw if complete; indeterminate if incomplete
    """
    outcome_by_key: Dict[Key, Outcome] = {}
    best_policy_move_by_key: Dict[Key, Move] = {}

    # Counts for proving losses:
    # remaining_non_opponent_wins[parent] is how many children are NOT proven opponent wins.
    remaining_non_opponent_wins: Dict[Key, int] = {
        key: len(edges) for key, edges in successors_by_key.items()
    }

    # Best move selection memory:
    best_win_choice: Dict[Key, Tuple[int, Move]] = {}          # (min remoteness, move)
    best_loss_delay_choice: Dict[Key, Tuple[int, Move]] = {}   # (max remoteness, move)

    stats = SolveStats(nodes=len(successors_by_key), explored_complete=explored_complete)

    queue: Deque[Key] = deque()

    # Seed terminals + indeterminate frontiers (if incomplete exploration)
    for key in successors_by_key.keys():
        terminal_label = terminal_label_by_key.get(key)
        if terminal_label in ("X", "O", "draw"):
            outcome_by_key[key] = Outcome(result=terminal_label, remoteness=0)
            queue.append(key)
            stats.terminals += 1
        else:
            if len(successors_by_key[key]) == 0 and not explored_complete:
                outcome_by_key[key] = Outcome(result="indeterminate", remoteness=None)
                queue.append(key)

    # Propagate solved information backward
    while queue:
        solved_child_key = queue.popleft()
        solved_child_outcome = outcome_by_key[solved_child_key]

        for parent_key in predecessors_by_key.get(solved_child_key, []):
            if parent_key in outcome_by_key:
                continue

            _, _, parent_player_to_move = parent_key
            parent_player = parent_player_to_move
            opponent_player = get_other_player(parent_player)

            child_result = solved_child_outcome.result

            # Case 1: parent can move to a child that is a WIN for parent_player
            if child_result == parent_player:
                child_remoteness = solved_child_outcome.remoteness or 0
                candidate_remoteness = child_remoteness + 1
                move_used = _get_edge_move_to_child(successors_by_key, parent_key, solved_child_key)

                current_best = best_win_choice.get(parent_key)
                if current_best is None or candidate_remoteness < current_best[0]:
                    best_win_choice[parent_key] = (candidate_remoteness, move_used)

                outcome_by_key[parent_key] = Outcome(result=parent_player, remoteness=best_win_choice[parent_key][0])
                best_policy_move_by_key[parent_key] = best_win_choice[parent_key][1]
                queue.append(parent_key)
                continue

            # Case 2: child is a WIN for opponent -> reduces parent's "escape" options
            if child_result == opponent_player:
                remaining_non_opponent_wins[parent_key] -= 1

                child_remoteness = solved_child_outcome.remoteness or 0
                candidate_remoteness = child_remoteness + 1
                move_used = _get_edge_move_to_child(successors_by_key, parent_key, solved_child_key)

                current_best_delay = best_loss_delay_choice.get(parent_key)
                if current_best_delay is None or candidate_remoteness > current_best_delay[0]:
                    best_loss_delay_choice[parent_key] = (candidate_remoteness, move_used)

                if remaining_non_opponent_wins[parent_key] == 0:
                    # All moves lead to opponent win => parent is a LOSS for parent_player
                    delay_remoteness, delay_move = best_loss_delay_choice[parent_key]
                    outcome_by_key[parent_key] = Outcome(result=opponent_player, remoteness=delay_remoteness)
                    best_policy_move_by_key[parent_key] = delay_move
                    queue.append(parent_key)

                continue

            # Case 3: draw/indeterminate child does not prove win/loss directly

    # Fill remaining outcomes
    for key in successors_by_key.keys():
        if key in outcome_by_key:
            continue
        if explored_complete:
            outcome_by_key[key] = Outcome(result="draw", remoteness=None)
        else:
            outcome_by_key[key] = Outcome(result="indeterminate", remoteness=None)

    stats.proven = sum(1 for out in outcome_by_key.values() if out.result in ("X", "O", "draw"))
    stats.indeterminate = sum(1 for out in outcome_by_key.values() if out.result == "indeterminate")

    trace_events.append(
        {
            "event": "solve_summary",
            "nodes": stats.nodes,
            "terminals": stats.terminals,
            "proven": stats.proven,
            "indeterminate": stats.indeterminate,
            "explored_complete": stats.explored_complete,
        }
    )

    return outcome_by_key, best_policy_move_by_key, stats

# endregion


# region Policy Rollout and Loop Detection

def _select_fallback_move(
    state: State,
    rows: int,
    cols: int,
    win_length: int,
    gravity_enabled: bool,
    stone_life_amount: int,
    stone_decay_amount: int,
    outcome_by_key: Dict[Key, Outcome],
) -> Optional[Move]:
    """
    If there is no policy move (e.g., draw/indeterminate nodes), pick a reasonable fallback:
    - Prefer a move leading to a known draw child (if available)
    - Else pick the first legal move
    - Else None if no legal moves exist
    """
    legal = list_legal_moves(state, rows, cols, gravity_enabled)
    if not legal:
        return None

    child_moves_and_keys: List[Tuple[Move, Key]] = []
    for candidate_move in legal:
        child_state = apply_move(
            state,
            candidate_move,
            rows,
            cols,
            gravity_enabled,
            stone_life_amount,
            stone_decay_amount,
        )
        child_moves_and_keys.append((candidate_move, to_key(child_state)))

    for candidate_move, child_key in child_moves_and_keys:
        if outcome_by_key.get(child_key, Outcome("indeterminate", None)).result == "draw":
            return candidate_move

    return child_moves_and_keys[0][0]


def rollout_policy(
    start_state: State,
    rows: int,
    cols: int,
    win_length: int,
    gravity_enabled: bool,
    stone_life_amount: int,
    stone_decay_amount: int,
    outcome_by_key: Dict[Key, Outcome],
    policy_move_by_key: Dict[Key, Move],
    max_rollout_plies: int,
    trace_events: List[Dict[str, Any]],
) -> Tuple[List[PlayedStep], Optional[LoopWitness], str]:
    """
    Roll out moves from start_state using policy_move_by_key when available.

    Returns:
      steps: played steps
      loop_witness: details if a loop is encountered on-policy
      final_label: "X"/"O"/"draw"/"indeterminate"
    """
    played_steps: List[PlayedStep] = []
    first_seen_step_index_by_key: Dict[Key, int] = {}

    current_state: State = (start_state[0].copy(), start_state[1].copy(), start_state[2])
    current_key = to_key(current_state)

    starting_ply_index = count_occupied_cells(current_state)

    for rollout_offset in range(max_rollout_plies):
        # Loop detection
        if current_key in first_seen_step_index_by_key:
            loop_start_step_index = first_seen_step_index_by_key[current_key]
            cycle_moves = [step.move for step in played_steps[loop_start_step_index:]]
            cycle_states = [step.state_before for step in played_steps[loop_start_step_index:]] + [current_state]

            trace_events.append(
                {
                    "event": "loop_on_policy",
                    "ply": starting_ply_index + rollout_offset,
                    "cycle_start_step_index": loop_start_step_index,
                    "cycle_len": len(cycle_moves),
                }
            )

            return played_steps, LoopWitness(loop_start_step_index, len(played_steps), cycle_moves, cycle_states), "draw"

        first_seen_step_index_by_key[current_key] = len(played_steps)

        # Terminal check
        terminal_label = check_winner(current_state, rows, cols, win_length)
        if terminal_label is not None:
            return played_steps, None, terminal_label

        # Outcome exists?
        if current_key not in outcome_by_key:
            return played_steps, None, "indeterminate"

        # Select move
        chosen_move = policy_move_by_key.get(current_key)
        if chosen_move is None:
            chosen_move = _select_fallback_move(
                current_state,
                rows,
                cols,
                win_length,
                gravity_enabled,
                stone_life_amount,
                stone_decay_amount,
                outcome_by_key,
            )
            if chosen_move is None:
                return played_steps, None, "indeterminate"

        state_before = (current_state[0].copy(), current_state[1].copy(), current_state[2])
        next_state = apply_move(
            current_state,
            chosen_move,
            rows,
            cols,
            gravity_enabled,
            stone_life_amount,
            stone_decay_amount,
        )
        state_after = (next_state[0].copy(), next_state[1].copy(), next_state[2])

        note: Optional[str] = None
        next_key = to_key(next_state)
        if next_key in outcome_by_key:
            note = f"[next_outcome] {outcome_by_key[next_key].result}"

        played_steps.append(
            PlayedStep(
                ply_index=starting_ply_index + rollout_offset,
                move=chosen_move,
                state_before=state_before,
                state_after=state_after,
                note=note,
            )
        )

        current_state = next_state
        current_key = to_key(current_state)

    trace_events.append({"event": "rollout_cutoff", "max_rollout": max_rollout_plies})
    return played_steps, None, "indeterminate"

# endregion


# region Printing and Formatting

def _format_final_label(result: str, loop_detected: bool, remoteness: Optional[int]) -> str:
    """Format final result for display."""
    if result in ("X", "O") and remoteness is not None:
        return f"{result} (in {remoteness} ply)"
    if result == "draw" and loop_detected:
        return "draw (loop_detected)"
    return result


def print_run(
    steps: List[PlayedStep],
    loop: Optional[LoopWitness],
    final_outcome_label: str,
    start_key: Key,
    outcome_by_key: Dict[Key, Outcome],
    rows: int,
    cols: int,
    stone_decay_amount: int,
    trace_events: List[Dict[str, Any]],
    trace_first_n: int,
) -> None:
    """Pretty-print the rollout, loop witness (if any), and trace events."""
    start_remoteness = outcome_by_key.get(start_key, Outcome("indeterminate", None)).remoteness
    header_label = _format_final_label(final_outcome_label, loop is not None, start_remoteness)

    print(f"FINAL RESULT: {header_label}\n")

    for step in steps:
        print(f"Step {step.ply_index} | move={step.move}")
        draw_board(step.state_before, rows, cols, stone_decay_amount)
        draw_board(step.state_after, rows, cols, stone_decay_amount)
        if step.note:
            print(f"  {step.note}")
        print()

    if loop is not None:
        print("LOOP DETECTED")
        print("LOOP START")
        first_cycle_state = loop.cycle_states[0]
        draw_board(first_cycle_state, rows, cols, stone_decay_amount)

        # cycle_states length = len(cycle_moves) + 1
        cycle_start_ply = steps[loop.start_step_index].ply_index if loop.start_step_index < len(steps) else (steps[-1].ply_index if steps else 0)

        for offset, move in enumerate(loop.cycle_moves):
            ply_index = cycle_start_ply + offset
            state_before = loop.cycle_states[offset]
            state_after = loop.cycle_states[offset + 1]

            print()
            print(f"Step {ply_index} | move={move}")
            draw_board(state_before, rows, cols, stone_decay_amount)
            draw_board(state_after, rows, cols, stone_decay_amount)

        print("\nLOOP END")
        draw_board(loop.cycle_states[-1], rows, cols, stone_decay_amount)
        print()

    print(f"--- TRACE EVENTS (first {trace_first_n}) ---")
    for event in trace_events[:trace_first_n]:
        print(event)
    print()

    footer_label = _format_final_label(final_outcome_label, loop is not None, start_remoteness)
    print(f"FINAL RESULT: {footer_label}")

# endregion


# region Example Main

if __name__ == "__main__":
    GAME_PRESETS: Dict[str, Dict[str, Any]] = {
        "tic-tac-toe": {
            "ROWS": 3,
            "COLS": 3,
            "WIN_LENGTH": 3,
            "GRAVITY_ENABLED": False,
            "STONE_LIFE": 1,
            "STONE_DECAY": 0,
        },
        "tic-tac-toe-decay": {
            "ROWS": 3,
            "COLS": 3,
            "WIN_LENGTH": 3,
            "GRAVITY_ENABLED": False,
            "STONE_LIFE": 3,
            "STONE_DECAY": 1,
        },
        "connect-4": {
            "ROWS": 6,
            "COLS": 7,
            "WIN_LENGTH": 4,
            "GRAVITY_ENABLED": True,
            "STONE_LIFE": 1,
            "STONE_DECAY": 0,
        },
        "connect-4-decay": {
            "ROWS": 6,
            "COLS": 7,
            "WIN_LENGTH": 4,
            "GRAVITY_ENABLED": True,
            "STONE_LIFE": 14,
            "STONE_DECAY": 1,
        },
    }

    selected_game_name = "tic-tac-toe-decay"
    selected_game = GAME_PRESETS[selected_game_name]

    rows = selected_game["ROWS"]
    cols = selected_game["COLS"]
    win_length = selected_game["WIN_LENGTH"]
    gravity_enabled = selected_game["GRAVITY_ENABLED"]
    stone_life_amount = selected_game["STONE_LIFE"]
    stone_decay_amount = selected_game["STONE_DECAY"]

    # Configure a starting position
    start_state = create_empty_state(rows, cols, player_to_move="X")

    # Solve settings
    ply_limit = 80
    max_rollout_plies = 200
    trace_first_n = 50

    trace_events: List[Dict[str, Any]] = []

    successors, predecessors, terminal_labels, depths, build_stats = build_state_graph(
        start_state=start_state,
        rows=rows,
        cols=cols,
        win_length=win_length,
        gravity_enabled=gravity_enabled,
        stone_life_amount=stone_life_amount,
        stone_decay_amount=stone_decay_amount,
        ply_limit=ply_limit,
        trace_events=trace_events,
    )

    outcome_by_key, policy_move_by_key, solve_stats = solve_graph_with_remoteness(
        successors_by_key=successors,
        predecessors_by_key=predecessors,
        terminal_label_by_key=terminal_labels,
        explored_complete=build_stats.explored_complete,
        trace_events=trace_events,
    )

    start_key = to_key(start_state)
    start_outcome = outcome_by_key[start_key].result

    steps, loop_witness, rollout_terminal = rollout_policy(
        start_state=start_state,
        rows=rows,
        cols=cols,
        win_length=win_length,
        gravity_enabled=gravity_enabled,
        stone_life_amount=stone_life_amount,
        stone_decay_amount=stone_decay_amount,
        outcome_by_key=outcome_by_key,
        policy_move_by_key=policy_move_by_key,
        max_rollout_plies=max_rollout_plies,
        trace_events=trace_events,
    )

    # Final label rules:
    # - If start is indeterminate, annotate ply-limit
    # - If start is draw, keep "draw" (loop witness is separately shown)
    if start_outcome == "indeterminate":
        final_outcome_label = f"reached ply limit of {ply_limit}, indeterminate"
    else:
        final_outcome_label = start_outcome

    print_run(
        steps=steps,
        loop=loop_witness,
        final_outcome_label=final_outcome_label,
        start_key=start_key,
        outcome_by_key=outcome_by_key,
        rows=rows,
        cols=cols,
        stone_decay_amount=stone_decay_amount,
        trace_events=trace_events,
        trace_first_n=trace_first_n,
    )

# endregion
