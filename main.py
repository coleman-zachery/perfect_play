from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

Player = str
Move = Union[int, Tuple[int, int]]
BoardKey = bytes
Key = Tuple[BoardKey, BoardKey, Player]
State = Tuple[List[int], List[int], Player]
WinLine = Tuple[int, ...]
Edge = Tuple[Move, Key]
PredEdge = Tuple[Key, Move]

OTHER = {"X": "O", "O": "X"}
ZERO = 0


@dataclass(slots=True)
class SolveStats:
    nodes: int = 0
    terminals: int = 0
    proven: int = 0
    indeterminate: int = 0
    explored_complete: bool = True


@dataclass(frozen=True, slots=True)
class Outcome:
    result: str
    remoteness: Optional[int]


@dataclass(frozen=True, slots=True)
class PlayedStep:
    ply_index: int
    move: Move
    state_before: State
    state_after: State
    note: Optional[str] = None


@dataclass(frozen=True, slots=True)
class LoopWitness:
    start_step_index: int
    end_step_index: int
    cycle_moves: List[Move]
    cycle_states: List[State]


@dataclass(frozen=True, slots=True)
class GameConfig:
    rows: int
    cols: int
    win_length: int
    gravity_enabled: bool
    stone_life_amount: int
    stone_decay_amount: int

    @property
    def cell_count(self) -> int:
        return self.rows * self.cols

    @property
    def placed_ttl_value(self) -> int:
        v = self.stone_life_amount + self.stone_decay_amount
        if not 0 <= v <= 255:
            raise ValueError("This bytes-optimized version requires placed_ttl_value <= 255")
        return v


@dataclass(frozen=True, slots=True)
class CompiledGame:
    config: GameConfig
    winning_lines: Tuple[WinLine, ...]
    gravity_columns_bottom_up: Tuple[Tuple[int, ...], ...]
    top_indices_by_col: Tuple[int, ...]
    non_gravity_cells: Tuple[Tuple[int, int, int], ...]  # row, col, flat index
    placed_ttl_value: int
    decay_amount: int


def flatten_index(row: int, col: int, cols: int) -> int:
    return row * cols + col


def generate_winning_lines(rows: int, cols: int, win_length: int) -> Tuple[WinLine, ...]:
    lines: List[WinLine] = []
    for row in range(rows):
        base = row * cols
        for col in range(cols - win_length + 1):
            lines.append(tuple(base + col + k for k in range(win_length)))
    for row in range(rows - win_length + 1):
        for col in range(cols):
            lines.append(tuple((row + k) * cols + col for k in range(win_length)))
    for row in range(rows - win_length + 1):
        for col in range(cols - win_length + 1):
            lines.append(tuple((row + k) * cols + col + k for k in range(win_length)))
    for row in range(rows - win_length + 1):
        for col in range(win_length - 1, cols):
            lines.append(tuple((row + k) * cols + col - k for k in range(win_length)))
    return tuple(lines)


def compile_game(config: GameConfig) -> CompiledGame:
    rows, cols = config.rows, config.cols
    columns = tuple(
        tuple(flatten_index(row, col, cols) for row in range(rows - 1, -1, -1))
        for col in range(cols)
    )
    return CompiledGame(
        config=config,
        winning_lines=generate_winning_lines(rows, cols, config.win_length),
        gravity_columns_bottom_up=columns,
        top_indices_by_col=tuple(col_indices[-1] for col_indices in columns),
        non_gravity_cells=tuple((r, c, r * cols + c) for r in range(rows) for c in range(cols)),
        placed_ttl_value=config.placed_ttl_value,
        decay_amount=config.stone_decay_amount,
    )


def create_empty_state(rows: int, cols: int, player_to_move: Player = "X") -> State:
    n = rows * cols
    return ([0] * n, [0] * n, player_to_move)


def to_key(state: State) -> Key:
    x, o, p = state
    return (bytes(x), bytes(o), p)


def from_key(key: Key) -> State:
    x, o, p = key
    return (list(x), list(o), p)


def count_occupied_cells_key(key: Key) -> int:
    x, o, _ = key
    count = 0
    for i in range(len(x)):
        if x[i] or o[i]:
            count += 1
    return count


def _format_cell(ttl_value: int, player_char: str, stone_decay_amount: int) -> str:
    if ttl_value <= 0:
        return " "
    if stone_decay_amount > 0 and ttl_value <= stone_decay_amount:
        return player_char.lower()
    return player_char


def draw_board(state: State, rows: int, cols: int, stone_decay_amount: int) -> None:
    x, o, player_to_move = state
    print(f"\n(to_play={player_to_move})")
    for row in range(rows):
        base = row * cols
        cells: List[str] = []
        for col in range(cols):
            i = base + col
            cells.append(
                _format_cell(x[i], "X", stone_decay_amount)
                if x[i]
                else _format_cell(o[i], "O", stone_decay_amount)
                if o[i]
                else " "
            )
        print(f'║{"│".join(cells)}║')


def _decay_bytearray_in_place(ttls: bytearray, decay_amount: int) -> None:
    if decay_amount <= 0:
        return
    for i in range(len(ttls)):
        ttl = ttls[i]
        if ttl:
            v = ttl - decay_amount
            ttls[i] = v if v > 0 else 0


def _apply_gravity_boards(x: Union[bytes, bytearray], o: Union[bytes, bytearray], game: CompiledGame) -> Tuple[bytes, bytes]:
    nx = bytearray(game.config.cell_count)
    no = bytearray(game.config.cell_count)
    for col_indices in game.gravity_columns_bottom_up:
        fill = 0
        for i in col_indices:
            xv = x[i]
            if xv:
                nx[col_indices[fill]] = xv
                fill += 1
            else:
                ov = o[i]
                if ov:
                    no[col_indices[fill]] = ov
                    fill += 1
    return bytes(nx), bytes(no)


def apply_move_key(key: Key, move: Move, game: CompiledGame) -> Key:
    """Hot path. Branches avoid unnecessary bytearray copies where possible."""
    x_key, o_key, player = key
    decay = game.decay_amount
    placed = game.placed_ttl_value
    next_player = OTHER[player]

    if game.config.gravity_enabled:
        col = int(move)
        col_indices = game.gravity_columns_bottom_up[col]

        if decay == 0:
            if player == "X":
                x = bytearray(x_key)
                for i in col_indices:
                    if not x_key[i] and not o_key[i]:
                        x[i] = placed
                        return (bytes(x), o_key, next_player)
            else:
                o = bytearray(o_key)
                for i in col_indices:
                    if not x_key[i] and not o_key[i]:
                        o[i] = placed
                        return (x_key, bytes(o), next_player)
            raise ValueError(f"Illegal move: column {col} is full")

        if player == "X":
            x = bytearray(x_key)
            for i in col_indices:
                if not x_key[i] and not o_key[i]:
                    x[i] = placed
                    break
            else:
                raise ValueError(f"Illegal move: column {col} is full")
            _decay_bytearray_in_place(x, decay)
            nx, no = _apply_gravity_boards(x, o_key, game)
            return (nx, no, next_player)
        else:
            o = bytearray(o_key)
            for i in col_indices:
                if not x_key[i] and not o_key[i]:
                    o[i] = placed
                    break
            else:
                raise ValueError(f"Illegal move: column {col} is full")
            _decay_bytearray_in_place(o, decay)
            nx, no = _apply_gravity_boards(x_key, o, game)
            return (nx, no, next_player)

    # Non-gravity branch.
    row, col = move  # type: ignore[misc]
    i = int(row) * game.config.cols + int(col)
    if x_key[i] or o_key[i]:
        raise ValueError(f"Illegal move: cell {(row, col)} is occupied")

    if decay == 0:
        if player == "X":
            x = bytearray(x_key)
            x[i] = placed
            return (bytes(x), o_key, next_player)
        o = bytearray(o_key)
        o[i] = placed
        return (x_key, bytes(o), next_player)

    if player == "X":
        x = bytearray(x_key)
        x[i] = placed
        _decay_bytearray_in_place(x, decay)
        return (bytes(x), o_key, next_player)
    o = bytearray(o_key)
    o[i] = placed
    _decay_bytearray_in_place(o, decay)
    return (x_key, bytes(o), next_player)


def _has_line(board: BoardKey, lines: Tuple[WinLine, ...]) -> bool:
    # Fast common specializations avoid inner variable-length loops.
    if lines:
        ln = len(lines[0])
        if ln == 3:
            for a, b, c in lines:  # type: ignore[misc]
                if board[a] and board[b] and board[c]:
                    return True
            return False
        if ln == 4:
            for a, b, c, d in lines:  # type: ignore[misc]
                if board[a] and board[b] and board[c] and board[d]:
                    return True
            return False
    for line in lines:
        ok = True
        for i in line:
            if not board[i]:
                ok = False
                break
        if ok:
            return True
    return False


def _is_full(x: BoardKey, o: BoardKey) -> bool:
    for i in range(len(x)):
        if not x[i] and not o[i]:
            return False
    return True


def check_winner_key(key: Key, game: CompiledGame) -> Optional[str]:
    x, o, _ = key
    lines = game.winning_lines
    if _has_line(x, lines):
        return "X"
    if _has_line(o, lines):
        return "O"
    if _is_full(x, o):
        return "draw"
    return None


def build_state_graph(
    start_state: State,
    game: CompiledGame,
    ply_limit: int,
    trace_events: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[Key, List[Edge]], Dict[Key, List[PredEdge]], Dict[Key, str], Dict[Key, int], SolveStats]:
    if trace_events is None:
        trace_events = []

    successors_by_key: Dict[Key, List[Edge]] = {}
    predecessors_by_key: Dict[Key, List[PredEdge]] = {}
    terminal_label_by_key: Dict[Key, str] = {}  # only terminal keys stored
    depth_by_key: Dict[Key, int] = {}

    start_key = to_key(start_state)
    queue: Deque[Key] = deque([start_key])
    depth_by_key[start_key] = 0
    stats = SolveStats()
    gravity = game.config.gravity_enabled

    while queue:
        current_key = queue.popleft()
        current_depth = depth_by_key[current_key]
        stats.nodes += 1

        terminal_label = check_winner_key(current_key, game)
        if terminal_label is not None:
            terminal_label_by_key[current_key] = terminal_label
            stats.terminals += 1
            successors_by_key[current_key] = []
            continue

        if current_depth >= ply_limit:
            stats.explored_complete = False
            successors_by_key[current_key] = []
            continue

        x_key, o_key, _ = current_key
        edges: List[Edge] = []

        if gravity:
            for col, top_i in enumerate(game.top_indices_by_col):
                if x_key[top_i] or o_key[top_i]:
                    continue
                child_key = apply_move_key(current_key, col, game)
                edges.append((col, child_key))
                predecessors_by_key.setdefault(child_key, []).append((current_key, col))
                if child_key not in depth_by_key:
                    depth_by_key[child_key] = current_depth + 1
                    queue.append(child_key)
        else:
            for r, c, i in game.non_gravity_cells:
                if x_key[i] or o_key[i]:
                    continue
                move = (r, c)
                child_key = apply_move_key(current_key, move, game)
                edges.append((move, child_key))
                predecessors_by_key.setdefault(child_key, []).append((current_key, move))
                if child_key not in depth_by_key:
                    depth_by_key[child_key] = current_depth + 1
                    queue.append(child_key)

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


def solve_graph_with_remoteness(
    successors_by_key: Dict[Key, List[Edge]],
    predecessors_by_key: Dict[Key, List[PredEdge]],
    terminal_label_by_key: Dict[Key, str],
    explored_complete: bool,
    trace_events: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[Key, Outcome], Dict[Key, Move], SolveStats]:
    if trace_events is None:
        trace_events = []

    outcome_by_key: Dict[Key, Outcome] = {}
    best_policy_move_by_key: Dict[Key, Move] = {}
    remaining_non_opponent_wins: Dict[Key, int] = {k: len(v) for k, v in successors_by_key.items()}
    best_win_choice: Dict[Key, Tuple[int, Move]] = {}
    best_loss_delay_choice: Dict[Key, Tuple[int, Move]] = {}

    stats = SolveStats(nodes=len(successors_by_key), explored_complete=explored_complete)
    queue: Deque[Key] = deque()

    # Only terminals are in terminal_label_by_key, avoiding a large optional-label dictionary.
    for key, label in terminal_label_by_key.items():
        outcome_by_key[key] = Outcome(label, 0)
        queue.append(key)
        stats.terminals += 1

    if not explored_complete:
        for key, edges in successors_by_key.items():
            if not edges and key not in outcome_by_key:
                outcome_by_key[key] = Outcome("indeterminate", None)
                queue.append(key)

    while queue:
        solved_child_key = queue.popleft()
        solved_child_outcome = outcome_by_key[solved_child_key]
        child_result = solved_child_outcome.result
        child_remote = solved_child_outcome.remoteness or 0

        for parent_key, move in predecessors_by_key.get(solved_child_key, ()):  # tuple object, no PredEdge attr lookup
            if parent_key in outcome_by_key:
                continue

            parent_player = parent_key[2]
            opponent_player = OTHER[parent_player]

            if child_result == parent_player:
                cand = child_remote + 1
                current_best = best_win_choice.get(parent_key)
                if current_best is None or cand < current_best[0]:
                    best_win_choice[parent_key] = (cand, move)
                win_remote, win_move = best_win_choice[parent_key]
                outcome_by_key[parent_key] = Outcome(parent_player, win_remote)
                best_policy_move_by_key[parent_key] = win_move
                queue.append(parent_key)

            elif child_result == opponent_player:
                remaining = remaining_non_opponent_wins[parent_key] - 1
                remaining_non_opponent_wins[parent_key] = remaining
                cand = child_remote + 1
                current_delay = best_loss_delay_choice.get(parent_key)
                if current_delay is None or cand > current_delay[0]:
                    best_loss_delay_choice[parent_key] = (cand, move)
                if remaining == 0:
                    delay_remote, delay_move = best_loss_delay_choice[parent_key]
                    outcome_by_key[parent_key] = Outcome(opponent_player, delay_remote)
                    best_policy_move_by_key[parent_key] = delay_move
                    queue.append(parent_key)

    fallback = "draw" if explored_complete else "indeterminate"
    fallback_outcome = Outcome(fallback, None)
    for key in successors_by_key:
        if key not in outcome_by_key:
            outcome_by_key[key] = fallback_outcome

    stats.proven = sum(out.result in ("X", "O", "draw") for out in outcome_by_key.values())
    stats.indeterminate = sum(out.result == "indeterminate" for out in outcome_by_key.values())
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


def _select_fallback_move_from_graph(
    current_key: Key,
    successors_by_key: Dict[Key, List[Edge]],
    outcome_by_key: Dict[Key, Outcome],
) -> Optional[Move]:
    edges = successors_by_key.get(current_key, [])
    if not edges:
        return None
    for move, child_key in edges:
        out = outcome_by_key.get(child_key)
        if out is not None and out.result == "draw":
            return move
    for move, child_key in edges:
        out = outcome_by_key.get(child_key)
        if out is not None and out.result != "indeterminate":
            return move
    return edges[0][0]


def rollout_policy(
    start_state: State,
    game: CompiledGame,
    successors_by_key: Dict[Key, List[Edge]],
    outcome_by_key: Dict[Key, Outcome],
    policy_move_by_key: Dict[Key, Move],
    max_rollout_plies: int,
    trace_events: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[PlayedStep], Optional[LoopWitness], str]:
    if trace_events is None:
        trace_events = []

    played_steps: List[PlayedStep] = []
    first_seen_step_index_by_key: Dict[Key, int] = {}
    current_key = to_key(start_state)
    starting_ply_index = count_occupied_cells_key(current_key)

    for rollout_offset in range(max_rollout_plies):
        previous_seen = first_seen_step_index_by_key.get(current_key)
        if previous_seen is not None:
            cycle_moves = [step.move for step in played_steps[previous_seen:]]
            cycle_states = [step.state_before for step in played_steps[previous_seen:]] + [from_key(current_key)]
            trace_events.append(
                {
                    "event": "loop_on_policy",
                    "ply": starting_ply_index + rollout_offset,
                    "cycle_start_step_index": previous_seen,
                    "cycle_len": len(cycle_moves),
                }
            )
            return played_steps, LoopWitness(previous_seen, len(played_steps), cycle_moves, cycle_states), "draw"

        first_seen_step_index_by_key[current_key] = len(played_steps)
        terminal_label = check_winner_key(current_key, game)
        if terminal_label is not None:
            return played_steps, None, terminal_label
        if current_key not in outcome_by_key:
            return played_steps, None, "indeterminate"

        chosen_move = policy_move_by_key.get(current_key)
        if chosen_move is None:
            chosen_move = _select_fallback_move_from_graph(current_key, successors_by_key, outcome_by_key)
            if chosen_move is None:
                return played_steps, None, "indeterminate"

        next_key = apply_move_key(current_key, chosen_move, game)
        next_outcome = outcome_by_key.get(next_key)
        played_steps.append(
            PlayedStep(
                ply_index=starting_ply_index + rollout_offset,
                move=chosen_move,
                state_before=from_key(current_key),
                state_after=from_key(next_key),
                note=f"[next_outcome] {next_outcome.result}" if next_outcome else None,
            )
        )
        current_key = next_key

    trace_events.append({"event": "rollout_cutoff", "max_rollout": max_rollout_plies})
    return played_steps, None, "indeterminate"


def _format_final_label(result: str, loop_detected: bool, remoteness: Optional[int]) -> str:
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
        draw_board(loop.cycle_states[0], rows, cols, stone_decay_amount)
        cycle_start_ply = steps[loop.start_step_index].ply_index if loop.start_step_index < len(steps) else 0
        for offset, move in enumerate(loop.cycle_moves):
            print(f"\nStep {cycle_start_ply + offset} | move={move}")
            draw_board(loop.cycle_states[offset], rows, cols, stone_decay_amount)
            draw_board(loop.cycle_states[offset + 1], rows, cols, stone_decay_amount)
        print("\nLOOP END")
        draw_board(loop.cycle_states[-1], rows, cols, stone_decay_amount)
        print()

    print(f"--- TRACE EVENTS (first {trace_first_n}) ---")
    for event in trace_events[:trace_first_n]:
        print(event)
    print(f"\nFINAL RESULT: {header_label}")


def solve_and_rollout(
    start_state: State,
    game: CompiledGame,
    ply_limit: int,
    max_rollout_plies: int,
) -> Tuple[Dict[Key, List[Edge]], Dict[Key, Outcome], Dict[Key, Move], List[PlayedStep], Optional[LoopWitness], str, List[Dict[str, Any]]]:
    trace_events: List[Dict[str, Any]] = []
    successors, predecessors, terminals, _depths, build_stats = build_state_graph(start_state, game, ply_limit, trace_events)
    outcome_by_key, policy_move_by_key, _solve_stats = solve_graph_with_remoteness(
        successors, predecessors, terminals, build_stats.explored_complete, trace_events
    )
    steps, loop_witness, _rollout_terminal = rollout_policy(
        start_state, game, successors, outcome_by_key, policy_move_by_key, max_rollout_plies, trace_events
    )
    start_outcome = outcome_by_key[to_key(start_state)].result
    final_label = f"reached ply limit of {ply_limit}, indeterminate" if start_outcome == "indeterminate" else start_outcome
    return successors, outcome_by_key, policy_move_by_key, steps, loop_witness, final_label, trace_events


if __name__ == "__main__":
    GAME_PRESETS: Dict[str, GameConfig] = {
        "tic-tac-toe": GameConfig(3, 3, 3, False, 1, 0),
        "tic-tac-toe-decay": GameConfig(3, 3, 3, False, 3, 1),
        "connect-4": GameConfig(6, 7, 4, True, 1, 0),
    }

    selected_game_name = "tic-tac-toe-decay"
    game = compile_game(GAME_PRESETS[selected_game_name])
    cfg = game.config

    start_state = create_empty_state(cfg.rows, cfg.cols, player_to_move="X")
    ply_limit = 80
    max_rollout_plies = 200
    trace_first_n = 50

    successors, outcome_by_key, policy_move_by_key, steps, loop_witness, final_label, trace_events = solve_and_rollout(
        start_state, game, ply_limit, max_rollout_plies
    )

    print_run(
        steps,
        loop_witness,
        final_label,
        to_key(start_state),
        outcome_by_key,
        cfg.rows,
        cfg.cols,
        cfg.stone_decay_amount,
        trace_events,
        trace_first_n,
    )
