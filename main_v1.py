from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

Player = str  # "X" or "O"
Move = int | Tuple[int, int]  # col (gravity) or (r,c)
Key = Tuple[Tuple[int, ...], Tuple[int, ...], Player]  # (X_ttls, O_ttls, to_play)
State = Tuple[List[int], List[int], Player]  # mutable version


# -------------------------
# Core helpers
# -------------------------
def other(p: Player) -> Player:
    return "O" if p == "X" else "X"


def idx_of(r: int, c: int, cols: int) -> int:
    return r * cols + c


def empty_state(rows: int, cols: int, to_play: Player = "X") -> State:
    n = rows * cols
    return ([0] * n, [0] * n, to_play)


def key_of(state: State) -> Key:
    x, o, tp = state
    return (tuple(x), tuple(o), tp)


def count_occupied(state: State) -> int:
    x, o, _ = state
    return sum(1 for i in range(len(x)) if x[i] > 0 or o[i] > 0)


# -------------------------
# Display
# -------------------------
def cell_char(
    ttl: int,
    player_char: str,
    stone_decay: int,
) -> str:
    if ttl <= 0:
        return " "
    if stone_decay > 0 and ttl <= stone_decay:
        return player_char.lower()
    return player_char


def draw_board(state: State, rows: int, cols: int, stone_decay: int) -> None:
    x, o, tp = state
    print()
    print(f"(to_play={tp})")
    for r in range(rows):
        row = []
        for c in range(cols):
            i = idx_of(r, c, cols)
            if x[i] > 0:
                row.append(cell_char(x[i], "X", stone_decay))
            elif o[i] > 0:
                row.append(cell_char(o[i], "O", stone_decay))
            else:
                row.append(" ")
        print(f'║{"│".join(row)}║')


# -------------------------
# Gravity (re-run after decay)
# -------------------------
def apply_gravity(x: List[int], o: List[int], rows: int, cols: int) -> Tuple[List[int], List[int]]:
    nx = [0] * (rows * cols)
    no = [0] * (rows * cols)

    for c in range(cols):
        stack: List[Tuple[str, int]] = []  # bottom-to-top: ("X"/"O", ttl)
        for r in range(rows - 1, -1, -1):
            i = idx_of(r, c, cols)
            if x[i] > 0:
                stack.append(("X", x[i]))
            elif o[i] > 0:
                stack.append(("O", o[i]))

        r_fill = rows - 1
        for owner, ttl in stack:
            j = idx_of(r_fill, c, cols)
            if owner == "X":
                nx[j] = ttl
            else:
                no[j] = ttl
            r_fill -= 1

    return nx, no


# -------------------------
# Move generation / applying
# -------------------------
def legal_moves(state: State, rows: int, cols: int, gravity: bool) -> List[Move]:
    x, o, _ = state
    moves: List[Move] = []
    if gravity:
        for c in range(cols):
            itop = idx_of(0, c, cols)
            if x[itop] == 0 and o[itop] == 0:
                moves.append(c)
        return moves

    for r in range(rows):
        for c in range(cols):
            i = idx_of(r, c, cols)
            if x[i] == 0 and o[i] == 0:
                moves.append((r, c))
    return moves


def _decay_in_place(arr: List[int], decay: int) -> None:
    if decay <= 0:
        return
    for i in range(len(arr)):
        if arr[i] > 0:
            arr[i] -= decay
            if arr[i] <= 0:
                arr[i] = 0


def apply_move(
    state: State,
    move: Move,
    rows: int,
    cols: int,
    gravity: bool,
    stone_life: int,
    stone_decay: int,
) -> State:
    """
    Fixed-player representation:
      state = (X_ttls, O_ttls, to_play)

    Rules:
      - Place a stone for to_play with ttl = stone_life + stone_decay
      - Then decay ALL stones belonging to to_play by stone_decay (including the new one)
      - If gravity and stone_decay > 0, re-run gravity after decay
      - Flip to_play
    """
    x, o, tp = state
    nx = x.copy()
    no = o.copy()

    place_ttl = stone_life + stone_decay
    me_arr = nx if tp == "X" else no
    opp_arr = no if tp == "X" else nx

    if gravity:
        c = int(move)
        placed = False
        for r in range(rows - 1, -1, -1):
            i = idx_of(r, c, cols)
            if me_arr[i] == 0 and opp_arr[i] == 0:
                me_arr[i] = place_ttl
                placed = True
                break
        if not placed:
            raise ValueError(f"Illegal move: column {c} is full")
    else:
        r, c = move  # type: ignore[misc]
        i = idx_of(int(r), int(c), cols)
        if me_arr[i] != 0 or opp_arr[i] != 0:
            raise ValueError(f"Illegal move: {move} occupied")
        me_arr[i] = place_ttl

    # decay phase
    _decay_in_place(me_arr, stone_decay)

    # gravity after decay
    if gravity and stone_decay > 0:
        nx, no = apply_gravity(nx, no, rows, cols)

    return (nx, no, other(tp))


# -------------------------
# Winner checking (ttl>0 counts as stone)
# -------------------------
def winning_lines(rows: int, cols: int, rule: int) -> List[List[int]]:
    lines: List[List[int]] = []

    # horizontal
    for r in range(rows):
        for c in range(cols - rule + 1):
            lines.append([idx_of(r, c + k, cols) for k in range(rule)])

    # vertical
    for r in range(rows - rule + 1):
        for c in range(cols):
            lines.append([idx_of(r + k, c, cols) for k in range(rule)])

    # diag down-right
    for r in range(rows - rule + 1):
        for c in range(cols - rule + 1):
            lines.append([idx_of(r + k, c + k, cols) for k in range(rule)])

    # diag down-left
    for r in range(rows - rule + 1):
        for c in range(rule - 1, cols):
            lines.append([idx_of(r + k, c - k, cols) for k in range(rule)])

    return lines


def check_winner(state: State, rows: int, cols: int, rule: int) -> Optional[str]:
    x, o, _ = state
    lines = winning_lines(rows, cols, rule)

    for line in lines:
        if all(x[i] > 0 for i in line):
            return "X"
        if all(o[i] > 0 for i in line):
            return "O"

    if all((x[i] > 0 or o[i] > 0) for i in range(rows * cols)):
        return "draw"

    return None


# -------------------------
# Graph building up to ply limit
# -------------------------
@dataclass
class Edge:
    move: Move
    child: Key


@dataclass
class SolveStats:
    nodes: int = 0
    terminals: int = 0
    proven: int = 0
    indeterminate: int = 0
    explored_complete: bool = True


def build_graph(
    start: State,
    rows: int,
    cols: int,
    rule: int,
    gravity: bool,
    stone_life: int,
    stone_decay: int,
    ply_limit: int,
    trace: List[Dict[str, Any]],
) -> Tuple[
    Dict[Key, List[Edge]],         # succ
    Dict[Key, List[Key]],          # preds
    Dict[Key, Optional[str]],      # terminal: "X"/"O"/"draw"/None
    Dict[Key, int],                # ply from start
    SolveStats,
]:
    succ: Dict[Key, List[Edge]] = {}
    preds: Dict[Key, List[Key]] = {}
    terminal: Dict[Key, Optional[str]] = {}
    depth: Dict[Key, int] = {}

    q: Deque[State] = deque()
    k0 = key_of(start)
    q.append(start)
    depth[k0] = 0

    stats = SolveStats()
    visited = set([k0])

    while q:
        s = q.popleft()
        k = key_of(s)
        d = depth[k]

        stats.nodes += 1
        t = check_winner(s, rows, cols, rule)
        terminal[k] = t
        if t is not None:
            stats.terminals += 1
            succ[k] = []
            continue

        if d >= ply_limit:
            # Depth boundary: treat as indeterminate frontier, do not expand.
            stats.explored_complete = False
            succ[k] = []
            continue

        edges: List[Edge] = []
        for mv in legal_moves(s, rows, cols, gravity):
            ns = apply_move(s, mv, rows, cols, gravity, stone_life, stone_decay)
            nk = key_of(ns)
            edges.append(Edge(mv, nk))
            preds.setdefault(nk, []).append(k)

            if nk not in visited:
                visited.add(nk)
                depth[nk] = d + 1
                q.append(ns)

        succ[k] = edges

    trace.append(
        {
            "event": "graph_built",
            "nodes": stats.nodes,
            "terminals": stats.terminals,
            "explored_complete": stats.explored_complete,
            "ply_limit": ply_limit,
        }
    )
    return succ, preds, terminal, depth, stats


# -------------------------
# Retrograde solve with "distance-to-win / distance-to-loss"
# -------------------------
@dataclass
class Outcome:
    result: str  # "X", "O", "draw", "indeterminate"
    remoteness: Optional[int]  # ply-to-end for win/loss; None for draw/indeterminate


def solve_graph_with_remoteness(
    succ: Dict[Key, List[Edge]],
    preds: Dict[Key, List[Key]],
    terminal: Dict[Key, Optional[str]],
    depth: Dict[Key, int],
    explored_complete: bool,
    trace: List[Dict[str, Any]],
) -> Tuple[Dict[Key, Outcome], Dict[Key, Move], SolveStats]:
    outcome: Dict[Key, Outcome] = {}
    policy: Dict[Key, Move] = {}

    # For determining forced LOSS:
    # non_oppwin_remaining[k] counts how many children are NOT proven "opponent wins".
    non_oppwin_remaining: Dict[Key, int] = {}

    # For choosing best loss-delay:
    best_loss_delay: Dict[Key, Tuple[int, Move]] = {}  # max child_remoteness, move

    # For choosing best win:
    best_win: Dict[Key, Tuple[int, Move]] = {}  # min child_remoteness, move

    stats = SolveStats(nodes=len(succ), explored_complete=explored_complete)

    # Initialize
    q: Deque[Key] = deque()
    for k, edges in succ.items():
        non_oppwin_remaining[k] = len(edges)

    # Seed terminals + frontier indeterminates
    for k in succ.keys():
        t = terminal.get(k)
        if t in ("X", "O", "draw"):
            outcome[k] = Outcome(result=t, remoteness=0)
            q.append(k)
            stats.terminals += 1
        else:
            # If no terminal, but this node has no successors and wasn't terminal,
            # it must be a ply-limit frontier (indeterminate) or a dead-end.
            if len(succ[k]) == 0 and not explored_complete:
                outcome[k] = Outcome(result="indeterminate", remoteness=None)
                q.append(k)

    # Propagate
    while q:
        v = q.popleft()
        v_out = outcome[v]

        for p in preds.get(v, []):
            if p in outcome:
                continue

            xk, ok, tp = p
            me = tp
            opp = other(me)

            # Figure out what v means for p:
            # If p->v and v_out.result == me, then p is WIN for me.
            if v_out.result == me:
                # WIN found; choose minimal remoteness
                child_r = v_out.remoteness if v_out.remoteness is not None else 0
                cand = child_r + 1
                mv = next(edge.move for edge in succ[p] if edge.child == v)
                cur = best_win.get(p)
                if cur is None or cand < cur[0]:
                    best_win[p] = (cand, mv)

                # Once a win exists, p is solved as WIN
                outcome[p] = Outcome(result=me, remoteness=best_win[p][0])
                policy[p] = best_win[p][1]
                q.append(p)
                continue

            # If v is proven win for opp, that reduces remaining non-oppwin options.
            if v_out.result == opp:
                non_oppwin_remaining[p] -= 1
                # Track best delay (maximize opponent's remoteness)
                child_r = v_out.remoteness if v_out.remoteness is not None else 0
                cand = child_r + 1
                mv = next(edge.move for edge in succ[p] if edge.child == v)
                cur = best_loss_delay.get(p)
                if cur is None or cand > cur[0]:
                    best_loss_delay[p] = (cand, mv)

                if non_oppwin_remaining[p] == 0:
                    # All moves lead to opp win => LOSS for me
                    # choose move that maximizes remoteness (delay loss)
                    delay = best_loss_delay[p][0]
                    outcome[p] = Outcome(result=opp, remoteness=delay)
                    policy[p] = best_loss_delay[p][1]
                    q.append(p)
                continue

            # v is draw/indeterminate -> doesn't prove win for either, doesn't reduce non_oppwin_remaining

    # Anything still unknown:
    # - If we explored complete state space: unresolved are draws (cycles with no forced win/loss).
    # - If we hit ply limit: unresolved are indeterminate.
    for k in succ.keys():
        if k in outcome:
            continue
        if explored_complete:
            outcome[k] = Outcome(result="draw", remoteness=None)
        else:
            outcome[k] = Outcome(result="indeterminate", remoteness=None)

    stats.proven = sum(1 for v in outcome.values() if v.result in ("X", "O", "draw"))
    stats.indeterminate = sum(1 for v in outcome.values() if v.result == "indeterminate")

    trace.append(
        {
            "event": "solve_summary",
            "nodes": stats.nodes,
            "terminals": stats.terminals,
            "proven": stats.proven,
            "indeterminate": stats.indeterminate,
            "explored_complete": stats.explored_complete,
        }
    )
    return outcome, policy, stats


# -------------------------
# Policy roll-out + loop witness (shortest cycle found on the played line)
# -------------------------
@dataclass
class PlayedStep:
    ply: int
    move: Move
    before: State
    after: State
    note: Optional[str] = None


@dataclass
class LoopWitness:
    start_index: int
    end_index: int
    cycle_moves: List[Move]
    cycle_states: List[State]


def rollout_policy(
    start: State,
    rows: int,
    cols: int,
    rule: int,
    gravity: bool,
    stone_life: int,
    stone_decay: int,
    outcome: Dict[Key, Outcome],
    policy: Dict[Key, Move],
    max_rollout: int,
    trace: List[Dict[str, Any]],
) -> Tuple[List[PlayedStep], Optional[LoopWitness], str]:
    """
    Returns:
      steps: list of PlayedStep
      loop: LoopWitness if a loop is encountered on-policy
      final_label: "X"/"O"/"draw"/"indeterminate" (and may append "(loop_detected)" externally)
    """
    steps: List[PlayedStep] = []
    seen_at: Dict[Key, int] = {}

    s = (start[0].copy(), start[1].copy(), start[2])
    k = key_of(s)
    ply0 = count_occupied(s)

    for i in range(max_rollout):
        # loop?
        if k in seen_at:
            j = seen_at[k]
            cycle_moves = [st.move for st in steps[j:]]
            cycle_states = [st.before for st in steps[j:]] + [s]
            trace.append(
                {
                    "event": "loop_on_policy",
                    "ply": ply0 + i,
                    "cycle_start_step_index": j,
                    "cycle_len": len(cycle_moves),
                }
            )
            return steps, LoopWitness(j, len(steps), cycle_moves, cycle_states), "draw"

        seen_at[k] = len(steps)

        term = check_winner(s, rows, cols, rule)
        if term is not None:
            return steps, None, term

        out = outcome.get(k)
        if out is None:
            return steps, None, "indeterminate"

        # Choose move
        mv = policy.get(k)
        if mv is None:
            # No policy for draw/indeterminate nodes; pick a "safe" move if possible
            # (prefer a move into a draw if known).
            edges = []
            # reconstruct successors on the fly (since we might not have succ map here)
            for cand in legal_moves(s, rows, cols, gravity):
                ns = apply_move(s, cand, rows, cols, gravity, stone_life, stone_decay)
                nk = key_of(ns)
                edges.append((cand, nk))
            picked: Optional[Move] = None
            for cand, nk in edges:
                if outcome.get(nk, Outcome("indeterminate", None)).result == "draw":
                    picked = cand
                    break
            if picked is None and edges:
                picked = edges[0][0]
            if picked is None:
                return steps, None, "indeterminate"
            mv = picked

        before = (s[0].copy(), s[1].copy(), s[2])
        ns = apply_move(s, mv, rows, cols, gravity, stone_life, stone_decay)
        after = (ns[0].copy(), ns[1].copy(), ns[2])

        note = None
        nk = key_of(ns)
        if nk in outcome:
            note = f"[next_outcome] {outcome[nk].result}"

        steps.append(PlayedStep(ply=ply0 + i, move=mv, before=before, after=after, note=note))

        s = ns
        k = key_of(s)

    trace.append({"event": "rollout_cutoff", "max_rollout": max_rollout})
    return steps, None, "indeterminate"


# -------------------------
# Printing (final result at top + bottom; loop start/end)
# -------------------------
def format_final(out: str, loop: bool, remoteness: Optional[int]) -> str:
    if out in ("X", "O") and remoteness is not None:
        return f"{out} (in {remoteness} ply)"
    if out == "draw" and loop:
        return "draw (loop_detected)"
    return out


def print_run(
    steps: List[PlayedStep],
    loop: Optional[LoopWitness],
    final_outcome: str,
    start_key: Key,
    outcome_map: Dict[Key, Outcome],
    rows: int,
    cols: int,
    stone_decay: int,
    trace: List[Dict[str, Any]],
    trace_first_n: int,
) -> None:
    start_rem = outcome_map.get(start_key, Outcome("indeterminate", None)).remoteness
    final_label_top = format_final(final_outcome, loop is not None, start_rem)

    # TOP
    print(f"FINAL RESULT: {final_label_top}\n")

    # MOVES PLAYED
    for st in steps:
        print(f"Step {st.ply} | move={st.move}")
        draw_board(st.before, rows, cols, stone_decay)
        draw_board(st.after, rows, cols, stone_decay)
        if st.note:
            print(f"  {st.note}")
        print()

    # LOOP SECTION
    if loop is not None:
        print("LOOP DETECTED")
        print("LOOP START")
        # Print the cycle as steps, using the stored cycle_states.
        # We print from the cycle start state through the moves, ending back at start.
        # cycle_states length = len(cycle_moves)+1
        cycle_start_ply = steps[loop.start_index].ply if loop.start_index < len(steps) else (steps[-1].ply if steps else 0)
        s0 = loop.cycle_states[0]
        draw_board(s0, rows, cols, stone_decay)
        for j, mv in enumerate(loop.cycle_moves):
            ply = cycle_start_ply + j
            before = loop.cycle_states[j]
            after = loop.cycle_states[j + 1]
            print()
            print(f"Step {ply} | move={mv}")
            draw_board(before, rows, cols, stone_decay)
            draw_board(after, rows, cols, stone_decay)
        print("\nLOOP END")
        draw_board(loop.cycle_states[-1], rows, cols, stone_decay)
        print()

    # TRACE
    print(f"--- TRACE EVENTS (first {trace_first_n}) ---")
    for e in trace[:trace_first_n]:
        print(e)
    print()

    # BOTTOM
    final_label_bottom = format_final(final_outcome, loop is not None, start_rem)
    print(f"FINAL RESULT: {final_label_bottom}")


# -------------------------
# Example main
# -------------------------
if __name__ == "__main__":
    GAME_TYPES = {
        "tic-tac-toe": {
            "ROWS": 3,
            "COLS": 3,
            "WIN_CON": 3,
            "GRAVITY": False,
            "STONE_LIFE": 1,
            "STONE_DECAY": 0,
        },
        "tic-tac-toe-decay": {
            "ROWS": 3,
            "COLS": 3,
            "WIN_CON": 3,
            "GRAVITY": False,
            "STONE_LIFE": 3,
            "STONE_DECAY": 1,
        },
        "connect-4": {
            "ROWS": 6,
            "COLS": 7,
            "WIN_CON": 4,
            "GRAVITY": True,
            "STONE_LIFE": 1,
            "STONE_DECAY": 0,
        },
        "connect-4-decay": {
            "ROWS": 6,
            "COLS": 7,
            "WIN_CON": 4,
            "GRAVITY": True,
            "STONE_LIFE": 14,
            "STONE_DECAY": 1,
        },
    }

    GAME_TYPE = "tic-tac-toe-decay"
    GAME = GAME_TYPES[GAME_TYPE]

    ROWS = GAME["ROWS"]
    COLS = GAME["COLS"]
    WIN_CON = GAME["WIN_CON"]
    GRAVITY = GAME["GRAVITY"]
    STONE_LIFE = GAME["STONE_LIFE"]
    STONE_DECAY = GAME["STONE_DECAY"]

    # ---------- Configure a starting position ----------
    # This is an example matching your typical "two moves played" setup:
    # X played center, O played (0,0), and now X to play.
    start = empty_state(ROWS, COLS, to_play="X")

    # Tic-Tac-Toe starting move examples
    #start = apply_move(start, (0, 1),
    #            ROWS, COLS, GRAVITY, STONE_LIFE, STONE_DECAY)  # X
    #start = apply_move(start, (1, 1),
    #            ROWS, COLS, GRAVITY, STONE_LIFE, STONE_DECAY)  # X

    # Adaptive starting move examples
    #start = apply_move(start, (ROWS // 2, COLS // 2) if not GRAVITY else (COLS // 2),
    #            ROWS, COLS, GRAVITY, STONE_LIFE, STONE_DECAY)  # X
    #start = apply_move(start, (0, 1) if not GRAVITY else (COLS // 2),
    #            ROWS, COLS, GRAVITY, STONE_LIFE, STONE_DECAY)  # O

    # ---------- Solve settings ----------
    # ply_limit: how far (in plies from the start position) we explore the graph.
    # If you want "complete" solving for tic-tac-toe-decay, set this high enough that
    # no frontier nodes remain (explored_complete=True).
    PLY_LIMIT = 80
    MAX_ROLLOUT = 200
    TRACE_FIRST_N = 50

    trace: List[Dict[str, Any]] = []

    succ, preds, terminal, depth, build_stats = build_graph(
        start,
        ROWS,
        COLS,
        WIN_CON,
        GRAVITY,
        STONE_LIFE,
        STONE_DECAY,
        ply_limit=PLY_LIMIT,
        trace=trace,
    )

    outcome_map, policy, solve_stats = solve_graph_with_remoteness(
        succ,
        preds,
        terminal,
        depth,
        explored_complete=build_stats.explored_complete,
        trace=trace,
    )

    start_key = key_of(start)
    start_out = outcome_map[start_key].result
    start_rem = outcome_map[start_key].remoteness

    # Rollout
    steps, loop_witness, rolled = rollout_policy(
        start,
        ROWS,
        COLS,
        WIN_CON,
        GRAVITY,
        STONE_LIFE,
        STONE_DECAY,
        outcome_map,
        policy,
        max_rollout=MAX_ROLLOUT,
        trace=trace,
    )

    # Decide final label:
    # If solver says WIN/LOSS and explored_complete, trust it.
    # If solver says indeterminate, report indeterminate with ply-limit note.
    final_outcome = start_out
    if start_out == "indeterminate":
        final_outcome = f"reached ply limit of {PLY_LIMIT}, indeterminate"
    elif start_out == "draw":
        # If we actually saw a loop on-policy, call it loop_detected; otherwise draw.
        final_outcome = "draw"

    # Print
    # (Remoteness shown only for X/O wins)
    print_run(
        steps=steps,
        loop=loop_witness,
        final_outcome="draw" if start_out == "draw" else start_out if start_out in ("X", "O") else final_outcome,
        start_key=start_key,
        outcome_map=outcome_map,
        rows=ROWS,
        cols=COLS,
        stone_decay=STONE_DECAY,
        trace=trace,
        trace_first_n=TRACE_FIRST_N,
    )
