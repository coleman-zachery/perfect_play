#!/usr/bin/env python3
"""
Connect-4 solver with bitboard alpha-beta search and optional persistent mmap cache.

Major upgrades over the original script:
  - Keeps the strong bitboard/negamax core: 49-bit board encoding, alpha-beta pruning,
    win-distance scoring, center-first move ordering, and mirror canonicalization.
  - Adds a persistent fixed-width binary transposition table inspired by MEMORY-PRIMER.md.
  - Uses mmap-backed 32-byte records so solved positions can survive process restarts and
    the OS can page cache state data instead of Python holding every entry as objects.
  - Maintains a compact in-process key->record index for O(1) lookup into the mmap file.
    This avoids repeatedly binary-searching or re-sorting during recursive search. The
    on-disk data remains raw fixed-width records, not JSON or pickle.
  - Includes helpers for extracting a weak strategy graph from solved/cached positions.

Record layout, little-endian, 32 bytes per record:
    uint64 position/current      offset 0
    uint64 mask                  offset 8
    int32  score                 offset 16
    int16  depth                 offset 20
    uint8  bound                 offset 22   0 exact, 1 lower, 2 upper, 255 empty
    int8   best_move             offset 23   -1 unknown, else 0..6
    uint64 reserved              offset 24   future use / alignment

CLI examples:
    python connect4_solver_updated.py 4443 --depth 12
    python connect4_solver_updated.py 4443 --time 2 --cache states.bin
    python z-1.py --depth 10 --cache-dir connect4-cache --cache-capacity 1000000
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any
import typer
import mmap
import os
import struct
import time
import math
import json
import sys
import shutil

WIDTH = 7
HEIGHT = 6
BOARD_SIZE = WIDTH * HEIGHT
SENTINEL_HEIGHT = HEIGHT + 1
DEFAULT_ORDER = (3, 4, 2, 5, 1, 6, 0)
DEFAULT_CACHE_DIR = Path("connect4-cache")
DEFAULT_TT_FILENAME = "connect4-game-states.bin"  # legacy single-file name; not used by tiered default
TIERED_TT_FILENAMES = (
    "1-early-game-states.bin",
    "2-mid-game-states.bin",
    "3-late-game-states.bin",
    "4-end-game-states.bin",
)
DEFAULT_TIER_CUTOFFS = (10, 22, 31)  # opening <=10, mid <=22, late <=31, end >=32
ROOTS_FILENAME = "root-checkpoints.json"
LEARNING_FILENAME = "search-learning.json"
PERFORMANCE_SNAPSHOT_FILENAME = "performance-state-snapshot.log"
ENDGAME_TT_FILENAME = "4-end-game-states.bin"  # legacy option alias


def bottom_mask(col: int) -> int:
    return 1 << (col * SENTINEL_HEIGHT)


def top_mask(col: int) -> int:
    return 1 << (HEIGHT - 1 + col * SENTINEL_HEIGHT)


def column_mask(col: int) -> int:
    return ((1 << HEIGHT) - 1) << (col * SENTINEL_HEIGHT)


BOTTOM_MASK = sum(bottom_mask(c) for c in range(WIDTH))
BOARD_MASK = sum(column_mask(c) for c in range(WIDTH))
TOP_MASKS = tuple(top_mask(c) for c in range(WIDTH))
COLUMN_MASKS = tuple(column_mask(c) for c in range(WIDTH))
BOTTOM_MASKS = tuple(bottom_mask(c) for c in range(WIDTH))


class Bound(IntEnum):
    EXACT = 0
    LOWER = 1
    UPPER = 2
    EMPTY = 255


@dataclass(slots=True)
class TTEntry:
    score: int
    depth: int
    bound: Bound
    best_col: Optional[int]


@dataclass(slots=True)
class SearchStats:
    states_visited: int = 0
    searched_children: int = 0
    legal_child_candidates: int = 0
    branching_positions: int = 0
    tt_hits: int = 0
    cutoffs: int = 0
    pruned_children_est: int = 0
    persistent_hits: int = 0
    persistent_writes: int = 0
    persistent_skipped_low_depth: int = 0
    persistent_skipped_bound: int = 0
    tt_stores: int = 0
    killer_uses: int = 0
    history_uses: int = 0
    initial_persistent_rows: int = 0
    initial_tier_rows: List[int] = None  # type: ignore[assignment]
    persistent_reads_by_tier: List[int] = None  # type: ignore[assignment]
    persistent_writes_by_tier: List[int] = None  # type: ignore[assignment]
    start_time: float = 0.0
    time_limit_s: Optional[float] = None
    root_depth: int = 0
    root_total_moves: int = 0
    root_started_moves: int = 0
    root_completed_moves: int = 0
    current_root_move: Optional[int] = None
    current_root_start_states: int = 0
    root_move_state_costs: List[int] = None  # type: ignore[assignment]
    depth_iteration: int = 0
    max_depth_iteration: int = 0
    max_ply_reached: int = 0
    states_by_turn: List[int] = None  # type: ignore[assignment]
    candidates_by_turn: List[int] = None  # type: ignore[assignment]
    tt_stores_by_turn: List[int] = None  # type: ignore[assignment]
    tt_hits_by_turn: List[int] = None  # type: ignore[assignment]
    mirror_canonicalized: int = 0
    tt_best_move_mirrored: int = 0
    endgame_hits: int = 0
    endgame_writes: int = 0
    last_report_states: int = 0

    def __post_init__(self) -> None:
        if self.root_move_state_costs is None:
            self.root_move_state_costs = []
        if self.states_by_turn is None:
            self.states_by_turn = [0 for _ in range(BOARD_SIZE + 1)]
        if self.candidates_by_turn is None:
            self.candidates_by_turn = [0 for _ in range(BOARD_SIZE + 1)]
        if self.tt_stores_by_turn is None:
            self.tt_stores_by_turn = [0 for _ in range(BOARD_SIZE + 1)]
        if self.tt_hits_by_turn is None:
            self.tt_hits_by_turn = [0 for _ in range(BOARD_SIZE + 1)]
        if self.initial_tier_rows is None:
            self.initial_tier_rows = [0, 0, 0, 0]
        if self.persistent_reads_by_tier is None:
            self.persistent_reads_by_tier = [0, 0, 0, 0]
        if self.persistent_writes_by_tier is None:
            self.persistent_writes_by_tier = [0, 0, 0, 0]

    def timed_out(self) -> bool:
        return self.time_limit_s is not None and (time.perf_counter() - self.start_time) >= self.time_limit_s

    @property
    def nodes(self) -> int:
        return self.states_visited

    @property
    def edges(self) -> int:
        return self.searched_children

    @property
    def hits(self) -> int:
        return self.tt_hits


class ConsoleProgress:
    """Compact terminal progress display that rewrites the same lines.

    Metric names used here:
      - states visited: negamax positions entered during this run.
      - states visited: recursive negamax calls entered during this run.
      - searched children: legal child moves actually descended into; this will usually be
        close to states visited because every child descent creates one state visit.
      - legal candidates: legal child moves observed before alpha-beta pruning. This is the
        better denominator for branching/pruning metrics.
      - alpha-beta cutoffs: times the search proved siblings cannot improve the result.
      - pruned children est: legal sibling moves skipped because of those cutoffs.
      - TT hits: transposition-table hits that avoided re-searching cached states.
      - unique states cached: new transposition-table stores this run; a practical proxy for
        how many non-terminal states became reusable knowledge.
      - effective branching: legal candidates / branching positions, the observed legal width.
      - projected total/remaining/ETA: based on completed root move costs when available.
      - cache rows at start/added: shows cache reuse across runs. This is not a
        full search-stack checkpoint; a restarted process begins again at the root,
        but previously solved states can return immediately as TT hits.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        interval_states: int = 25_000,
        snapshot_path: Optional[Path] = None,
        snapshot_interval_states: int = 1_000_000,
        snapshot_interval_seconds: float = 60.0,
    ) -> None:
        self.enabled = enabled and sys.stderr.isatty()
        self.interval_states = max(1, int(interval_states))
        self.snapshot_path = snapshot_path
        self.snapshot_interval_states = max(0, int(snapshot_interval_states))
        self.snapshot_interval_seconds = max(0.0, float(snapshot_interval_seconds))
        self._line_count = 0
        self._last_time = 0.0
        self._last_snapshot_states = 0
        self._last_snapshot_time = 0.0

    @staticmethod
    def _fmt_int(n: Optional[float | int]) -> str:
        if n is None or (isinstance(n, float) and not math.isfinite(n)):
            return "n/a"
        n = int(max(0, n))
        if n >= 1_000_000_000_000:
            return f"{n / 1_000_000_000_000:.2f}T"
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    @staticmethod
    def _fmt_seconds(seconds: Optional[float]) -> str:
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            return "n/a"
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        if m:
            return f"{m}m {s:02d}s"
        return f"{s}s"


    @staticmethod
    def _turn_axis(width: int = BOARD_SIZE) -> Tuple[str, str]:
        # Two compact rows: ones digits plus sparse tens markers.
        # Use spaces instead of "0" and filler dots so decade marks stand out less noisily.
        ones = "".join(" " if i % 10 == 0 else str(i % 10) for i in range(1, width + 1))
        tens = "".join(str(i // 10) if i % 10 == 0 else " " for i in range(1, width + 1))
        return ones, tens

    @staticmethod
    def _log_chart_rows(
        label: str,
        values: Sequence[int],
        *,
        width: int = BOARD_SIZE,
        height: int = 4,
        inner_width: int = 96,
    ) -> List[str]:
        """Return a 4-row log10 histogram using fractional block heights.

        Each column is one turn/ply.  Counts are converted to log10(count) and
        scaled into 32 vertical steps: zero is blank, then 4 rows * 8 Unicode
        block heights.  This preserves the compact 4-row layout while giving
        much more detail than binary filled/empty cells.
        """
        blocks = "▁▂▃▄▅▆▇█"
        vals = [max(0, int(v)) for v in list(values[:width])]
        logs = [math.log10(v) if v > 0 else 0.0 for v in vals]
        max_log = max(logs) if logs else 0.0
        max_exp = max(0, int(math.ceil(max_log)))
        scale = max(1.0, float(max_exp))
        label_w = 12
        tick_w = 4
        chart_w = min(width, max(10, inner_width - label_w - tick_w - 4))

        # 0 means blank; 1..32 are divided across the 4 display rows.
        heights: List[int] = []
        for value in logs[:chart_w]:
            if value <= 0:
                heights.append(0)
            else:
                heights.append(max(1, min(height * 8, int(round((value / scale) * height * 8)))))

        rows: List[str] = []
        for row_idx in range(height):
            # Top row is row_idx 0; bottom row is row_idx height-1.
            row_top_units = (height - row_idx) * 8
            row_bottom_units = row_top_units - 8
            tick_exp = int(round((row_top_units / (height * 8)) * scale))
            chars = []
            for h in heights:
                if h <= row_bottom_units:
                    chars.append(" ")
                else:
                    partial = min(8, h - row_bottom_units)
                    chars.append(blocks[partial - 1])
            label_text = label if row_idx == 0 else ""
            rows.append(f"{label_text:<{label_w}} e{tick_exp:<2}│{''.join(chars)}")

        ones, tens = ConsoleProgress._turn_axis(chart_w)
        rows.append(f"{'turn ones':<{label_w}}    │{ones}")
        rows.append(f"{'turn tens':<{label_w}}    │{tens}")
        return rows

    @staticmethod
    def _confidence(completed_roots: int, using_running_root: bool = False) -> str:
        if completed_roots <= 0:
            return "first root in progress; ETA is a moving lower bound" if using_running_root else "warming up"
        if completed_roots == 1:
            return "low"
        if completed_roots < 4:
            return "medium"
        return "high"

    def _projection(self, stats: SearchStats) -> Tuple[Optional[float], Optional[float], Optional[float], str, Optional[float]]:
        total_root = max(0, stats.root_total_moves)
        completed = len(stats.root_move_state_costs)
        if total_root <= 0:
            return None, None, None, self._confidence(completed), None
        using_running = False
        if completed > 0:
            avg_root_cost = sum(stats.root_move_state_costs) / completed
        elif stats.root_started_moves > 0 and stats.current_root_start_states > 0:
            # Until one root move finishes, this is only a moving lower-bound style projection.
            # It normally rises as the first root keeps expanding, so do not treat ETA as stable.
            avg_root_cost = max(1.0, stats.states_visited - stats.current_root_start_states)
            using_running = True
        else:
            return None, None, None, self._confidence(completed), None
        projected_total = avg_root_cost * total_root
        remaining = max(0.0, projected_total - stats.states_visited)
        convergence = min(100.0, 100.0 * stats.states_visited / max(1.0, projected_total))
        return projected_total, remaining, convergence, self._confidence(completed, using_running), avg_root_cost

    def maybe_render(self, solver: "Connect4Solver", pos: Position, *, force: bool = False, phase: str = "search") -> None:
        if not self.enabled and self.snapshot_path is None:
            return
        stats = solver.stats
        now = time.perf_counter()
        if not force and stats.states_visited - stats.last_report_states < self.interval_states and now - self._last_time < 0.25:
            return
        stats.last_report_states = stats.states_visited
        self._last_time = now
        elapsed = max(1e-9, now - stats.start_time)
        sps = stats.states_visited / elapsed
        total_root = max(1, stats.root_total_moves)
        completed = min(stats.root_completed_moves, total_root)
        started = min(stats.root_started_moves, total_root)
        projected_total, remaining, convergence, confidence, avg_root_projected = self._projection(stats)
        eta = None if remaining is None or sps <= 0 else remaining / sps
        cache_rows = len(solver.persistent_tt) if solver.persistent_tt is not None else 0
        cache_start_rows = stats.initial_persistent_rows
        cache_added_rows = max(0, cache_rows - cache_start_rows)
        mem_rows = len(solver.tt)
        tt_hit_rate = 100.0 * stats.tt_hits / max(1, stats.states_visited)
        total_observed_candidates = stats.legal_child_candidates
        prune_rate = 100.0 * stats.pruned_children_est / max(1, total_observed_candidates)
        search_rate = 100.0 * stats.searched_children / max(1, total_observed_candidates)
        effective_bf = stats.legal_child_candidates / max(1, stats.branching_positions)
        searched_bf = stats.searched_children / max(1, stats.branching_positions)
        unique_rate = 100.0 * stats.tt_stores / max(1, stats.states_visited)
        avg_root = avg_root_projected
        root_label = "n/a" if stats.current_root_move is None else str(stats.current_root_move + 1)
        current_root_cost = None if stats.current_root_move is None else max(0, stats.states_visited - stats.current_root_start_states)
        width = min(100, max(72, shutil.get_terminal_size((100, 20)).columns))
        inner = width - 4

        def cell(label: str, value: str, w: int = 23) -> str:
            text = f"{label}: {value}"
            return text[:w].ljust(w)

        def row(*cells: str) -> str:
            body = " | ".join(cells)
            return "│ " + body[:inner].ljust(inner) + " │"

        disk_policy_skips = stats.persistent_skipped_low_depth + stats.persistent_skipped_bound
        disk_skip_note = f"{self._fmt_int(disk_policy_skips)} filtered"
        if disk_policy_skips:
            disk_skip_note += f" ({self._fmt_int(stats.persistent_skipped_low_depth)} shallow, {self._fmt_int(stats.persistent_skipped_bound)} bound)"

        def fit(text: str, w: int = inner) -> str:
            text = str(text)
            if len(text) <= w:
                return text.ljust(w)
            if w <= 1:
                return text[:w]
            return (text[: w - 1] + "…").ljust(w)

        def full(text: str) -> str:
            return "│ " + fit(text) + " │"

        def two(left: str, right: str) -> str:
            gap = " │ "
            lw = (inner - len(gap)) // 2
            rw = inner - len(gap) - lw
            return "│ " + fit(left, lw) + gap + fit(right, rw) + " │"

        depth_label = stats.root_depth
        if stats.max_depth_iteration and stats.depth_iteration and stats.depth_iteration != stats.max_depth_iteration:
            depth_text = f"d{stats.depth_iteration}/{stats.max_depth_iteration}"
        else:
            depth_text = f"d{depth_label}"
        root_text = f"roots {completed}/{total_root} done"
        if stats.current_root_move is not None:
            root_text += f" · active {root_label}"
        elif started:
            root_text += f" · started {started}/{total_root}"

        disk_delta = cache_added_rows
        progress_text = None if convergence is None else f"{convergence:.1f}%"
        current_root_text = self._fmt_int(current_root_cost)
        avg_root_text = self._fmt_int(avg_root)

        visit_chart = self._log_chart_rows("visits", stats.states_by_turn[1 : BOARD_SIZE + 1], inner_width=inner)
        hit_chart = self._log_chart_rows("TT hits", stats.tt_hits_by_turn[1 : BOARD_SIZE + 1], inner_width=inner)
        tt_chart = self._log_chart_rows("TT stores", stats.tt_stores_by_turn[1 : BOARD_SIZE + 1], inner_width=inner)

        tier_rows = solver.persistent_tt.rows_by_tier() if hasattr(solver.persistent_tt, "rows_by_tier") else []
        tier_added = [max(0, tier_rows[i] - stats.initial_tier_rows[i]) for i in range(min(len(tier_rows), len(stats.initial_tier_rows)))]
        tier_labels = ("O", "M", "L", "E")
        if tier_rows:
            tier_rows_text = "  ".join(f"{lab}:{self._fmt_int(v)}" for lab, v in zip(tier_labels, tier_rows))
            tier_add_text = "  ".join(f"{lab}:+{self._fmt_int(v)}" for lab, v in zip(tier_labels, tier_added))
            tier_rw_text = "  ".join(
                f"{lab}:{self._fmt_int(r)}/{self._fmt_int(w)}"
                for lab, r, w in zip(tier_labels, stats.persistent_reads_by_tier, stats.persistent_writes_by_tier)
            )
        else:
            tier_rows_text = "single:" + self._fmt_int(cache_rows)
            tier_add_text = "+" + self._fmt_int(disk_delta)
            tier_rw_text = "n/a"

        def three(a: str, b: str, c: str) -> str:
            gap = " │ "
            w1 = (inner - 2 * len(gap)) // 3
            w2 = w1
            w3 = inner - 2 * len(gap) - w1 - w2
            return "│ " + fit(a, w1) + gap + fit(b, w2) + gap + fit(c, w3) + " │"

        mirror_rate = 100.0 * stats.mirror_canonicalized / max(1, stats.states_visited)
        flip_rate = 100.0 * stats.tt_best_move_mirrored / max(1, stats.mirror_canonicalized)
        root_run = f"root {root_label} running" if stats.current_root_move is not None else f"roots {completed}/{total_root} done"
        header_parts = [f"{phase} {depth_text}", root_run, self._fmt_seconds(elapsed)]
        if progress_text is not None:
            header_parts.append(progress_text)
        header = " | ".join(header_parts)
        disk_skip_text = f"skips shallow/bound {self._fmt_int(stats.persistent_skipped_low_depth)}/{self._fmt_int(stats.persistent_skipped_bound)}"

        # Keep dashboard <=35 rows: 2 header rows + 10 metrics rows + 1 title + 12 chart rows + 2 axis rows + borders.
        lines = [
            "┌" + "─" * (width - 2) + "┐",
            full(header),
            "├" + "─" * (width - 2) + "┤",
            three(f"states {self._fmt_int(stats.states_visited)}", f"speed {self._fmt_int(sps)}/s", f"cutoffs {self._fmt_int(stats.cutoffs)}"),
            three(f"moves legal {self._fmt_int(stats.legal_child_candidates)}", f"searched {self._fmt_int(stats.searched_children)} ({search_rate:.1f}%)", f"pruned {self._fmt_int(stats.pruned_children_est)} ({prune_rate:.1f}%)"),
            three(f"branch legal/search {effective_bf:.2f}/{searched_bf:.2f}", f"TT hits {self._fmt_int(stats.tt_hits)} ({tt_hit_rate:.1f}%)", f"RAM stores {self._fmt_int(stats.tt_stores)} ({unique_rate:.1f}%)"),
            three(f"mirror keys {mirror_rate:.1f}%", f"flipped TT moves {flip_rate:.1f}%", f"hints k/h {self._fmt_int(stats.killer_uses)}/{self._fmt_int(stats.history_uses)}"),
            "├" + "─" * (width - 2) + "┤",
            two(f"tier rows {tier_rows_text}", f"tier adds {tier_add_text}"),
            two(f"tier R/W {tier_rw_text}", f"disk R/W {self._fmt_int(stats.persistent_hits)}/{self._fmt_int(stats.persistent_writes)} · {disk_skip_text}"),
            two(f"RAM rows {self._fmt_int(mem_rows)}", "tiers O:0-10 M:11-22 L:23-31 E:32-42"),
            two(f"root active {current_root_text} · done avg {avg_root_text}", f"projected {self._fmt_int(projected_total)} · left {self._fmt_int(remaining)} · ETA {self._fmt_seconds(eta)}"),
            two(f"estimate {confidence}", "resume roots + tiered TT + learning"),
            "├" + "─" * (width - 2) + "┤",
            full("turn distributions, log10 scale; each column is one move number"),
            *[full(r) for r in visit_chart[:4]],
            *[full(r) for r in hit_chart[:4]],
            *[full(r) for r in tt_chart[:4]],
            full(visit_chart[4]),
            full(visit_chart[5]),
            "└" + "─" * (width - 2) + "┘",
        ]
        if self._snapshot_due(stats, now, force=force):
            self._write_snapshot(solver, stats, lines, phase=phase, now=now)

        if not self.enabled:
            return
        if self._line_count:
            sys.stderr.write("\r" + "\x1b[F" * (self._line_count - 1))
        clipped = []
        for line in lines:
            clipped.append(line[:width].ljust(width))
        sys.stderr.write("\r" + "\n".join(clipped))
        sys.stderr.flush()
        self._line_count = len(lines)

    def _snapshot_due(self, stats: SearchStats, now: float, *, force: bool = False) -> bool:
        if self.snapshot_path is None:
            return False
        if force:
            return True
        by_states = self.snapshot_interval_states > 0 and (
            stats.states_visited - self._last_snapshot_states >= self.snapshot_interval_states
        )
        by_time = self.snapshot_interval_seconds > 0 and (
            now - self._last_snapshot_time >= self.snapshot_interval_seconds
        )
        return by_states or by_time

    def _write_snapshot(self, solver: "Connect4Solver", stats: SearchStats, lines: Sequence[str], *, phase: str, now: float) -> None:
        """Write one current, analysis-oriented performance snapshot.

        This intentionally overwrites the prior file instead of appending history.
        The terminal table is included, but the main payload is structured JSON with
        raw per-turn counts and derived ratios so it can be pasted back into ChatGPT
        or analyzed by scripts without re-parsing the UI.
        """
        if self.snapshot_path is None:
            return
        try:
            path = self.snapshot_path
            path.parent.mkdir(parents=True, exist_ok=True)
            elapsed = max(1e-9, now - stats.start_time)
            disk_rows = len(solver.persistent_tt) if solver.persistent_tt is not None else 0
            endgame_rows = len(solver.endgame_tt) if getattr(solver, "endgame_tt", None) is not None else 0
            ram_rows = len(solver.tt)
            legal = max(1, stats.legal_child_candidates)
            states = max(1, stats.states_visited)
            branching = max(1, stats.branching_positions)

            def pct(num: int, den: int) -> float:
                return round(100.0 * num / max(1, den), 4)

            def per_turn_payload() -> List[Dict[str, float | int]]:
                rows: List[Dict[str, float | int]] = []
                for turn in range(1, BOARD_SIZE + 1):
                    visits = int(stats.states_by_turn[turn])
                    candidates = int(stats.candidates_by_turn[turn])
                    hits = int(stats.tt_hits_by_turn[turn])
                    stores = int(stats.tt_stores_by_turn[turn])
                    rows.append({
                        "turn": turn,
                        "visits": visits,
                        "legal_candidates": candidates,
                        "tt_hits": hits,
                        "tt_stores": stores,
                        "avg_legal_candidates_per_visit": round(candidates / max(1, visits), 6),
                        "tt_hit_rate_pct_of_visits": pct(hits, visits),
                        "tt_store_rate_pct_of_visits": pct(stores, visits),
                        "log10_visits": None if visits <= 0 else round(math.log10(visits), 6),
                        "log10_tt_hits": None if hits <= 0 else round(math.log10(hits), 6),
                        "log10_tt_stores": None if stores <= 0 else round(math.log10(stores), 6),
                    })
                return rows

            projected_total, remaining, convergence, confidence, avg_root_projected = self._projection(stats)
            eta = None if remaining is None or stats.states_visited <= 0 else remaining / (stats.states_visited / elapsed)
            payload = {
                "schema": "connect4-performance-snapshot-v3",
                "note": "Single current snapshot; overwritten periodically. Heuristic caches affect speed only, not correctness.",
                "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "phase": phase,
                "runtime": {
                    "elapsed_s": round(elapsed, 3),
                    "states_per_s": round(stats.states_visited / elapsed, 3),
                    "eta_s": None if eta is None or not math.isfinite(eta) else round(eta, 3),
                },
                "search": {
                    "depth": stats.root_depth,
                    "depth_iteration": stats.depth_iteration,
                    "max_depth_iteration": stats.max_depth_iteration,
                    "states_visited": stats.states_visited,
                    "legal_child_candidates": stats.legal_child_candidates,
                    "searched_children": stats.searched_children,
                    "branching_positions": stats.branching_positions,
                    "cutoffs": stats.cutoffs,
                    "pruned_children_est": stats.pruned_children_est,
                    "avg_legal_width": round(stats.legal_child_candidates / branching, 6),
                    "avg_searched_width": round(stats.searched_children / branching, 6),
                    "searched_pct_of_candidates": pct(stats.searched_children, legal),
                    "pruned_pct_of_candidates": pct(stats.pruned_children_est, legal),
                },
                "transposition_tables": {
                    "tt_hits": stats.tt_hits,
                    "tt_hit_rate_pct_of_states": pct(stats.tt_hits, states),
                    "ram_tt_rows": ram_rows,
                    "ram_stores_this_run": stats.tt_stores,
                    "ram_store_rate_pct_of_states": pct(stats.tt_stores, states),
                    "disk_tt_rows_start": stats.initial_persistent_rows,
                    "disk_tt_rows_current": disk_rows,
                    "disk_tt_rows_added": max(0, disk_rows - stats.initial_persistent_rows),
                    "tier_rows_start": list(stats.initial_tier_rows),
                    "tier_rows_current": solver.persistent_tt.rows_by_tier() if hasattr(solver.persistent_tt, "rows_by_tier") else [],
                    "tier_reads": list(stats.persistent_reads_by_tier),
                    "tier_writes": list(stats.persistent_writes_by_tier),
                    "tier_cutoffs": list(getattr(solver.persistent_tt, "cutoffs", [])) if solver.persistent_tt is not None else [],
                    "tier_names": list(getattr(solver.persistent_tt, "NAMES", [])) if solver.persistent_tt is not None else [],
                    "tier_ranges": {"opening": [0, 10], "mid": [11, 22], "late": [23, 31], "end": [32, 42]},
                    "disk_reads": stats.persistent_hits,
                    "disk_writes": stats.persistent_writes,
                    "disk_skipped_shallow": stats.persistent_skipped_low_depth,
                    "disk_skipped_bound": stats.persistent_skipped_bound,
                    "endgame_tt_rows": endgame_rows,
                    "endgame_hits": stats.endgame_hits,
                    "endgame_writes": stats.endgame_writes,
                },
                "ordering_and_symmetry": {
                    "killer_uses": stats.killer_uses,
                    "history_uses": stats.history_uses,
                    "mirror_canonicalized_keys": stats.mirror_canonicalized,
                    "tt_best_move_mirrored": stats.tt_best_move_mirrored,
                    "mirror_key_rate_pct_of_states": pct(stats.mirror_canonicalized, states),
                    "tt_best_move_mirror_flip_rate_pct_of_mirror_keys": pct(stats.tt_best_move_mirrored, stats.mirror_canonicalized),
                },
                "roots": {
                    "completed": stats.root_completed_moves,
                    "total": stats.root_total_moves,
                    "started": stats.root_started_moves,
                    "current_root_move_1_based": None if stats.current_root_move is None else stats.current_root_move + 1,
                    "current_root_cost_states": None if stats.current_root_move is None else max(0, stats.states_visited - stats.current_root_start_states),
                    "completed_root_state_costs": list(stats.root_move_state_costs),
                    "avg_root_projected_states": None if avg_root_projected is None else round(avg_root_projected, 3),
                    "projected_total_states": None if projected_total is None else round(projected_total, 3),
                    "remaining_states_est": None if remaining is None else round(remaining, 3),
                    "convergence_pct": None if convergence is None else round(convergence, 4),
                    "estimate_confidence": confidence,
                },
                "turn_distributions": {
                    "description": "Index 1..42 are move numbers/ply. Index 0 is unused and omitted here.",
                    "by_turn": per_turn_payload(),
                    "raw_arrays": {
                        "states_by_turn": list(stats.states_by_turn[1 : BOARD_SIZE + 1]),
                        "legal_candidates_by_turn": list(stats.candidates_by_turn[1 : BOARD_SIZE + 1]),
                        "tt_hits_by_turn": list(stats.tt_hits_by_turn[1 : BOARD_SIZE + 1]),
                        "tt_stores_by_turn": list(stats.tt_stores_by_turn[1 : BOARD_SIZE + 1]),
                    },
                },
                "terminal_table": list(lines),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            self._last_snapshot_states = stats.states_visited
            self._last_snapshot_time = now
        except Exception:
            # Progress snapshots are diagnostic only; never let them interrupt the solve.
            pass


    def finish(self) -> None:
        if self.enabled and self._line_count:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._line_count = 0


@dataclass(frozen=True, slots=True)
class SolveResult:
    score: int
    best_move: Optional[int]
    solved_exactly: bool
    nodes: int
    edges: int
    prunes: int
    transposition_hits: int
    persistent_hits: int
    persistent_writes: int
    elapsed_s: float

    @property
    def outcome(self) -> str:
        if self.score > 0:
            return "win"
        if self.score < 0:
            return "loss"
        return "draw"


def has_alignment(pos: int) -> bool:
    """True iff bitboard `pos` contains four connected stones."""
    m = pos & (pos >> 1)  # vertical
    if m & (m >> 2):
        return True
    m = pos & (pos >> SENTINEL_HEIGHT)  # horizontal
    if m & (m >> (2 * SENTINEL_HEIGHT)):
        return True
    m = pos & (pos >> HEIGHT)  # diagonal /
    if m & (m >> (2 * HEIGHT)):
        return True
    m = pos & (pos >> (HEIGHT + 2))  # diagonal \
    if m & (m >> (2 * (HEIGHT + 2))):
        return True
    return False


def mirror_bits(bits: int) -> int:
    """Mirror a Connect-4 bitboard horizontally."""
    out = 0
    for c in range(WIDTH):
        col = bits & COLUMN_MASKS[c]
        shift = (WIDTH - 1 - 2 * c) * SENTINEL_HEIGHT
        if shift > 0:
            out |= col << shift
        elif shift < 0:
            out |= col >> (-shift)
        else:
            out |= col
    return out


def mirror_col(col: Optional[int]) -> Optional[int]:
    if col is None:
        return None
    return WIDTH - 1 - int(col)


def canonical_key_info(current: int, mask: int, use_symmetry: bool = True) -> Tuple[Tuple[int, int], bool]:
    """Return (canonical_key, mirrored).

    If mirrored is true, the canonical key is the horizontal mirror of the
    caller's position.  Scores are unchanged by mirroring, but any stored
    best-move column must be mirrored when crossing this boundary.
    """
    if not use_symmetry:
        return (current, mask), False
    original = (current, mask)
    mirrored = (mirror_bits(current), mirror_bits(mask))
    if mirrored < original:
        return mirrored, True
    return original, False


def canonical_key(current: int, mask: int, use_symmetry: bool = True) -> Tuple[int, int]:
    return canonical_key_info(current, mask, use_symmetry)[0]


class Position:
    """Bitboard state. `current` stores stones belonging to the side to move."""

    __slots__ = ("current", "mask", "moves")

    def __init__(self, current: int = 0, mask: int = 0, moves: int = 0) -> None:
        self.current = current
        self.mask = mask
        self.moves = moves

    def copy(self) -> "Position":
        return Position(self.current, self.mask, self.moves)

    def can_play(self, col: int) -> bool:
        return 0 <= col < WIDTH and (self.mask & TOP_MASKS[col]) == 0

    def play_col(self, col: int) -> None:
        if not self.can_play(col):
            raise ValueError(f"Column {col} is full or invalid")
        self.current ^= self.mask
        self.mask |= (self.mask + BOTTOM_MASKS[col]) & COLUMN_MASKS[col]
        self.moves += 1

    def winning_move(self, col: int) -> bool:
        if not self.can_play(col):
            return False
        move = (self.mask + BOTTOM_MASKS[col]) & COLUMN_MASKS[col]
        return bool(move) and has_alignment(self.current | move)

    def legal_moves(self, order: Sequence[int] = DEFAULT_ORDER) -> List[int]:
        return [c for c in order if self.can_play(c)]

    def key(self, canonical: bool = True) -> Tuple[int, int]:
        return canonical_key(self.current, self.mask, canonical)

    @classmethod
    def from_columns(cls, columns: Sequence[int]) -> "Position":
        pos = cls()
        for c in columns:
            pos.play_col(c)
            # A move list should stop when a previous player has just won.
            if has_alignment(pos.mask ^ pos.current):
                break
        return pos

    @classmethod
    def from_grid(cls, grid: Sequence[str], to_move: Optional[str] = None) -> "Position":
        if len(grid) != HEIGHT or any(len(row) != WIDTH for row in grid):
            raise ValueError("Grid must have exactly 6 rows of 7 characters")
        xs = os = mask = 0
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                bit = 1 << (c * SENTINEL_HEIGHT + (HEIGHT - 1 - r))
                if ch.upper() in ("X", "R"):
                    xs |= bit
                    mask |= bit
                elif ch.upper() in ("O", "Y"):
                    os |= bit
                    mask |= bit
                elif ch in (".", " ", "_"):
                    pass
                else:
                    raise ValueError(f"Bad grid character {ch!r}")
        for c in range(WIDTH):
            seen_empty = False
            for h in range(HEIGHT):
                occupied = bool(mask & (1 << (c * SENTINEL_HEIGHT + h)))
                if not occupied:
                    seen_empty = True
                elif seen_empty:
                    raise ValueError("Grid violates gravity: floating stone detected")
        x_count, o_count = xs.bit_count(), os.bit_count()
        if x_count not in (o_count, o_count + 1):
            raise ValueError("Invalid stone counts for normal Connect-4")
        side = ("X" if x_count == o_count else "O") if to_move is None else to_move.upper()
        current = xs if side == "X" else os
        return cls(current=current, mask=mask, moves=x_count + o_count)

    def render(self) -> str:
        us = self.current
        them = self.mask ^ self.current
        rows = []
        for r in range(HEIGHT - 1, -1, -1):
            row = []
            for c in range(WIDTH):
                bit = 1 << (c * SENTINEL_HEIGHT + r)
                row.append("X" if us & bit else "O" if them & bit else ".")
            rows.append("".join(row))
        return "\n".join(rows)


class PersistentTranspositionTable:
    """Raw mmap-backed fixed-record transposition table.

    The file begins with a 4 KiB header and then fixed-width records. Records are appended
    as states are discovered. A small Python dict maps the 16-byte key to the record index;
    values stay in the mmap and are updated in place. This design is faster for recursive
    search than maintaining a physically sorted file after every insert, while keeping the
    storage raw, fixed-width, and OS-page-cache-friendly.
    """

    MAGIC = b"C4TTv2\0\0"
    HEADER_STRUCT = struct.Struct("<8sQQ")  # magic, capacity_records, used_records
    HEADER_SIZE = 4096
    RECORD_STRUCT = struct.Struct("<QQihBbQ")  # current, mask, score, depth, bound, best_move, reserved
    RECORD_SIZE = RECORD_STRUCT.size

    def __init__(self, path: str | os.PathLike[str], *, capacity_records: int = 1_000_000, flush_every: int = 10_000) -> None:
        if capacity_records <= 0:
            raise ValueError("capacity_records must be positive")
        self.path = Path(path)
        self.capacity_records = int(capacity_records)
        self.flush_every = max(1, int(flush_every))
        self._writes_since_flush = 0
        self._header_dirty = False
        self._file = None
        self._mmap: Optional[mmap.mmap] = None
        self._index: Dict[Tuple[int, int], int] = {}
        self.used_records = 0
        self._open()

    def _open(self) -> None:
        exists = self.path.exists() and self.path.stat().st_size >= self.HEADER_SIZE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+b" if exists else "w+b"
        self._file = open(self.path, mode)
        expected_size = self.HEADER_SIZE + self.capacity_records * self.RECORD_SIZE
        if not exists:
            self._file.truncate(expected_size)
            self._file.seek(0)
            self._file.write(self.HEADER_STRUCT.pack(self.MAGIC, self.capacity_records, 0))
            self._file.flush()
        else:
            self._file.seek(0)
            header = self._file.read(self.HEADER_STRUCT.size)
            magic, stored_capacity, used = self.HEADER_STRUCT.unpack(header)
            if magic != self.MAGIC:
                raise ValueError(f"{self.path} is not a compatible Connect-4 TT file")
            if stored_capacity > self.capacity_records:
                self.capacity_records = stored_capacity
                expected_size = self.HEADER_SIZE + self.capacity_records * self.RECORD_SIZE
            current_size = self.path.stat().st_size
            if current_size < expected_size:
                self._file.truncate(expected_size)
            self.used_records = min(int(used), self.capacity_records)
        self._mmap = mmap.mmap(self._file.fileno(), 0)
        self._load_index()

    def _load_index(self) -> None:
        assert self._mmap is not None
        self._index.clear()
        magic, cap, used = self.HEADER_STRUCT.unpack_from(self._mmap, 0)
        if magic != self.MAGIC:
            raise ValueError("Corrupt transposition table header")
        self.used_records = min(int(used), int(cap))
        base = self.HEADER_SIZE
        rs = self.RECORD_SIZE
        unpack = self.RECORD_STRUCT.unpack_from
        for i in range(self.used_records):
            current, mask, _score, _depth, bound, _best, _reserved = unpack(self._mmap, base + i * rs)
            if bound != Bound.EMPTY:
                self._index[(current, mask)] = i

    def __len__(self) -> int:
        return self.used_records

    def close(self) -> None:
        if self._mmap is not None:
            self.flush()
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "PersistentTranspositionTable":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def flush(self) -> None:
        if self._mmap is not None:
            if self._header_dirty:
                self._write_header_used()
                self._header_dirty = False
            self._mmap.flush()
        if self._file is not None:
            self._file.flush()
        self._writes_since_flush = 0

    def _write_header_used(self) -> None:
        assert self._mmap is not None
        self.HEADER_STRUCT.pack_into(self._mmap, 0, self.MAGIC, self.capacity_records, self.used_records)

    def grow(self, min_extra: int = 1) -> None:
        """Grow the mmap file when capacity is exhausted."""
        if self._file is None:
            raise RuntimeError("Persistent table is closed")
        new_capacity = max(self.capacity_records * 2, self.capacity_records + min_extra)
        self.flush()
        if self._mmap is not None:
            self._mmap.close()
        new_size = self.HEADER_SIZE + new_capacity * self.RECORD_SIZE
        self._file.truncate(new_size)
        self.capacity_records = new_capacity
        self._mmap = mmap.mmap(self._file.fileno(), 0)
        self._write_header_used()

    def get(self, key: Tuple[int, int], min_depth: int) -> Optional[TTEntry]:
        idx = self._index.get(key)
        if idx is None:
            return None
        assert self._mmap is not None
        current, mask, score, depth, bound, best, _reserved = self.RECORD_STRUCT.unpack_from(
            self._mmap, self.HEADER_SIZE + idx * self.RECORD_SIZE
        )
        if (current, mask) != key or bound == Bound.EMPTY or depth < min_depth:
            return None
        return TTEntry(score=int(score), depth=int(depth), bound=Bound(bound), best_col=None if best < 0 else int(best))

    def put(self, key: Tuple[int, int], entry: TTEntry) -> None:
        if self._mmap is None:
            raise RuntimeError("Persistent table is closed")
        idx = self._index.get(key)
        if idx is None:
            if self.used_records >= self.capacity_records:
                self.grow(1)
            idx = self.used_records
            self.used_records += 1
            self._index[key] = idx
            self._header_dirty = True
        else:
            old = self.get(key, min_depth=-1)
            if old is not None and old.depth > entry.depth and old.bound == Bound.EXACT:
                return
        best = -1 if entry.best_col is None else int(entry.best_col)
        self.RECORD_STRUCT.pack_into(
            self._mmap,
            self.HEADER_SIZE + idx * self.RECORD_SIZE,
            int(key[0]),
            int(key[1]),
            int(entry.score),
            int(entry.depth),
            int(entry.bound),
            best,
            0,
        )
        self._writes_since_flush += 1
        if self._writes_since_flush >= self.flush_every:
            self.flush()

    def export_sorted_snapshot(self, output_path: str | os.PathLike[str]) -> None:
        """Write a sorted-by-key compact snapshot for binary-search-only consumers."""
        if self._mmap is None:
            raise RuntimeError("Persistent table is closed")
        out = Path(output_path)
        items = sorted(self._index.items(), key=lambda kv: kv[0])
        with open(out, "wb") as f:
            capacity = len(items)
            f.truncate(self.HEADER_SIZE + capacity * self.RECORD_SIZE)
            f.seek(0)
            f.write(self.HEADER_STRUCT.pack(self.MAGIC, capacity, capacity))
            for out_idx, (_key, old_idx) in enumerate(items):
                data = self._mmap[self.HEADER_SIZE + old_idx * self.RECORD_SIZE : self.HEADER_SIZE + (old_idx + 1) * self.RECORD_SIZE]
                f.seek(self.HEADER_SIZE + out_idx * self.RECORD_SIZE)
                f.write(data)



class TieredPersistentTranspositionTables:
    """Four mmap-backed TT files partitioned by move number/ply.

    Default tiers:
      1 early: turns 0..12
      2 mid:   turns 13..24
      3 late:  turns 25..34
      4 end:   turns 35..42

    The record schema is identical to PersistentTranspositionTable, so each tier
    is restartable by itself. Root checkpoints and learning hints remain separate.
    """

    NAMES = ("opening", "mid", "late", "end")

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        *,
        capacity_records: int = 1_000_000,
        cutoffs: Sequence[int] = DEFAULT_TIER_CUTOFFS,
        filenames: Sequence[str] = TIERED_TT_FILENAMES,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cutoffs = tuple(int(x) for x in cutoffs)
        if len(self.cutoffs) != 3:
            raise ValueError("tier cutoffs must contain exactly three integers: early_end, mid_end, late_end")
        if not (self.cutoffs[0] < self.cutoffs[1] < self.cutoffs[2]):
            raise ValueError("tier cutoffs must be strictly increasing")
        caps = self._split_capacity(max(4, int(capacity_records)))
        self.tables = [
            PersistentTranspositionTable(self.cache_dir / filenames[i], capacity_records=caps[i])
            for i in range(4)
        ]

    @staticmethod
    def _split_capacity(total: int) -> List[int]:
        # Bias capacity toward late/end tiers because observed work concentrates there.
        weights = [1, 2, 3, 4]
        base = max(25_000, total // 20)
        caps = [max(base, total * w // sum(weights)) for w in weights]
        return caps

    def tier_index_for_turn(self, turn: Optional[int]) -> int:
        t = 0 if turn is None else max(0, min(BOARD_SIZE, int(turn)))
        if t <= self.cutoffs[0]:
            return 0
        if t <= self.cutoffs[1]:
            return 1
        if t <= self.cutoffs[2]:
            return 2
        return 3

    def tier_name_for_turn(self, turn: Optional[int]) -> str:
        return self.NAMES[self.tier_index_for_turn(turn)]

    def get(self, key: Tuple[int, int], min_depth: int, *, turn: Optional[int] = None) -> Optional[TTEntry]:
        return self.tables[self.tier_index_for_turn(turn)].get(key, min_depth=min_depth)

    def put(self, key: Tuple[int, int], entry: TTEntry, *, turn: Optional[int] = None) -> int:
        idx = self.tier_index_for_turn(turn)
        self.tables[idx].put(key, entry)
        return idx

    def flush(self) -> None:
        for table in self.tables:
            table.flush()

    def close(self) -> None:
        for table in self.tables:
            table.close()

    def __len__(self) -> int:
        return sum(len(t) for t in self.tables)

    def rows_by_tier(self) -> List[int]:
        return [len(t) for t in self.tables]

    @property
    def path(self) -> Path:
        return self.cache_dir

    def export_sorted_snapshot(self, output_path: str | os.PathLike[str]) -> None:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        for name, table in zip(self.NAMES, self.tables):
            table.export_sorted_snapshot(out / f"{name}-sorted.bin")

class RootSearchCheckpoint:
    """Small JSON sidecar that remembers completed root moves for a specific solve.

    This is deliberately conservative: it only checkpoints completed root children, not
    the recursive call stack inside the current root. If a process is interrupted in the
    middle of root move 4, the next run still restarts that root, but any root moves that
    finished earlier can be skipped exactly.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.data: Dict[str, Any] = {"version": 1, "roots": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("roots"), dict):
                self.data = data
        except Exception:
            # Corrupt checkpoint should not stop solving; just start a fresh sidecar.
            self.data = {"version": 1, "roots": {}}

    def solve_key(self, pos: "Position", depth: int, *, use_symmetry: bool, move_order: Sequence[int]) -> str:
        current, mask = pos.key(use_symmetry)
        order = ",".join(str(c) for c in move_order)
        return f"current={current}:mask={mask}:moves={pos.moves}:depth={depth}:sym={int(use_symmetry)}:order={order}"

    def completed(self, solve_key: str) -> Dict[int, Dict[str, Any]]:
        raw = self.data.setdefault("roots", {}).setdefault(solve_key, {})
        out: Dict[int, Dict[str, Any]] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    col = int(k)
                except Exception:
                    continue
                if isinstance(v, dict) and "score" in v:
                    out[col] = v
        return out

    def record(self, solve_key: str, col: int, *, score: int, states: int, elapsed_s: float) -> None:
        roots = self.data.setdefault("roots", {}).setdefault(solve_key, {})
        roots[str(int(col))] = {
            "score": int(score),
            "states": int(max(0, states)),
            "elapsed_s": float(max(0.0, elapsed_s)),
            "saved_at": time.time(),
        }
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)


class SearchLearningCache:
    """Small, safe-to-delete JSON cache for move-ordering hints.

    This does not prove anything and does not affect correctness. It only preserves
    killer/history ordering information between runs so alpha-beta can find cutoffs
    earlier after a restart. The big binary TT remains the correctness/reuse cache.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.history_scores: List[int] = [0 for _ in range(WIDTH)]
        self.killer_moves: List[List[Optional[int]]] = [[None, None] for _ in range(BOARD_SIZE + 1)]
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or int(data.get("version", 1)) not in (1, 2):
                return
            hist = data.get("history_scores")
            if isinstance(hist, list) and len(hist) == WIDTH:
                self.history_scores = [int(max(0, x)) for x in hist]
            killers = data.get("killer_moves")
            if isinstance(killers, list):
                cleaned: List[List[Optional[int]]] = []
                for row in killers[: BOARD_SIZE + 1]:
                    pair: List[Optional[int]] = [None, None]
                    if isinstance(row, list):
                        for i, val in enumerate(row[:2]):
                            if isinstance(val, int) and 0 <= val < WIDTH:
                                pair[i] = val
                    cleaned.append(pair)
                while len(cleaned) < BOARD_SIZE + 1:
                    cleaned.append([None, None])
                self.killer_moves = cleaned
        except Exception:
            # Search hints are disposable; corrupt learning data should not stop solving.
            self.history_scores = [0 for _ in range(WIDTH)]
            self.killer_moves = [[None, None] for _ in range(BOARD_SIZE + 1)]

    def apply_to(self, solver: "Connect4Solver") -> None:
        solver.history_scores = list(self.history_scores)
        solver.killer_moves = [list(row) for row in self.killer_moves]

    def update_from(self, solver: "Connect4Solver") -> None:
        self.history_scores = [int(max(0, x)) for x in solver.history_scores]
        self.killer_moves = [[None if c is None else int(c) for c in row[:2]] for row in solver.killer_moves]

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        data = {
            "version": 2,
            "updated_at": time.time(),
            "note": "Move-ordering hints only; safe to delete. History is mirror-smoothed; killer moves are saved in board orientation.",
            "history_scores": self.history_scores,
            "killer_moves": self.killer_moves,
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)


class Connect4Solver:
    def __init__(
        self,
        use_symmetry: bool = True,
        move_order: Sequence[int] = DEFAULT_ORDER,
        persistent_tt: Optional[PersistentTranspositionTable] = None,
        endgame_tt: Optional[PersistentTranspositionTable] = None,
        endgame_min_turn: int = 30,
        memory_tt_limit: int = 500_000,
        progress: Optional[ConsoleProgress] = None,
        root_checkpoint: Optional[RootSearchCheckpoint] = None,
        persistent_min_depth: int = 8,
        persistent_exact_only: bool = False,
        learning_cache: Optional["SearchLearningCache"] = None,
        learning_flush_interval_states: int = 1_000_000,
    ) -> None:
        self.tt: Dict[Tuple[int, int], TTEntry] = {}
        self.use_symmetry = use_symmetry
        self.move_order = tuple(move_order)
        self.stats = SearchStats()
        self.persistent_tt = persistent_tt
        self.endgame_tt = endgame_tt
        self.endgame_min_turn = max(0, int(endgame_min_turn))
        self.memory_tt_limit = max(0, int(memory_tt_limit))
        self.progress = progress
        self.root_checkpoint = root_checkpoint
        self.persistent_min_depth = max(0, int(persistent_min_depth))
        self.persistent_exact_only = bool(persistent_exact_only)
        self.learning_cache = learning_cache
        self.learning_flush_interval_states = max(0, int(learning_flush_interval_states))
        self._last_learning_flush_states = 0
        self.killer_moves: List[List[Optional[int]]] = [[None, None] for _ in range(BOARD_SIZE + 1)]
        self.history_scores: List[int] = [0 for _ in range(WIDTH)]

    def clear(self, *, clear_memory: bool = True) -> None:
        if clear_memory:
            self.tt.clear()

    @staticmethod
    def theoretical_max_score(moves: int) -> int:
        return (BOARD_SIZE + 1 - moves) // 2

    @staticmethod
    def theoretical_min_score(moves: int) -> int:
        return -((BOARD_SIZE - moves) // 2)

    def solve(self, position: Optional[Position] = None, *, time_limit_s: Optional[float] = None) -> SolveResult:
        """Exact root solve with optional completed-root checkpointing.

        The recursive search still uses alpha-beta and the TT. This wrapper exists so a
        long solve can resume completed root moves after Ctrl-C/restart. It cannot resume
        the middle of the current root move; that would require a full stack checkpoint.
        """
        pos = position.copy() if position is not None else Position()
        root_depth = BOARD_SIZE - pos.moves
        root_moves = list(pos.legal_moves(self.move_order))
        self.stats = SearchStats(
            start_time=time.perf_counter(),
            time_limit_s=time_limit_s,
            root_depth=root_depth,
            root_total_moves=len(root_moves),
            depth_iteration=root_depth,
            max_depth_iteration=root_depth,
        )
        if self.persistent_tt is not None:
            self.stats.initial_persistent_rows = len(self.persistent_tt)
            if hasattr(self.persistent_tt, "rows_by_tier"):
                self.stats.initial_tier_rows = list(self.persistent_tt.rows_by_tier())

        checkpoint_key: Optional[str] = None
        completed_from_checkpoint: Dict[int, Dict[str, Any]] = {}
        if self.root_checkpoint is not None:
            checkpoint_key = self.root_checkpoint.solve_key(pos, root_depth, use_symmetry=self.use_symmetry, move_order=self.move_order)
            completed_from_checkpoint = self.root_checkpoint.completed(checkpoint_key)

        if self.progress is not None:
            self.progress.maybe_render(self, pos, force=True, phase="exact solve")

        alpha = self.theoretical_min_score(pos.moves)
        beta = self.theoretical_max_score(pos.moves)
        best_score = self.theoretical_min_score(pos.moves)
        best_move: Optional[int] = None
        exact = True

        try:
            for col in root_moves:
                # Reuse a previously completed root move. This is stronger than TT-only
                # restart behavior because it skips the whole root child.
                saved = completed_from_checkpoint.get(col)
                if saved is not None:
                    score = int(saved["score"])
                    self.stats.root_started_moves += 1
                    self.stats.root_completed_moves += 1
                    self.stats.root_move_state_costs.append(int(saved.get("states", 0)))
                    if score > best_score:
                        best_score, best_move = score, col
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        break
                    continue

                self.stats.root_started_moves += 1
                self.stats.current_root_move = col
                self.stats.current_root_start_states = self.stats.states_visited
                root_start_time = time.perf_counter()

                child = pos.copy()
                child.play_col(col)
                score, _ = self._negamax(child, -beta, -alpha, root_depth - 1)
                score = -score

                root_cost = max(0, self.stats.states_visited - self.stats.current_root_start_states)
                self.stats.root_completed_moves += 1
                self.stats.root_move_state_costs.append(root_cost)
                self.stats.current_root_move = None
                self.stats.current_root_start_states = 0

                if checkpoint_key is not None and self.root_checkpoint is not None:
                    self.root_checkpoint.record(
                        checkpoint_key,
                        col,
                        score=int(score),
                        states=int(root_cost),
                        elapsed_s=time.perf_counter() - root_start_time,
                    )

                if score > best_score:
                    best_score, best_move = score, col
                alpha = max(alpha, score)
                if self.progress is not None:
                    self.progress.maybe_render(self, pos, force=True, phase="exact solve")
                if alpha >= beta:
                    self.stats.cutoffs += 1
                    break
        except TimeoutError:
            exact = False
            if best_move is None:
                fallback = self.best_move_limited(pos, max_depth=8, time_limit_s=None)
                best_score, best_move = fallback.score, fallback.best_move

        elapsed = time.perf_counter() - self.stats.start_time
        if self.persistent_tt is not None:
            self.persistent_tt.flush()
        self._maybe_flush_learning(force=True)
        if self.progress is not None:
            self.progress.maybe_render(self, pos, force=True, phase="exact solve")
            self.progress.finish()
        return SolveResult(int(best_score), best_move, exact, self.stats.states_visited, self.stats.searched_children, self.stats.cutoffs, self.stats.tt_hits, self.stats.persistent_hits, self.stats.persistent_writes, elapsed)

    def best_move_limited(self, position: Position, *, max_depth: int = 12, time_limit_s: Optional[float] = None) -> SolveResult:
        start = time.perf_counter()
        deadline = None if time_limit_s is None else start + time_limit_s
        best_score = -math.inf
        best_move: Optional[int] = None
        exact = False
        total_states = 0
        total_child_searches = 0
        total_prunes = 0
        total_tt_hits = 0
        total_phits = 0
        total_writes = 0
        for depth in range(1, max_depth + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            limit = None if deadline is None else max(0.0, deadline - time.perf_counter())
            self.stats = SearchStats(
                start_time=time.perf_counter(),
                time_limit_s=limit,
                root_depth=depth,
                root_total_moves=len(position.legal_moves(self.move_order)),
                depth_iteration=depth,
                max_depth_iteration=max_depth,
            )
            if self.persistent_tt is not None:
                self.stats.initial_persistent_rows = len(self.persistent_tt)
            if hasattr(self.persistent_tt, "rows_by_tier"):
                self.stats.initial_tier_rows = list(self.persistent_tt.rows_by_tier())
            if self.progress is not None:
                self.progress.maybe_render(self, position, force=True, phase="iterative search")
            try:
                score, move = self._negamax(position.copy(), -10_000, 10_000, depth)
                best_score, best_move = score, move
                exact = depth >= BOARD_SIZE - position.moves
            except TimeoutError:
                pass
            if self.persistent_tt is not None:
                self.stats.initial_persistent_rows = len(self.persistent_tt)
            if hasattr(self.persistent_tt, "rows_by_tier"):
                self.stats.initial_tier_rows = list(self.persistent_tt.rows_by_tier())
            if self.progress is not None:
                self.progress.maybe_render(self, position, force=True, phase="iterative search")
            total_states += self.stats.states_visited
            total_child_searches += self.stats.searched_children
            total_prunes += self.stats.cutoffs
            total_tt_hits += self.stats.tt_hits
            total_phits += self.stats.persistent_hits
            total_writes += self.stats.persistent_writes
            if deadline is not None and time.perf_counter() >= deadline:
                break
        elapsed = time.perf_counter() - start
        if self.persistent_tt is not None:
            self.persistent_tt.flush()
        self._maybe_flush_learning(force=True)
        if self.progress is not None:
            self.progress.finish()
        return SolveResult(
            int(best_score if best_score != -math.inf else 0),
            best_move,
            exact,
            total_states,
            total_child_searches,
            total_prunes,
            total_tt_hits,
            total_phits,
            total_writes,
            elapsed,
        )

    def _orient_entry(self, entry: TTEntry, mirrored: bool) -> TTEntry:
        if not mirrored or entry.best_col is None:
            return entry
        self.stats.tt_best_move_mirrored += 1
        return TTEntry(entry.score, entry.depth, entry.bound, mirror_col(entry.best_col))

    def _lookup_tt(self, key: Tuple[int, int], depth: int, *, turn: int, mirrored: bool = False) -> Optional[TTEntry]:
        entry = self.tt.get(key)
        if entry is not None and entry.depth >= depth:
            self.stats.tt_hits += 1
            if 0 <= turn <= BOARD_SIZE:
                self.stats.tt_hits_by_turn[turn] += 1
            return self._orient_entry(entry, mirrored)

        # Endgame cache is checked first for deep played positions. Those positions
        # dominate the histogram and tend to be exact/high-reuse.
        if self.endgame_tt is not None and turn >= self.endgame_min_turn:
            entry = self.endgame_tt.get(key, min_depth=depth)
            if entry is not None:
                self.stats.tt_hits += 1
                self.stats.persistent_hits += 1
                self.stats.endgame_hits += 1
                if 0 <= turn <= BOARD_SIZE:
                    self.stats.tt_hits_by_turn[turn] += 1
                if self.memory_tt_limit != 0 and len(self.tt) < self.memory_tt_limit:
                    self.tt[key] = entry
                return self._orient_entry(entry, mirrored)

        if self.persistent_tt is not None:
            entry = self.persistent_tt.get(key, min_depth=depth, turn=turn) if hasattr(self.persistent_tt, "tier_index_for_turn") else self.persistent_tt.get(key, min_depth=depth)
            if entry is not None:
                self.stats.tt_hits += 1
                self.stats.persistent_hits += 1
                if hasattr(self.persistent_tt, "tier_index_for_turn"):
                    self.stats.persistent_reads_by_tier[self.persistent_tt.tier_index_for_turn(turn)] += 1
                if 0 <= turn <= BOARD_SIZE:
                    self.stats.tt_hits_by_turn[turn] += 1
                if self.memory_tt_limit != 0 and len(self.tt) < self.memory_tt_limit:
                    self.tt[key] = entry
                return self._orient_entry(entry, mirrored)
        return None

    def _store_tt(self, key: Tuple[int, int], entry: TTEntry, *, turn: Optional[int] = None, mirrored: bool = False) -> None:
        # Store best_move in canonical-key orientation so mirrored lookups can flip it back.
        canonical_entry = entry
        if mirrored and entry.best_col is not None:
            canonical_entry = TTEntry(entry.score, entry.depth, entry.bound, mirror_col(entry.best_col))
        if turn is not None and 0 <= turn <= BOARD_SIZE:
            self.stats.tt_stores_by_turn[turn] += 1
        if self.memory_tt_limit != 0:
            if len(self.tt) >= self.memory_tt_limit:
                # Cheap bounded-memory policy: clear shallow Python cache, keep disk cache.
                self.tt.clear()
            self.tt[key] = canonical_entry
        self.stats.tt_stores += 1

        wrote_endgame = False
        if self.endgame_tt is not None and turn is not None and turn >= self.endgame_min_turn and canonical_entry.bound == Bound.EXACT:
            self.endgame_tt.put(key, canonical_entry)
            self.stats.endgame_writes += 1
            wrote_endgame = True

        if self.persistent_tt is not None:
            if canonical_entry.depth < self.persistent_min_depth:
                self.stats.persistent_skipped_low_depth += 1
            elif self.persistent_exact_only and canonical_entry.bound != Bound.EXACT:
                self.stats.persistent_skipped_bound += 1
            else:
                tier_written = self.persistent_tt.put(key, canonical_entry, turn=turn) if hasattr(self.persistent_tt, "tier_index_for_turn") else (self.persistent_tt.put(key, canonical_entry) or None)
                self.stats.persistent_writes += 1
                if tier_written is not None:
                    self.stats.persistent_writes_by_tier[int(tier_written)] += 1

    def _move_center_distance(self, col: int) -> int:
        return abs(col - 3)

    def _opponent_can_win_next(self, pos_after_our_move: Position) -> bool:
        return any(pos_after_our_move.can_play(c) and pos_after_our_move.winning_move(c) for c in range(WIDTH))

    def _maybe_flush_learning(self, *, force: bool = False) -> None:
        if self.learning_cache is None:
            return
        if not force and self.learning_flush_interval_states > 0:
            if self.stats.states_visited - self._last_learning_flush_states < self.learning_flush_interval_states:
                return
        self.learning_cache.update_from(self)
        self.learning_cache.flush()
        self._last_learning_flush_states = self.stats.states_visited

    def _note_cutoff_move(self, col: int, depth: int) -> None:
        idx = max(0, min(BOARD_SIZE, int(depth)))
        killers = self.killer_moves[idx]
        if killers[0] != col:
            killers[1] = killers[0]
            killers[0] = col
        bonus = max(1, depth * depth)
        self.history_scores[col] += bonus
        # Mirror smoothing: because the TT canonicalizes mirrored boards, a cutoff
        # learned on one wing is usually useful on the opposite wing too.  The
        # smaller mirrored bonus preserves local evidence while respecting symmetry.
        mcol = mirror_col(col)
        if mcol is not None and mcol != col:
            self.history_scores[mcol] += max(1, bonus // 2)
        self._maybe_flush_learning()

    def _ordered_moves(self, pos: Position, tt_move: Optional[int], depth: int) -> Iterable[int]:
        """Move ordering only; correctness does not depend on this order.

        Priority:
          1. Immediate winning moves.
          2. TT best move from a previous search/cache entry.
          3. Killer moves that caused cutoffs at this remaining depth.
          4. Remaining safe moves sorted by history score, then center distance.
          5. Unsafe moves that allow an immediate opponent win.
        """
        yielded: set[int] = set()
        legal = [c for c in self.move_order if pos.can_play(c)]

        wins = [c for c in legal if pos.winning_move(c)]
        for c in wins:
            yielded.add(c)
            yield c

        if tt_move is not None and tt_move not in yielded and pos.can_play(tt_move):
            yielded.add(tt_move)
            yield tt_move

        killers = self.killer_moves[max(0, min(BOARD_SIZE, int(depth)))]
        killer_candidates: List[Optional[int]] = []
        for c in killers:
            killer_candidates.append(c)
            killer_candidates.append(mirror_col(c))
        for c in killer_candidates:
            if c is not None and c not in yielded and pos.can_play(c):
                self.stats.killer_uses += 1
                yielded.add(c)
                yield c

        safe: List[int] = []
        unsafe: List[int] = []
        for c in legal:
            if c in yielded:
                continue
            child = pos.copy()
            child.play_col(c)
            (unsafe if self._opponent_can_win_next(child) else safe).append(c)

        def key(c: int) -> Tuple[int, int]:
            return (-self.history_scores[c], self._move_center_distance(c))

        safe.sort(key=key)
        unsafe.sort(key=key)
        for c in safe + unsafe:
            if self.history_scores[c] > 0:
                self.stats.history_uses += 1
            yield c

    def _negamax(self, pos: Position, alpha: int, beta: int, depth: int) -> Tuple[int, Optional[int]]:
        if self.stats.timed_out():
            raise TimeoutError
        self.stats.states_visited += 1
        if 0 <= pos.moves <= BOARD_SIZE:
            self.stats.states_by_turn[pos.moves] += 1
        self.stats.max_ply_reached = max(self.stats.max_ply_reached, self.stats.root_depth - depth)

        if pos.moves == BOARD_SIZE:
            return 0, None

        for c in self.move_order:
            if pos.can_play(c) and pos.winning_move(c):
                return self.theoretical_max_score(pos.moves), c

        if depth <= 0:
            return self._static_eval(pos), None

        original_alpha = alpha
        original_beta = beta
        key, mirrored_key = canonical_key_info(pos.current, pos.mask, self.use_symmetry)
        if mirrored_key:
            self.stats.mirror_canonicalized += 1
        entry = self._lookup_tt(key, depth, turn=pos.moves, mirrored=mirrored_key)
        tt_move = None
        if entry is not None:
            tt_move = entry.best_col
            if entry.bound == Bound.EXACT:
                return entry.score, entry.best_col
            if entry.bound == Bound.LOWER:
                alpha = max(alpha, entry.score)
            elif entry.bound == Bound.UPPER:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score, entry.best_col

        beta = min(beta, self.theoretical_max_score(pos.moves))
        if alpha >= beta:
            return beta, None

        best_score = self.theoretical_min_score(pos.moves)
        best_col: Optional[int] = None
        moved = False

        ordered_moves = list(self._ordered_moves(pos, tt_move, depth))
        if ordered_moves:
            self.stats.branching_positions += 1
            self.stats.legal_child_candidates += len(ordered_moves)
            if 0 <= pos.moves <= BOARD_SIZE:
                self.stats.candidates_by_turn[pos.moves] += len(ordered_moves)
        for idx, col in enumerate(ordered_moves):
            moved = True
            self.stats.searched_children += 1
            root_child_start_states = self.stats.states_visited
            if depth == self.stats.root_depth:
                self.stats.root_started_moves += 1
                self.stats.current_root_move = col
                self.stats.current_root_start_states = root_child_start_states
            child = pos.copy()
            child.play_col(col)
            score, _ = self._negamax(child, -beta, -alpha, depth - 1)
            score = -score
            if depth == self.stats.root_depth:
                self.stats.root_completed_moves += 1
                self.stats.root_move_state_costs.append(max(0, self.stats.states_visited - root_child_start_states))
                self.stats.current_root_move = None
                self.stats.current_root_start_states = 0
            if self.progress is not None:
                self.progress.maybe_render(self, pos)
            if score > best_score:
                best_score, best_col = score, col
            alpha = max(alpha, score)
            if alpha >= beta:
                self.stats.cutoffs += 1
                self.stats.pruned_children_est += max(0, len(ordered_moves) - idx - 1)
                self._note_cutoff_move(col, depth)
                break

        if not moved:
            return 0, None

        if best_score <= original_alpha:
            bound = Bound.UPPER
        elif best_score >= original_beta:
            bound = Bound.LOWER
        else:
            bound = Bound.EXACT
        self._store_tt(key, TTEntry(int(best_score), int(depth), bound, best_col), turn=pos.moves, mirrored=mirrored_key)
        return best_score, best_col

    def _static_eval(self, pos: Position) -> int:
        score = 0
        center_mask = COLUMN_MASKS[3]
        score += 3 * (pos.current & center_mask).bit_count()
        score -= 3 * ((pos.mask ^ pos.current) & center_mask).bit_count()
        for c in range(WIDTH):
            if pos.can_play(c) and pos.winning_move(c):
                score += 50
            child = pos.copy()
            if child.can_play(c):
                child.play_col(c)
                for oc in range(WIDTH):
                    if child.can_play(oc) and child.winning_move(oc):
                        score -= 20
                        break
        return int(score)

    def best_moves(self, position: Position, root_perspective: bool = True) -> List[Tuple[int, int]]:
        """Return legal moves scored by a one-ply solve call from side-to-move perspective."""
        out: List[Tuple[int, int]] = []
        alpha = self.theoretical_min_score(position.moves)
        beta = self.theoretical_max_score(position.moves)
        for c in position.legal_moves(self.move_order):
            child = position.copy()
            child.play_col(c)
            score, _ = self._negamax(child, -beta, -alpha, BOARD_SIZE - child.moves)
            out.append((c, -score))
        return sorted(out, key=lambda x: x[1], reverse=True)

    def extract_weak_solution(self, position: Position, *, max_nodes: int = 1000, score_depth: int = 8) -> Dict[str, Any]:
        """Build a compact weak-solution style graph from the current search policy.

        At strategy turns, it keeps one best move; at opponent turns, it keeps all replies.
        This is bounded by max_nodes so it remains an inspectable artifact rather than a
        second full-state solver.
        """
        seen: set[Tuple[int, int]] = set()
        count = 0

        def rec(pos: Position, strategy_turn: bool) -> Dict[str, Any]:
            nonlocal count
            count += 1
            if count > max_nodes:
                return {"type": "truncated"}
            if pos.moves == BOARD_SIZE:
                return {"type": "terminal", "value": 0}
            for c in self.move_order:
                if pos.can_play(c) and pos.winning_move(c):
                    return {"type": "terminal", "value": 1 if strategy_turn else -1, "winning_move": c}
            key = pos.key(self.use_symmetry)
            if key in seen:
                return {"type": "transpose", "key": f"{key[0]}:{key[1]}"}
            seen.add(key)
            moves = pos.legal_moves(self.move_order)
            if strategy_turn:
                scores = []
                for c in moves:
                    child = pos.copy(); child.play_col(c)
                    try:
                        score, _ = self._negamax(child, -10_000, 10_000, min(score_depth, BOARD_SIZE - child.moves))
                        score = -score
                    except TimeoutError:
                        score = self._static_eval(child)
                    scores.append((score, c))
                _score, chosen = max(scores)
                child = pos.copy(); child.play_col(chosen)
                return {"type": "choice", "move": chosen, "child": rec(child, False)}
            children = {}
            for c in moves:
                child = pos.copy(); child.play_col(c)
                children[str(c)] = rec(child, True)
            return {"type": "opponent", "children": children}

        return rec(position.copy(), True)


def parse_columns(text: str) -> List[int]:
    """Accept 0-based strings containing 0, or human 1-based strings otherwise."""
    digits = [int(ch) for ch in text if ch.isdigit()]
    if not digits:
        return []
    if any(d == 7 for d in digits) or not any(d == 0 for d in digits):
        return [d - 1 for d in digits]
    return digits


def main(
    moves: str = typer.Argument("4", help="Played columns, e.g. 4443 using human 1-based columns."),
    time_limit: Optional[float] = typer.Option(None, "--time", help="Optional seconds before returning best completed search."),
    depth: Optional[int] = typer.Option(None, "--depth", help="Use depth-limited iterative deepening instead of exact proof."),
    no_symmetry: bool = typer.Option(False, "--no-symmetry", help="Disable mirror canonicalization."),
    cache: Optional[Path] = typer.Option(None, "--cache", help="Legacy single-file TT path. If omitted, uses four tiered TT files in <cache-dir>."),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir", help="Directory for TT, root checkpoint, and learning caches."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the persistent mmap transposition table."),
    cache_capacity: int = typer.Option(1_000_000, "--cache-capacity", help="Initial persistent record capacity."),
    tier_cutoffs: str = typer.Option("10,22,31", "--tier-cutoffs", help="Move-number cutoffs for opening,mid,late tiers; end is above last cutoff."),
    root_checkpoint: Optional[Path] = typer.Option(None, "--root-checkpoint", help="JSON sidecar for completed root moves; default is <cache-dir>/root-checkpoints.json."),
    learning_cache: Optional[Path] = typer.Option(None, "--learning-cache", help="JSON cache for killer/history move-ordering hints; default is <cache-dir>/search-learning.json."),
    no_learning_cache: bool = typer.Option(False, "--no-learning-cache", help="Disable persisted move-ordering hints."),
    learning_flush_interval_states: int = typer.Option(1_000_000, "--learning-flush-interval-states", help="Autosave killer/history hints after about this many states."),
    no_root_checkpoint: bool = typer.Option(False, "--no-root-checkpoint", help="Disable completed-root checkpointing."),
    memory_tt_limit: int = typer.Option(750_000, "--memory-tt-limit", help="Max Python TT entries; 0 disables RAM TT."),
    disk_min_depth: int = typer.Option(8, "--disk-min-depth", help="Only persist TT entries with at least this remaining depth; RAM TT still stores all depths."),
    disk_exact_only: bool = typer.Option(False, "--disk-exact-only", help="Persist only exact TT entries; lowers disk writes but reduces cross-run reuse."),
    quiet: bool = typer.Option(False, "--quiet", help="Disable live progress display."),
    progress_interval_states: int = typer.Option(25_000, "--progress-interval-states", help="Refresh progress after about this many new states."),
    performance_snapshot: Optional[Path] = typer.Option(None, "--performance-snapshot", help="Append periodic diagnostic snapshots; default is <cache-dir>/performance-state-snapshot.log."),
    no_performance_snapshot: bool = typer.Option(False, "--no-performance-snapshot", help="Disable periodic performance-state-snapshot.log writes."),
    snapshot_interval_states: int = typer.Option(1_000_000, "--snapshot-interval-states", help="Append a performance snapshot after about this many states."),
    snapshot_interval_seconds: float = typer.Option(60.0, "--snapshot-interval-seconds", help="Append a performance snapshot after about this many seconds."),
    export_sorted_cache: Optional[Path] = typer.Option(None, "--export-sorted-cache", help="Write a sorted compact snapshot of the cache and exit."),
    weak_solution_json: Optional[Path] = typer.Option(None, "--weak-solution-json", help="Write a bounded weak-solution graph after solving."),
    weak_max_nodes: int = typer.Option(1000, "--weak-max-nodes"),
) -> None:
    """Solve or depth-search a Connect-4 position.

    Make executable with `chmod +x z-1.py`, then run for example:

        ./z-1.py 4443 --depth 12
        ./z-1.py 4443 --depth 12 --no-cache
        python z-1.py 4443 --depth 12

    Do not run this as `typer z-1.py`; that command is Typer's developer runner,
    not the normal way to execute this script.
    """
    persistent: Optional[PersistentTranspositionTable] = None
    endgame_persistent: Optional[PersistentTranspositionTable] = None  # unused; kept for compatibility
    learning_obj: Optional[SearchLearningCache] = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tt_path = cache
        roots_path = root_checkpoint if root_checkpoint is not None else cache_dir / ROOTS_FILENAME
        learning_path = learning_cache if learning_cache is not None else cache_dir / LEARNING_FILENAME
        snapshot_path = performance_snapshot if performance_snapshot is not None else cache_dir / PERFORMANCE_SNAPSHOT_FILENAME
        if no_performance_snapshot:
            snapshot_path = None

        if not no_cache:
            cutoffs = tuple(int(x.strip()) for x in tier_cutoffs.split(",") if x.strip())
            if tt_path is not None:
                persistent = PersistentTranspositionTable(tt_path, capacity_records=cache_capacity)
            else:
                persistent = TieredPersistentTranspositionTables(cache_dir, capacity_records=cache_capacity, cutoffs=cutoffs)
            if export_sorted_cache:
                persistent.export_sorted_snapshot(export_sorted_cache)
                typer.echo(f"exported_sorted_cache: {export_sorted_cache}")
                return

        parsed_moves = parse_columns(moves)
        pos = Position.from_columns(parsed_moves)
        checkpoint_obj: Optional[RootSearchCheckpoint] = None
        if not no_root_checkpoint:
            checkpoint_obj = RootSearchCheckpoint(roots_path)
        if not no_learning_cache:
            learning_obj = SearchLearningCache(learning_path)
        progress = ConsoleProgress(
            enabled=not quiet,
            interval_states=progress_interval_states,
            snapshot_path=snapshot_path,
            snapshot_interval_states=snapshot_interval_states,
            snapshot_interval_seconds=snapshot_interval_seconds,
        )
        solver = Connect4Solver(
            use_symmetry=not no_symmetry,
            persistent_tt=persistent,
            endgame_tt=None,
            endgame_min_turn=BOARD_SIZE + 1,
            memory_tt_limit=memory_tt_limit,
            progress=progress,
            root_checkpoint=checkpoint_obj,
            persistent_min_depth=disk_min_depth,
            persistent_exact_only=disk_exact_only,
            learning_cache=learning_obj,
            learning_flush_interval_states=learning_flush_interval_states,
        )
        if learning_obj is not None:
            learning_obj.apply_to(solver)

        result = solver.best_move_limited(pos, max_depth=depth, time_limit_s=time_limit) if depth else solver.solve(pos, time_limit_s=time_limit)

        if learning_obj is not None:
            learning_obj.update_from(solver)
            learning_obj.flush()

        typer.echo(pos.render())
        typer.echo()
        typer.echo(f"outcome: {result.outcome}")
        typer.echo(f"score: {result.score}")
        typer.echo(f"best_move_0_based: {result.best_move}")
        typer.echo(f"best_move_1_based: {None if result.best_move is None else result.best_move + 1}")
        typer.echo(f"solved_exactly: {result.solved_exactly}")
        typer.echo(f"states_visited: {result.nodes}")
        typer.echo(f"child_searches: {result.edges}")
        typer.echo(f"alpha_beta_cutoffs: {result.prunes}")
        typer.echo(f"tt_hits: {result.transposition_hits}")
        typer.echo(f"persistent_hits: {result.persistent_hits}")
        typer.echo(f"persistent_writes: {result.persistent_writes}")
        typer.echo(f"persistent_skipped_low_depth: {solver.stats.persistent_skipped_low_depth}")
        typer.echo(f"persistent_skipped_bound: {solver.stats.persistent_skipped_bound}")
        typer.echo(f"killer_uses: {solver.stats.killer_uses}")
        typer.echo(f"history_uses: {solver.stats.history_uses}")
        typer.echo(f"mirror_canonicalized: {solver.stats.mirror_canonicalized}")
        typer.echo(f"tt_best_move_mirrored: {solver.stats.tt_best_move_mirrored}")
        typer.echo(f"tier_reads: {solver.stats.persistent_reads_by_tier}")
        typer.echo(f"tier_writes: {solver.stats.persistent_writes_by_tier}")
        if persistent is not None:
            typer.echo(f"persistent_records: {len(persistent)}")
            typer.echo(f"tt_cache: {persistent.path}")
            typer.echo("resume_mode: root-checkpoint for completed root moves + TT cache for internal states + learning hints")
        if checkpoint_obj is not None:
            typer.echo(f"root_checkpoint: {checkpoint_obj.path}")
        if persistent is not None and hasattr(persistent, "rows_by_tier"):
            typer.echo(f"tier_records: {persistent.rows_by_tier()}")
            typer.echo(f"tier_cache_dir: {persistent.path}")
        if learning_obj is not None:
            typer.echo(f"learning_cache: {learning_obj.path}")
        if not no_performance_snapshot:
            typer.echo(f"performance_snapshot: {snapshot_path}")
        typer.echo(f"elapsed_s: {result.elapsed_s:.3f}")

        if weak_solution_json:
            graph = solver.extract_weak_solution(pos, max_nodes=weak_max_nodes, score_depth=max(1, depth or 8))
            with open(weak_solution_json, "w", encoding="utf-8") as f:
                json.dump(graph, f, indent=2)
            typer.echo(f"weak_solution_json: {weak_solution_json}")
    finally:
        # tiered TT is closed through persistent.close()
        if persistent is not None:
            persistent.close()


if __name__ == "__main__":
    typer.run(main)
