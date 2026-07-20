# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory monitoring and enforcement for subprocess workers.

Two-tier approach:
- **Soft limit** (default 85% of cap): polling thread captures diagnostics
  (tracemalloc snapshot, GC stats, RSS history, stack traces) to a file.
- **Hard limit**: thread-based kill at 100% — writes error result to stdout
  via ``os.write()`` and calls ``os._exit(137)``.  Also attempts
  ``resource.setrlimit(RLIMIT_AS)`` as a secondary safety net (Linux only;
  macOS does not enforce it).
"""

from __future__ import annotations

import gc
import logging
import platform
import resource
import sys
import threading
import time
import traceback
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path


def enable_tracking() -> None:
    """Start tracemalloc (25 frames deep). Safe to call multiple times."""
    if not tracemalloc.is_tracing():
        tracemalloc.start(25)


def get_rss_mb() -> float:
    """Return current RSS in MB.

    Uses ``/proc/self/status`` on Linux (accurate current RSS).
    Falls back to ``resource.getrusage().ru_maxrss`` on macOS, which
    reports *peak* RSS — the value never decreases within a process.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except FileNotFoundError:
        pass
    # macOS: ru_maxrss is in bytes; Linux: kB
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def _get_vas_mb() -> float:
    """Return current virtual address space size in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except FileNotFoundError:
        pass
    # Fallback: estimate from RLIMIT_AS or use RSS * 3 as rough proxy
    rss = get_rss_mb()
    return rss * 3


def set_hard_limit(limit_mb: int) -> bool:
    """Set RLIMIT_AS soft limit (virtual address space) as a safety net.

    The cap is set to ``current_vas + limit_mb`` — i.e. the process keeps
    its existing virtual mappings (shared libs, Python heap, etc.) and is
    allowed to grow by *limit_mb* on top.  This avoids killing workers whose
    baseline VAS already exceeds a naïve ``limit_mb * N`` multiplier.

    Only the *soft* limit is lowered so ``clear_hard_limit()`` can raise it
    back without requiring root privileges.

    Returns True if the limit was set, False if it couldn't be applied
    (e.g. macOS where RLIMIT_AS is not enforced and the OS may reject the
    value).
    """
    current_vas = _get_vas_mb()
    vas_cap_bytes = int((current_vas + limit_mb) * 1024 * 1024)
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        # Clamp to OS hard ceiling (macOS maps RLIMIT_AS → RLIMIT_RSS and
        # may have a hard limit lower than our requested cap).
        if hard != resource.RLIM_INFINITY and vas_cap_bytes > hard:
            vas_cap_bytes = hard
        resource.setrlimit(resource.RLIMIT_AS, (vas_cap_bytes, hard))
        return True
    except (ValueError, OSError) as exc:
        logging.getLogger(__name__).debug("set_hard_limit skipped: %s", exc)
        return False


def clear_hard_limit() -> None:
    """Remove the RLIMIT_AS soft cap (reset to the hard ceiling)."""
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
    except (ValueError, OSError) as exc:
        logging.getLogger(__name__).debug("clear_hard_limit skipped: %s", exc)


class MemoryMonitor:
    """Daemon thread that polls RSS and captures diagnostics at the soft limit.

    Parameters
    ----------
    limit_mb:
        Memory cap in MB (matches the CLI ``--memory-limit`` value).
    trace_dir:
        Directory where the diagnostics file is written.
    sample_id:
        Unique sample identifier — used in the diagnostics filename.
    soft_pct:
        Fraction of *limit_mb* at which the soft limit fires (default 0.85).
    poll_interval:
        Seconds between RSS checks (default 2.0).
    """

    def __init__(
        self,
        limit_mb: int,
        trace_dir: str,
        sample_id: str,
        proto_out: object | None = None,
        task_meta: dict | None = None,
        soft_pct: float = 0.85,
        poll_interval: float = 2.0,
    ) -> None:
        self.limit_mb = limit_mb
        self.soft_limit_mb = limit_mb * soft_pct
        self.trace_dir = Path(trace_dir)
        self.sample_id = sample_id
        self.poll_interval = poll_interval
        self._proto_out = proto_out  # stdout buffer for writing JSON result before exit
        self._task_meta = task_meta or {}  # identity fields for error results

        self.peak_rss_mb: float = 0.0
        self.soft_limit_hit: bool = False
        self.diag_file: str | None = None

        self._rss_history: list[tuple[float, float]] = []  # (elapsed_s, rss_mb)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._poll, name="mem-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # --------------------------------------------------------------------- #
    # Polling loop
    # --------------------------------------------------------------------- #

    def _poll(self) -> None:
        _log = logging.getLogger(__name__)
        while not self._stop.wait(self.poll_interval):
            try:
                rss = get_rss_mb()
            except Exception:
                _log.debug("get_rss_mb() failed", exc_info=True)
                continue
            elapsed = time.monotonic() - self._start_time
            self._rss_history.append((elapsed, rss))
            self.peak_rss_mb = max(self.peak_rss_mb, rss)

            if rss >= self.soft_limit_mb and not self.soft_limit_hit:
                self.soft_limit_hit = True
                self._capture_diagnostics(rss)

            if rss >= self.limit_mb:
                self._hard_kill(rss)

    # --------------------------------------------------------------------- #
    # Diagnostics capture
    # --------------------------------------------------------------------- #

    def _capture_diagnostics(self, current_rss: float) -> None:
        """Write a diagnostics file with tracemalloc, GC, and stack info."""
        diag_path = self.trace_dir / f"{self.sample_id}_memory_diag.txt"
        lines: list[str] = []

        ts = datetime.now(UTC).isoformat()
        lines.append(f"MEMORY DIAGNOSTICS — {self.sample_id}")
        lines.append(f"Captured at: {ts}")
        lines.append(f"RSS at capture: {current_rss:.1f} MB")
        pct = self.soft_limit_mb / self.limit_mb * 100
        lines.append(f"Soft limit: {self.soft_limit_mb:.1f} MB ({pct:.0f}% of {self.limit_mb} MB)")
        lines.append(f"Peak RSS so far: {self.peak_rss_mb:.1f} MB")
        lines.append("")

        # -- tracemalloc top allocators ----------------------------------- #
        lines.append("=" * 60)
        lines.append("TOP 25 MEMORY ALLOCATORS (tracemalloc)")
        lines.append("=" * 60)
        try:
            snapshot = tracemalloc.take_snapshot()
            # Filter out tracemalloc/importlib internals
            snapshot = snapshot.filter_traces(
                [
                    tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                    tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
                    tracemalloc.Filter(False, tracemalloc.__file__),
                ]
            )
            top_stats = snapshot.statistics("lineno")
            for stat in top_stats[:25]:
                lines.append(f"  {stat}")
            lines.append("")
            lines.append("--- Detailed traces for top 10 ---")
            top_traces = snapshot.statistics("traceback")
            for stat in top_traces[:10]:
                lines.append(f"\n  {stat.size / (1024 * 1024):.2f} MB — {stat.count} blocks")
                for frame_line in stat.traceback.format():
                    lines.append(f"    {frame_line}")
        except MemoryError:
            lines.append("  [tracemalloc snapshot failed — MemoryError]")
        except Exception as exc:
            lines.append(f"  [tracemalloc snapshot failed — {exc}]")
        lines.append("")

        # -- GC object counts by type ------------------------------------- #
        lines.append("=" * 60)
        lines.append("TOP 25 OBJECT TYPES BY COUNT (gc)")
        lines.append("=" * 60)
        try:
            type_counts: dict[str, int] = {}
            for obj in gc.get_objects():
                t = type(obj).__qualname__
                type_counts[t] = type_counts.get(t, 0) + 1
            for name, count in sorted(type_counts.items(), key=lambda x: -x[1])[:25]:
                lines.append(f"  {name}: {count:,}")
        except MemoryError:
            lines.append("  [gc.get_objects() failed — MemoryError]")
        except Exception as exc:
            lines.append(f"  [gc.get_objects() failed — {exc}]")
        lines.append("")

        # -- GC generation stats ------------------------------------------ #
        lines.append("=" * 60)
        lines.append("GC STATS")
        lines.append("=" * 60)
        try:
            for i, stat in enumerate(gc.get_stats()):
                lines.append(f"  Gen {i}: {stat}")
            lines.append(f"  Total tracked objects: {gc.get_count()}")
        except Exception as exc:
            lines.append(f"  [gc.get_stats() failed — {exc}]")
        lines.append("")

        # -- Thread stack traces ------------------------------------------ #
        lines.append("=" * 60)
        lines.append("THREAD STACK TRACES")
        lines.append("=" * 60)
        try:
            frames = sys._current_frames()
            for thread_id, frame in frames.items():
                # Find thread name
                thread_name = "unknown"
                for t in threading.enumerate():
                    if t.ident == thread_id:
                        thread_name = t.name
                        break
                lines.append(f"\n  Thread '{thread_name}' (id={thread_id}):")
                for line in traceback.format_stack(frame):
                    for sub in line.splitlines():
                        lines.append(f"    {sub}")
        except Exception as exc:
            lines.append(f"  [stack trace capture failed — {exc}]")
        lines.append("")

        # -- RSS history -------------------------------------------------- #
        lines.append("=" * 60)
        lines.append(f"RSS HISTORY (sampled every {self.poll_interval:.0f}s)")
        lines.append("=" * 60)
        for elapsed, rss in self._rss_history:
            lines.append(f"  +{elapsed:6.1f}s: {rss:8.1f} MB")
        lines.append("")

        # -- Write file --------------------------------------------------- #
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            diag_path.write_text("\n".join(lines))
            self.diag_file = str(diag_path)
        except Exception:
            # Last resort: write to stderr
            sys.stderr.write("\n".join(lines) + "\n")
            sys.stderr.flush()

    def _hard_kill(self, current_rss: float) -> None:
        """Kill the process when RSS exceeds the hard limit.

        The soft limit (85%) has already fired and written diagnostics.
        This appends a final note to the diagnostics file, flushes traces
        so the viewer has the full execution, writes an error result to
        stdout (so the parent gets a proper EvalTestResult), and terminates.
        Works on all platforms (macOS, Linux) — no RLIMIT_AS dependency.
        """
        import os

        # Ensure diagnostics were captured (in case RSS jumped past both thresholds
        # in the same poll interval)
        if not self.soft_limit_hit:
            self.soft_limit_hit = True
            self._capture_diagnostics(current_rss)

        # Append kill notice to diagnostics file
        if self.diag_file:
            try:
                with open(self.diag_file, "a") as f:
                    f.write(
                        f"\n*** HARD KILL at {current_rss:.1f} MB (limit: {self.limit_mb} MB) ***\n"
                    )
            except Exception:
                pass

        # Flush traces so the viewer has the full execution up to this point
        try:
            from nooa.tracing import flush_traces

            flush_traces()
        except Exception:
            pass

        # Write error result to stdout so the parent gets a proper
        # EvalTestResult instead of "Worker closed stdout unexpectedly".
        if self._proto_out:
            try:
                from .eval_types import EvalTestResult

                diag_name = Path(self.diag_file).name if self.diag_file else "N/A"
                m = self._task_meta
                err_result = EvalTestResult(
                    test_id=m.get("test_id", "unknown"),
                    base_test_id=m.get("base_test_id", "unknown"),
                    run_id=m.get("run_id", 1),
                    test_case=m.get("test_case", "unknown"),
                    agent_class=m.get("agent_class", "unknown"),
                    method=m.get("method", "unknown"),
                    test_name=m.get("test_name"),
                    display_name=m.get("display_name"),
                    model=m.get("model", "unknown"),
                    variant=m.get("variant", "run1"),
                    passed=False,
                    scores={},
                    input=None,
                    output=None,
                    expected=None,
                    trace_file=m.get("trace_file"),
                    error=(
                        f"MemoryError: RSS {current_rss:.1f} MB "
                        f"exceeds {self.limit_mb} MB limit. "
                        f"Diagnostics: {diag_name}"
                    ),
                    error_type="MemoryError",
                    memory_diag_file=self.diag_file,
                    peak_rss_mb=self.peak_rss_mb,
                )
                # Use os.write() directly on the file descriptor — this is a
                # single syscall with no Python buffering, so os._exit() can't
                # race with incomplete flushes.
                data = err_result.model_dump_json().encode() + b"\n"
                fd = self._proto_out.fileno()
                os.write(fd, data)
            except Exception:
                pass

        sys.stderr.write(
            f"\n[memory-monitor] KILLED: RSS {current_rss:.1f} MB "
            f"exceeds {self.limit_mb} MB limit. "
            f"Diagnostics: {self.diag_file or 'N/A'}\n"
        )
        sys.stderr.flush()

        # os._exit skips cleanup (atexit, finally blocks) — necessary because
        # the main thread may be blocked in an LLM call or allocation.
        os._exit(137)  # 128 + 9 (SIGKILL convention)
