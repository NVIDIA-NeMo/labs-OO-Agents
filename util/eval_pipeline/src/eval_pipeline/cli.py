# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for eval_pipeline.

Thin wrapper around the Evaluator Python API.

Usage:
    python -m eval_pipeline --config config.yaml
    python -m eval_pipeline --config config.yaml --runs 3 --parallel 10
    python -m eval_pipeline --config config.yaml --default_strategy codeact
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

# Valid strategy names for CLI/config
VALID_STRATEGIES = [
    "pure_python",
    "codeact",
    "codeact_lite",
    "reflexion",
    "predict",
    "structured_output",
]


class HangWatchdog:
    """Watchdog that sends SIGUSR2 when no progress is made for a specified duration.

    Uses a separate thread (not asyncio) to detect hangs even when the event loop
    is blocked by synchronous code like input() or time.sleep().

    Usage:
        watchdog = HangWatchdog(timeout=120)  # 2 minutes
        watchdog.start()
        try:
            for item in items:
                process(item)
                watchdog.ping()  # Reset the timer
        finally:
            watchdog.stop()
    """

    def __init__(self, timeout: float, quiet: bool = False):
        self.timeout = timeout
        self.quiet = quiet
        self._last_ping = time.monotonic()
        self._running = False
        self._thread: threading.Thread | None = None

    def ping(self):
        """Reset the watchdog timer. Call this on each progress update."""
        self._last_ping = time.monotonic()

    def _monitor(self):
        """Background thread that checks for hangs."""
        while self._running:
            time.sleep(1)  # Check every second (thread-based, not asyncio)
            if not self._running:
                break

            elapsed = time.monotonic() - self._last_ping
            if elapsed > self.timeout:
                # No progress for too long - send debug signal
                if not self.quiet:
                    sys.stderr.write(
                        f"\n[watchdog] No progress for {elapsed:.0f}s, sending SIGUSR2 for debug dump\n"
                    )
                    sys.stderr.flush()
                try:
                    os.kill(os.getpid(), signal.SIGUSR2)
                except (OSError, AttributeError):
                    pass  # Signal not available on this platform
                # Reset timer after sending signal (don't spam)
                self._last_ping = time.monotonic()

    def start(self):
        """Start the watchdog background thread."""
        if self._running:
            return
        self._running = True
        self._last_ping = time.monotonic()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the watchdog."""
        self._running = False
        # Thread is daemon, will exit when main program exits
        self._thread = None


def get_strategy_instance(strategy_name: str):
    """Create a strategy instance from a strategy name.

    Args:
        strategy_name: One of "pure_python", "codeact", "reflexion", "structured_output"

    Returns:
        GenerationStrategy instance

    Raises:
        ValueError: If strategy_name is not recognized
    """
    from nooa import (
        CodeActLiteStrategy,
        CodeActStrategy,
        PredictStrategy,
        ReflexionStrategy,
    )
    from nooa.strategies.pure_python import PurePythonStrategy

    strategies = {
        "pure_python": PurePythonStrategy,
        "codeact": CodeActStrategy,
        "codeact_lite": CodeActLiteStrategy,
        "reflexion": ReflexionStrategy,
        "predict": PredictStrategy,
        "structured_output": PredictStrategy,  # Backward-compatible alias
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. Valid options: {', '.join(VALID_STRATEGIES)}"
        )

    return strategies[strategy_name]()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run agent evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m eval_pipeline --config config.yaml
  python -m eval_pipeline --config config.yaml --runs 3 --parallel 10
  python -m eval_pipeline --config config.yaml --test sentiment --limit 5
  python -m eval_pipeline --config config.yaml --models gpt-4,claude-3 -q
  python -m eval_pipeline --config config.yaml --default_strategy codeact
        """,
    )
    parser.add_argument("--config", type=Path, required=True, help="Config file path")
    parser.add_argument("--output-dir", type=Path, help="Override output directory from config")
    parser.add_argument("--test", type=str, help="Run only these tests (comma-separated)")
    parser.add_argument("--models", type=str, help="Use only these models (comma-separated)")
    parser.add_argument("--runs", type=int, default=1, help="Runs per test (default: 1)")
    parser.add_argument("--limit", type=int, help="Limit samples per test")
    parser.add_argument(
        "--parallel", type=int, default=1, help="Max concurrent samples (default: 1)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout per sample in seconds (default: no timeout)",
    )
    parser.add_argument(
        "--engine",
        choices=["asyncio", "subprocess"],
        default="asyncio",
        help="Execution engine: asyncio (I/O-bound LLM APIs) or subprocess (CPU-bound local models)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Tasks per subprocess when using --engine subprocess (default: 1)",
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=None,
        metavar="MB",
        help="Memory limit per subprocess worker in MB (default: no limit). "
        "Captures diagnostics at 85%% and kills the worker if exceeded. "
        "Only effective with --engine subprocess.",
    )
    parser.add_argument(
        "--hang-timeout",
        type=float,
        default=None,
        help="Send SIGUSR2 debug dump if no progress for N seconds (default: disabled)",
    )
    parser.add_argument(
        "--default_strategy",
        type=str,
        choices=VALID_STRATEGIES,
        help=f"Default strategy for all agents ({', '.join(VALID_STRATEGIES)})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable NVIDIA inference API caching (needed for pass@k diversity)",
    )
    parser.add_argument(
        "--no-files",
        action="store_true",
        help="Suppress all file output (no experiment dir, no trace files, no results JSONL). "
        "Traces still go to OTLP if a viewer is running.",
    )
    parser.add_argument(
        "--trace-files",
        action="store_true",
        help="Write JSONL trace files even when a viewer is running. "
        "By default, file traces are skipped when the viewer is active (traces go to OTLP only). "
        "Useful for cross-validating viewer traces against local files.",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        metavar="ID",
        help="Run only tasks with these IDs (e.g. kdd_demo_001 kdd_demo_007)",
    )
    agent_group = parser.add_mutually_exclusive_group()
    agent_group.add_argument(
        "--agent",
        type=str,
        metavar="[OLD=]SPEC",
        help="Override agent class. "
        "Use 'module.Class' to replace all agents, or 'OldClass=module.Class' "
        "to replace only tests using OldClass. "
        "Supports file paths: 'path/to/agent.py' or 'path/to/agent.py::ClassName'.",
    )
    agent_group.add_argument(
        "--agents",
        nargs="+",
        metavar="[LABEL:]SPEC",
        help="Compare multiple agents on the same data. Each SPEC uses the same syntax as "
        "--agent. Optionally prefix with 'label:' to name the agent in the output table "
        "(e.g. 'pf:path/to/agent-pf.py::Agent'). Results are written to per-agent "
        "subdirectories and a comparison table is printed at the end.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    parser.add_argument(
        "--http-logging",
        type=str,
        nargs="?",
        const="tmp/debug_logs",
        metavar="DIR",
        help="Enable HTTP request logging to DIR (default: debug_logs)",
    )
    parser.add_argument(
        "--http-logging-url-filter",
        type=str,
        default="nvidia.com",
        help="URL filter for HTTP logging (default: nvidia.com)",
    )
    parser.add_argument(
        "--http-logging-responses",
        action="store_true",
        help="Also save HTTP responses (larger files)",
    )
    return parser.parse_args()


def _load_class_from_file(file_path: str, class_name: str | None) -> tuple[type | None, str | None]:
    """Load a class from a Python file by path.

    Args:
        file_path: Absolute or relative path to a .py file.
        class_name: Class name to load, or None to auto-detect (single class).

    Returns:
        Tuple of ``(class, error_message)``. On success error_message is None.
    """
    import importlib.util
    import inspect

    path = Path(file_path)
    if not path.exists():
        return None, f"--agent: file not found: '{file_path}'"

    module_name = f"_agent_override_{path.stem}_{hash(str(path.resolve())) & 0xFFFFFFFF:08x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None, f"--agent: cannot load file: '{file_path}'"

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        del sys.modules[module_name]
        return None, f"--agent: error executing '{file_path}': {e}"

    if class_name:
        cls = getattr(mod, class_name, None)
        if cls is None:
            return None, f"--agent: '{file_path}' has no class '{class_name}'"
        return cls, None

    # Auto-detect: find non-private classes defined in this file
    candidates = [
        obj
        for name, obj in vars(mod).items()
        if not name.startswith("_") and inspect.isclass(obj) and obj.__module__ == module_name
    ]
    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) == 0:
        return None, f"--agent: no classes found in '{file_path}'"
    names = [c.__name__ for c in candidates]
    return None, (
        f"--agent: multiple classes in '{file_path}': {names}. "
        "Use 'path/to/file.py::ClassName' to specify one."
    )


def apply_agent_override(evaluator, spec: str) -> tuple[list[str], str | None]:
    """Apply an --agent override to all matching tests in an evaluator.

    Args:
        evaluator: Evaluator whose tests will be updated.
        spec: One of:
            - ``"module.Class"`` — import by module name (file must be on sys.path)
            - ``"path/to/file.py"`` — load from file, auto-detect single class
            - ``"path/to/file.py::ClassName"`` — load specific class from file
            - Prepend ``"OldClass="`` to any of the above to replace only tests
              whose current agent is named OldClass.

    Returns:
        Tuple of ``(overridden_test_names, error_message)``.
        On success ``error_message`` is None; on failure ``overridden_test_names``
        is empty and ``error_message`` describes the problem.
    """
    import importlib

    if "=" in spec:
        old_class_name, new_spec = spec.split("=", 1)
    else:
        old_class_name, new_spec = None, spec

    if old_class_name is not None and "." in old_class_name:
        return [], (
            f"--agent: old class name '{old_class_name}' must be a bare class name, "
            "not a dotted path (e.g. use 'OldClass=module.NewClass', not 'module.OldClass=...')."
        )

    # File path: contains '/' or '\', or ends with '.py', or contains '::'
    is_file_path = (
        "/" in new_spec or "\\" in new_spec or new_spec.endswith(".py") or "::" in new_spec
    )
    if is_file_path:
        if "::" in new_spec:
            file_path, class_name = new_spec.split("::", 1)
        else:
            file_path, class_name = new_spec, None
        new_class, err = _load_class_from_file(file_path, class_name)
        if err:
            return [], err
    else:
        if "." not in new_spec:
            return [], (
                f"--agent: '{new_spec}' is not a fully-qualified class path. "
                "Expected 'module.ClassName', 'path/to/file.py', or 'path/to/file.py::ClassName'."
            )
        module_path, class_name = new_spec.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            return [], f"--agent: cannot import module '{module_path}': {e}"
        try:
            new_class = getattr(mod, class_name)
        except AttributeError:
            return [], f"--agent: module '{module_path}' has no class '{class_name}'"

    overridden = []
    for test in evaluator.tests.values():
        if old_class_name is None or test.agent_class.__name__ == old_class_name:
            test.agent_class = new_class
            overridden.append(test.name)

    return overridden, None


def _parse_agent_spec(raw: str) -> tuple[str | None, str]:
    """Split an optional ``label:`` prefix from an agent spec.

    Returns ``(label, spec)`` where ``label`` is None when no prefix is present.

    A label prefix is detected when the spec starts with ``word:`` and the
    word contains none of ``/``, ``\\``, ``.``, or ``=`` (to distinguish it
    from Windows paths, module paths, and OLD=NEW filters).

    Examples::

        "pf:path/to/agent.py::Agent"  →  ("pf",  "path/to/agent.py::Agent")
        "v2:module.AgentV2"           →  ("v2",  "module.AgentV2")
        "path/to/agent.py::Agent"     →  (None,  "path/to/agent.py::Agent")
        "OldClass=module.New"         →  (None,  "OldClass=module.New")
    """
    colon_idx = raw.find(":")
    if colon_idx > 0:
        candidate = raw[:colon_idx]
        rest = raw[colon_idx + 1 :]
        if candidate and not any(c in candidate for c in r"/\\.="):
            return candidate, rest
    return None, raw


def _derive_agent_label_from_spec(raw: str) -> str:
    """Derive a display label from an agent spec (or ``label:spec`` pair).

    Label priority:
    1. Explicit ``label:`` prefix in the spec.
    2. Filename stem for file-path specs (``/`` or ``.py`` in the spec).
    3. Class name for module specs (last dotted component).
    """
    label, spec = _parse_agent_spec(raw)
    if label:
        return label

    # Strip OLD= prefix if present
    new_spec = spec.split("=", 1)[-1] if "=" in spec else spec

    # File-path spec: use the filename stem
    is_file_path = (
        "/" in new_spec or "\\" in new_spec or new_spec.endswith(".py") or "::" in new_spec
    )
    if is_file_path:
        file_part = new_spec.split("::")[0]
        return Path(file_part).stem

    # Module spec: use the class name (last component)
    if "." in new_spec:
        return new_spec.rsplit(".", 1)[-1]

    return new_spec


def _unique_label(label: str, existing: list[str]) -> str:
    """Return ``label`` if not in ``existing``, else append ``_1``, ``_2``, …"""
    if label not in existing:
        return label
    i = 1
    while f"{label}_{i}" in existing:
        i += 1
    return f"{label}_{i}"


def _clone_evaluator_with_tests(evaluator, tests: dict):
    """Shallow-clone an Evaluator, replacing its tests dict."""
    from .evaluator import Evaluator

    clone = Evaluator.__new__(Evaluator)
    clone.__dict__.update(evaluator.__dict__)
    clone.tests = tests
    return clone


def _print_comparison_table(
    labels: list[str],
    quiet: bool,
    results,
) -> None:
    """Print a per-agent comparison table to stdout.

    ``labels`` is the ordered list of agent labels used in the run.
    ``results`` is the single ``EvalResults`` from the bundled run.
    Each result's ``variant`` field is ``"{label}_run{n}"``.
    """
    if not labels:
        return

    def results_for(label: str):
        """Return all EvalTestResult objects belonging to this label."""
        prefix = f"{label}_run"
        return [
            r for r in results.results if hasattr(r, "variant") and r.variant.startswith(prefix)
        ]

    def cell(passed: int, total: int) -> str:
        pct = f"{100 * passed / total:.0f}%" if total else "—"
        return f"{passed}/{total} {pct:>4}"

    if quiet:
        for label in labels:
            rs = results_for(label)
            p = sum(1 for r in rs if r.passed)
            t = len(rs)
            print(f"{label}: {cell(p, t)}")
        return

    # Collect all test names in insertion order
    test_names: list[str] = []
    seen: set[str] = set()
    for r in results.results:
        if not hasattr(r, "test_name"):
            continue
        if r.test_name and r.test_name not in seen:
            test_names.append(r.test_name)
            seen.add(r.test_name)

    def per_test_counts(label: str, test_name: str) -> tuple[int, int]:
        rs = [r for r in results_for(label) if r.test_name == test_name]
        return sum(1 for r in rs if r.passed), len(rs)

    # Build table rows
    rows: list[list[str]] = []
    for label in labels:
        row = [label]
        for name in test_names:
            p, t = per_test_counts(label, name)
            row.append(cell(p, t))
        rs = results_for(label)
        row.append(cell(sum(1 for r in rs if r.passed), len(rs)))
        rows.append(row)

    headers = ["Agent"] + test_names + ["TOTAL"]
    widths = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]
    sep = "─" * (sum(widths) + 3 * (len(widths) - 1))

    print()
    print("   ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(sep)
    for row in rows:
        print("   ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def load_custom_strategy(module_path: str, class_name: str):
    """Load a custom strategy class from a module.

    Args:
        module_path: Module path (e.g., "agents.strategy")
        class_name: Class name (e.g., "MyCustomStrategy")

    Returns:
        Strategy instance
    """
    import importlib

    module = importlib.import_module(module_path)
    strategy_class = getattr(module, class_name)
    return strategy_class()


async def main_async():
    """Async main entry point."""
    args = parse_args()
    from nooa import set_default_strategy

    from .config import StrategyConfig, evaluator_from_config, load_config

    # Enable HTTP logging BEFORE any LLM imports (must patch httpx first)
    disable_http_logging = None
    if args.http_logging:
        from nooa.unifiedllm.http_logging import enable_http_request_logging

        disable_http_logging = enable_http_request_logging(
            output_dir=args.http_logging,
            url_filter=args.http_logging_url_filter,
            save_responses=args.http_logging_responses,
            verbose=not args.quiet,
        )

    # Configure logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("nooa").setLevel(logging.ERROR)
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("litellm").setLevel(logging.ERROR)

    # Add config directory to path for importing test modules
    config_dir = str(args.config.parent.resolve())
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)

    # Load config to get default_strategy (before creating evaluator)
    config = load_config(args.config)

    # Apply default strategy override
    # Priority: CLI --default_strategy > config default_strategy > library default
    strategy_display_name = None
    if args.default_strategy:
        # CLI override (built-in strategy name)
        strategy_instance = get_strategy_instance(args.default_strategy)
        set_default_strategy(strategy_instance)
        strategy_display_name = args.default_strategy
    elif config.default_strategy is not None:
        if isinstance(config.default_strategy, str):
            # Config specifies built-in strategy name
            strategy_instance = get_strategy_instance(config.default_strategy)
            set_default_strategy(strategy_instance)
            strategy_display_name = config.default_strategy
        elif isinstance(config.default_strategy, StrategyConfig):
            # Config specifies custom strategy class
            strategy_instance = load_custom_strategy(
                config.default_strategy.module,
                config.default_strategy.class_name,
            )
            set_default_strategy(strategy_instance)
            strategy_display_name = (
                f"{config.default_strategy.module}.{config.default_strategy.class_name}"
            )

    # Load evaluator from config (YAML -> pure Python)
    if not args.quiet:
        print(f"Loading config: {args.config}")
    evaluator = evaluator_from_config(args.config, no_cache_override=args.no_cache or None)

    # Override output directory if specified
    if args.output_dir:
        evaluator.output_dir = args.output_dir

    # Apply --agent override: swap agent_class on matching tests
    if args.agent:
        overridden, err = apply_agent_override(evaluator, args.agent)
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 1
        if not overridden:
            old_part = args.agent.split("=", 1)[0] if "=" in args.agent else None
            match_desc = f" matching '{old_part}'" if old_part else ""
            print(f"WARNING: --agent: no tests found{match_desc}", file=sys.stderr)
        elif not args.quiet:
            new_spec = args.agent.split("=", 1)[-1]
            print(f"Agent override: {new_spec} → tests: {', '.join(overridden)}")

    # Parse comma-separated lists
    tests = [t.strip() for t in args.test.split(",") if t.strip()] if args.test else None
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None

    # Show run info
    if not args.quiet:
        print(f"Experiment: {evaluator.name}")
        if not args.no_files:
            print(f"Output: {evaluator.output_dir}")
        print(f"Runs: {args.runs}, Parallel: {args.parallel}")
        if args.timeout:
            print(f"Timeout: {args.timeout}s per sample")
        if args.engine != "asyncio":
            engine_info = f"Engine: {args.engine}"
            if args.batch_size > 1:
                engine_info += f" (batch_size={args.batch_size})"
            print(engine_info)
        if args.memory_limit:
            if args.engine == "asyncio":
                print(
                    "WARNING: --memory-limit has no effect with --engine asyncio. "
                    "Use --engine subprocess.",
                    file=sys.stderr,
                )
            else:
                print(f"Memory limit: {args.memory_limit} MB per worker")
        if args.hang_timeout:
            print(f"Hang detection: {args.hang_timeout}s")
        if strategy_display_name:
            print(f"Strategy: {strategy_display_name}")
        if args.no_cache:
            print("Cache: disabled (--no-cache)")
        print()

    # Setup hang detection watchdog if requested
    watchdog = None
    if args.hang_timeout:
        watchdog = HangWatchdog(timeout=args.hang_timeout, quiet=args.quiet)
        watchdog.start()

    # Common kwargs shared across all runs
    run_kwargs = {
        "tests": tests,
        "models": models,
        "runs": args.runs,
        "limit": args.limit,
        "task_ids": args.task_ids,
        "parallel": args.parallel,
        "timeout": args.timeout,
        "quiet": args.quiet,
        "engine_type": args.engine,
        "batch_size": args.batch_size,
        "on_progress_hook": watchdog.ping if watchdog else None,
        "no_files": args.no_files,
        "trace_files": args.trace_files,
        "memory_limit_mb": args.memory_limit,
    }

    try:
        if args.agents:
            # --agents: build one sample set per agent spec, bundle into a single run
            import dataclasses

            agent_variants: list[tuple[str, list]] = []
            seen_labels: list[str] = []

            for raw_spec in args.agents:
                _, spec = _parse_agent_spec(raw_spec)

                # Snapshot tests and apply this agent's override
                tests_snapshot = {k: dataclasses.replace(v) for k, v in evaluator.tests.items()}

                # Temporarily wrap in a minimal object apply_agent_override can use
                class _Holder:
                    tests = tests_snapshot

                holder = _Holder()
                overridden, err = apply_agent_override(holder, spec)
                if err:
                    print(f"ERROR: {err}", file=sys.stderr)
                    return 1
                if not overridden:
                    old_part = spec.split("=", 1)[0] if "=" in spec else None
                    match_desc = f" matching '{old_part}'" if old_part else ""
                    print(f"WARNING: --agents: no tests found{match_desc}", file=sys.stderr)

                label = _derive_agent_label_from_spec(raw_spec)
                deduped = _unique_label(label, seen_labels)
                if deduped != label:
                    print(
                        f"WARNING: --agents: label '{label}' already used; renamed to '{deduped}'",
                        file=sys.stderr,
                    )
                seen_labels.append(deduped)
                agent_variants.append((deduped, list(tests_snapshot.values())))

            results = await evaluator.run(**run_kwargs, agent_variants=agent_variants)

            _print_comparison_table(seen_labels, args.quiet, results)

        else:
            # Single run (--agent or no override)
            agent_label = _derive_agent_label_from_spec(args.agent) if args.agent else None
            results = await evaluator.run(**run_kwargs, agent_label=agent_label)

            # Print summary
            if args.quiet:
                summary = results.summary()
                if results.output_file:
                    print(f"TOTAL: {summary} → {results.output_file}")
                else:
                    print(f"TOTAL: {summary}")
            else:
                print(f"\n{'=' * 50}")
                print(f"TOTAL: {results.summary()}")
                if results.output_file:
                    print(f"Output: {results.output_file}")

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    finally:
        if watchdog:
            watchdog.stop()

    # Cleanup HTTP logging
    if disable_http_logging:
        disable_http_logging()


def main():
    """Main entry point."""
    try:
        sys.exit(asyncio.run(main_async()) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
