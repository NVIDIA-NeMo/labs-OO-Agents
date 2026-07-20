# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test: OTLP spans must be flushed even when a subprocess worker task times out.

Regression test for the bug where asyncio.timeout cancels agent.run(),
leaving spans un-ended. OTel SDK only exports ended spans, so
shutdown_traces() silently drops all execution data from timed-out tasks.
"""

import json
import subprocess
import sys


def _build_timeout_task_input(trace_dir: str, timeout_seconds: float = 1.0) -> dict:
    """Build a SubprocessTaskInput that will time out."""
    return {
        "agent_spec": {
            "agent_module": "eval_pipeline._test_fixtures",
            "agent_class": "SlowAgent",
            "method": "classify",
            "client_config": {},
        },
        "task": {
            "id": "timeout_test_001",
            "input": [[], {"text": "great product"}],
            "expected": "positive",
            "run_id": 1,
        },
        "scorers": [
            {
                "name": "exact_match",
                "weight": 1.0,
                "scorer_class": "eval_pipeline.scoring.ExactMatchScorer",
                "scorer_kwargs": {},
            }
        ],
        "sample_id": "timeout_sample_001",
        "model": "dummy",
        "test_name": "timeout_test",
        "display_name": "timeout_test/001/dummy/run1",
        "tier": "stable",
        "trace_dir": trace_dir,
        "use_otlp": True,
        "write_trace_file": True,
        "otlp_endpoint": "http://localhost:59999/v1/traces",
        "pass_threshold": 0.5,
        "timeout_seconds": timeout_seconds,
    }


class TestTimeoutSpanFlush:
    """Verify that spans are flushed even when a task times out."""

    def test_spans_exported_on_timeout(self, tmp_path):
        """When a task times out, the AGENT span must still be exported (ended + flushed).

        Before the fix: asyncio.timeout cancels agent.run(), the before_agent_call
        span is never ended, and shutdown_traces() silently drops it.
        """
        # Use a string path to avoid any PosixPath/symlink issues
        trace_dir = str(tmp_path.resolve())
        task_input = _build_timeout_task_input(
            trace_dir=trace_dir,
            timeout_seconds=1.0,  # Agent sleeps for 60s, so this will timeout
        )
        proc = subprocess.run(
            [sys.executable, "-m", "eval_pipeline.subprocess_worker"],
            input=json.dumps(task_input).encode(),
            capture_output=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"Worker crashed: {proc.stderr.decode()}"

        # The result should indicate a timeout error
        result = json.loads(proc.stdout)
        assert result["passed"] is False
        assert "Timeout" in (result.get("error") or ""), (
            f"Expected timeout error, got: {result.get('error')}"
        )

        # Now check that trace spans were actually written to the jsonl file
        import os

        actual_files = os.listdir(trace_dir)
        trace_files = [f for f in actual_files if f.endswith(".jsonl")]
        assert len(trace_files) >= 1, (
            f"Expected at least one .jsonl trace file in {trace_dir}, "
            f"found files: {actual_files}. "
            f"stderr: {proc.stderr.decode()[-500:]}"
        )

        # Read the trace file and verify it contains an AGENT span
        trace_file_path = os.path.join(trace_dir, trace_files[0])
        with open(trace_file_path) as tf:
            trace_content = tf.read().strip()
        assert trace_content, f"Trace file {trace_files[0]} is empty"

        # Parse OTLP JSON lines and look for spans
        spans_found = []
        for line in trace_content.split("\n"):
            if not line.strip():
                continue
            payload = json.loads(line)
            for rs in payload.get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    for span in ss.get("spans", []):
                        spans_found.append(span.get("name", ""))

        # The critical assertion: an AGENT span must be present even on timeout.
        # Before the fix, this list would be empty because the span was never ended.
        agent_spans = [s for s in spans_found if "method." in s]
        assert len(agent_spans) >= 1, (
            f"Expected at least one method.* (AGENT) span in trace, "
            f"but found only: {spans_found}. "
            f"This means spans were not flushed before the worker exited."
        )
