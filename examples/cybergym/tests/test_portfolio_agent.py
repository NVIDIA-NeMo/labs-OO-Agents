# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the portfolio-based CyberGym agent."""

import asyncio
import hashlib
import inspect
import json
import shlex
from types import SimpleNamespace

import pytest

pytest.importorskip("nooa")

from opentelemetry import trace as otel_trace  # noqa: E402

from examples.cybergym.nooa_cybergym import agent as nooa_cybergym_agent  # noqa: E402
from examples.cybergym.nooa_cybergym import main as nooa_cybergym_main  # noqa: E402
from examples.cybergym.nooa_cybergym import submissions as cybergym_submissions  # noqa: E402
from nooa.prompts import build_prompt_data  # noqa: E402
from nooa.tracing import flush_traces  # noqa: E402
from nooa.unifiedllm.fake import FakeLLMClient  # noqa: E402


def _unused_shell() -> SimpleNamespace:
    """Return isolated placeholder shell state for tests that never execute commands."""
    return SimpleNamespace()


def _submission_manager(
    *,
    submission_count: int = 0,
    submissions: list[cybergym_submissions.PocSubmission] | None = None,
) -> cybergym_submissions.SubmissionManager:
    return cybergym_submissions.SubmissionManager(
        shell=_unused_shell(),
        submission_count=submission_count,
        submissions=submissions or [],
    )


def test_fingerprint_uses_dedup_token_for_asan_crash():
    output = """
==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x123
    #0 0xabc in tt_face_palette_set /src/freetype2/src/sfnt/ttcpal.c:268:18
    #1 0xdef in tt_face_load_cpal /src/freetype2/src/sfnt/ttcpal.c:209:5
DEDUP_TOKEN: tt_face_palette_set--tt_face_load_cpal--sfnt_load_face
"""

    fp = cybergym_submissions.SubmissionManager.fingerprint_output("crashed", 1, output)

    assert fp.kind == "crash"
    assert fp.sanitizer == "AddressSanitizer"
    assert fp.error_type == "heap-buffer-overflow"
    assert fp.dedup_token == "tt_face_palette_set--tt_face_load_cpal--sfnt_load_face"
    assert "tt_face_palette_set--tt_face_load_cpal" in fp.cluster_key


def test_fingerprint_ignores_volatile_asan_addresses_for_same_crash_site():
    first = """
==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x512000000bd4 at pc 0x562ca6ebe9db bp 0x7fff2c54a6d0 sp 0x7fff2c549e98
    #0 0x562ca6ebe9db in strlen /src/string.c:10:1
    #1 0x562ca6e00111 in Set /src/string.h:20:1
    #2 0x562ca6e00222 in Assimp::MD3Importer::InternReadFile /src/MD3Loader.cpp:30:1
"""
    second = """
==2==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x513000000508 at pc 0x560a0fc9c9db bp 0x7ffee3c3db70 sp 0x7ffee3c3d338
    #0 0x560a0fc9c9db in strlen /src/string.c:10:1
    #1 0x560a0fc00111 in Set /src/string.h:20:1
    #2 0x560a0fc00222 in Assimp::MD3Importer::InternReadFile /src/MD3Loader.cpp:30:1
"""

    first_fp = cybergym_submissions.SubmissionManager.fingerprint_output(
        "crashed", 1, first
    )
    second_fp = cybergym_submissions.SubmissionManager.fingerprint_output(
        "crashed", 1, second
    )

    assert first_fp.error_type == "heap-buffer-overflow"
    assert second_fp.error_type == "heap-buffer-overflow"
    assert first_fp.cluster_key == second_fp.cluster_key


def test_fingerprint_keeps_distinct_asan_error_categories_separate():
    heap_overflow = """
==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x512000000bd4
    #0 0xabc in parse_tag /src/parser.c:10:1
"""
    segv = """
==2==ERROR: AddressSanitizer: SEGV on unknown address 0x512000000bd4
    #0 0xdef in parse_tag /src/parser.c:10:1
"""

    heap_fp = cybergym_submissions.SubmissionManager.fingerprint_output(
        "crashed", 1, heap_overflow
    )
    segv_fp = cybergym_submissions.SubmissionManager.fingerprint_output(
        "crashed", 1, segv
    )

    assert heap_fp.cluster_key != segv_fp.cluster_key


def test_fingerprint_classifies_msan_personality_as_infra():
    output = """
MemorySanitizer: CHECK failed: msan_linux.cpp:192
"((personality(old_personality | ADDR_NO_RANDOMIZE))) != ((-1))"
    <empty stack>
"""

    fp = cybergym_submissions.SubmissionManager.fingerprint_output("crashed", 1, output)

    assert fp.kind == "infra"
    assert fp.cluster_key == "infra:msan_personality"


def test_fingerprint_classifies_cryptofuzz_assertion():
    output = """
Difference detected
Assertion failure: Botan-wolfCrypt-BignumCalc-(no algorithm)-difference
==2==ERROR: AddressSanitizer: ABRT on unknown address
    #0 0xaaa in raise (/lib/x86_64-linux-gnu/libc.so.6+0x35438)
    #1 0xbbb in abort (/lib/x86_64-linux-gnu/libc.so.6+0x37039)
"""

    fp = cybergym_submissions.SubmissionManager.fingerprint_output("crashed", 1, output)

    assert fp.kind == "assertion"
    assert fp.assertion == "Botan-wolfCrypt-BignumCalc-(no algorithm)-difference"
    assert fp.cluster_key.startswith("assertion:Botan-wolfCrypt")


def test_submit_result_can_carry_fingerprint():
    fp = cybergym_submissions.SubmissionManager.fingerprint_output(
        "no_crash", 0, "Execution successful"
    )
    result = cybergym_submissions.SubmitResult(
        status="no_crash",
        exit_code=0,
        output="Execution successful",
        submission_number=1,
        fingerprint=fp,
    )

    assert result.fingerprint is not None
    assert result.fingerprint.kind == "no_crash"


def test_classify_submit_treats_bare_fault_signals_as_crashes():
    assert cybergym_submissions.SubmissionManager.classify_submit(-11, "") == "crashed"
    assert cybergym_submissions.SubmissionManager.classify_submit(139, "") == "crashed"


def test_classify_submit_keeps_external_kills_and_safe_exits_non_crashing():
    assert cybergym_submissions.SubmissionManager.classify_submit(-15, "") == "crashed_suspect"
    assert cybergym_submissions.SubmissionManager.classify_submit(137, "") == "no_crash"


def test_classify_submit_detects_bannerless_ubsan_runtime_error():
    output = "x.c:1:1: runtime error: signed integer overflow: 1 + 2147483647"
    assert cybergym_submissions.SubmissionManager.classify_submit(1, output) == "crashed"


def test_submit_runner_quotes_poc_path():
    class FakeShell:
        command = ""

        async def run(self, command, timeout):
            self.command = command
            assert timeout == 300.0
            return SimpleNamespace(stdout='{"exit_code": 0, "output": "Execution successful"}')

    shell = FakeShell()
    manager = cybergym_submissions.SubmissionManager(shell=shell)
    poc_path = "/tmp/poc with spaces;touch /tmp/nope"

    result = asyncio.run(manager._run_submit_script(poc_path, submission_number=1))

    assert result.status == "no_crash"
    assert shell.command == (f"bash {shlex.quote(manager.SUBMIT_SCRIPT)} {shlex.quote(poc_path)}")


def test_submit_stores_hypothesis_in_submission_and_jsonl(tmp_path):
    class FakeShell:
        async def run(self, command, timeout):
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "exit_code": 1,
                        "output": (
                            "ERROR: AddressSanitizer: heap-buffer-overflow\n"
                            "#0 0xabc in parse_header /src/parser.c:10:1"
                        ),
                    }
                )
            )

    manager = cybergym_submissions.SubmissionManager(shell=FakeShell())
    manager.SUBMISSION_LOG_PATH = tmp_path / "submissions.jsonl"
    hypothesis = "A short length field reaches parse_header and overruns the heap buffer."

    result = asyncio.run(manager.submit("/tmp/poc", hypothesis=hypothesis))

    submission = manager.get_submission(result.submission_number)
    assert submission is not None
    assert submission.hypothesis == hypothesis
    record = json.loads(manager.SUBMISSION_LOG_PATH.read_text().strip())
    assert record["hypothesis"] == hypothesis


def test_submit_preserves_candidate_in_persistent_artifacts(tmp_path):
    class FakeShell:
        async def run(self, command, timeout):
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "exit_code": 1,
                        "output": "ERROR: AddressSanitizer: heap-use-after-free",
                    }
                )
            )

    source = tmp_path / "candidate.otf"
    source.write_bytes(b"persistent-candidate")
    manager = cybergym_submissions.SubmissionManager(shell=FakeShell())
    manager.SUBMISSIONS_DIR = tmp_path / "verifier-does-not-copy"
    manager.SUBMISSION_LOG_PATH = tmp_path / "artifacts" / "submissions.jsonl"
    manager.CANDIDATE_DIR = tmp_path / "artifacts" / "candidates"

    result = asyncio.run(manager.submit(str(source), hypothesis="Exercises the CFF parser."))

    submission = manager.get_submission(result.submission_number)
    assert submission is not None
    assert submission.submitted_path == str(manager.CANDIDATE_DIR / "submission_1.poc")
    assert (manager.CANDIDATE_DIR / "submission_1.poc").read_bytes() == b"persistent-candidate"
    record = json.loads(manager.SUBMISSION_LOG_PATH.read_text().strip())
    assert record["submitted_path"] == submission.submitted_path


def test_submit_rejects_an_empty_hypothesis_before_running_verifier():
    class FakeShell:
        async def run(self, command, timeout):
            pytest.fail("verifier should not run without a hypothesis")

    manager = cybergym_submissions.SubmissionManager(shell=FakeShell())

    with pytest.raises(ValueError, match="hypothesis must briefly explain"):
        asyncio.run(manager.submit("/tmp/poc", hypothesis="  \n  "))


def test_finalize_writes_one_immutable_model_selected_poc(tmp_path):
    source = tmp_path / "candidate.bin"
    source.write_bytes(b"chosen-poc")
    fingerprint = cybergym_submissions.SubmissionManager.fingerprint_output(
        "crashed", 139, "SIGSEGV"
    )
    submission = cybergym_submissions.PocSubmission(
        submission_number=7,
        original_path=str(source),
        submitted_path=str(source),
        status="crashed",
        exit_code=139,
        fingerprint=fingerprint,
        hypothesis="The length field reaches the vulnerable copy.",
    )
    manager = _submission_manager(submission_count=7, submissions=[submission])
    manager.FINAL_SUBMISSION_DIR = tmp_path / "final_submission"

    artifact = manager.finalize(7, selection_reason="Strongest patch-relevant crash.")

    final_poc = manager.FINAL_SUBMISSION_DIR / "poc"
    manifest_path = manager.FINAL_SUBMISSION_DIR / "selection.json"
    assert final_poc.read_bytes() == b"chosen-poc"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["submission_number"] == 7
    assert manifest["selection_reason"] == "Strongest patch-relevant crash."
    assert manifest["sha256"] == hashlib.sha256(b"chosen-poc").hexdigest()
    assert artifact.sha256 == manifest["sha256"]

    with pytest.raises(FileExistsError, match="already exists"):
        manager.finalize(7, selection_reason="A second choice must never replace it.")
    assert final_poc.read_bytes() == b"chosen-poc"


def test_finalize_rejects_a_non_crashing_candidate(tmp_path):
    source = tmp_path / "safe.bin"
    source.write_bytes(b"safe")
    submission = cybergym_submissions.PocSubmission(
        submission_number=2,
        original_path=str(source),
        submitted_path=str(source),
        status="no_crash",
        exit_code=0,
        fingerprint=cybergym_submissions.SubmissionManager.fingerprint_output(
            "no_crash", 0, "Execution successful"
        ),
        hypothesis="Does not crash.",
    )
    manager = _submission_manager(submission_count=2, submissions=[submission])
    manager.FINAL_SUBMISSION_DIR = tmp_path / "final_submission"

    with pytest.raises(ValueError, match="verified crash"):
        manager.finalize(2, selection_reason="Invalid selection")

    assert not manager.FINAL_SUBMISSION_DIR.exists()


@pytest.mark.asyncio
async def test_agent_uses_model_selection_to_finalize_portfolio(tmp_path, monkeypatch):
    source = tmp_path / "candidate.bin"
    source.write_bytes(b"agent-choice")
    submission = cybergym_submissions.PocSubmission(
        submission_number=4,
        original_path=str(source),
        submitted_path=str(source),
        status="crashed",
        exit_code=139,
        fingerprint=cybergym_submissions.SubmissionManager.fingerprint_output(
            "crashed", 139, "SIGSEGV"
        ),
        hypothesis="Triggers the vulnerable parser branch.",
    )
    manager = _submission_manager(submission_count=4, submissions=[submission])
    manager.FINAL_SUBMISSION_DIR = tmp_path / "final_submission"
    portfolio = nooa_cybergym_agent.Portfolio(manager)
    portfolio.submissions = [submission]
    agent = nooa_cybergym_agent.CyberGymAgent(llm=FakeLLMClient())
    agent._portfolio = portfolio

    async def choose(self, current_portfolio_state):
        assert "crash_families=1" in current_portfolio_state
        return nooa_cybergym_agent.FinalSelection(
            submission_number=4,
            reasoning="Most direct and reproducible trigger.",
        )

    monkeypatch.setattr(nooa_cybergym_agent.CyberGymAgent, "_select_final", choose)

    artifact = await agent._finalize_portfolio()

    assert artifact.submission_number == 4
    assert (manager.FINAL_SUBMISSION_DIR / "poc").read_bytes() == b"agent-choice"


@pytest.mark.asyncio
async def test_shutdown_cancels_workers_and_closes_every_shell():
    class CloseableShell:
        def __init__(self):
            self.closed = 0

        async def close(self):
            self.closed += 1

    root_shell = CloseableShell()
    worker_shell = CloseableShell()
    manager = _submission_manager()
    manager._owner._shell = root_shell
    agent = nooa_cybergym_agent.CyberGymAgent(llm=FakeLLMClient())
    agent._portfolio = nooa_cybergym_agent.Portfolio(manager)
    agent._worker_agents = [SimpleNamespace(shell=worker_shell, llm=FakeLLMClient())]
    sleeper = asyncio.create_task(asyncio.sleep(30))
    agent._active_tasks = {sleeper}

    await agent.shutdown()

    assert sleeper.cancelled()
    assert root_shell.closed == 1
    assert worker_shell.closed == 1


def test_finder_uses_feedback_history_for_portfolio_context():
    portfolio = nooa_cybergym_agent.Portfolio(
        cybergym_submissions.SubmissionManager(shell=_unused_shell())
    )

    finder = nooa_cybergym_agent.Finder(
        llm=FakeLLMClient(), portfolio=portfolio, model_name="test-model"
    )

    blocks = finder.context_manager._blocks
    static = finder.context_manager._static
    assert "current_portfolio" not in blocks
    assert "state" in blocks
    assert finder.context_manager.is_disabled("state")
    assert "tools_reminder" in blocks
    assert static["tools_reminder"] is True

    events = list(finder.event_manager.values())
    assert len(events) == 1
    assert events[0].content.startswith("<current_portfolio_update reason='initial'>")
    assert "crash_families=0" in events[0].content


def test_finder_records_portfolio_feedback_only_when_changed():
    portfolio = nooa_cybergym_agent.Portfolio(
        cybergym_submissions.SubmissionManager(shell=_unused_shell())
    )
    finder = nooa_cybergym_agent.Finder(
        llm=FakeLLMClient(), portfolio=portfolio, model_name="test-model"
    )

    finder.record_portfolio_context_if_changed("unchanged")
    assert len(list(finder.event_manager.values())) == 1

    portfolio.guidance = "Try a different parser path."
    finder.record_portfolio_context_if_changed("review")

    events = list(finder.event_manager.values())
    assert len(events) == 2
    assert "reason='review'" in events[-1].content
    assert "Try a different parser path." in events[-1].content


def test_expander_uses_static_tool_reminder_without_dynamic_context():
    portfolio = nooa_cybergym_agent.Portfolio(
        cybergym_submissions.SubmissionManager(shell=_unused_shell())
    )

    expander = nooa_cybergym_agent.Expander(
        llm=FakeLLMClient(), portfolio=portfolio, model_name="test-model"
    )

    blocks = expander.context_manager._blocks
    static = expander.context_manager._static
    assert "current_portfolio" not in blocks
    assert "state" in blocks
    assert expander.context_manager.is_disabled("state")
    assert "tools_reminder" in blocks
    assert static["tools_reminder"] is True


def test_cybergym_agent_disables_default_state_context():
    agent = nooa_cybergym_agent.CyberGymAgent(llm=FakeLLMClient())

    assert "state" in agent.context_manager._blocks
    assert agent.context_manager.is_disabled("state")


@pytest.mark.asyncio
async def test_reviewer_prompt_uses_the_effective_minimum_exploration_window():
    agent = nooa_cybergym_agent.CyberGymAgent(llm=FakeLLMClient())
    prompt_template = inspect.getdoc(nooa_cybergym_agent.CyberGymAgent._review)
    prompt = await build_prompt_data(agent._review, "empty portfolio")

    assert agent._minimum_exploration_sec == nooa_cybergym_agent.MIN_EXPLORATION_SEC
    assert "{self._minimum_exploration_sec} seconds" in prompt_template
    assert "default: 20 minutes" not in prompt_template
    assert (
        f"minimum exploration window ({nooa_cybergym_agent.MIN_EXPLORATION_SEC} seconds)"
        in prompt.task_prompt
    )


def test_cybergym_agents_have_isolated_shell_sessions():
    first = nooa_cybergym_agent.CyberGymAgent(llm=FakeLLMClient())
    second = nooa_cybergym_agent.CyberGymAgent(llm=FakeLLMClient())

    assert first.shell is not second.shell
    assert first.shell._session is not second.shell._session


def test_glm52_is_the_agent_default_with_three_finder_lanes():
    assert nooa_cybergym_agent.DEFAULT_MODEL_NAME == "glm-5.2"
    assert nooa_cybergym_agent.LANES == [
        nooa_cybergym_agent.Lane(label="glm-5.2", model_name="glm-5.2"),
        nooa_cybergym_agent.Lane(label="nemotron-3-ultra", model_name="nvidia/nemotron-3-ultra"),
        nooa_cybergym_agent.Lane(label="deepseek-v4-flash", model_name="deepseek-v4-flash"),
    ]


def test_finder_provenance_uses_resolved_provider_model(monkeypatch):
    resolved_llm = FakeLLMClient()
    resolved_llm.model = "openai/deepseek-v4-flash"
    monkeypatch.setattr(nooa_cybergym_agent, "make_llm", lambda *args, **kwargs: resolved_llm)
    monkeypatch.setattr(nooa_cybergym_agent, "install_summarizer", lambda *args: None)

    agent = nooa_cybergym_agent.CyberGymAgent(llm=FakeLLMClient())
    agent._portfolio = nooa_cybergym_agent.Portfolio(_submission_manager())
    finder = agent._make_finder(
        nooa_cybergym_agent.Lane(label="configured-alias", model_name="glm-5.2")
    )
    expander, _ = agent._make_expander(SimpleNamespace())

    assert finder._model_name == "openai/deepseek-v4-flash"
    assert expander._model_name == "openai/deepseek-v4-flash"


@pytest.mark.parametrize(
    "method",
    [nooa_cybergym_agent.Finder.find, nooa_cybergym_agent.Expander.expand],
)
def test_worker_cells_use_a_hard_out_of_process_timeout(method):
    config = method._plan_strategy.config

    assert config.execution_backend == "sandbox"
    assert config.cell_timeout == 60
    assert config.sandbox.broker_timeout_s == 360


def test_submission_manager_digest_clusters_submissions_without_llm_constructor():
    fp = cybergym_submissions.SubmissionManager.fingerprint_output(
        "crashed",
        1,
        """
==1==ERROR: AddressSanitizer: heap-buffer-overflow
    #0 0xabc in foo /x.c:1:1
""",
    )
    manager = _submission_manager(
        submission_count=2,
        submissions=[
            cybergym_submissions.PocSubmission(
                submission_number=1,
                original_path="/tmp/poc_a",
                submitted_path="/workspace/submissions/poc_001",
                status="crashed",
                exit_code=1,
                fingerprint=fp,
                hypothesis="Candidate A reaches foo through the primary parser path.",
            ),
            cybergym_submissions.PocSubmission(
                submission_number=2,
                original_path="/tmp/poc_b",
                submitted_path="/workspace/submissions/poc_002",
                status="crashed",
                exit_code=1,
                fingerprint=fp,
                hypothesis="Candidate B reaches the same crash through a variant input.",
            ),
        ],
    )

    digest = manager.digest()

    assert "Total public self.submit() calls: 2" in digest
    assert "Distinct fingerprint clusters: 1" in digest
    assert "submissions=[1, 2]" in digest


def test_configure_tracing_installs_atif_with_nooa_api(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    trajectory_path = tmp_path / "trajectory.json"
    monkeypatch.setenv("NOOA_CYBERGYM_TRACE_DIR", str(trace_dir))
    monkeypatch.setenv("NOOA_CYBERGYM_TRAJECTORY_PATH", str(trajectory_path))
    monkeypatch.setenv("NOOA_CYBERGYM_SESSION_ID", "test-session")
    monkeypatch.delenv("NOOA_CYBERGYM_OTLP_ENDPOINT", raising=False)

    agent = nooa_cybergym_main.CyberGymAgent(llm=FakeLLMClient())

    nooa_cybergym_main.configure_tracing(agent, "fake-model")
    try:
        uninstall = agent._atif_uninstall
        assert callable(uninstall)
        assert uninstall.exporter.path == trajectory_path
        assert uninstall.exporter.get_trajectory().session_id == "test-session"
        assert trace_dir.is_dir()
        duplicated_message = "journal exporter must strip this repeated message body"
        with otel_trace.get_tracer(__name__).start_as_current_span("journal-smoke") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("input.value", duplicated_message)
            span.set_attribute("llm.input_messages.0.message.content", duplicated_message)
        flush_traces()
        journal_path = trace_dir / "test-session.nooa.jsonl"
        assert journal_path.is_file()
        journal_text = journal_path.read_text()
        bodies = [json.loads(line) for line in journal_text.splitlines()]
        assert any("nooaJournal" in body for body in bodies)
        assert any("resourceSpans" in body for body in bodies)
        assert duplicated_message not in journal_text
    finally:
        uninstall()
        nooa_cybergym_main.shutdown_tracing()


def test_portfolio_apply_review_updates_guidance_and_stop_flag():
    portfolio = nooa_cybergym_agent.Portfolio(
        cybergym_submissions.SubmissionManager(shell=_unused_shell())
    )
    review = nooa_cybergym_agent.Review(
        on_target=True,
        guidance="Explore a different parser branch.",
        stop=True,
        reasoning="One strong on-target crash is enough after exploration.",
    )

    portfolio.apply_review(review)

    assert portfolio.guidance == "Explore a different parser branch."
    assert portfolio.stop is True
    assert portfolio.changed.is_set()
    assert "Explore a different parser branch." in str(portfolio)


def test_portfolio_renders_hypothesis_for_each_crash_family():
    manager = _submission_manager()
    portfolio = nooa_cybergym_agent.Portfolio(manager)
    fingerprint = cybergym_submissions.SubmissionManager.fingerprint_output(
        "crashed",
        1,
        """
==1==ERROR: AddressSanitizer: heap-buffer-overflow
    #0 0xabc in parse_header /src/parser.c:10:1
""",
    )
    portfolio.submissions = [
        cybergym_submissions.PocSubmission(
            submission_number=1,
            original_path="/tmp/poc",
            submitted_path="/workspace/submissions/poc_001",
            status="crashed",
            exit_code=1,
            fingerprint=fingerprint,
            hypothesis="The declared payload length exceeds the available input.",
        )
    ]

    rendered = str(portfolio)

    assert "Hypothesis: The declared payload length exceeds the available input." in rendered
    assert "poc=/workspace/submissions/poc_001" in rendered


def test_portfolio_pending_crash_clusters_skips_duplicates_and_expanders():
    manager = cybergym_submissions.SubmissionManager(shell=_unused_shell())
    portfolio = nooa_cybergym_agent.Portfolio(manager)
    fp_a = cybergym_submissions.SubmissionManager.fingerprint_output(
        "crashed",
        1,
        """
==1==ERROR: AddressSanitizer: heap-buffer-overflow
    #0 0xabc in parse_a /src/parser.c:10:1
""",
    )
    fp_b = cybergym_submissions.SubmissionManager.fingerprint_output(
        "crashed",
        1,
        """
==1==ERROR: AddressSanitizer: heap-buffer-overflow
    #0 0xdef in parse_b /src/parser.c:20:1
""",
    )
    portfolio.submissions = [
        cybergym_submissions.PocSubmission(
            submission_number=1,
            original_path="/tmp/a1",
            submitted_path="/workspace/submissions/poc_001",
            status="crashed",
            exit_code=1,
            fingerprint=fp_a,
            source_agent="finder",
            hypothesis="The first parser path reaches parse_a.",
        ),
        cybergym_submissions.PocSubmission(
            submission_number=2,
            original_path="/tmp/a2",
            submitted_path="/workspace/submissions/poc_002",
            status="crashed",
            exit_code=1,
            fingerprint=fp_a,
            source_agent="finder",
            hypothesis="A variant input reaches the same parse_a family.",
        ),
        cybergym_submissions.PocSubmission(
            submission_number=3,
            original_path="/tmp/b",
            submitted_path="/workspace/submissions/poc_003",
            status="crashed",
            exit_code=1,
            fingerprint=fp_b,
            source_agent="expander",
            hypothesis="The expanded input selects the parse_b branch.",
        ),
    ]

    pending = portfolio.pending_crash_clusters()

    assert [item.submission_number for item in pending] == [1]
    portfolio.mark_expanded(1)
    assert portfolio.pending_crash_clusters() == []


def test_finder_and_expander_are_distinct_worker_agent_types():
    portfolio = nooa_cybergym_agent.Portfolio(
        cybergym_submissions.SubmissionManager(shell=_unused_shell())
    )

    finder = nooa_cybergym_agent.Finder(llm=FakeLLMClient(), portfolio=portfolio)
    expander = nooa_cybergym_agent.Expander(llm=FakeLLMClient(), portfolio=portfolio)

    assert isinstance(finder, nooa_cybergym_agent.Finder)
    assert isinstance(expander, nooa_cybergym_agent.Expander)
    assert not isinstance(expander, nooa_cybergym_agent.Finder)
    assert finder is not expander
    assert finder.shell is not expander.shell


def test_submission_owner_serializes_callers_and_hides_shell():
    class FakeShell:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        async def run(self, command, timeout):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return SimpleNamespace(
                stdout='{"exit_code": 0, "output": "Execution successful"}',
                returncode=0,
            )

    async def scenario():
        shell = FakeShell()
        manager = cybergym_submissions.SubmissionManager(shell=shell)
        assert not hasattr(manager, "shell")
        await asyncio.gather(
            manager._run_submit_script("/tmp/a", submission_number=1),
            manager._run_submit_script("/tmp/b", submission_number=2),
            manager._run_submit_script("/tmp/c", submission_number=3),
        )
        await manager.close()
        return shell.max_active

    assert asyncio.run(scenario()) == 1


def test_submission_owner_isolates_caller_cancellation():
    class FakeShell:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.completed = False

        async def run(self, command, timeout):
            self.started.set()
            await self.release.wait()
            self.completed = True
            return SimpleNamespace(
                stdout='{"exit_code": 0, "output": "Execution successful"}',
                returncode=0,
            )

    async def scenario():
        shell = FakeShell()
        manager = cybergym_submissions.SubmissionManager(shell=shell)
        caller = asyncio.create_task(manager._run_submit_script("/tmp/a", submission_number=1))
        await shell.started.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        shell.release.set()
        for _ in range(20):
            if shell.completed:
                break
            await asyncio.sleep(0)
        assert shell.completed
        await manager.close()

    asyncio.run(scenario())


def test_submission_owner_poison_respawns_and_retries_once():
    class PoisonedShell:
        def __init__(self):
            self.closed = False

        async def run(self, command, timeout):
            return SimpleNamespace(stdout="", returncode=124, timed_out=True)

        async def close(self):
            self.closed = True

    class HealthyShell:
        async def run(self, command, timeout):
            return SimpleNamespace(
                stdout='{"exit_code": 0, "output": "Execution successful"}',
                returncode=0,
                timed_out=False,
            )

    async def scenario():
        poisoned = PoisonedShell()
        manager = cybergym_submissions.SubmissionManager(
            shell=poisoned,
            shell_factory=HealthyShell,
        )
        result = await manager._run_submit_script("/tmp/a", submission_number=1)
        assert result.status == "no_crash"
        assert poisoned.closed
        await manager.close()

    asyncio.run(scenario())


def test_submission_owner_preserves_verifier_timeout_exit_124():
    factory_calls = 0

    class VerifierTimeoutShell:
        async def run(self, command, timeout):
            return SimpleNamespace(
                stdout='{"exit_code": 124, "output": "candidate timed out"}\n',
                returncode=124,
                timed_out=False,
            )

        async def close(self):
            pass

    def shell_factory():
        nonlocal factory_calls
        factory_calls += 1
        return VerifierTimeoutShell()

    async def scenario():
        manager = cybergym_submissions.SubmissionManager(
            VerifierTimeoutShell(),
            shell_factory=shell_factory,
        )
        result = await manager._run_submit_script("/tmp/a", submission_number=1)
        assert result.status == "timeout"
        assert result.exit_code == 124
        assert factory_calls == 0
        assert manager._owner._consecutive_respawns == 0
        await manager.close()

    asyncio.run(scenario())


def test_submission_owner_routes_at_production_process_boundary(tmp_path):
    factory_calls = 0
    submit_script = tmp_path / "submit.sh"
    submit_script.write_text("#!/usr/bin/env bash\nexit 124\n")
    submit_script.chmod(0o755)

    def shell_factory():
        nonlocal factory_calls
        factory_calls += 1
        return cybergym_submissions.ShellTools(cwd=tmp_path)

    async def scenario():
        owner = cybergym_submissions.SubmissionShellOwner(
            cybergym_submissions.ShellTools(cwd=tmp_path),
            shell_factory=shell_factory,
            timeout=0.25,
            max_consecutive_respawns=3,
        )

        child_exit = await owner.execute(f"bash {submit_script}")
        assert child_exit.returncode == 124
        assert child_exit.timed_out is False
        assert factory_calls == 0
        assert owner._consecutive_respawns == 0

        shell_survives = await owner.execute("printf shell-survived")
        assert shell_survives.stdout == "shell-survived"
        assert shell_survives.returncode == 0
        assert shell_survives.timed_out is False
        assert factory_calls == 0

        with pytest.raises(
            cybergym_submissions.SubmissionShellCircuitOpen,
            match="failed after one clean-session retry",
        ):
            await owner.execute("exit 23")
        assert factory_calls == 2
        assert owner._consecutive_respawns == 2
        await owner.close()

    asyncio.run(scenario())


def test_submission_owner_circuit_breaks_after_bounded_respawns():
    class PoisonedShell:
        async def run(self, command, timeout):
            return SimpleNamespace(stdout="", returncode=-1)

        async def close(self):
            pass

    async def scenario():
        owner = cybergym_submissions.SubmissionShellOwner(
            PoisonedShell(),
            shell_factory=PoisonedShell,
            max_consecutive_respawns=2,
        )
        with pytest.raises(
            cybergym_submissions.SubmissionShellCircuitOpen,
            match="failed after one clean-session retry",
        ):
            await owner.execute("first")
        with pytest.raises(
            cybergym_submissions.SubmissionShellCircuitOpen,
            match="circuit open after 2 respawns",
        ):
            await owner.execute("second")
        await owner.close()

    asyncio.run(scenario())
