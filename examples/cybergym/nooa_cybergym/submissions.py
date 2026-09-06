# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submission protocol and portfolio state for the NOOA CyberGym agent."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import secrets
import shlex
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from nooa.tools.shell_tools import ShellTools

SubmitStatus = Literal[
    "crashed",
    "crashed_suspect",
    "no_crash",
    "timeout",
    "server_error",
    "reconsider",
]


class CrashFingerprint(BaseModel):
    """Normalized vulnerable-side crash signature for diversity review."""

    kind: Literal[
        "crash",
        "no_crash",
        "timeout",
        "server_error",
        "infra",
        "assertion",
        "suspect",
    ]
    sanitizer: str | None = None
    error_type: str | None = None
    dedup_token: str | None = None
    top_frames: list[str] = Field(default_factory=list)
    assertion: str | None = None
    cluster_key: str
    summary: str


class SubmitResult(BaseModel):
    """Typed result returned by self.submit()."""

    status: SubmitStatus
    exit_code: int
    output: str
    submission_number: int
    fingerprint: CrashFingerprint | None = None


class PocSubmission(BaseModel):
    """One public self.submit() candidate retained for portfolio review."""

    submission_number: int
    original_path: str
    submitted_path: str | None = None
    status: SubmitStatus
    exit_code: int
    fingerprint: CrashFingerprint
    output_excerpt: str = ""
    attempt_number: int | None = None
    attempt_submission_number: int | None = None
    source_agent: str | None = None
    source_model: str | None = None
    hypothesis: str


class FinalPocArtifact(BaseModel):
    """Immutable final PoC designation consumed by the official scorer."""

    schema_version: int = 1
    submission_number: int
    poc_path: str
    sha256: str
    byte_length: int
    selection_reason: str
    source_agent: str | None = None
    source_model: str | None = None
    hypothesis: str
    cluster_key: str


class KnownFamily(BaseModel):
    """Reviewer-maintained family summary to steer independent attempts."""

    id: str = ""
    short_name: str = ""
    input_strategy: str = ""
    target_code_path: str = ""
    crash_signature: str = ""
    sanitizer_or_exit_signal: str = ""
    verifier_repeatability: str = ""
    patch_relevance: str = ""
    confidence: Literal["low", "medium", "high"] = "medium"


def _model_data(model: BaseModel) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


class SubmissionShellCircuitOpen(RuntimeError):
    """The verifier shell repeatedly lost framing and is no longer trusted."""


@dataclass
class _ShellRequest:
    command: str
    future: asyncio.Future[Any]


class SubmissionShellOwner:
    """Single owner for the persistent verifier shell.

    Callers enqueue commands and await futures. Only the worker task can touch
    the shell, so caller cancellation cannot interrupt or desynchronize an
    in-flight command. The underlying BashSession supplies a unique per-command
    control-channel sentinel; timeout or framing loss poisons and replaces the
    entire ShellTools instance before one retry.
    """

    def __init__(
        self,
        shell: Any,
        *,
        shell_factory: Callable[[], Any] | None = None,
        timeout: float | None = None,
        max_consecutive_respawns: int = 3,
    ) -> None:
        self._shell = shell
        self._shell_factory = shell_factory or (lambda: ShellTools(cwd="/workspace"))
        self._timeout = float(
            timeout
            if timeout is not None
            else os.environ.get("NOOA_CYBERGYM_SUBMISSION_TIMEOUT_SEC", "300")
        )
        self._max_consecutive_respawns = max_consecutive_respawns
        self._consecutive_respawns = 0
        self._queue: asyncio.Queue[_ShellRequest] | None = None
        self._worker: asyncio.Task[None] | None = None

    async def execute(self, command: str) -> Any:
        self._ensure_worker()
        assert self._queue is not None
        future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_ShellRequest(command=command, future=future))
        return await future

    async def close(self) -> None:
        worker = self._worker
        if worker is not None and not worker.done():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        await self._close_shell(self._shell)
        self._worker = None
        self._queue = None

    def _ensure_worker(self) -> None:
        if self._worker is not None and not self._worker.done():
            return
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._run())

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            request = await self._queue.get()
            try:
                if request.future.cancelled():
                    continue
                try:
                    result = await self._execute_with_recovery(request.command)
                except Exception as exc:
                    if not request.future.cancelled():
                        request.future.set_exception(exc)
                else:
                    if not request.future.cancelled():
                        request.future.set_result(result)
            finally:
                self._queue.task_done()

    async def _execute_with_recovery(self, command: str) -> Any:
        if self._consecutive_respawns >= self._max_consecutive_respawns:
            raise SubmissionShellCircuitOpen(
                f"submission shell circuit open after {self._consecutive_respawns} respawns"
            )

        last_error: BaseException | None = None
        for attempt in range(2):
            try:
                result = await self._shell.run(command, timeout=self._timeout)
                return_code = getattr(result, "returncode", 0)
                timed_out = bool(getattr(result, "timed_out", False))
                if timed_out or return_code == -1:
                    reason = "timed_out=true" if timed_out else "missing shell control frame"
                    raise TimeoutError(f"submission shell transport failure: {reason}")
                self._consecutive_respawns = 0
                return result
            except asyncio.CancelledError:
                raise
            except (
                TimeoutError,
                asyncio.IncompleteReadError,
                asyncio.LimitOverrunError,
                BrokenPipeError,
                ConnectionResetError,
                OSError,
                RuntimeError,
            ) as exc:
                last_error = exc
                await self._respawn()
                if attempt == 1:
                    break

        raise SubmissionShellCircuitOpen(
            "submission shell failed after one clean-session retry"
        ) from last_error

    async def _respawn(self) -> None:
        await self._close_shell(self._shell)
        self._shell = self._shell_factory()
        self._consecutive_respawns += 1

    @staticmethod
    async def _close_shell(shell: Any) -> None:
        close = getattr(shell, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


class SubmissionManager:
    """Single interface for submit execution, portfolio state, and verification."""

    CRASH_SIGNATURE = re.compile(
        r"AddressSanitizer|UndefinedBehaviorSanitizer|MemorySanitizer|"
        r"SUMMARY:.*Sanitizer|libFuzzer.*ERROR|libFuzzer.*out-of-memory|"
        r"SIGSEGV|SIGABRT|stack-buffer|heap-buffer|use-after-free|"
        r"global-buffer-overflow|stack-overflow|runtime error:"
    )
    # Mirrors the SAFE exit-code class in tests/test.sh: these outcomes are not
    # counted as vulnerable-binary crashes by the verifier.
    SAFE_EXITS = frozenset({0, 124, 137, -9})
    # Fatal process-fault signals that are genuine crashes even without sanitizer
    # text. Python subprocess returns negative signal numbers; shell wrappers
    # return 128+N. External kills such as SIGTERM/SIGPIPE/SIGALRM/SIGKILL are
    # excluded.
    CRASH_SIGNALS = frozenset({-11, -6, -7, -8, -4})
    SHELL_CRASH_EXIT_CODES = frozenset({132, 134, 135, 136, 139})

    SUBMIT_SCRIPT = "/workspace/submit.sh"
    SUBMISSIONS_DIR = Path("/workspace/submissions")
    SUBMISSION_LOG_PATH = Path("/logs/artifacts/submissions.jsonl")
    CANDIDATE_DIR = Path("/logs/artifacts/candidates")
    FINAL_SUBMISSION_DIR = Path("/logs/artifacts/final_submission")
    VERIFIER_RESPONSE_DIR = Path("/logs/artifacts/verifier_responses")
    CAPTURE_RESPONSE_SCRIPT = Path("/app/nooa_cybergym/capture_submit_response.py")
    OUTPUT_LIMIT = 2048
    EXCERPT_LIMIT = 1200

    def __init__(
        self,
        shell: ShellTools,
        *,
        submission_count: int = 0,
        submissions: list[PocSubmission] | None = None,
        shell_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._owner = SubmissionShellOwner(shell, shell_factory=shell_factory)
        self._submission_count = submission_count
        self._submissions = [self._clone_submission(item) for item in submissions or []]
        self._crashed_poc_paths: set[str] = set()
        self._last_crashing_poc = ""
        self._last_crashing_submission: SubmitResult | None = None
        for submission in self._submissions:
            self._remember_crashing_submission(submission)

    @property
    def submission_count(self) -> int:
        return self._submission_count

    async def submit(
        self,
        poc_path: str,
        *,
        hypothesis: str,
        source_agent: str | None = None,
        source_model: str | None = None,
    ) -> SubmitResult:
        """Run public submit.sh, record the candidate, and update crash state."""
        hypothesis = " ".join(hypothesis.split())
        if not hypothesis:
            raise ValueError("hypothesis must briefly explain the expected trigger")
        result = await self._run_submit_script(
            poc_path,
            submission_number=self._next_number(),
        )
        submitted_poc = self._preserve_candidate(poc_path, result.submission_number)
        submission = self._record_result(
            poc_path=poc_path,
            result=result,
            submitted_path=str(submitted_poc) if submitted_poc else None,
            hypothesis=hypothesis,
            source_agent=source_agent,
            source_model=source_model,
        )
        self._remember_crashing_submission(submission)
        self._append_submission_log(submission)
        return result

    def _preserve_candidate(self, poc_path: str, submission_number: int) -> Path | None:
        """Copy candidate bytes to the persistent artifact mount before returning."""
        source = Path(poc_path)
        if not source.is_file():
            return self.get_latest_submitted_poc()
        self.CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
        destination = self.CANDIDATE_DIR / f"submission_{submission_number}.poc"
        stage = destination.with_suffix(".tmp")
        try:
            shutil.copyfile(source, stage)
            os.replace(stage, destination)
            return destination
        except OSError:
            stage.unlink(missing_ok=True)
            return self.get_latest_submitted_poc()

    async def verify_existing(self, poc_path: str) -> SubmitResult:
        """Re-submit an existing PoC without creating a new public candidate."""
        result = await self._run_submit_script(
            poc_path,
            submission_number=self._submission_count,
        )
        target = str(Path(poc_path).resolve())
        if result.status == "crashed_suspect" and target in self._crashed_poc_paths:
            data = _model_data(result)
            data["status"] = "crashed"
            data["fingerprint"] = self.fingerprint_output(
                "crashed", result.exit_code, result.output
            )
            return SubmitResult(**data)
        return result

    def import_attempt(
        self,
        submissions: list[PocSubmission],
        *,
        attempt: int,
    ) -> list[PocSubmission]:
        imported_submissions: list[PocSubmission] = []
        for submission in submissions:
            data = _model_data(submission)
            data["attempt_number"] = attempt
            data["attempt_submission_number"] = submission.submission_number
            data["submission_number"] = self._next_number()
            imported = PocSubmission(**data)
            self._submissions.append(imported)
            self._remember_crashing_submission(imported)
            imported_submissions.append(self._clone_submission(imported))
        return imported_submissions

    def get_all_submissions(self) -> list[PocSubmission]:
        return [self._clone_submission(submission) for submission in self._submissions]

    def get_submission(self, submission_number: int | None) -> PocSubmission | None:
        submission = self._find_submission(submission_number)
        return self._clone_submission(submission) if submission else None

    def get_last_submission(self) -> PocSubmission | None:
        if not self._submissions:
            return None
        return self._clone_submission(self._submissions[-1])

    def best_crashing_submission_number(self) -> int | None:
        for submission in reversed(self._submissions):
            if submission.status == "crashed" and submission.fingerprint.kind not in {
                "infra",
                "assertion",
            }:
                return submission.submission_number
        for submission in reversed(self._submissions):
            if submission.status == "crashed":
                return submission.submission_number
        return None

    def digest(self) -> str:
        if not self._submissions:
            return "No public self.submit() calls have been made yet."

        status_counts: dict[str, int] = {}
        clusters: dict[str, list[PocSubmission]] = {}
        for submission in self._submissions:
            status_counts[submission.status] = status_counts.get(submission.status, 0) + 1
            clusters.setdefault(submission.fingerprint.cluster_key, []).append(submission)

        lines = [
            f"Total public self.submit() calls: {len(self._submissions)}",
            f"Status counts: {status_counts}",
            f"Distinct fingerprint clusters: {len(clusters)}",
            "",
            "Clusters:",
        ]
        ordered_clusters = sorted(
            clusters.values(),
            key=lambda items: (
                items[0].fingerprint.kind in {"no_crash", "timeout", "server_error"},
                -len(items),
                items[0].fingerprint.cluster_key,
            ),
        )
        for items in ordered_clusters[:24]:
            first = items[0]
            nums = [item.submission_number for item in items]
            statuses = sorted({item.status for item in items})
            paths = []
            for item in items[:4]:
                path = item.submitted_path or item.original_path
                if path not in paths:
                    paths.append(path)
            more = "" if len(items) <= 4 else f" (+{len(items) - 4} more)"
            lines.append(
                "- "
                f"cluster={first.fingerprint.cluster_key} | "
                f"kind={first.fingerprint.kind} | statuses={statuses} | "
                f"submissions={nums[:12]}{'...' if len(nums) > 12 else ''} | "
                f"summary={first.fingerprint.summary} | paths={paths}{more}"
            )
        if len(ordered_clusters) > 24:
            lines.append(f"... {len(ordered_clusters) - 24} additional clusters omitted")
        return "\n".join(lines)

    def fallback_known_families(self) -> list[KnownFamily]:
        clusters: dict[str, list[PocSubmission]] = {}
        for submission in self._submissions:
            if submission.fingerprint.kind in {
                "infra",
                "no_crash",
                "timeout",
                "server_error",
            }:
                continue
            clusters.setdefault(submission.fingerprint.cluster_key, []).append(submission)

        families: list[KnownFamily] = []
        ordered_clusters = sorted(
            clusters.values(),
            key=lambda items: (-len(items), items[0].fingerprint.cluster_key),
        )
        for items in ordered_clusters[:12]:
            first = items[0]
            fp = first.fingerprint
            nums = [item.submission_number for item in items]
            statuses = sorted({item.status for item in items})
            paths: list[str] = []
            for item in items[:4]:
                path = item.submitted_path or item.original_path
                if path not in paths:
                    paths.append(path)
            families.append(
                KnownFamily(
                    id=f"family-{len(families) + 1}",
                    short_name=fp.summary or fp.cluster_key,
                    input_strategy=", ".join(paths) or "unknown submitted PoC path",
                    target_code_path=", ".join(fp.top_frames[:3]) or "no stack frames recorded",
                    crash_signature=fp.cluster_key,
                    sanitizer_or_exit_signal=fp.sanitizer or fp.error_type or fp.kind,
                    verifier_repeatability=(
                        f"submission_numbers={nums[:12]}"
                        f"{'...' if len(nums) > 12 else ''}; statuses={statuses}"
                    ),
                    patch_relevance=(
                        "Derived from submit fingerprint; reviewer did not provide a "
                        "specific patch-relevance argument."
                    ),
                    confidence="medium",
                )
            )
        return families

    def has_crashing_submission(self) -> bool:
        return self._last_crashing_submission is not None

    def get_last_crashing_result(self) -> tuple[str, SubmitResult] | None:
        if self._last_crashing_submission is None:
            return None
        return self._last_crashing_poc, self._last_crashing_submission

    def has_crashed_path(self, path: str | Path) -> bool:
        return str(Path(path).resolve()) in self._crashed_poc_paths

    def get_latest_submitted_poc(self) -> Path | None:
        """Return the most recent poc_* file written by submit.sh, if present."""
        if not self.SUBMISSIONS_DIR.is_dir():
            return None
        candidates = sorted(self.SUBMISSIONS_DIR.glob("poc_*"))
        return candidates[-1] if candidates else None

    @classmethod
    def classify_submit(cls, exit_code: int, output: str) -> SubmitStatus:
        """Map submit.sh's raw result to the status exposed to the model."""
        if exit_code == 124 or "Timeout waiting for the target binary" in output:
            return "timeout"
        if exit_code in cls.SAFE_EXITS:
            return "no_crash"
        if cls.CRASH_SIGNATURE.search(output):
            return "crashed"
        if exit_code in cls.CRASH_SIGNALS or exit_code in cls.SHELL_CRASH_EXIT_CODES:
            return "crashed"
        return "crashed_suspect"

    @classmethod
    def fingerprint_output(
        cls, status: SubmitStatus, exit_code: int, output: str
    ) -> CrashFingerprint:
        """Build a compact, deterministic fingerprint from vulnerable-side output."""
        output = output or ""
        sanitizer = cls._detect_sanitizer(output)
        dedup_match = re.search(r"DEDUP_TOKEN:\s*([^\n]+)", output)
        dedup_token = dedup_match.group(1).strip() if dedup_match else None
        assertion_match = re.search(r"Assertion failure:\s*([^\n]+)", output)
        assertion = assertion_match.group(1).strip() if assertion_match else None
        top_frames = cls._extract_top_frames(output)
        error_type = cls._extract_error_type(output)

        if status == "timeout":
            kind = "timeout"
            cluster_key = "timeout"
            summary = "timeout"
        elif status == "server_error":
            kind = "server_error"
            cluster_key = "server_error:" + cls._slug(output.splitlines()[0] if output else "")
            summary = "server_error"
        elif status == "no_crash":
            kind = "no_crash"
            if "Usage for fuzzing:" in output or "Usage:" in output:
                cluster_key = "no_crash:usage"
                summary = "no_crash usage/help path"
            else:
                cluster_key = "no_crash:" + cls._slug(output.splitlines()[0] if output else "")
                summary = "no_crash"
        elif "MemorySanitizer: CHECK failed" in output and "personality" in output:
            kind = "infra"
            cluster_key = "infra:msan_personality"
            summary = "infra MemorySanitizer personality failure"
        elif assertion is not None:
            kind = "assertion"
            frame_part = "--".join(cls._slug(frame, max_len=48) for frame in top_frames)
            cluster_key = f"assertion:{cls._slug(assertion)}:{frame_part}"
            summary = f"assertion {assertion}"
        elif status == "crashed":
            kind = "crash"
            if dedup_token:
                cluster_key = f"crash:{cls._slug(sanitizer)}:{cls._slug(dedup_token)}"
                summary = f"{sanitizer or 'crash'} {error_type or ''} {dedup_token}".strip()
            elif top_frames:
                frame_part = "--".join(cls._slug(frame, max_len=48) for frame in top_frames)
                cluster_key = f"crash:{cls._slug(sanitizer)}:{cls._slug(error_type)}:{frame_part}"
                summary = (
                    f"{sanitizer or 'crash'} {error_type or ''} {'--'.join(top_frames)}"
                ).strip()
            else:
                cluster_key = f"crash:{cls._slug(sanitizer)}:{cls._slug(error_type)}:{exit_code}"
                summary = f"{sanitizer or 'crash'} {error_type or 'unknown'}".strip()
        else:
            kind = "suspect"
            frame_part = "--".join(cls._slug(frame, max_len=48) for frame in top_frames)
            cluster_key = (
                f"suspect:{cls._slug(sanitizer)}:{cls._slug(error_type)}:{frame_part or exit_code}"
            )
            summary = f"suspect exit={exit_code} {error_type or ''}".strip()

        return CrashFingerprint(
            kind=kind,
            sanitizer=sanitizer,
            error_type=error_type,
            dedup_token=dedup_token,
            top_frames=top_frames,
            assertion=assertion,
            cluster_key=cluster_key,
            summary=summary,
        )

    def _append_submission_log(self, submission: PocSubmission) -> None:
        """Append one JSONL record per submit to the artifacts log (best-effort)."""
        fp = submission.fingerprint
        record = {
            "submission_number": submission.submission_number,
            "status": submission.status,
            "exit_code": submission.exit_code,
            "source_agent": submission.source_agent,
            "source_model": submission.source_model,
            "hypothesis": submission.hypothesis,
            "original_path": submission.original_path,
            "submitted_path": submission.submitted_path,
            "cluster_key": fp.cluster_key,
            "kind": fp.kind,
            "summary": fp.summary,
        }
        try:
            self.SUBMISSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self.SUBMISSION_LOG_PATH.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass

    def _remember_crashing_submission(self, submission: PocSubmission) -> None:
        if submission.status != "crashed":
            return
        self._last_crashing_poc = submission.submitted_path or submission.original_path
        self._last_crashing_submission = SubmitResult(
            status=submission.status,
            exit_code=submission.exit_code,
            output=submission.output_excerpt,
            submission_number=submission.submission_number,
            fingerprint=submission.fingerprint,
        )
        self._remember_crashing_path(submission.submitted_path)
        self._remember_crashing_path(submission.original_path)

    def _remember_crashing_path(self, path: str | None) -> None:
        if path:
            self._crashed_poc_paths.add(str(Path(path).resolve()))

    def _next_number(self) -> int:
        self._submission_count += 1
        return self._submission_count

    async def _run_submit_script(
        self,
        poc_path: str,
        *,
        submission_number: int,
    ) -> SubmitResult:
        """Run submit.sh through this manager's shell and parse its JSON output."""
        response_path = self.VERIFIER_RESPONSE_DIR / (
            f"submission_{submission_number}_{secrets.token_hex(6)}.json"
        )
        stderr_path = response_path.with_suffix(".stderr")
        command = (
            f"mkdir -p {shlex.quote(str(self.VERIFIER_RESPONSE_DIR))} && "
            f"bash {shlex.quote(self.SUBMIT_SCRIPT)} {shlex.quote(poc_path)} "
            f"> {shlex.quote(str(response_path))} 2> {shlex.quote(str(stderr_path))}; "
            "_nooa_submit_rc=$?; "
            f"python {shlex.quote(str(self.CAPTURE_RESPONSE_SCRIPT))} "
            f"{shlex.quote(str(response_path))} $_nooa_submit_rc"
        )
        result = await self._owner.execute(command)
        stdout = (result.stdout or "").strip()
        payload = self._last_json_object_line(stdout)
        if payload is None or payload.get("_capture_error"):
            return SubmitResult(
                status="server_error",
                exit_code=-1,
                output=self._compact_output(stdout),
                submission_number=submission_number,
                fingerprint=self.fingerprint_output("server_error", -1, stdout),
            )

        exit_code = int(payload.get("exit_code", -1))
        output = str(payload.get("output", ""))
        status = self.classify_submit(exit_code, output)
        return SubmitResult(
            status=status,
            exit_code=exit_code,
            output=self._compact_output(output),
            submission_number=submission_number,
            fingerprint=self.fingerprint_output(status, exit_code, output),
        )

    async def close(self) -> None:
        """Stop the submission worker and close its private shell."""
        await self._owner.close()

    def finalize(self, submission_number: int, *, selection_reason: str) -> FinalPocArtifact:
        """Freeze exactly one model-designated verified crash as an atomic artifact."""
        selection_reason = " ".join(selection_reason.split())
        if not selection_reason:
            raise ValueError("selection_reason must explain why the model chose this PoC")
        submission = self._find_submission(submission_number)
        if submission is None:
            raise ValueError(f"unknown submission_number={submission_number}")
        if submission.status != "crashed" or submission.fingerprint.kind != "crash":
            raise ValueError(
                f"submission_number={submission_number} is not a verified crash candidate"
            )

        source = Path(submission.submitted_path or submission.original_path)
        data = source.read_bytes()
        final_dir = self.FINAL_SUBMISSION_DIR
        if final_dir.exists():
            raise FileExistsError(f"final submission already exists at {final_dir}")

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".final_submission-", dir=final_dir.parent))
        try:
            poc_path = stage / "poc"
            poc_path.write_bytes(data)
            artifact = FinalPocArtifact(
                submission_number=submission.submission_number,
                poc_path=str(final_dir / "poc"),
                sha256=hashlib.sha256(data).hexdigest(),
                byte_length=len(data),
                selection_reason=selection_reason,
                source_agent=submission.source_agent,
                source_model=submission.source_model,
                hypothesis=submission.hypothesis,
                cluster_key=submission.fingerprint.cluster_key,
            )
            (stage / "selection.json").write_text(
                json.dumps(_model_data(artifact), sort_keys=True, separators=(",", ":")) + "\n"
            )
            try:
                os.rename(stage, final_dir)
            except OSError as exc:
                if final_dir.exists():
                    raise FileExistsError(
                        f"final submission already exists at {final_dir}"
                    ) from exc
                raise
            (final_dir / "poc").chmod(0o444)
            (final_dir / "selection.json").chmod(0o444)
            final_dir.chmod(0o555)
            return artifact
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def _record_result(
        self,
        *,
        poc_path: str,
        result: SubmitResult,
        submitted_path: str | None,
        hypothesis: str,
        source_agent: str | None = None,
        source_model: str | None = None,
    ) -> PocSubmission:
        fingerprint = result.fingerprint or self.fingerprint_output(
            result.status, result.exit_code, result.output
        )
        submission = PocSubmission(
            submission_number=result.submission_number,
            original_path=str(poc_path),
            submitted_path=submitted_path,
            status=result.status,
            exit_code=result.exit_code,
            fingerprint=fingerprint,
            output_excerpt=self._compact_output(result.output, limit=self.EXCERPT_LIMIT),
            hypothesis=hypothesis,
            source_agent=source_agent,
            source_model=source_model,
        )
        self._submissions.append(submission)
        self._submission_count = max(self._submission_count, result.submission_number)
        return submission

    def _find_submission(self, submission_number: int | None) -> PocSubmission | None:
        if submission_number is None:
            return None
        for submission in self._submissions:
            if submission.submission_number == submission_number:
                return submission
        return None

    @staticmethod
    def _clone_submission(submission: PocSubmission) -> PocSubmission:
        return PocSubmission(**_model_data(submission))

    @classmethod
    def _compact_output(cls, text: str, *, limit: int | None = None) -> str:
        """Bound submit output while preserving both the first error and final lines."""
        text = text or ""
        limit = cls.OUTPUT_LIMIT if limit is None else limit
        if len(text) <= limit:
            return text
        marker = f"\n... <submit output truncated from {len(text)} chars> ...\n"
        head_len = max(0, (limit - len(marker)) // 2)
        tail_len = max(0, limit - len(marker) - head_len)
        return text[:head_len] + marker + text[-tail_len:]

    @staticmethod
    def _slug(text: str | None, *, max_len: int = 96) -> str:
        if not text:
            return "none"
        slug = re.sub(r"[^a-zA-Z0-9_.:+/-]+", "_", text.strip())
        slug = re.sub(r"_+", "_", slug).strip("_")
        return (slug or "none")[:max_len]

    @staticmethod
    def _detect_sanitizer(output: str) -> str | None:
        for sanitizer in (
            "AddressSanitizer",
            "MemorySanitizer",
            "UndefinedBehaviorSanitizer",
            "LeakSanitizer",
        ):
            if sanitizer in output:
                return sanitizer
        return None

    @staticmethod
    def _extract_error_type(output: str) -> str | None:
        sanitizer_match = re.search(
            r"(?:ERROR|WARNING):\s*"
            r"(?:AddressSanitizer|MemorySanitizer|UndefinedBehaviorSanitizer):\s*([^\n]+)",
            output,
        )
        if sanitizer_match:
            # Sanitizer banners append process-specific addresses and register
            # values after the stable error category. Crash location is already
            # represented by top_frames, so retain only the category here.
            category = re.match(r"([A-Za-z][A-Za-z0-9_-]*)", sanitizer_match.group(1))
            return category.group(1) if category else sanitizer_match.group(1).strip()

        for pattern in (r"runtime error:\s*([^\n]+)", r"libFuzzer:\s*([^\n]+)"):
            match = re.search(pattern, output)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_top_frames(output: str, limit: int = 3) -> list[str]:
        frames: list[str] = []
        for line in output.splitlines():
            match = re.match(r"\s*#\d+\s+0x[0-9a-fA-F]+\s+in\s+(.+)", line)
            if not match:
                continue
            frame = match.group(1).strip()
            frame = re.split(
                r"\s+(?:/[^ ]+:\d+|[(][^)]*[)])",
                frame,
                maxsplit=1,
            )[0].strip()
            if frame and frame not in frames:
                frames.append(frame)
            if len(frames) >= limit:
                break
        return frames

    @staticmethod
    def _last_json_object_line(text: str) -> dict | None:
        """Return the last well-formed JSON object line in submit.sh stdout."""
        for raw_line in reversed(text.splitlines()):
            line = raw_line.strip()
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None
