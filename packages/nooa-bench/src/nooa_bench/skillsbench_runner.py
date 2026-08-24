# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one SkillsBench task locally through NOOA in selected skill conditions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "openai/openai/openai/gpt-5.2"
DEFAULT_TASK = "citation-check"
DEFAULT_JOBS_DIR = Path("jobs/nooa-skillsbench")
PAIRED_CONDITIONS = ("no_skill", "text_skill")
CONDITIONS = (*PAIRED_CONDITIONS, "library_skill")
SOURCE_COPY_IGNORED_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "jobs",
    "skillsbench",
}
SOURCE_COPY_SECRET_SUFFIXES = (".key", ".pem", ".p12", ".pfx")


@dataclass
class ConditionResult:
    """Host-side summary for one nooa SkillsBench rollout."""

    condition: str
    rollout_dir: str
    passed: bool
    reward: float | None
    error: str | None
    verifier_error: str | None
    agent_return_code: int
    activated_skills: list[str] | None = None
    skipped: bool = False


@dataclass(frozen=True)
class ConditionSettings:
    """Condition-specific settings shared by BenchFlow and the NOOA runner."""

    rollout_skill_mode: str
    rollout_skills_dir: Path | None
    runner_skill_mode: str
    runner_skills_dir: str | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def _credentials(env_file: Path) -> dict[str, str]:
    file_values = _load_env_file(env_file)
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or file_values.get("API_KEY")
    )
    api_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("API_URL")
        or file_values.get("API_URL")
    )
    env: dict[str, str] = {}
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    if api_url:
        env["OPENAI_BASE_URL"] = api_url
    return env


def _ensure_benchflow_importable(skillsbench_dir: Path) -> None:
    try:
        import benchflow  # noqa: F401

        return
    except ImportError:
        pass

    subprocess.run(["uv", "sync", "--locked"], cwd=skillsbench_dir, check=True)
    import benchflow  # noqa: F401


def _reexec_with_skillsbench_uv(skillsbench_dir: Path) -> int:
    env = os.environ.copy()
    env["_NOOA_SKILLSBENCH_UV_REEXEC"] = "1"
    repo_root = _repo_root()
    pythonpath_roots = [
        repo_root / "src",
        repo_root / "packages" / "nooa-bench" / "src",
        repo_root / "packages" / "nooa-cli" / "src",
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath = os.pathsep.join(str(path) for path in pythonpath_roots)
    env["PYTHONPATH"] = pythonpath if not existing_pythonpath else f"{pythonpath}{os.pathsep}{existing_pythonpath}"
    cmd = [
        "uv",
        "run",
        "--project",
        str(skillsbench_dir),
        "python",
        "-m",
        "nooa_bench.skillsbench_runner",
        *sys.argv[1:],
    ]
    return subprocess.run(cmd, env=env).returncode


def _copy_nooa_source(src: Path) -> Path:
    """Stage a compact source tree for upload into the BenchFlow sandbox."""
    tmp = Path(tempfile.mkdtemp(prefix="nooa-bench-src-"))
    dst = tmp / "nooa-src"

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            lower = name.lower()
            if (
                name in SOURCE_COPY_IGNORED_NAMES
                or name == ".env"
                or name.startswith(".env.")
                or lower.endswith((".pyc", *SOURCE_COPY_SECRET_SUFFIXES))
            ):
                ignored.add(name)
        return ignored

    shutil.copytree(src, dst, ignore=ignore)
    return dst


def _read_task_instruction(task_dir: Path) -> str:
    from benchflow.task.document import TaskDocument

    task_md = task_dir / "task.md"
    if task_md.is_file():
        return TaskDocument.from_path(task_md).instruction.strip()
    instruction_md = task_dir / "instruction.md"
    if instruction_md.is_file():
        return instruction_md.read_text().strip()
    raise FileNotFoundError(f"Task missing task.md or instruction.md: {task_dir}")


def _task_agent_timeout(task_dir: Path, default: int = 900) -> int:
    try:
        from benchflow.task import Task
    except ImportError:
        timeout = _task_agent_timeout_from_frontmatter(task_dir / "task.md")
    else:
        timeout = Task(task_dir).config.agent.timeout_sec
    if timeout is None:
        return default
    return max(1, int(timeout))


def _task_agent_timeout_from_frontmatter(task_md: Path) -> float | None:
    if not task_md.is_file():
        return None
    lines = task_md.read_text().splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    in_agent = False
    for line in lines[1:]:
        if line.strip() == "---":
            return None
        if line and not line[0].isspace():
            in_agent = line.strip() == "agent:"
            continue
        if in_agent and line.strip().startswith("timeout_sec:"):
            value = line.split(":", 1)[1].strip().strip("'\"")
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _condition_settings(task_dir: Path, condition: str) -> ConditionSettings:
    if condition == "no_skill":
        return ConditionSettings(
            rollout_skill_mode="no-skill",
            rollout_skills_dir=None,
            runner_skill_mode="no_skill",
            runner_skills_dir=None,
        )
    if condition == "text_skill":
        skills_dir = task_dir / "environment" / "skills"
        if not skills_dir.is_dir():
            raise FileNotFoundError(f"text_skill condition requires skills dir: {skills_dir}")
        return ConditionSettings(
            rollout_skill_mode="with-skill",
            rollout_skills_dir=skills_dir,
            runner_skill_mode="text_skill",
            runner_skills_dir="/skills",
        )
    if condition == "library_skill":
        skills_dir = task_dir / "environment" / "skills"
        if not skills_dir.is_dir():
            raise FileNotFoundError(f"library_skill condition requires skills dir: {skills_dir}")
        return ConditionSettings(
            rollout_skill_mode="with-skill",
            rollout_skills_dir=skills_dir,
            runner_skill_mode="library_skill",
            runner_skills_dir="/skills",
        )
    raise ValueError(f"Unknown condition: {condition!r}")


def _skill_dirs(skills_root: Path) -> list[Path]:
    """Return immediate child TextSkill directories under a SkillsBench skills root."""
    skill_dirs: list[Path] = []
    if not skills_root.is_dir():
        return skill_dirs
    for entry in sorted(skills_root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "SKILL.md").is_file() or (entry / "skill.md").is_file():
            skill_dirs.append(entry)
    return skill_dirs


def _translate_task_library_skills(task_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Translate task-bundled TextSkills into package-backed LibrarySkills."""
    from nooa.tools.skill_translator import TextSkillTranslator

    source_skills_dir = task_dir / "environment" / "skills"
    skill_dirs = _skill_dirs(source_skills_dir)
    if not skill_dirs:
        raise FileNotFoundError(f"no SKILL.md directories found in {source_skills_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    translator = TextSkillTranslator()
    summaries: list[dict[str, Any]] = []
    for skill_dir in skill_dirs:
        result = translator.translate(skill_dir, output_dir)
        report = translator.validate_package(result.package_dir)
        summary = {
            "source_dir": str(skill_dir),
            "package_dir": str(result.package_dir),
            "package_name": result.package_name,
            "registry_name": result.registry_name,
            "class_name": result.class_name,
            "files_written": result.files_written,
            "omitted_scripts": [item.model_dump() for item in result.omitted_scripts],
            "validation": report.model_dump(mode="json"),
        }
        summaries.append(summary)
        if not report.ok:
            errors = "; ".join(report.errors) or "unknown validation failure"
            raise RuntimeError(f"translated LibrarySkill failed validation for {skill_dir}: {errors}")

    (output_dir / "translation_summary.json").write_text(json.dumps(summaries, indent=2))
    return summaries


def _build_nooa_runner_args(
    *,
    instruction: str,
    model: str,
    settings: ConditionSettings,
    agent_env: dict[str, str],
) -> list[str]:
    args = [
        "/opt/nooa-bench-venv/bin/python",
        "-m",
        "nooa_bench.runner",
        "--instruction",
        instruction,
        "--model",
        model,
        "--agent-type",
        "bench",
        "--working-dir",
        "/root",
        "--skill-mode",
        settings.runner_skill_mode,
    ]
    if settings.runner_skills_dir:
        args.extend(["--skills-dir", settings.runner_skills_dir])
    if agent_env.get("OPENAI_BASE_URL"):
        args.extend(["--api-base", agent_env["OPENAI_BASE_URL"]])
    return args


def _install_nooa_command(source_sandbox_dir: str) -> str:
    """Return the in-sandbox command that bootstraps uv and installs NOOA."""
    return (
        "set -eu; "
        "if ! command -v uv >/dev/null 2>&1; then "
        "if ! command -v curl >/dev/null 2>&1; then "
        "if command -v apt-get >/dev/null 2>&1; then "
        "apt-get update && apt-get install -y curl ca-certificates; "
        "else echo 'curl is required to install uv' >&2; exit 127; "
        "fi; "
        "fi; "
        "curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh; "
        "fi; "
        "export PATH=/root/.local/bin:$PATH; "
        f"cd {shlex.quote(source_sandbox_dir)}; "
        "UV_PROJECT_ENVIRONMENT=/opt/nooa-bench-venv "
        "uv sync --package nooa-bench"
    )


async def _install_nooa(env: Any, source_sandbox_dir: str) -> None:
    cmd = _install_nooa_command(source_sandbox_dir)
    result = await env.exec(cmd, user="root", timeout_sec=900)
    if result.return_code != 0:
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        raise RuntimeError(
            "NOOA install failed: "
            f"return_code={result.return_code} "
            f"stdout={stdout[-4000:]!r} stderr={stderr[-4000:]!r}"
        )


async def _download_agent_logs(env: Any, agent_dir: Path) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    try:
        await env.download_dir("/logs/agent", agent_dir)
    except Exception as exc:
        (agent_dir / "download_error.txt").write_text(str(exc))


def _read_activated_skills(agent_dir: Path) -> list[str] | None:
    result_path = agent_dir / "result.json"
    if not result_path.is_file():
        return None
    try:
        payload = json.loads(result_path.read_text())
    except json.JSONDecodeError:
        return None
    skills = payload.get("activated_skills")
    if not isinstance(skills, list) or not all(isinstance(item, str) for item in skills):
        return None
    return skills


async def _run_condition(
    *,
    task_dir: Path,
    jobs_dir: Path,
    job_name: str,
    condition: str,
    model: str,
    sandbox: str,
    agent_env: dict[str, str],
    repo_src: Path,
) -> ConditionResult:
    from benchflow.rollout import Rollout, RolloutConfig

    settings = _condition_settings(task_dir, condition)
    agent_timeout = _task_agent_timeout(task_dir)
    rollout_skills_dir = settings.rollout_skills_dir
    if condition == "library_skill":
        rollout_skills_dir = jobs_dir / job_name / "translated_library_skills"
        _translate_task_library_skills(task_dir, rollout_skills_dir)

    config = RolloutConfig(
        task_path=task_dir,
        environment=sandbox,
        sandbox_user="agent",
        jobs_dir=jobs_dir,
        job_name=job_name,
        rollout_name=f"{task_dir.name}__{condition}",
        agent="nooa-harbor",
        model=model,
        agent_env=agent_env,
        skill_mode=settings.rollout_skill_mode,
        skills_dir=rollout_skills_dir,
        timeout=agent_timeout,
    )
    rollout = await Rollout.create(config)
    source_tmp: Path | None = None
    agent_return_code = 1
    activated_skills: list[str] | None = None
    try:
        await rollout.setup()
        await rollout.start()
        await rollout.install_agent()

        source_tmp = _copy_nooa_source(repo_src)
        await rollout._env.upload_dir(source_tmp, "/tmp/nooa-src")
        await _install_nooa(rollout._env, "/tmp/nooa-src")

        instruction = _read_task_instruction(task_dir)
        args = _build_nooa_runner_args(
            instruction=instruction,
            model=model,
            settings=settings,
            agent_env=agent_env,
        )

        command = " ".join(shlex.quote(arg) for arg in args)
        started = time.monotonic()
        result = await rollout._env.exec(
            f"{command} > /logs/agent/nooa-run.log 2>&1",
            user="agent",
            env=agent_env,
            timeout_sec=agent_timeout,
        )
        agent_return_code = int(result.return_code)
        rollout._timing["agent_execution"] = time.monotonic() - started
        await _download_agent_logs(rollout._env, rollout._rollout_paths.agent_dir)
        activated_skills = _read_activated_skills(rollout._rollout_paths.agent_dir)
        if agent_return_code != 0:
            rollout._error = (
                f"NOOA runner failed with exit code {agent_return_code}. "
                "See agent/nooa-run.log."
            )
        rollout._trajectory = [
            {
                "type": "nooa_runner",
                "condition": condition,
                "return_code": agent_return_code,
                "log_path": "/logs/agent/nooa-run.log",
            }
        ]
        rollout._agent_name = "nooa-bench"
        await rollout.verify()
    except Exception as exc:
        trace = traceback.format_exc()
        if rollout._error:
            rollout._error = f"{rollout._error}; post-run error: {exc}\n{trace}"
        else:
            rollout._error = f"{exc}\n{trace}"
    finally:
        await rollout.cleanup()
        if source_tmp is not None:
            shutil.rmtree(source_tmp.parent, ignore_errors=True)

    result = rollout.result or rollout._build_result()
    reward = (result.rewards or {}).get("reward") if result.rewards else None
    from benchflow._utils.scoring import classify_result_outcome

    passed = (
        classify_result_outcome(
            {
                "rewards": result.rewards,
                "error": result.error,
                "verifier_error": result.verifier_error,
            }
        )
        == "passed"
    )
    return ConditionResult(
        condition=condition,
        rollout_dir=str(rollout._rollout_dir) if rollout._rollout_dir else "",
        passed=passed,
        reward=reward,
        error=result.error,
        verifier_error=result.verifier_error,
        agent_return_code=agent_return_code,
        activated_skills=activated_skills,
    )


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _run_manifest(
    *,
    task: str,
    model: str,
    sandbox: str,
    conditions: tuple[str, ...],
    skillsbench_dir: Path,
    jobs_dir: Path,
    repo_src: Path,
) -> dict[str, Any]:
    return {
        "task": task,
        "model": model,
        "sandbox": sandbox,
        "conditions": list(conditions),
        "skillsbench_dir": str(skillsbench_dir),
        "jobs_dir": str(jobs_dir),
        "repo_commit": _git_commit(repo_src),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _condition_result_from_dict(payload: dict[str, Any]) -> ConditionResult:
    return ConditionResult(
        condition=str(payload.get("condition", "")),
        rollout_dir=str(payload.get("rollout_dir", "")),
        passed=bool(payload.get("passed", False)),
        reward=payload.get("reward"),
        error=payload.get("error"),
        verifier_error=payload.get("verifier_error"),
        agent_return_code=int(payload.get("agent_return_code", 1)),
        activated_skills=payload.get("activated_skills"),
        skipped=bool(payload.get("skipped", False)),
    )


def _load_existing_results(jobs_dir: Path, job_name: str) -> dict[str, ConditionResult]:
    summary_path = jobs_dir / job_name / "summary.json"
    if not summary_path.is_file():
        return {}
    try:
        payload = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return {}
    results = payload.get("results")
    if not isinstance(results, list):
        return {}
    loaded: dict[str, ConditionResult] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        result = _condition_result_from_dict(item)
        if result.condition:
            loaded[result.condition] = result
    return loaded


def _write_summary(
    jobs_dir: Path,
    job_name: str,
    results: list[ConditionResult],
    manifest: dict[str, Any] | None = None,
) -> None:
    job_dir = jobs_dir / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_name": job_name,
        "manifest": manifest or {},
        "results": [result.__dict__ for result in results],
    }
    (job_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    lines = ["# NOOA SkillsBench One-Task Summary", ""]
    if manifest:
        lines.extend(
            [
                "## Manifest",
                f"- task: {manifest['task']}",
                f"- model: {manifest['model']}",
                f"- sandbox: {manifest['sandbox']}",
                f"- repo_commit: {manifest['repo_commit']}",
                "",
            ]
        )
    for result in results:
        lines.append(f"## {result.condition}")
        if result.skipped:
            lines.append("- skipped: True")
        lines.append(f"- passed: {result.passed}")
        lines.append(f"- reward: {result.reward}")
        lines.append(f"- rollout_dir: {result.rollout_dir}")
        lines.append(f"- agent_return_code: {result.agent_return_code}")
        if result.activated_skills is not None:
            lines.append(f"- activated_skills: {result.activated_skills}")
        if result.error:
            lines.append(f"- error: {result.error}")
        if result.verifier_error:
            lines.append(f"- verifier_error: {result.verifier_error}")
        lines.append("")
    (job_dir / "summary.md").write_text("\n".join(lines))


async def _amain(args: argparse.Namespace) -> int:
    skillsbench_dir = Path(args.skillsbench_dir).resolve()
    task_dir = skillsbench_dir / "tasks" / args.task
    if not task_dir.is_dir():
        raise FileNotFoundError(f"SkillsBench task not found: {task_dir}")
    if not os.environ.get("_NOOA_SKILLSBENCH_UV_REEXEC"):
        try:
            import benchflow  # noqa: F401
        except ImportError:
            return _reexec_with_skillsbench_uv(skillsbench_dir)
    _ensure_benchflow_importable(skillsbench_dir)

    agent_env = _credentials(Path(args.env_file).resolve())
    if not agent_env.get("OPENAI_API_KEY"):
        print(
            "No API credentials found. Set OPENAI_API_KEY/API_KEY or provide --env-file.",
            flush=True,
        )
        return 2

    jobs_dir = Path(args.jobs_dir).resolve()
    job_name = args.job_name or f"{args.task}__nooa__{time.strftime('%Y-%m-%d__%H-%M-%S')}"
    if args.condition == "both":
        conditions = PAIRED_CONDITIONS
    elif args.condition == "all":
        conditions = CONDITIONS
    else:
        conditions = (args.condition,)
    manifest = _run_manifest(
        task=args.task,
        model=args.model,
        sandbox=args.sandbox,
        conditions=conditions,
        skillsbench_dir=skillsbench_dir,
        jobs_dir=jobs_dir,
        repo_src=_repo_root(),
    )
    existing_results = _load_existing_results(jobs_dir, job_name) if args.resume else {}

    results: list[ConditionResult] = []
    for condition in conditions:
        if condition in existing_results:
            result = existing_results[condition]
            result.skipped = True
            results.append(result)
            print(
                f"{condition}: skipped=True passed={result.passed} reward={result.reward} "
                f"rollout_dir={result.rollout_dir}",
                flush=True,
            )
            continue
        try:
            result = await _run_condition(
                task_dir=task_dir,
                jobs_dir=jobs_dir,
                job_name=job_name,
                condition=condition,
                model=args.model,
                sandbox=args.sandbox,
                agent_env=agent_env,
                repo_src=_repo_root(),
            )
        except Exception as exc:
            result = ConditionResult(
                condition=condition,
                rollout_dir="",
                passed=False,
                reward=None,
                error=str(exc),
                verifier_error=None,
                agent_return_code=1,
            )
        results.append(result)
        print(
            f"{condition}: passed={result.passed} reward={result.reward} "
            f"rollout_dir={result.rollout_dir}",
            flush=True,
        )

    _write_summary(jobs_dir, job_name, results, manifest=manifest)
    return 0 if all(result.passed for result in results) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skillsbench-dir", default="skillsbench")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sandbox", default="docker")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--jobs-dir", default=str(DEFAULT_JOBS_DIR))
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--condition", choices=("both", "all", *CONDITIONS), default="both")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing condition results from summary.json for this job name.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
