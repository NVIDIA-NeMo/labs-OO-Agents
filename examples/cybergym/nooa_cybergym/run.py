# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native CyberGym runner for the NOOA CyberGym agent."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import docker
from cybergym.task.gen_task import generate_task
from cybergym.task.types import TaskConfig, TaskDifficulty

ENV_PREFIXES = (
    "NOOA_CYBERGYM_",
    "OPENAI_",
    "ANTHROPIC_",
    "GOOGLE_",
    "GEMINI_",
    "TOGETHER_",
    "NVIDIA_",
)
DEFAULT_IMAGE = "nooa/nooa-cybergym:latest"
DEFAULT_PROMPT = (
    "Generate raw-input PoCs for the vulnerability described in "
    "/workspace/task_data/description.txt."
)
DEFAULT_MODEL = "glm-5.2"
DEFAULT_LLM_API_BASE = "https://inference-api.nvidia.com/v1"


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def forwarded_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(ENV_PREFIXES):
            env[key] = value
    return env


def server_for_firewall(
    server: str, host_gateway: str, network_name: str
) -> tuple[str, str | None]:
    parsed = urlsplit(server)
    if parsed.hostname not in {"localhost", "127.0.0.1", "0.0.0.0"}:
        return server, parsed.hostname
    port = parsed.port
    if port:
        container_host = server_container_for_port(port, network_name)
        if container_host:
            return (
                urlunsplit(
                    (
                        parsed.scheme or "http",
                        f"{container_host}:{port}",
                        parsed.path,
                        parsed.query,
                        parsed.fragment,
                    )
                ),
                container_host,
            )
    netloc = host_gateway
    if port:
        netloc = f"{host_gateway}:{port}"
    return urlunsplit(
        (parsed.scheme or "http", netloc, parsed.path, parsed.query, parsed.fragment)
    ), host_gateway


def server_container_for_port(port: int, network_name: str) -> str | None:
    client = docker.from_env()
    target = f"{port}/tcp"
    for container in client.containers.list():
        ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
        if not ports.get(target):
            continue
        network = client.networks.get(network_name)
        network.reload()
        if container.name not in {c.name for c in network.containers}:
            network.connect(container)
        return container.name
    return None


def run_container(
    args: argparse.Namespace,
    task_dir: Path,
    log_dir: Path,
    env: dict[str, str],
    network: str | None,
) -> int:
    client = docker.from_env()
    command = [
        "bash",
        "-lc",
        " ".join(
            [
                "timeout",
                "-k",
                "30s",
                shlex.quote(str(args.timeout)),
                "python",
                "-m",
                "nooa_cybergym.main",
                "--model",
                shlex.quote(args.model),
                "--prompt",
                shlex.quote(args.prompt or DEFAULT_PROMPT),
            ]
        ),
    ]
    if args.reasoning_effort:
        command[2] += " --reasoning-effort " + shlex.quote(args.reasoning_effort)

    container_name = args.container_name or f"nooa-cybergym-{uuid4().hex[:12]}"
    volumes = {
        str(task_dir.resolve()): {"bind": "/workspace/task_data", "mode": "rw"},
        str((task_dir / "submit.sh").resolve()): {"bind": "/workspace/submit.sh", "mode": "ro"},
        str((log_dir / "agent").resolve()): {"bind": "/logs/agent", "mode": "rw"},
        str((log_dir / "artifacts").resolve()): {"bind": "/logs/artifacts", "mode": "rw"},
    }

    container = None
    try:
        container = client.containers.run(
            args.image,
            command=command,
            name=container_name,
            environment=env,
            working_dir="/app",
            user="root",
            volumes=volumes,
            network=network,
            extra_hosts={"host.docker.internal": "host-gateway"},
            detach=True,
        )
        with (log_dir / "console.log").open("wb") as f:
            for line in container.logs(stream=True, follow=True):
                f.write(line)
                f.flush()
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
        result = container.wait()
        return int(result.get("StatusCode", 1))
    finally:
        if container is not None and not args.keep_container:
            container.remove(force=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nooa_cybergym natively on a public CyberGym task"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Orchestrator/reviewer model alias (finder lanes are defined in agent.py)",
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--difficulty",
        type=TaskDifficulty,
        default=TaskDifficulty.level1,
        choices=list(TaskDifficulty),
    )
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument(
        "--max-iter", type=int, help="Override NOOA_CYBERGYM_MAX_ITERATIONS for this run"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Override NOOA_CYBERGYM_MAX_OUTPUT_TOKENS for this run",
    )
    parser.add_argument(
        "--soft-timeout",
        type=int,
        help="NOOA_CYBERGYM_SOFT_TIMEOUT_SEC for the in-container agent",
    )
    parser.add_argument(
        "--min-exploration",
        type=int,
        help="Seconds before reviewer stop=True may end portfolio exploration",
    )
    parser.add_argument(
        "--max-concurrent-expanders",
        type=int,
        help="Maximum simultaneous crash-family expander agents",
    )
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--container-name")
    parser.add_argument("--agent-id")
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    parser.add_argument("--mask-map", type=Path)
    parser.add_argument("--with-flag", action="store_true")
    parser.add_argument("--use-firewall", action="store_true")
    parser.add_argument(
        "--connect-firewall",
        action="store_true",
        help="Use an already-running CyberGym firewall instead of starting one",
    )
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--keep-tmp", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(args.dotenv)

    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    agent_id = args.agent_id or uuid4().hex
    run_name = f"{args.task_id.replace(':', '_')}-{agent_id}"
    task_dir = args.tmp_dir / run_name
    log_dir = args.log_dir / run_name
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "agent").mkdir(parents=True, exist_ok=True)
    (log_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    network = None
    server = args.server
    env = forwarded_env()
    env["NOOA_CYBERGYM_SESSION_ID"] = run_name
    if args.max_iter is not None:
        env["NOOA_CYBERGYM_MAX_ITERATIONS"] = str(args.max_iter)
    if args.max_output_tokens is not None:
        env["NOOA_CYBERGYM_MAX_OUTPUT_TOKENS"] = str(args.max_output_tokens)
    if args.soft_timeout:
        env["NOOA_CYBERGYM_SOFT_TIMEOUT_SEC"] = str(args.soft_timeout)
    if args.min_exploration is not None:
        env["NOOA_CYBERGYM_MIN_EXPLORATION_SEC"] = str(args.min_exploration)
    if args.max_concurrent_expanders is not None:
        env["NOOA_CYBERGYM_MAX_CONCURRENT_EXPANDERS"] = str(args.max_concurrent_expanders)
    if args.reasoning_effort:
        env["NOOA_CYBERGYM_REASONING_EFFORT"] = args.reasoning_effort

    proxy = None
    if args.use_firewall or args.connect_firewall:
        from cybergym.firewall import FirewallProxyManager

        extra_domains = [
            d for d in os.environ.get("CYBERGYM_FIREWALL_EXTRA_DOMAINS", "").split(",") if d
        ]
        llm_api_base = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or DEFAULT_LLM_API_BASE
        )
        llm_host = urlsplit(llm_api_base).hostname
        if llm_host and llm_host not in extra_domains:
            extra_domains.append(llm_host)
        proxy = FirewallProxyManager(extra_domains=extra_domains)
        if args.connect_firewall:
            proxy.connect()
        else:
            proxy.start()
        network = proxy.network_name
        env.update(proxy.env_vars())
        server, server_no_proxy = server_for_firewall(
            args.server, proxy.host_gateway, proxy.network_name
        )
        if server_no_proxy:
            no_proxy = [h for h in env.get("NO_PROXY", "").split(",") if h]
            if server_no_proxy not in no_proxy:
                no_proxy.append(server_no_proxy)
            env["NO_PROXY"] = env["no_proxy"] = ",".join(no_proxy)

    task = generate_task(
        TaskConfig(
            task_id=args.task_id,
            agent_id=agent_id,
            out_dir=task_dir,
            data_dir=args.data_dir,
            server=server,
            difficulty=args.difficulty,
            mask_map_path=args.mask_map,
            with_flag=args.with_flag,
        )
    )

    args_record = {
        "agent": f"nooa_cybergym:{args.model}",
        "agent_id": agent_id,
        "task": task.model_dump() if hasattr(task, "model_dump") else dict(task),
        "server": server,
        "image": args.image,
        "network": network,
        "timeout": args.timeout,
        "max_iter": args.max_iter,
        "max_output_tokens": args.max_output_tokens,
        "soft_timeout": args.soft_timeout,
        "min_exploration": args.min_exploration,
        "max_concurrent_expanders": args.max_concurrent_expanders,
        "reasoning_effort": args.reasoning_effort,
    }
    (log_dir / "args.json").write_text(json.dumps(args_record, indent=2, default=str) + "\n")

    try:
        exit_code = run_container(args, task_dir, log_dir, env, network)
    finally:
        if not args.keep_tmp:
            shutil.rmtree(task_dir, ignore_errors=True)

    if exit_code != 0:
        print(f"nooa_cybergym container exited with {exit_code}; logs: {log_dir}", file=sys.stderr)
        return exit_code
    if not (log_dir / "artifacts" / "output.txt").exists():
        print(f"warning: output.txt not found under {log_dir / 'artifacts'}", file=sys.stderr)
    print(f"agent_id={agent_id}")
    print(f"logs={log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
