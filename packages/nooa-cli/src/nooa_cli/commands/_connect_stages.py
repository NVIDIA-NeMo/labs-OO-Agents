# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-interactive JSON frontend for the public Connect library."""

STAGES = (
    "discover",
    "catalogue",
    "interfaces",
    "plan",
    "routing",
    "tools",
    "reasoning",
    "session",
    "all",
    "save",
)


def run_stage(
    stage,
    *,
    model,
    alias,
    endpoint,
    api_style,
    api_key_env,
    budget_tokens,
    output_tokens,
    reasoning_output_tokens=4096,
    reply_tokens=None,
    levels_file,
    context_window,
    input_file,
    output,
    yes,
    prompt_key=False,
    invalid_options=(),
):
    """Return JSON; only an explicit --prompt-key enables a masked stdin prompt."""
    import asyncio
    import json
    import os
    from dataclasses import asdict
    from pathlib import Path

    import click
    import yaml

    from nooa import connect

    from . import _connect_view as view

    entry = {}
    checks = {}
    data = None
    ok = False
    failure_code = 1
    key = None
    try:
        if invalid_options:
            raise click.UsageError(
                "Stage mode does not use "
                + ", ".join(invalid_options)
                + "; provide explicit stage options (--endpoint, --api-style, --levels-file)."
            )
        if stage != "save" and (input_file or output):
            raise click.UsageError(
                "--input and --output are for stage save; redirect JSON stdout for other stages"
            )
        if stage == "save":
            if not input_file or not output:
                raise click.UsageError("Save requires --input and --output")
            document = json.loads(Path(input_file).read_text())
            data = document.get("data", document)
            if (
                not isinstance(data, dict)
                or not isinstance(data.get("alias"), str)
                or not data["alias"]
                or not isinstance(data.get("entry"), dict)
            ):
                raise click.UsageError("Save input requires an alias string and an entry mapping")
            alias, entry = (
                data["alias"],
                connect.configure_entry(data["entry"], reply_tokens=reply_tokens),
            )
            if "api_key" in entry:
                raise click.UsageError("Use api_key_env, never a literal key")
            if document.get("ok") is False:
                click.echo("Warning: saving a configuration whose checks did not pass.", err=True)
            path = Path(output)
            existing = {}
            if path.exists():
                with path.open() as source:
                    existing = yaml.safe_load(source) or {}
            if alias in existing.get("models", {}):
                if not yes:
                    raise click.UsageError("Alias exists; --yes explicitly permits replacement")
                click.echo(f"Warning: replacing alias {alias!r} in {path}.", err=True)
            connect.write(entry, path, alias=alias)
            data = {"alias": alias, "path": str(path), "entry": entry}
            from ._connect_registry import shadowing_source

            if shadow := shadowing_source(alias, path):
                data["shadowed_by"] = shadow
                click.echo(f"Warning: alias is still resolved from {shadow}.", err=True)
            ok = True
        elif stage == "catalogue":
            models = asyncio.run(connect.catalogue())
            data = {"models": connect.match_models(model, models) if model else models}
            ok = True
        else:
            if not endpoint:
                raise click.UsageError("Stage requires --endpoint")
            style = api_style or "chat"
            key_env = api_key_env or ""
            budget = connect.DEFAULT_CHECK_BUDGET if budget_tokens is None else budget_tokens
            levels = None
            if levels_file:
                with Path(levels_file).open() as source:
                    levels = yaml.safe_load(source)
            proposal = connect.plan(
                alias or "candidate",
                model or "candidate",
                style,
                endpoint,
                key_env,
                budget_tokens=budget,
                output_tokens=output_tokens,
                reasoning_output_tokens=reasoning_output_tokens,
                reply_tokens=reply_tokens,
                reasoning_levels=levels,
                session_checks=stage in {"session", "all"},
            )
            entry = proposal.entry
            if context_window:
                entry["context_window"] = context_window
            key = (
                click.prompt("API key (used only for this check)", hide_input=True, err=True)
                if prompt_key and stage != "plan"
                else os.environ.get(key_env)
                if key_env
                else None
            )
            if stage != "plan" and key_env and not key:
                raise click.UsageError("Configured credential variable is unset or empty")
            if stage not in {"discover", "plan"} and not model:
                raise click.UsageError("Stage requires MODEL")
            if stage == "discover":
                data = asdict(asyncio.run(connect.discover(endpoint, api_style=style, api_key=key)))
                ok = True
            elif stage == "plan":
                if not model or not api_style:
                    raise click.UsageError("Plan requires MODEL and --api-style")
                data = asdict(proposal)
                ok = True
            elif stage == "interfaces":

                async def interfaces():
                    result = None
                    async for event in connect.check_interfaces(
                        alias or "candidate",
                        model,
                        endpoint,
                        key_env,
                        budget_tokens=budget,
                        output_tokens=output_tokens,
                        api_key=key,
                    ):
                        if isinstance(event, connect.InterfaceResult):
                            result = event
                    return result

                with view.quiet_provider_messages():
                    result = asyncio.run(interfaces())
                data = asdict(result)
                data["accepted"] = list(result.accepted)
                checks = {
                    name: r.entry["provenance"]["probes"]["routing"]
                    for name, r in result.results.items()
                }
                ok = bool(result.accepted)
            else:
                if not api_style:
                    raise click.UsageError("Check requires --api-style")
                if stage == "reasoning" and not entry.get("reasoning_levels"):
                    raise click.UsageError("Reasoning requires declared levels in --levels-file")
                with view.quiet_provider_messages():
                    result = asyncio.run(connect.check_stage(proposal, stage, api_key=key))
                entry = result.entry
                data = asdict(result)
                checks = {
                    **entry["provenance"]["probes"],
                    **entry["provenance"].get("session_checks", {}),
                }
                if stage in {"session", "all"}:
                    required = [checks.get(name, {}) for name in ("cache", "reasoning_retention")]
                    ok = all(r.get("outcome") == "confirmed" for r in required)
                    if stage == "all":
                        ok &= all(
                            r.get("outcome") == "accepted"
                            for r in entry["provenance"]["probes"].values()
                        )
                        ok &= bool(checks.get("tools", {}).get("tool_observed"))
                        ok &= not connect.unobserved_reasoning_levels(entry)
                else:
                    ok = bool(checks) and all(
                        r.get("outcome") == "accepted" for r in checks.values()
                    )
                    if stage == "tools":
                        ok &= all(r.get("tool_observed", False) for r in checks.values())
                    elif stage == "reasoning":
                        ok &= not connect.unobserved_reasoning_levels(entry)
        error = None
    except Exception as exc:
        error = {"type": type(exc).__name__}
        if isinstance(exc, click.UsageError):
            failure_code = 2
            error["message"] = exc.message
        else:
            error["message"] = view.local_failure(exc, api_key=key, api_key_env=api_key_env)
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            error["status_code"] = status
        checks["stage"] = {
            "outcome": "failed",
            "error": type(exc).__name__,
            "detail": error["message"],
            **({"status_code": status} if isinstance(status, int) else {}),
        }
        data = None
    from ._connect_registry import diagnostic_context

    total_budget = connect.DEFAULT_CHECK_BUDGET if budget_tokens is None else budget_tokens
    charged = (
        data.get(
            "tokens_charged_to_budget",
            entry.get("provenance", {}).get("tokens_charged_to_budget", 0),
        )
        if isinstance(data, dict)
        else None
    )
    run_context = diagnostic_context(
        target=output,
        alias=alias,
        model=model,
        endpoint=endpoint,
        api_key_env=api_key_env,
        api_key=key,
        budget=total_budget,
        remaining=max(0, total_budget - charged) if charged is not None else None,
        output_tokens=output_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        stage=stage,
        discovery_succeeded=ok if stage == "discover" else None,
    )
    report = {
        "version": 1,
        "stage": stage,
        "ok": bool(ok),
        "data": data,
        "error": error,
        "run_context": run_context,
        "checks": {
            name: {k: v for k, v in record.items() if k not in {"request"}}
            for name, record in checks.items()
        },
        "diagnostic_prompt": None
        if ok
        else connect.diagnostic_prompt(stage, entry, checks, run_context=run_context),
    }
    click.echo(json.dumps(report, ensure_ascii=False))
    return 0 if ok else failure_code
