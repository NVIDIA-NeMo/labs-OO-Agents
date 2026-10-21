# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-interactive JSON frontend for the public Connect library."""

STAGES = (
    "discover",
    "catalogue",
    "interfaces",
    "plan",
    "routing",
    "decision",
    "tools",
    "reasoning",
    "media",
    "session",
    "all",
    "save",
)


def read_discovery(path, endpoint, *, models_url=None):
    """Compatibility list view of a validated discovery listing."""
    return list(read_discovery_listing(path, endpoint, models_url=models_url).models)


def read_discovery_listing(path, endpoint, *, models_url=None):
    """Read bounded metadata with listing source separate from model records."""
    import json
    from pathlib import Path

    from nooa.unifiedllm import connect

    with Path(path).open("rb") as source_file:
        raw = source_file.read(5 * 1024 * 1024 + 1)
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError("Discovery file exceeds 5 MiB")
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("Discovery metadata requires an object")
    data = document.get("data", document)
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("api_base"), str)
        or (
            data.get("models_url") != connect.validate_models_url(models_url)
            if models_url is not None
            else connect.normalize_endpoint(data["api_base"]).removesuffix("/v1")
            != connect.normalize_endpoint(endpoint).removesuffix("/v1")
        )
    ):
        raise ValueError("Discovery metadata belongs to a different endpoint")
    models = data.get("models")
    if (
        not isinstance(models, list)
        or len(models) > 5000
        or not all(
            isinstance(m, dict)
            and isinstance(m.get("id"), str)
            and m["id"].strip()
            and len(m["id"]) <= 500
            and not any(ord(c) < 32 for c in m["id"])
            for m in models
        )
        or len({m["id"] for m in models}) != len(models)
    ):
        raise ValueError("Discovery metadata requires at most 5,000 models with unique valid IDs")
    if data.get("models_url") is not None and not isinstance(data["models_url"], str):
        raise ValueError("Discovery models_url must be a full HTTP(S) URL string")
    source = (
        models_url
        or data.get("models_url")
        or connect.normalize_endpoint(data["api_base"]) + "/models"
    )
    source = connect.validate_models_url(source)
    records = [
        {
            **{
                key: value
                for key, value in item.items()
                if key not in {"media_model_id", "limit_sources", "endpoint_limits", "media_source"}
            },
            **({"media_source": source} if connect.media_metadata(item) else {}),
        }
        for item in models
    ]
    return connect.Discovery(data["api_base"], tuple(records), source)


def read_catalogue(path, model, *, catalogue_model=None):
    """Read explicitly sourced OpenRouter facts; never guess a native model match."""
    import json
    from copy import deepcopy
    from pathlib import Path

    from nooa.unifiedllm import connect

    source = Path(path)
    if source.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("Catalogue file exceeds 5 MiB")
    document = json.loads(source.read_text())
    if not isinstance(document, dict):
        raise ValueError("Catalogue file requires an object with source and models")
    if "stage" in document:
        if (
            document.get("stage") != "catalogue"
            or document.get("version") != 1
            or document.get("ok") is not True
        ):
            raise ValueError("Catalogue file must be a successful version 1 catalogue report")
        data = document.get("data")
    else:
        data = document
    if not isinstance(data, dict) or data.get("source") != connect.CATALOGUE_URL:
        raise ValueError("Catalogue source must be " + connect.CATALOGUE_URL)
    models = data.get("models")
    if (
        not isinstance(models, list)
        or len(models) > 5000
        or not all(
            isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and item["id"].strip()
            and len(item["id"]) <= 500
            and not any(ord(c) < 32 for c in item["id"])
            for item in models
        )
    ):
        raise ValueError("Catalogue file requires at most 5,000 model objects with valid IDs")
    selected_id = catalogue_model or model
    matches = [item for item in models if item["id"] == selected_id]
    if len(matches) != 1:
        raise ValueError(
            "Catalogue model must match exactly once; supply --catalogue-model for an explicit OpenRouter ID mapping"
        )
    # Internal attribution keys are never provider facts. Do not accept a forged
    # media_model_id/limit_sources/endpoint_limits from a local catalogue record.
    selected = {
        key: deepcopy(value)
        for key, value in matches[0].items()
        if key
        in {
            "id",
            "description",
            "context_length",
            "top_provider",
            "reasoning",
            "default_parameters",
            "architecture",
            "input_modalities",
            "capabilities",
            "pricing",
        }
    }
    # Same safe shaping as the public catalogue reader. Chat plan uses only
    # allowlisted limits/reasoning/reply recommendations; decision plan keeps
    # bounded descriptive/media facts only. Neither copies arbitrary parameters.
    for key in ("top_provider", "reasoning", "default_parameters"):
        if not isinstance(selected.get(key), dict):
            selected[key] = {}
    efforts = selected["reasoning"].get("supported_efforts")
    if isinstance(efforts, list) and len(efforts) > 100:
        raise ValueError("Catalogue reasoning efforts exceed 100 values")
    selected["reasoning"]["supported_efforts"] = (
        [
            value
            for value in efforts
            if isinstance(value, str) and value.strip() and len(value) <= 200
        ]
        if isinstance(efforts, list)
        else []
    )
    if not isinstance(selected["reasoning"].get("default_effort"), str):
        selected["reasoning"].pop("default_effort", None)
    selected["media_source"] = data["source"]
    return selected


def listing_key(models_url, endpoint, key, key_env=None):
    """Do not forward an inference credential to a different catalogue origin."""
    import os
    import re
    from urllib.parse import urlsplit

    from nooa.unifiedllm import connect

    if models_url is not None:
        connect.validate_models_url(models_url)
    if key_env is not None:
        if key_env in {"", "-"}:
            return None
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env):
            raise ValueError("--discovery-api-key-env must be a variable name or -")
        value = os.environ.get(key_env)
        if not value:
            raise ValueError("Configured discovery credential variable is unset or empty")
        return value
    if models_url is not None:

        def origin(url):
            value = urlsplit(url)
            return (
                value.scheme,
                value.hostname,
                value.port or (443 if value.scheme == "https" else 80),
            )

        if origin(endpoint) != origin(models_url):
            raise ValueError("A different catalogue origin requires --discovery-api-key-env (or -)")
    return key


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
    systemone_endpoint=None,
    share_interface_key=False,
    reasoning_output_tokens=4096,
    reasoning_template=None,
    reasoning_level="medium",
    reply_tokens=None,
    media_inputs=None,
    test_media_inputs=None,
    decision_image_transport=None,
    typesafe_danger_zone_images=None,
    levels_file,
    context_window,
    input_file,
    output,
    yes,
    prompt_key=False,
    discovery_file=None,
    catalogue_file=None,
    catalogue_model=None,
    models_url=None,
    listing_style=None,
    discovery_api_key_env=None,
    invalid_options=(),
    working_dir=False,
):
    """Return JSON; only an explicit --prompt-key enables a masked stdin prompt."""
    import asyncio
    import json
    import os
    from dataclasses import asdict
    from pathlib import Path

    import click
    import yaml

    from nooa.unifiedllm import connect

    from . import _connect_view as view

    entry = {}
    checks = {}
    data = None
    ok = False
    failure_code = 1
    key = None
    discovery_key = None
    from ._connect_prompts import parse_media_inputs

    try:
        if catalogue_file and stage not in {
            "plan",
            "routing",
            "decision",
            "tools",
            "reasoning",
            "media",
            "session",
            "all",
        }:
            raise click.UsageError("--catalogue-file is only for plan/check stages")
        if catalogue_model and not catalogue_file:
            raise click.UsageError("--catalogue-model in stage mode requires --catalogue-file")
        if systemone_endpoint:
            connect.normalize_endpoint(systemone_endpoint)
            if stage != "interfaces":
                raise click.UsageError("--systemone-endpoint is for interface discovery only")
        declared_media = parse_media_inputs(media_inputs)
        if media_inputs is not None and stage in {"discover", "catalogue", "interfaces"}:
            raise click.UsageError(
                "--media-inputs is a connection declaration; use a plan or save stage"
            )
        if invalid_options:
            raise click.UsageError(
                "For this stage, use --endpoint, --api-style and --levels-file. These wizard options are unavailable: "
                + ", ".join(invalid_options)
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
            try:
                alias, entry = (
                    data["alias"],
                    connect.configure_entry(
                        {
                            **data["entry"],
                            **(
                                {"decision_image_transport": decision_image_transport}
                                if decision_image_transport is not None
                                else {}
                            ),
                            **(
                                {"typesafe_danger_zone_images": typesafe_danger_zone_images}
                                if typesafe_danger_zone_images is not None
                                else {}
                            ),
                            **(
                                {"media_inputs": declared_media}
                                if declared_media is not None
                                else {}
                            ),
                        },
                        reply_tokens=reply_tokens,
                    ),
                )
            except ValueError as exc:
                raise click.UsageError(str(exc)) from None
            if document.get("ok") is False:
                click.echo("Warning: saving a configuration whose checks did not pass.", err=True)
            path = Path(output)
            existing = {}
            if path.exists():
                with path.open() as source:
                    existing = yaml.safe_load(source) or {}
            if alias in (existing.get("models") or {}):
                if not yes:
                    raise click.UsageError("Alias exists; --yes explicitly permits replacement")
                click.echo(f"Warning: replacing alias {alias!r} in {path}.", err=True)
            connect.write(entry, path, alias=alias)
            data = {"alias": alias, "path": str(path), "entry": entry}
            from ._connect_registry import shadowing_source

            if shadow := shadowing_source(alias, path, extra_priority=bool(working_dir)):
                data["shadowed_by"] = shadow
                click.echo(f"Warning: alias is still resolved from {shadow}.", err=True)
            ok = True
        elif stage == "catalogue":
            models = asyncio.run(connect.catalogue())
            matches = connect.match_models(model, models) if model else models
            data = {"source": connect.CATALOGUE_URL, "models": matches}
            if model and not matches:
                # An exact/suffix match found nothing — likely a gateway
                # prefix (aws/..., bedrock-..., vertex/...) the catalogue
                # never records. Surface fuzzy suggestions for the caller to
                # confirm; never auto-select one, matching the wizard's own
                # "did you mean" behavior for the same situation.
                fuzzy = connect.fuzzy_match_models(model, models)
                if fuzzy:
                    data["fuzzy_models"] = fuzzy
            ok = True
        else:
            if not endpoint:
                raise click.UsageError("Stage requires --endpoint")
            style = api_style or "chat"
            if style in connect.DECISION_API_STYLES and stage == "discover" and not models_url:
                raise click.UsageError(
                    "Decision endpoints do not list decision models; pass the exact MODEL "
                    "or supply --models-url for a shared catalogue"
                )
            if style in connect.DECISION_API_STYLES and discovery_file and not models_url:
                raise click.UsageError("--discovery-file for decision routes requires --models-url")
            if style in connect.DECISION_API_STYLES and (reasoning_template or reasoning_level):
                raise click.UsageError(
                    "--reasoning-template and --reasoning-level do not apply to "
                    f"--api-style {style}"
                )
            key_env = api_key_env or ""
            budget = connect.DEFAULT_CHECK_BUDGET if budget_tokens is None else budget_tokens
            levels = None
            if levels_file:
                with Path(levels_file).open() as source:
                    levels = yaml.safe_load(source)
            try:
                listing = (
                    read_discovery_listing(discovery_file, endpoint, models_url=models_url)
                    if discovery_file and stage != "discover"
                    else None
                )
                proposal = (
                    None
                    if stage == "discover"
                    else connect.plan(
                        alias or "candidate",
                        model or "candidate",
                        style,
                        endpoint,
                        key_env,
                        catalogue=read_catalogue(
                            catalogue_file, model, catalogue_model=catalogue_model
                        )
                        if catalogue_file
                        else None,
                        budget_tokens=budget,
                        output_tokens=output_tokens,
                        reasoning_output_tokens=reasoning_output_tokens,
                        reply_tokens=reply_tokens,
                        media_inputs=declared_media,
                        decision_image_transport=decision_image_transport,
                        typesafe_danger_zone_images=typesafe_danger_zone_images,
                        test_media_inputs=parse_media_inputs(test_media_inputs),
                        reasoning_levels=levels,
                        session_checks=stage in {"session", "all"}
                        and style not in connect.DECISION_API_STYLES,
                        endpoint_model=next((m for m in listing.models if m["id"] == model), None)
                        if listing
                        else None,
                        endpoint_models_url=listing.models_url if listing else None,
                    )
                )
                entry = proposal.entry if proposal is not None else {}
                if "advertised_model" in entry.get("provenance", {}) and models_url:
                    entry["provenance"]["models_url"] = connect.validate_models_url(models_url)
                if context_window and proposal is not None:
                    entry["context_window"] = context_window
                    proposal = connect.refresh_plan(proposal)
                    entry = proposal.entry
            except ValueError as exc:
                # Local plan/configuration validation, not request execution.
                raise click.UsageError(str(exc)) from None
            key = (
                click.prompt("API key (used only for this check)", hide_input=True, err=True)
                if prompt_key and stage != "plan"
                else os.environ.get(key_env)
                if key_env
                else None
            )
            if stage not in {"plan", "discover"} and key_env and not key:
                raise click.UsageError("Set the configured key variable before running this check")
            if stage not in {"discover", "plan"} and not model:
                raise click.UsageError("Stage requires MODEL")
            if stage == "discover":
                discovery_key = listing_key(models_url, endpoint, key, discovery_api_key_env)
                if key_env and not discovery_key and discovery_api_key_env is None:
                    raise click.UsageError(
                        "Set the configured key variable before running this check"
                    )
                data = asdict(
                    asyncio.run(
                        connect.discover(
                            endpoint,
                            api_style=style,
                            api_key=discovery_key,
                            models_url=models_url,
                            listing_style=listing_style,
                        )
                    )
                )
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
                        reasoning_template=reasoning_template,
                        reasoning_level=reasoning_level or "medium",
                        reasoning_output_tokens=reasoning_output_tokens,
                        styles=(api_style,)
                        if api_style
                        else ("chat", "responses", "anthropic", "systemone")
                        if systemone_endpoint
                        else ("chat", "responses", "anthropic"),
                        endpoints={"systemone": systemone_endpoint} if systemone_endpoint else None,
                        allow_cross_origin_credentials=share_interface_key,
                    ):
                        if isinstance(event, connect.InterfaceResult):
                            result = event
                    return result

                with view.quiet_provider_messages():
                    result = asyncio.run(interfaces())
                data = asdict(result)
                data["accepted"] = list(result.accepted)
                data["accepted_client_types"] = list(result.accepted_client_types)
                data["reasoning_observed"] = result.reasoning_observed
                data["recommended_style"] = result.recommended_style
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
                if test_media_inputs is not None:
                    retained = list(entry.get("media_inputs", []))
                    entry["media_inputs"] = retained + [
                        kind
                        for kind in parse_media_inputs(test_media_inputs)
                        if kind not in retained
                        and entry["provenance"]["probes"].get(f"media:{kind}", {}).get("outcome")
                        == "passed"
                    ]
                data = asdict(result)
                checks = {
                    **entry["provenance"]["probes"],
                    **entry["provenance"].get("session_checks", {}),
                }
                ok = connect.verdict(entry).ok
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
        api_style=api_style,
        api_key_env=api_key_env,
        api_key=key,
        budget=total_budget,
        remaining=max(0, total_budget - charged) if charged is not None else None,
        output_tokens=output_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        stage=stage,
        discovery_succeeded=ok if stage == "discover" else None,
    )
    from nooa.unifiedllm.connect._diagnostics import scrub_report
    from nooa.unifiedllm.connect._records import public_record

    # Public results retain safe media facts, not inline generated payloads.
    # Plan previews intentionally include their proposed synthetic request bodies.
    if stage != "plan" and isinstance(data, dict):

        def hide_media_requests(value):
            if not isinstance(value, dict):
                return
            for name, child in value.items():
                if name == "probes" and isinstance(child, dict):
                    for probe_name, record in child.items():
                        if probe_name.startswith("media:"):
                            record.pop("request", None)
                hide_media_requests(child)

        hide_media_requests(data)
    report = {
        "version": 1,
        "stage": stage,
        "ok": bool(ok),
        "data": data,
        "error": error,
        "run_context": run_context,
        "warnings": connect.entry_warnings(entry) if entry else [],
        "checks": {name: public_record(record) for name, record in checks.items()},
        "diagnostic_prompt": None
        if ok
        else connect.diagnostic_prompt(stage, entry, checks, run_context=run_context, api_key=key),
    }
    # Both credentials can be reflected by catalogue metadata or local error paths.
    report = scrub_report(report, api_key=key, api_key_env=api_key_env)
    report = scrub_report(report, api_key=discovery_key, api_key_env=discovery_api_key_env)
    click.echo(json.dumps(report, ensure_ascii=False))
    return 0 if ok else failure_code
