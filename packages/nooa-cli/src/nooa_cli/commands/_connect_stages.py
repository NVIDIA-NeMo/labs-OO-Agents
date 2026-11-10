# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``nooa connect --stage``: run one step of connect without prompts and print JSON.

Each stage calls one part of ``nooa.unifiedllm.connect`` and prints a single
JSON report on standard output (``version``, ``stage``, ``ok``, ``data``,
``error``, ``checks``, ``run_context``, ...). This lets scripts and agents run
the setup step by step: ``discover`` (model listing), ``model-info`` (server
details for one model), ``catalogue`` (OpenRouter catalogue lookup),
``interfaces`` (find which API styles the endpoint accepts), ``plan`` (build
the test calls without sending them), the test-call stages (``routing``,
``decision``, ``tools``, ``reasoning``, ``media``, ``session``, ``all``), and
``save`` (write a planned or tested model entry to a registry file).

The only prompt is the masked API key prompt enabled by ``--prompt-key``.
Saved files from earlier stages (``--discovery-file``, ``--catalogue-file``,
``--input``) can be read back instead of repeating requests; they are size
limited and checked to belong to the same endpoint and model.
"""

STAGES = (
    "discover",
    "model-info",
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


def read_discovery(path, endpoint, *, models_url=None, api_style=None):
    """Return the model records of a saved model listing as a list.

    Kept for older callers; see ``read_discovery_listing`` for the checks.
    """
    return list(
        read_discovery_listing(path, endpoint, models_url=models_url, api_style=api_style).models
    )


def read_discovery_listing(path, endpoint, *, models_url=None, api_style=None):
    """Read a saved ``--stage discover`` result and return it as a ``connect.Discovery``.

    ``path`` is the JSON file (at most 5 MiB, either the full stage report or
    just its ``data``). The listing must belong to ``endpoint``: the same API
    base, or the same base plus ``/v1``. For decision API styles with an
    explicit ``models_url``, any listing from the same origin (scheme, host,
    port) is accepted. If ``models_url`` is given it must match the URL the
    listing was read from.

    Only known, size-limited fields are kept from each model record. The URL
    the listing came from is returned separately from the records.
    Raises ``ValueError`` when the file is too large, malformed, or belongs
    to another endpoint.
    """
    import json
    from pathlib import Path
    from urllib.parse import urlsplit

    from nooa.unifiedllm import connect

    with Path(path).open("rb") as source_file:
        raw = source_file.read(5 * 1024 * 1024 + 1)
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError("Discovery file exceeds 5 MiB")
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("Discovery metadata requires an object")
    data = document.get("data", document)
    if not isinstance(data, dict) or not isinstance(data.get("api_base"), str):
        raise ValueError("Discovery metadata belongs to a different endpoint")
    listed_base = connect.normalize_endpoint(data["api_base"])
    requested_base = connect.normalize_endpoint(endpoint)
    same_base = listed_base == requested_base or (
        not urlsplit(requested_base).path and listed_base == requested_base + "/v1"
    )
    if api_style in connect.DECISION_API_STYLES and models_url is not None:

        def origin(url):
            address = urlsplit(url)
            return (
                address.scheme,
                address.hostname,
                address.port or (443 if address.scheme == "https" else 80),
            )

        # A decision endpoint has no model listing of its own and uses a shared
        # one on the same server, so accept a listing from the same origin. A
        # listing from a different server (which may need a different key) is
        # still rejected.
        same_base = same_base or origin(listed_base) == origin(requested_base)
    if not same_base or (
        models_url is not None and data.get("models_url") != connect.validate_models_url(models_url)
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
    from nooa.unifiedllm.connect._server_details import listing_facts, openrouter_facts

    records = [
        {
            **listing_facts(item),
            **(
                {
                    "metadata_fields": {
                        key: value
                        for key, value in item["metadata_fields"].items()
                        if key in {"context_window", "max_input_tokens", "max_output_tokens"}
                        and isinstance(value, str)
                        and len(value) <= 100
                    }
                }
                if isinstance(item.get("metadata_fields"), dict)
                and len(item["metadata_fields"]) <= 10
                else {}
            ),
            **(
                {"public_model": openrouter_facts(item["public_model"])}
                if source == connect.CATALOGUE_URL and isinstance(item.get("public_model"), dict)
                else {}
            ),
            **({"media_source": source} if connect.media_metadata(item) else {}),
        }
        for item in models
    ]
    return connect.Discovery(data["api_base"], tuple(records), source)


def read_catalogue(path, model, *, catalogue_model=None):
    """Read a saved OpenRouter catalogue file and return the details of one model.

    The file (at most 5 MiB) is either a successful ``--stage catalogue``
    report or its ``data``, and must name the OpenRouter catalogue as its
    source. The model is looked up by exact ID: ``catalogue_model`` if given,
    else ``model``. Exactly one record must match; no similar names are tried,
    because the endpoint's model ID may not equal the OpenRouter ID. Raises
    ``ValueError`` otherwise.
    """
    import json
    from pathlib import Path

    from nooa.unifiedllm import connect

    source = Path(path)
    with source.open("rb") as source_file:
        raw = source_file.read(5 * 1024 * 1024 + 1)
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError("Catalogue file exceeds 5 MiB")
    document = json.loads(raw)
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
    from nooa.unifiedllm.connect._server_details import openrouter_facts

    efforts = (
        (matches[0].get("reasoning") or {}).get("supported_efforts")
        if isinstance(matches[0].get("reasoning"), dict)
        else None
    )
    if isinstance(efforts, list) and len(efforts) > 100:
        raise ValueError("Catalogue reasoning efforts exceed 100 values")
    selected = openrouter_facts(matches[0])
    selected["media_source"] = data["source"]
    return selected


def listing_key(models_url, endpoint, key, key_env=None):
    """Return the API key to send with the model listing request (or None).

    ``key`` is the key for model requests. ``key_env`` is the value of
    ``--discovery-api-key-env``: ``-`` or empty means no key, otherwise the
    named variable is read and must be set. Without ``key_env``, ``key`` is
    used only if the listing URL has the same origin as ``endpoint``; a
    listing on another server raises ``ValueError``, so the model request key
    is never sent to a different server by accident.
    """
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
    cache_enabled=None,
    structured_output_mode=None,
    structured_output_mode_set=False,
    media_inputs=None,
    test_media_inputs=None,
    decision_image_transport=None,
    typesafe_danger_zone_images=None,
    levels_file,
    context_window,
    input_file,
    output,
    yes,
    allow_unverified=False,
    prompt_key=False,
    discovery_file=None,
    catalogue_file=None,
    catalogue_model=None,
    models_url=None,
    listing_style=None,
    discovery_api_key_env=None,
    invalid_options=(),
    working_dir=False,
    no_metadata=False,
    server_metadata="listing",
):
    """Run one ``--stage`` step, print its JSON report and return the exit code.

    The arguments are the command-line options of ``nooa connect``. Option
    combinations that do not fit the stage raise ``click.UsageError`` inside
    the stage and are reported in the JSON like other errors.

    Returns 0 when the stage succeeded, 2 for a usage error and 1 for any
    other failure. The report never contains the API key: key values are
    replaced before printing. The terminal is only read when ``--prompt-key``
    is given (a masked key prompt).
    """
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
    saved = False
    verification = None
    input_diagnostic = None
    known_failed = False
    from ._connect_prompts import parse_media_inputs

    try:
        mode_kwargs = (
            {"structured_output_mode": structured_output_mode} if structured_output_mode_set else {}
        )
        if structured_output_mode_set:
            if stage in {"discover", "catalogue", "model-info", "interfaces", "decision"}:
                raise click.UsageError(
                    "--structured-output-mode is a Completion profile control; use a plan/check/save stage"
                )
            if api_style in {"responses", "systemone", "openai"}:
                raise click.UsageError(
                    "--structured-output-mode requires Completion (chat or anthropic)"
                )
        if cache_enabled is not None and stage in {
            "discover",
            "catalogue",
            "model-info",
            "interfaces",
        }:
            raise click.UsageError("--no-cache is a profile control; use a plan/check/save stage")
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
        if allow_unverified and stage != "save":
            raise click.UsageError("--allow-unverified requires --stage save")
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
                imported_entry = data["entry"]
                if context_window is not None or (
                    discovery_file and imported_entry.get("context_window") is not None
                ):
                    imported_model = imported_entry.get("model_name", "").split("/", 1)[-1]
                    imported_endpoint = imported_entry.get("api_base")
                    imported_listing = (
                        read_discovery_listing(
                            discovery_file,
                            imported_endpoint,
                            models_url=models_url,
                            api_style=imported_entry.get("api_style", "chat"),
                        )
                        if discovery_file
                        else None
                    )
                    server = (
                        next(
                            (m for m in imported_listing.models if m["id"] == imported_model), None
                        )
                        if imported_listing
                        else connect.scoped_endpoint_limits(imported_entry)
                    )
                    imported_entry = connect.apply_context_window(
                        imported_entry,
                        context_window
                        if context_window is not None
                        else imported_entry["context_window"],
                        model=imported_model,
                        endpoint_model=server,
                    )
                alias, entry = (
                    data["alias"],
                    connect.configure_entry(
                        {
                            **imported_entry,
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
                        cache_enabled=cache_enabled,
                        **mode_kwargs,
                    ),
                )
            except ValueError as exc:
                raise click.UsageError(str(exc)) from None
            from nooa.unifiedllm.connect._verification import save_verification

            verification, current_checks = save_verification(entry)
            checks = {
                **(document.get("checks") or {}),
                **current_checks,
            }
            checks = {name: record for name, record in checks.items() if isinstance(record, dict)}
            for name, record in checks.items():
                if name not in current_checks:
                    record["verification_current"] = False
                    if name not in verification["unverified_scopes"]:
                        verification["unverified_scopes"].append(name)
            if verification["state"] == "verified" and verification["unverified_scopes"]:
                verification["state"] = "partial"
                verification["verified"] = False
            verification["checks"] = checks
            input_diagnostic = document.get("diagnostic_prompt")
            known_failed = document.get("ok") is False or verification["state"] == "failed"
            if known_failed:
                if not verification["baseline_verified"]:
                    verification["state"] = "failed"
                elif verification["state"] == "verified":
                    verification["state"] = "partial"
                    verification["unverified_scopes"].append("input_report")
                verification["verified"] = False
            # The command reports success when the file is written, nothing more.
            # The "ok" value in the input report does not mark the entry as tested
            # and does not allow saving a failed result without --allow-unverified.
            if not connect.routing_allows_save(data["entry"], entry):
                view.warning(
                    "Warning: saving a configuration whose current settings have not passed checks.",
                    err=True,
                )
            if known_failed and not allow_unverified:
                raise click.UsageError(
                    "Failed input was not saved; --allow-unverified explicitly acknowledges failed checks"
                )
            path = Path(output)
            existing = {}
            if path.exists():
                with path.open() as source:
                    existing = yaml.safe_load(source) or {}
            if alias in (existing.get("models") or {}):
                if not yes:
                    raise click.UsageError("Alias exists; --yes explicitly permits replacement")
                view.warning(f"Warning: replacing alias {alias!r} in {path}.", err=True)
            expected_alias = connect.alias_target(existing.get("models") or {}, alias)
            connect.write(entry, path, alias=alias, expected_alias=expected_alias)
            saved = True
            data = {"alias": alias, "path": str(path), "entry": entry}
            from ._connect_registry import shadowing_source

            if shadow := shadowing_source(alias, path, extra_priority=bool(working_dir)):
                data["shadowed_by"] = shadow
                view.warning(f"Warning: alias is still resolved from {shadow}.", err=True)
            ok = True
        elif stage == "model-info":
            if not endpoint or not model:
                raise click.UsageError("Model-info requires MODEL and --endpoint")
            key = (
                click.prompt("API key (metadata only)", hide_input=True, err=True)
                if prompt_key
                else os.environ.get(api_key_env)
                if api_key_env
                else None
            )
            discovery_key = listing_key(models_url, endpoint, key, discovery_api_key_env)
            listing = (
                read_discovery_listing(
                    discovery_file, endpoint, models_url=models_url, api_style=api_style
                )
                if discovery_file
                else None
            )
            data = asdict(
                asyncio.run(
                    connect.server_details(
                        endpoint,
                        model,
                        api_style=api_style or "chat",
                        api_key=key,
                        listing_api_key=discovery_key,
                        listing=listing,
                        models_url=models_url,
                        listing_style=listing_style,
                        metadata=not no_metadata,
                        litellm=server_metadata == "litellm",
                    )
                )
            )
            ok = True
        elif stage == "catalogue":
            models = asyncio.run(connect.catalogue())
            matches = connect.match_models(model, models) if model else models
            data = {"source": connect.CATALOGUE_URL, "models": matches}
            if model and not matches:
                # No exact or suffix match. The model ID probably has a gateway
                # prefix (aws/..., bedrock-..., vertex/...) that the catalogue
                # does not use. Report similar IDs for the caller to confirm,
                # but never pick one, as the wizard's "did you mean" does.
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
                    read_discovery_listing(
                        discovery_file, endpoint, models_url=models_url, api_style=style
                    )
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
                        context_window=context_window,
                        cache_enabled=cache_enabled,
                        **mode_kwargs,
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
            except ValueError as exc:
                # These come from checking the options locally, before any request,
                # so they are reported as usage errors.
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
                    from ._connect_input_policy import available_media

                    retained = list(entry.get("media_inputs", []))
                    entry["media_inputs"] = retained + [
                        kind
                        for kind in parse_media_inputs(test_media_inputs)
                        if kind not in retained
                        and kind in available_media(state=None, entry=entry)
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
        data = {"alias": alias, "entry": entry} if stage == "save" and verification else None
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

    # Remove the request bodies of media test calls from the report: they hold
    # the generated sample files. The plan stage keeps them on purpose, since
    # it shows the requests that would be sent.
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
        "diagnostic_prompt": (
            input_diagnostic
            or connect.diagnostic_prompt(stage, entry, checks, run_context=run_context, api_key=key)
        )
        if not ok or known_failed or (verification and verification["state"] == "partial")
        else None,
        **(
            {
                "saved": saved,
                "save_status": "saved" if saved else "not_saved",
                "verification": {
                    **verification,
                    "checks": {name: public_record(record) for name, record in checks.items()},
                }
                if verification
                else None,
            }
            if stage == "save"
            else {}
        ),
    }
    # Either key (model requests or model listing) may appear in listing data or
    # error messages, so remove both from the report.
    report = scrub_report(report, api_key=key, api_key_env=api_key_env)
    report = scrub_report(report, api_key=discovery_key, api_key_env=discovery_api_key_env)
    click.echo(json.dumps(report, ensure_ascii=False))
    return 0 if ok else failure_code
