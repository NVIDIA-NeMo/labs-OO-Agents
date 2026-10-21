# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal approval and display for the shared nooa.unifiedllm.connect library."""

import click

from ._connect_stages import STAGES


@click.command()
@click.argument("model", required=False)
@click.option(
    "--edit-model",
    is_flag=False,
    flag_value="",
    default=None,
    help="Edit an existing alias; omit NAME to select from the registry.",
)
@click.option(
    "--stage", type=click.Choice(STAGES), help="Run one non-interactive stage; emit JSON and exit."
)
@click.option(
    "--input",
    "input_file",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON plan/result to save with --stage save.",
)
@click.option(
    "--provider",
    help="Connection preset: nvidia, hub (NVIDIA Inference Hub), openai, anthropic, google, openrouter, or custom.",
)
@click.option("--as", "alias", help="Local model alias to save (otherwise prompted).")
@click.option("--endpoint", help="API base URL, or full decision request URL (otherwise prompted).")
@click.option(
    "--systemone-endpoint",
    help="Full System One POST URL for mixed interface discovery; never inferred for custom servers.",
)
@click.option(
    "--share-interface-key",
    is_flag=True,
    help="Explicitly allow mixed interface checks to share the inference key across origins.",
)
@click.option(
    "--api-style",
    type=click.Choice(["chat", "responses", "anthropic", "systemone", "openai"]),
    help=(
        "Request interface used by this model route. systemone and openai are "
        "decision-model interfaces (System One, or OpenAI's /v1/decisions)."
    ),
)
@click.option(
    "--api-key-env",
    help="Environment variable name, never the key itself (otherwise prompted).",
)
@click.option(
    "--catalogue-model", help="Explicit OpenRouter metadata ID; stages require --catalogue-file."
)
@click.option(
    "--catalogue-file",
    type=click.Path(exists=True, dir_okay=False),
    help="Read a saved OpenRouter catalogue offline for plan/check stages; exact mapping only.",
)
@click.option(
    "--discovery-file",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON output from --stage discover; reuse endpoint limits without another request.",
)
@click.option(
    "--models-url", help="Full shared catalogue GET URL; required for custom decision listing."
)
@click.option(
    "--listing-style",
    type=click.Choice(["chat", "responses", "anthropic"]),
    help="Catalogue auth/pagination convention, independent of --api-style.",
)
@click.option(
    "--discovery-api-key-env",
    help="Catalogue key variable; - for no authentication. Required for a different origin.",
)
@click.option("--prompt-key", is_flag=True, help="Read a masked key; offer to save it at the end.")
@click.option("--no-catalogue", is_flag=True, help="Do not fetch public model metadata.")
@click.option(
    "--reasoning-template",
    type=click.Choice(["effort", "adaptive", "budget", "toggle", "thinking"]),
    help="Reasoning settings to try on this server.",
)
@click.option("--levels", help="Comma-separated reasoning levels to check.")
@click.option(
    "--levels-file",
    type=click.Path(exists=True, dir_okay=False),
    help="YAML mapping from labels to complete request settings.",
)
@click.option(
    "--context-window",
    type=click.IntRange(min=1),
    help="User-supplied context limit, not a request allocation.",
)
@click.option(
    "--probe",
    type=click.Choice(["all", "minimal", "none"]),
    default="all",
    show_default=True,
    help="All checks, connection only, or no model calls.",
)
@click.option("--no-probe", is_flag=True, help="Save an untested entry without model calls.")
@click.option(
    "--budget-tokens",
    type=click.IntRange(min=1),
    help="Estimated token budget for checks (default: unlimited); not a spending cap.",
)
@click.option(
    "--output-tokens",
    type=click.IntRange(1, 4096),
    # Same value as nooa.unifiedllm.connect.DISCOVERY_OUTPUT_TOKENS, written out so
    # the CLI need not import nooa.unifiedllm at load; a test keeps them equal.
    default=2048,
    show_default=True,
    help="Chat-family interface discovery cap only, before the saved reply budget is chosen; no cap for native decisions.",
)
@click.option(
    "--reasoning-output-tokens",
    type=click.IntRange(1, 32768),
    default=4096,
    show_default=True,
    help="Deprecated compatibility option; reasoning checks now use the configured reply cap.",
)
@click.option(
    "--reasoning-level",
    default=None,
    help=(
        "Effort/label probed when interfaces are also checked for reasoning. "
        "Defaults to the first --levels entry, or 'medium'."
    ),
)
@click.option(
    "--max-tokens",
    "--reply-tokens",
    "reply_tokens",
    type=click.IntRange(min=1),
    help="Chat-family reply token limit to save and use for checks; native decisions have no reply cap.",
)
@click.option(
    "--media-inputs",
    help="Manually declare comma-separated image,audio,video,pdf inputs (unverified); native decisions support image only. Empty or - means text-only.",
)
@click.option(
    "--test-media-inputs",
    help="Test selected image,audio,video,pdf formats with generated samples in the approved budget; native decisions support image only with a valid transport.",
)
@click.option(
    "--decision-image-transport",
    type=click.Choice(["json", "typesafe_multipart"]),
    help="Explicit native decision image transport; never inferred from the endpoint.",
)
@click.option(
    "--typesafe-danger-zone-images/--no-typesafe-danger-zone-images",
    default=None,
    help="Explicit experimental TypeSafe multipart image opt-in.",
)
@click.option("--show-config", is_flag=True, help="Show full YAML details before saving.")
@click.option(
    "--show-diagnostics",
    is_flag=True,
    help="Show the full troubleshooting handoff when wizard checks fail; stage JSON always includes it.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    help="Registry path; defaults to the user llm_config.yaml. Mutually exclusive with --working-dir.",
)
@click.option(
    "--working-dir",
    "-w",
    # No exists=True here: click validates the raw argument before this
    # command ever runs, so a literal "~" would fail as "does not exist"
    # even though it's a real directory — expand it ourselves below, then
    # check existence on the expanded path.
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    help=(
        "Save to this project's registry (<working-dir>/.nooa/llm_config.yaml) "
        "instead of the user-global one — the same file `nooa tui -w <working-dir>` "
        "reads. Mutually exclusive with --output."
    ),
)
@click.option(
    "--yes",
    is_flag=True,
    help="Approve the checks and save without prompting; supply the connection options.",
)
def command(
    model,
    edit_model,
    stage,
    input_file,
    provider,
    alias,
    endpoint,
    systemone_endpoint,
    share_interface_key,
    api_style,
    api_key_env,
    prompt_key,
    catalogue_model,
    catalogue_file,
    discovery_file,
    models_url,
    listing_style,
    discovery_api_key_env,
    no_catalogue,
    reasoning_template,
    levels,
    levels_file,
    context_window,
    probe,
    no_probe,
    budget_tokens,
    output_tokens,
    reasoning_output_tokens,
    reasoning_level,
    reply_tokens,
    media_inputs,
    test_media_inputs,
    decision_image_transport,
    typesafe_danger_zone_images,
    show_config,
    show_diagnostics,
    output,
    working_dir,
    yes,
):
    """Walk through model setup, check the connection, and save an alias.

    Run `uv run nooa connect` with no arguments for guided setup. Flags prefill
    the answers; --yes requires MODEL, --as and either --provider or an
    explicit --endpoint and --api-style.
    MODEL is the exact endpoint model ID, without a LiteLLM routing prefix.
    """
    from ._connect_prompts import parse_media_inputs

    try:
        parse_media_inputs(media_inputs)
        selected_tests = parse_media_inputs(test_media_inputs)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from None
    if test_media_inputs is not None:
        if not selected_tests:
            raise click.UsageError("--test-media-inputs requires at least one format")
        if media_inputs is not None:
            raise click.UsageError("Choose --test-media-inputs OR --media-inputs")
        if no_probe or probe != "all":
            raise click.UsageError("Media tests require --probe all without --no-probe")
        if stage and stage not in {"plan", "media", "all"}:
            raise click.UsageError("Media tests use --stage plan, media or all")
    if working_dir:
        if output:
            raise click.UsageError("--working-dir and --output are mutually exclusive.")
        if stage and stage != "save":
            raise click.UsageError(
                "--working-dir only applies to the full interactive run or --stage save; "
                "other stages emit JSON and never write a registry file."
            )
        from pathlib import Path

        from nooa import paths

        resolved_dir = Path(working_dir).expanduser().resolve()
        # click's own file_okay/dir_okay checks only run when its (raw,
        # unexpanded) argument happens to already exist, so a "~"-relative
        # path skips them entirely — check both cases explicitly here,
        # after expansion, rather than reporting "does not exist" for a
        # real path that just isn't a directory.
        if not resolved_dir.exists():
            raise click.UsageError(f"--working-dir directory {str(resolved_dir)!r} does not exist.")
        if not resolved_dir.is_dir():
            raise click.UsageError(f"--working-dir path {str(resolved_dir)!r} is not a directory.")
        output = str(resolved_dir / paths.DIR_NAME / "llm_config.yaml")
    from nooa.tracing import quiet_trace_target

    # Checks create model clients, which may auto-enable tracing; its
    # "OTel tracing enabled" line would land mid-way through the wizard output.
    with quiet_trace_target():
        if stage:
            from ._connect_stages import run_stage

            code = run_stage(
                stage,
                model=model,
                alias=alias,
                endpoint=endpoint,
                systemone_endpoint=systemone_endpoint,
                share_interface_key=share_interface_key,
                api_style=api_style,
                api_key_env=api_key_env,
                budget_tokens=budget_tokens,
                output_tokens=output_tokens,
                reasoning_output_tokens=reasoning_output_tokens,
                reasoning_template=reasoning_template if stage == "interfaces" else None,
                reasoning_level=reasoning_level,
                reply_tokens=reply_tokens,
                media_inputs=media_inputs,
                test_media_inputs=test_media_inputs,
                decision_image_transport=decision_image_transport,
                typesafe_danger_zone_images=typesafe_danger_zone_images,
                levels_file=levels_file,
                context_window=context_window,
                input_file=input_file,
                output=output,
                yes=yes,
                prompt_key=prompt_key,
                catalogue_file=catalogue_file,
                catalogue_model=catalogue_model,
                discovery_file=discovery_file,
                models_url=models_url,
                listing_style=listing_style,
                discovery_api_key_env=discovery_api_key_env,
                invalid_options=[
                    name
                    for name, used in {
                        "--no-probe": no_probe,
                        "--probe": probe != "all",
                        "--provider": provider,
                        "--catalogue-model": catalogue_model and not catalogue_file,
                        "--reasoning-template": reasoning_template and stage != "interfaces",
                        "--reasoning-level": reasoning_level and stage != "interfaces",
                        "--levels": levels,
                        "--show-config": show_config,
                        "--edit-model": edit_model is not None,
                    }.items()
                    if used
                ],
                working_dir=bool(working_dir),
            )
            raise click.exceptions.Exit(code)
        if input_file:
            raise click.UsageError("--input requires --stage save")
        if catalogue_file:
            raise click.UsageError("--catalogue-file requires a plan/check --stage")
        from ._connect_wizard import run_wizard

        run_wizard(
            model=model,
            edit_model=edit_model,
            provider=provider,
            alias=alias,
            endpoint=endpoint,
            systemone_endpoint=systemone_endpoint,
            share_interface_key=share_interface_key,
            api_style=api_style,
            api_key_env=api_key_env,
            prompt_key=prompt_key,
            catalogue_model=catalogue_model,
            discovery_file=discovery_file,
            models_url=models_url,
            listing_style=listing_style,
            discovery_api_key_env=discovery_api_key_env,
            no_catalogue=no_catalogue,
            reasoning_template=reasoning_template,
            reasoning_level=reasoning_level,
            levels=levels,
            levels_file=levels_file,
            context_window=context_window,
            probe=probe,
            no_probe=no_probe,
            budget_tokens=budget_tokens,
            output_tokens=output_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
            reply_tokens=reply_tokens,
            media_inputs=media_inputs,
            test_media_inputs=test_media_inputs,
            decision_image_transport=decision_image_transport,
            typesafe_danger_zone_images=typesafe_danger_zone_images,
            show_config=show_config,
            show_diagnostics=show_diagnostics,
            output=output,
            working_dir=working_dir,
            yes=yes,
        )
