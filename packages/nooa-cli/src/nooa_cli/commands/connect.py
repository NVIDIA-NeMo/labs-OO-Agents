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
    help="Connection preset: nvidia, openai, anthropic, google, openrouter, or custom.",
)
@click.option("--as", "alias", help="Local model alias to save (otherwise prompted).")
@click.option("--endpoint", help="API base URL (otherwise prompted).")
@click.option(
    "--api-style",
    type=click.Choice(["chat", "responses", "anthropic"]),
    help="Request interface used by this model route.",
)
@click.option(
    "--api-key-env",
    help="Environment variable name, never the key itself (otherwise prompted).",
)
@click.option("--catalogue-model", help="Explicit OpenRouter model ID to use as metadata.")
@click.option(
    "--discovery-file",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON output from --stage discover; reuse endpoint limits without another request.",
)
@click.option("--prompt-key", is_flag=True, help="Read a masked key; offer to save it at the end.")
@click.option("--no-catalogue", is_flag=True, help="Do not fetch public model metadata.")
@click.option(
    "--reasoning-template",
    type=click.Choice(["effort", "adaptive", "budget", "toggle", "thinking"]),
    help="Candidate request shape; verify support on the actual endpoint.",
)
@click.option("--levels", help="Comma-separated candidate labels to probe.")
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
    help="All checks, configured routing only, or no inference calls.",
)
@click.option("--no-probe", is_flag=True, help="Save an untested entry without model calls.")
@click.option(
    "--budget-tokens",
    type=click.IntRange(min=1),
    help="Shared estimated-token budget for all checks (default: unlimited); never increased after approval.",
)
@click.option(
    "--output-tokens",
    type=click.IntRange(1, 4096),
    default=200,
    show_default=True,
    help="Cap for interface discovery only, before the saved reply budget is chosen.",
)
@click.option(
    "--reasoning-output-tokens",
    type=click.IntRange(1, 32768),
    default=4096,
    show_default=True,
    help="Deprecated compatibility option; reasoning checks now use the configured reply cap.",
)
@click.option(
    "--max-tokens",
    "--reply-tokens",
    "reply_tokens",
    type=click.IntRange(min=1),
    help="Reply cap saved for agents and sent by configured checks.",
)
@click.option("--show-config", is_flag=True, help="Show full YAML details before saving.")
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
    api_style,
    api_key_env,
    prompt_key,
    catalogue_model,
    discovery_file,
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
    reply_tokens,
    show_config,
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
        if not resolved_dir.is_dir():
            raise click.UsageError(f"--working-dir directory {str(resolved_dir)!r} does not exist.")
        output = str(resolved_dir / paths.DIR_NAME / "llm_config.yaml")
    if stage:
        from ._connect_stages import run_stage

        code = run_stage(
            stage,
            model=model,
            alias=alias,
            endpoint=endpoint,
            api_style=api_style,
            api_key_env=api_key_env,
            budget_tokens=budget_tokens,
            output_tokens=output_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
            reply_tokens=reply_tokens,
            levels_file=levels_file,
            context_window=context_window,
            input_file=input_file,
            output=output,
            yes=yes,
            prompt_key=prompt_key,
            discovery_file=discovery_file,
            invalid_options=[
                name
                for name, used in {
                    "--no-probe": no_probe,
                    "--probe": probe != "all",
                    "--provider": provider,
                    "--catalogue-model": catalogue_model,
                    "--reasoning-template": reasoning_template,
                    "--levels": levels,
                    "--show-config": show_config,
                    "--edit-model": edit_model is not None,
                }.items()
                if used
            ],
        )
        raise click.exceptions.Exit(code)
    if input_file:
        raise click.UsageError("--input requires --stage save")
    from ._connect_wizard import run_wizard

    run_wizard(
        model=model,
        edit_model=edit_model,
        provider=provider,
        alias=alias,
        endpoint=endpoint,
        api_style=api_style,
        api_key_env=api_key_env,
        prompt_key=prompt_key,
        catalogue_model=catalogue_model,
        discovery_file=discovery_file,
        no_catalogue=no_catalogue,
        reasoning_template=reasoning_template,
        levels=levels,
        levels_file=levels_file,
        context_window=context_window,
        probe=probe,
        no_probe=no_probe,
        budget_tokens=budget_tokens,
        output_tokens=output_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        reply_tokens=reply_tokens,
        show_config=show_config,
        output=output,
        yes=yes,
    )
