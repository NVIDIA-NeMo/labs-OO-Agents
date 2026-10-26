# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read the model registry files as NOOA loads them, without creating any clients.

The registry is a chain of YAML files (bundled defaults, the user file, the
project file, files named in ``NEMO_OO_LLM_CONFIG``); a later file overrides
an alias defined in an earlier one. The wizard uses this module to list
existing aliases for editing, to suggest API key variable names already used
for the same server, to warn when a saved alias would be overridden by another
file, and to describe the run in troubleshooting output.
"""


def entries(extra_path=None):
    """Return ``{alias: (entry, file)}`` for every alias in the registry chain.

    Each alias maps to the entry that wins (the one in the last file defining
    it) and the file it came from. ``extra_path``, if it exists, is read last,
    so it overrides all other files. Raises ``ValueError`` if a file's
    ``models`` value is not a mapping. Model clients are not created.
    """
    from pathlib import Path

    import yaml

    from nooa.llm_config import llm_config_chain

    paths = llm_config_chain()
    if extra_path is not None and extra_path.exists():
        paths = [p for p in paths if p.resolve() != extra_path.resolve()] + [extra_path]
    resolved = {}
    for path in paths:
        with Path(path).open() as source:
            data = yaml.safe_load(source) or {}
        if isinstance(data, dict) and data.get("models") is None:
            data["models"] = {}
        if not isinstance(data, dict) or not isinstance(data.get("models", {}), dict):
            raise ValueError(f"Registry {path} must contain a models mapping")
        for alias, entry in data.get("models", {}).items():
            if isinstance(alias, str) and isinstance(entry, dict):
                resolved[alias] = (entry, Path(path))
            else:
                resolved.pop(alias, None)
    return resolved


def credential_names(registry, endpoint):
    """Return the API key variable names used by saved entries for the same server.

    Endpoints are compared after normalisation and with a trailing ``/v1``
    removed, so ``https://host`` and ``https://host/v1`` count as the same
    server. Only variable names are returned, never key values.
    """
    from nooa.unifiedllm.connect import normalize_endpoint

    def normalized(value):
        return normalize_endpoint(value).removesuffix("/v1")

    target = normalized(endpoint)
    names = []
    for entry, _ in registry.values():
        address, name = entry.get("endpoint") or entry.get("api_base"), entry.get("api_key_env")
        if not isinstance(address, str) or not isinstance(name, str):
            continue
        try:
            matches = normalized(address) == target
        except ValueError:
            continue
        if matches and name not in names:
            names.append(name)
    return names


def shadowing_source(alias, path, *, extra_priority=False):
    """Return the file that would override ``alias`` if it were saved to ``path``, or None.

    Another file "shadows" the save when it already defines ``alias`` and NOOA
    reads it after ``path``, so its definition would win. Returns that file's
    path as a string, or None if the saved alias would be the one in effect.

    Set ``extra_priority`` when ``path`` is a ``--working-dir``/``-w`` target.
    Such a file is read last (highest priority) by ``entries()`` and by a later
    ``nooa tui -w`` on the same directory, so nothing overrides it. Any other
    ``--output`` path that NOOA does not load by default is treated as
    overridden by any existing definition elsewhere.
    """
    import os
    from pathlib import Path

    from nooa.llm_config import bundled_config_paths
    from nooa.paths import get_project_dir, get_user_dir

    found = entries().get(alias)
    if not found or found[1].resolve() == path.resolve():
        return None
    # List the standard user and project files even if they do not exist yet,
    # so a file about to be created gets its real place in the order. Other
    # --output paths are not loaded automatically and so have no place here.
    paths = [
        *bundled_config_paths(),
        get_user_dir("llm_config.yaml"),
        get_project_dir("llm_config.yaml"),
        *(
            Path(p.strip()).expanduser()
            for p in os.environ.get("NEMO_OO_LLM_CONFIG", "").split(",")
            if p.strip()
        ),
        *([path] if extra_priority else []),
    ]
    priority = {p.resolve(): i for i, p in enumerate(paths)}
    target = priority.get(path.resolve())
    source = priority.get(found[1].resolve())
    if target is None or source is None or source > target:
        return str(found[1])
    return None


def diagnostic_context(
    *,
    target=None,
    alias=None,
    model=None,
    endpoint=None,
    api_style=None,
    api_key_env=None,
    api_key=None,
    budget=None,
    remaining=None,
    output_tokens=2048,  # == nooa.unifiedllm.connect.DISCOVERY_OUTPUT_TOKENS (tested)
    reasoning_output_tokens=4096,
    stage=None,
    discovery_succeeded=None,
    interface_timeout_seconds=30,
):
    """Return a dict describing this run, for troubleshooting output.

    Includes the installation, target file, alias, model, token budgets, where
    the API key came from (but not the key), proxy variables that are set, the
    registry files, and, for a failed ``interfaces`` stage, a command to rerun
    it. Key values, registry contents and the raw command line are left out;
    any occurrence of the key in the values is replaced with ``[redacted]``.
    """
    import os
    import shlex
    from pathlib import Path

    import yaml

    from nooa._version import __version__
    from nooa.llm_config import llm_config_chain
    from nooa.unifiedllm.connect import DECISION_API_STYLES, normalize_endpoint
    from nooa.unifiedllm.connect._diagnostics import installation_context

    context = {
        **installation_context(),
        "target_file": str(Path(target).expanduser().resolve()) if target else None,
        "working_directory": str(Path.cwd()),
        "alias": alias,
        "wire_model": model,
        "nooa_version": __version__,
        "approved_budget_tokens": budget,
        "remaining_budget_tokens": remaining,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "credential_source": "pasted"
        if api_key and (not api_key_env or api_key != os.environ.get(api_key_env))
        else "environment_or_secrets"
        if api_key_env
        else "none",
        "credential_available": bool(api_key or (api_key_env and os.environ.get(api_key_env))),
        "interface_timeout_seconds": interface_timeout_seconds,
        "reasoning_timeout_seconds": 120,
        "discovery_succeeded": discovery_succeeded,
        "proxy_variables_set": {
            name: bool(os.environ.get(name))
            for name in (
                "HTTPS_PROXY",
                "HTTP_PROXY",
                "NO_PROXY",
                "https_proxy",
                "http_proxy",
                "no_proxy",
            )
        },
    }
    override = os.environ.get("NOOA_LLM_TRANSPORT")
    context["transport_override"] = (
        override if override in {None, "direct", "litellm"} else "invalid"
    )
    try:
        context["registry_files"] = [str(p.resolve()) for p in llm_config_chain()]
        if target:
            context["target_in_registry_chain"] = (
                context["target_file"] in context["registry_files"]
            )
            if not context["target_in_registry_chain"]:
                context["target_load_note"] = (
                    "To load this file from the current working directory, add it to the registry "
                    "chain or load it explicitly. Entries may already load in the target project."
                )
        found = entries().get(alias) if alias else None
        context["effective_alias_source"] = str(found[1]) if found else None
    except (OSError, ValueError, yaml.YAMLError):
        pass  # A broken registry file must not hide the error being reported.
    if stage == "interfaces" and model and endpoint and remaining and remaining > 0:
        try:
            address = normalize_endpoint(endpoint)
        except (TypeError, ValueError):
            address = None
        if address:
            context["rerun_command"] = shlex.join(
                [
                    "uv",
                    "run",
                    "nooa",
                    "connect",
                    model,
                    "--stage",
                    "interfaces",
                    "--endpoint",
                    address,
                    *(
                        ["--prompt-key"]
                        if context["credential_source"] == "pasted"
                        else ["--api-key-env", api_key_env or ""]
                    ),
                    *(
                        ["--api-style", api_style, "--budget-tokens", str(min(remaining, 512))]
                        if api_style in DECISION_API_STYLES
                        else [
                            "--output-tokens",
                            str(output_tokens),
                            "--budget-tokens",
                            str(min(remaining, 3 * (output_tokens + 512))),
                        ]
                    ),
                ]
            )
    if context["credential_source"] == "pasted":
        context["credential_note"] = (
            "The pasted key is not included. Rerun with --prompt-key; --api-key-env only works "
            "if that variable contains the intended key in the rerun environment."
        )

    # A key pasted by mistake into a file name or model ID must not be shown either.
    def scrub(value):
        if isinstance(value, str):
            for secret in (api_key, os.environ.get(api_key_env) if api_key_env else None):
                if secret:
                    value = value.replace(secret, "[redacted]")
        elif isinstance(value, list):
            value = [scrub(item) for item in value]
        elif isinstance(value, dict):
            value = {key: scrub(item) for key, item in value.items()}
        return value

    return {key: scrub(value) for key, value in context.items()}
