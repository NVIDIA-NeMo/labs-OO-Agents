# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``/connect``: connect a model provider with client forms, without the model.

A host-side built-in command. It runs inside ``Session.invoke_command`` with
the host's asker bound and asks each step with ``nooa.elicit.ask``; the
answers never reach the agent. It is built only on the library seam
``nooa.unifiedllm.connect`` and follows the ``nooa connect`` wizard's order
and defaults: provider, server URL, the name of the environment variable
that holds the key (never the key itself), model, connection checks, reply
limits, alias, then a write to the user's ``llm_config.yaml``.

When the client cannot show forms it answers with :data:`CONNECT_FALLBACK`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path
from typing import Any

from nooa.elicit import Choice, Confirm, ElicitationUnavailable, Integer, Text, ask

CONNECT_FALLBACK = (
    "Connecting a model provider needs the terminal for now: run `nooa connect` in a "
    "shell, then pick the new model alias here."
)
"""The reply when the client cannot show forms."""

CANCELLED = "Cancelled; nothing written."

MAX_CHOICES = 25
"""More discovered models than this are narrowed with a filter first."""

_KEY_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class _Stop(Exception):
    """End the wizard with ``message`` (nothing written)."""


class _Cancelled(Exception):
    """The person declined or dismissed a form."""


def _connect() -> Any:
    from nooa.unifiedllm import connect

    return connect


def default_registry_path() -> Path:
    """The ``llm_config.yaml`` that ``nooa connect`` writes by default."""
    from nooa import paths

    return paths.get_user_dir("llm_config.yaml")


async def _form(message: str, **fields: Any) -> dict[str, Any]:
    answer = await ask(message, **fields)
    if not answer.accepted:
        raise _Cancelled
    return answer.values


def _existing_models(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = data.get("models") if isinstance(data, dict) else None
    return models if isinstance(models, dict) else {}


class _Wizard:
    def __init__(self, path: Path, environ: Mapping[str, str]) -> None:
        self.connect = _connect()
        self.path = path
        self.environ = environ
        self.notes: list[str] = []
        self.api_key: str | None = None

    def note(self, line: str) -> None:
        self.notes.append(line)

    def progress(self) -> str:
        """What happened so far, shown above the next form."""
        return "\n".join(self.notes[-6:])

    def with_progress(self, message: str) -> str:
        done = self.progress()
        return f"{done}\n\n{message}" if done else message

    def scrub(self, text: str) -> str:
        return text.replace(self.api_key, "***") if self.api_key else text

    # ---- steps -------------------------------------------------------

    async def provider(self) -> tuple[str, str, str, str]:
        """Provider, server URL and key variable: ``(provider, endpoint, style, key_env)``."""
        connect = self.connect
        options = {name: preset.label for name, preset in connect.PROVIDERS.items()}
        options["custom"] = "Custom endpoint"
        values = await _form(
            "Connect a model provider. Setup lists the server's models and sends a few "
            "small test requests; nothing is saved until the last step.",
            provider=Choice(title="Provider", options=options),
        )
        provider = values["provider"]
        preset = connect.PROVIDERS.get(provider)
        values = await _form(
            f"Server for {options[provider]}.",
            endpoint=Text(
                title="Model server URL",
                description="The API base, for example https://host/v1.",
                default=preset.api_base if preset else None,
            ),
        )
        try:
            endpoint = connect.normalize_endpoint(values["endpoint"])
        except ValueError as exc:
            raise _Stop(f"That server URL cannot be used: {exc}. Nothing written.") from None
        style = preset.api_style if preset else "chat"
        default_env = preset.api_key_env if preset else "OPENAI_API_KEY"
        values = await _form(
            "Which environment variable holds the API key? Enter its name, not the key; "
            "- means no authentication.",
            api_key_env=Text(
                title="Key environment variable",
                default=default_env,
                pattern=r"^([A-Za-z_][A-Za-z0-9_]*|-)$",
            ),
        )
        key_env = values["api_key_env"].strip()
        if key_env == "-":
            key_env = ""
        elif not _KEY_ENV.fullmatch(key_env):
            raise _Stop(f"{key_env!r} is not an environment variable name. Nothing written.")
        if key_env:
            key = self.environ.get(key_env)
            if not key:
                raise _Stop(
                    f"{key_env} is not set in the environment NOOA runs in, so the server "
                    "cannot be reached with a key. Set it where you start your editor or "
                    f"agent (for example `export {key_env}=...`), restart it, and run "
                    "/connect again; or run `nooa connect` in a shell, which can take the "
                    "key there. Nothing written."
                )
            self.api_key = key
            self.note(f"Using the key in {key_env}.")
        return provider, endpoint, style, key_env

    async def model(self, endpoint: str, style: str) -> tuple[str, str, dict[str, Any] | None]:
        """Discover and pick a model: ``(model, endpoint, endpoint_model)``."""
        connect = self.connect
        try:
            found = await connect.discover(endpoint, api_style=style, api_key=self.api_key)
        except connect.DiscoveryError as exc:
            if exc.status_code in {401, 403}:
                raise _Stop(
                    "The server rejected the key. Check the variable's value, restart, "
                    "and run /connect again. Nothing written."
                ) from None
            self.note(f"Could not list models: {self.scrub(str(exc))}")
            values = await _form(
                self.with_progress("Enter the exact model ID."),
                model=Text(title="Model ID", min_length=1),
            )
            return values["model"].strip(), endpoint, None
        endpoint = found.api_base
        names = [item["id"] for item in found.models]
        if not names:
            values = await _form(
                "The server listed no models. Enter the exact model ID.",
                model=Text(title="Model ID", min_length=1),
            )
            return values["model"].strip(), endpoint, None
        self.note(f"Found {len(names)} models at {endpoint}.")
        shown = names
        message = f"Found {len(names)} models. Type part of a name to narrow the list."
        while len(shown) > MAX_CHOICES:
            values = await _form(
                message, filter=Text(title="Filter", description="Part of a model ID.")
            )
            query = values["filter"].strip().lower()
            matching = [name for name in names if query in name.lower()]
            if not matching:
                message = f"No model ID contains {values['filter']!r}. Try another filter."
            elif len(matching) > MAX_CHOICES:
                message = f"{len(matching)} models match {values['filter']!r}. Narrow it further."
            else:
                shown = matching
        values = await _form(
            f"Choose a model ({len(shown)} shown).",
            model=Choice(title="Model", options=shown),
        )
        model = values["model"]
        endpoint_model = next((item for item in found.models if item.get("id") == model), None)
        return model, endpoint, endpoint_model

    async def interfaces(self, model: str, endpoint: str, key_env: str, preferred: str) -> Any:
        """Probe the request interfaces; return ``(style, InterfaceResult)``."""
        connect = self.connect
        result = None
        async with aclosing(
            connect.check_interfaces(
                "candidate",
                model,
                endpoint,
                key_env,
                api_key=self.api_key,
                styles=("chat", "responses", "anthropic"),
                reasoning_template="effort",
            )
        ) as updates:
            async for update in updates:
                if isinstance(update, connect.InterfaceResult):
                    result = update
        if result is None or not result.accepted:
            raise _Stop(
                "Could not confirm a working connection to "
                f"{model}: no request interface accepted a test request. Check the server "
                "URL, the key and the model ID. Nothing written."
            )
        accepted = result.accepted
        self.note("Interfaces that answered: " + ", ".join(accepted) + ".")
        default = (
            result.recommended_style
            if result.recommended_style in accepted
            else preferred
            if preferred in accepted
            else accepted[0]
        )
        if len(accepted) == 1:
            return accepted[0], result
        values = await _form(
            self.with_progress(f"Choose the request interface for {model}."),
            api_style=Choice(title="API format", options=list(accepted), default=default),
        )
        return values["api_style"], result

    async def metadata(self, model: str, endpoint_model: dict[str, Any] | None) -> Any:
        """Public catalogue limits for ``model``, if one entry matches."""
        connect = self.connect
        candidate = None
        try:
            matches = connect.match_models(model, await connect.catalogue())
        except Exception:
            self.note("Public catalogue unavailable; limits come from the server or defaults.")
            matches = []
        if len(matches) == 1:
            candidate = matches[0]
        if endpoint_model is not None:
            combined = connect.model_metadata(model, candidate, endpoint_model)
            if combined.get("endpoint_limits"):
                candidate = combined
        return candidate

    async def limits(self, proposal: Any) -> Any:
        """Review the context window and the reply limit; return the refreshed plan."""
        connect = self.connect
        entry = connect.configure_entry(proposal.entry)
        context = entry.get("context_window")
        values = await _form(
            self.with_progress("Review the model's limits."),
            context_window=Integer(
                title="Context window (tokens)",
                description="Leave empty if unknown; NOOA then uses its default.",
                default=context,
                minimum=1,
                required=False,
            ),
            max_tokens=Integer(
                title="Reply limit (tokens)",
                description="Room for each reply, including thinking and the answer.",
                default=entry["max_tokens"],
                minimum=1,
            ),
        )
        if values["context_window"] and values["context_window"] != context:
            entry["context_window"] = values["context_window"]
            entry["provenance"]["context_window"] = {
                "source": "user",
                "value": values["context_window"],
            }
        reply = values["max_tokens"]
        entry = connect.configure_entry(
            entry, reply_tokens=reply if reply != entry["max_tokens"] else None
        )
        return connect.refresh_plan(replace(proposal, entry=entry))

    async def checks(self, proposal: Any) -> Any:
        """Run the approved checks; return the result to save."""
        connect = self.connect
        result = None
        async with aclosing(
            connect.run_steps(proposal, approved="all", api_key=self.api_key)
        ) as updates:
            async for update in updates:
                if isinstance(update, connect.ConnectResult):
                    result = update
        if result is None:
            raise _Stop("The checks ended without a result. Nothing written.")
        decision = connect.verdict(result.entry)
        if "routing" not in decision.passed:
            raise _Stop(
                "The connection check did not pass, so the model was not saved. "
                "Review the server URL and settings, or run `nooa connect "
                "--show-diagnostics` in a shell. Nothing written."
            )
        self.note("Checks passed: " + (", ".join(decision.passed) or "none") + ".")
        if decision.needs_attention:
            self.note("Needs attention: " + ", ".join(decision.needs_attention) + ".")
        return result

    async def save(self, result: Any, model: str) -> str:
        suggested = model.rsplit("/", 1)[-1]
        values = await _form(
            self.with_progress(f"Save this model in {self.path} as:"),
            alias=Text(title="Alias", default=suggested, min_length=1),
        )
        alias = values["alias"].strip()
        if not alias:
            raise _Stop("An alias is required. Nothing written.")
        if alias in _existing_models(self.path):
            values = await _form(
                f"{alias!r} already exists in {self.path}. Replace it?",
                replace=Confirm(title="Replace", default=False),
            )
            if not values["replace"]:
                raise _Cancelled
        self.connect.write(result.entry, self.path, alias=alias)
        return alias

    async def run(self) -> str:
        connect = self.connect
        provider, endpoint, style, key_env = await self.provider()
        model, endpoint, endpoint_model = await self.model(endpoint, style)
        style, interfaces = await self.interfaces(model, endpoint, key_env, style)
        candidate = await self.metadata(model, endpoint_model)
        proposal = connect.plan(
            "candidate",
            model,
            style,
            endpoint,
            key_env,
            catalogue=candidate,
            existing_entry=interfaces.results[style].entry,
            session_checks=True,
        )
        proposal = await self.limits(proposal)
        result = await self.checks(proposal)
        alias = await self.save(result, model)
        entry = result.entry
        self.note(
            f"Saved {alias} ({model} · {style} · reply limit {entry['max_tokens']:,} tokens) "
            f"to {self.path}."
        )
        self.note(f"Pick it with the model selector or `/model {alias}`.")
        del provider
        return "\n".join(self.notes)


async def run_connect(
    args: str = "",
    *,
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Connect a model provider and save it as a model alias."""
    del args  # the wizard asks everything
    wizard = _Wizard(path or default_registry_path(), os.environ if environ is None else environ)
    try:
        return await wizard.run()
    except ElicitationUnavailable:
        return CONNECT_FALLBACK
    except _Cancelled:
        return "\n".join([*wizard.notes, CANCELLED])
    except _Stop as stop:
        return "\n".join([*wizard.notes, str(stop)])
    except (OSError, ValueError, KeyError) as exc:
        detail = wizard.scrub(f"{type(exc).__name__}: {exc}")
        return "\n".join([*wizard.notes, f"/connect failed: {detail}. Nothing written."])
    except Exception as exc:
        import httpx

        if isinstance(exc, httpx.HTTPError):
            detail = wizard.scrub(f"{type(exc).__name__}: {exc}")
            return "\n".join([*wizard.notes, f"/connect failed: {detail}. Nothing written."])
        raise


__all__ = ["CANCELLED", "CONNECT_FALLBACK", "default_registry_path", "run_connect"]
