# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral provider identity and capability contracts.

Models reach NOOA through routing strings (``openai/nvidia/zai-org/glm-5.3``,
``openai/azure/openai/gpt-5.6-sol``, ``kimi-k3:free``) that mix gateway
routes, vendor namespaces, tier suffixes, and provider aliases into one
identifier. Inferring semantic identity from those substrings is how a
routing prefix like ``openai/`` gets mistaken for the OpenAI provider. This
module resolves a routing string to a logical provider exactly once, at the
adapter edge, so no other code has to guess.

It also types two things providers disagree about:

- **reasoning artifacts** (encrypted items, plain text, checkpoints) as a
  kind + payload + provenance record, so they can be stored and compared
  without holding provider SDK objects; and
- **capabilities** (what a provider can capture/replay, which effort levels
  it supports), as explicit declarations where ``None`` means "unsupported"
  and unknown providers are never promoted to "supported".

Fail-closed rules encoded here:

- Gateway/routing prefixes never determine the logical provider; a model
  string that cannot be resolved raises UnknownProviderIdentityError
  instead of guessing.
- Opaque replay compatibility requires a *declared* model compat group;
  undeclared models derive no key rather than a speculative one.
- Unknown capability lookups return None.

The module is intentionally additive: nothing here is consumed by existing
clients yet, and importing it does not import LiteLLM or any provider SDK.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

__all__ = [
    "ModelCompatGroup",
    "NormalizedModel",
    "OPAQUE_REPLAY_KEY_VERSION",
    "ProviderIdentity",
    "ReasoningCapabilities",
    "ReasoningKind",
    "ReasoningRecord",
    "ReasoningReplayMode",
    "RedactionClass",
    "UnknownProviderIdentityError",
    "compat_group_for",
    "derive_opaque_replay_key",
    "get_reasoning_capabilities",
    "parse_model_string",
    "register_compat_group",
    "register_reasoning_capabilities",
]

#: Recursive JSON value.  Provider objects are converted to this shape at
#: the adapter edge so no SDK object ever persists or crosses the boundary.
#:
#: Defined here because neither Python nor Pydantic ships one: ``json`` is a
#: (de)serializer, not a type; ``typing.Any`` validates nothing; and the
#: old-style recursive alias (``JsonValue = list["JsonValue"]``) recurses
#: infinitely under Pydantic 2.x schema generation.  The PEP 695 ``type``
#: statement is the one recursive form Pydantic supports as a field
#: annotation, which is why this module requires pydantic>=2.11.
type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

#: Version tag mixed into every opaque replay key: when the normalization
#: rules change the key must change with them, so rollout is a new version.
OPAQUE_REPLAY_KEY_VERSION = "nooa.opaque-replay-key.v1"


class ReasoningKind(StrEnum):
    """Kind of retained reasoning artifact."""

    OPAQUE = "opaque"  # encrypted/signed/redacted provider state
    TEXT = "text"  # provider-exposed plain reasoning
    CHECKPOINT = "checkpoint"  # compaction/continuation checkpoint


class ReasoningReplayMode(StrEnum):
    """Replay policy modes; AUTO is the default."""

    OFF = "off"
    AUTO = "auto"
    NATIVE_ONLY = "native_only"
    TEXT_CONTEXT = "text_context"  # force plain text to labeled context


#: Redaction class is attached independently of the payload kind.
RedactionClass = Literal["opaque", "plain_reasoning"]


class ProviderIdentity(BaseModel):
    """Logical provider identity, independent of transport routing strings.

    Owned by NOOA rather than borrowed from a transport library (litellm or a
    successor) because the two answer different questions. litellm parses a
    routing string to pick *where to send the request*; that parsing is
    transport-specific, mutable across library versions, and confuses gateway
    routes with providers — an ``openai/`` prefix in a routed id is not
    evidence the model is served by OpenAI. This identity answers *who
    produced a stored artifact and who may receive it back*: it is stamped
    once at the adapter edge, persisted alongside captured reasoning, and
    compared long after the original request. Making that durable,
    provider-independent, and testable in isolation is why it lives here.

    endpoint_id and account_scope are non-secret fingerprints (never raw
    credentials).  transport is *not* part of replay compatibility, so it is
    excluded from derive_opaque_replay_key.
    """

    provider: str  # openai, anthropic, moonshot, zai, ...
    api_style: str  # responses, chat-completions, messages, ...
    model: str
    endpoint_id: str | None = None  # stable non-secret endpoint fingerprint
    account_scope: str | None = None  # non-secret hash if replay-bound
    transport: str  # litellm, anyllm, direct, fake
    opaque_replay_key: str | None = None  # adapter-declared compat boundary

    @classmethod
    def from_model_string(
        cls,
        model_string: str,
        *,
        api_style: str,
        transport: str,
        endpoint_id: str | None = None,
        account_scope: str | None = None,
        compat_groups: Mapping[str, ModelCompatGroup] | None = None,
    ) -> ProviderIdentity:
        """Build an identity from a (legacy) LiteLLM-style model string.

        This is the adapter-edge parser: routing and
        namespace prefixes, provider aliases, and tier suffixes are stripped
        here, once, so no runtime or formatter ever infers semantic identity
        from model substrings.  Unresolvable strings fail closed with
        UnknownProviderIdentityError; adapters with private model knowledge
        may construct the model directly instead.
        """
        parsed = parse_model_string(model_string)
        if parsed.provider is None:
            raise UnknownProviderIdentityError(model_string)
        return cls(
            provider=parsed.provider,
            api_style=api_style,
            model=parsed.model,
            endpoint_id=endpoint_id,
            account_scope=account_scope,
            transport=transport,
            opaque_replay_key=derive_opaque_replay_key(
                provider=parsed.provider,
                api_style=api_style,
                model=parsed.model,
                endpoint_id=endpoint_id,
                account_scope=account_scope,
                compat_groups=compat_groups,
            ),
        )


class ReasoningRecord(BaseModel):
    """One retained reasoning artifact with provenance.

    payload is NOOA-owned JSON only; provider SDK objects are converted at
    the adapter edge and never persist.
    """

    version: int = 1
    kind: ReasoningKind
    payload: JsonValue
    provenance: ProviderIdentity
    provider_item_type: str | None = None
    sequence: int  # exact order within provider output
    provider_token_count: int | None = None
    replayable: bool = True
    redaction_class: RedactionClass


class ReasoningCapabilities(BaseModel):
    """Reasoning capability profile for a provider.

    All fields are explicit declarations — unknowns are never promoted to
    supported.  effort_map maps neutral effort levels to the provider wire
    value, with None meaning "unsupported".
    """

    capture_kinds: frozenset[ReasoningKind]
    native_replay_kinds: frozenset[ReasoningKind]
    effort_map: dict[str, str | None]  # null means unsupported
    supports_reasoning_when_disabled: bool | None = None
    requires_tool_turn_reasoning_replay: bool | None = None
    replay_field: str | None = None
    signature_prefix_sensitive: bool | None = None
    terminal_backfill: bool = False


class ModelCompatGroup(BaseModel):
    """Explicitly declared model compatibility group.

    Group membership is hand-registered only after live verification; model-id
    equality stays out of the replay hot path.  A group is scoped to one
    logical provider, so the same model string can never silently grant
    compatibility across providers.
    """

    name: str
    provider: str
    models: frozenset[str]


class NormalizedModel(BaseModel):
    """Result of normalizing a routing-style model string."""

    provider: str | None  # None = unknown, callers must fail closed
    model: str  # normalized model id (prefixes/tier stripped)
    namespace: str | None = None  # org-style prefix, e.g. "zai-org"
    tier: str | None = None  # stripped tier suffix, e.g. "free", "cloud"


class UnknownProviderIdentityError(ValueError):
    """A model string could not be resolved to a logical provider.

    This is the fail-closed path: never guess a provider
    from a gateway prefix.  Adapters with private knowledge can construct
    ProviderIdentity directly with an explicit provider.
    """


# --- Identity normalization -------------------------------------------------

#: Gateway/routing prefixes stripped from the left of a model string.  These
#: are routing hints, never logical providers.
_ROUTING_PREFIXES: frozenset[str] = frozenset(
    {"openai", "nvidia", "azure", "aws", "bedrock", "vertex_ai", "nvidia_nim", "huggingface"}
)

#: Free-tier / capacity tier suffixes stripped before any comparison or hash
#: (e.g. "kimi-k3:free", "deepseek-v4-flash:cloud").
_TIER_SUFFIXES: frozenset[str] = frozenset({"free", "cloud"})

#: Provider aliases: a leading segment that names a vendor but is not the
#: canonical logical provider.  Canonical names map to themselves.
_PROVIDER_ALIASES: dict[str, str] = {
    "claude": "anthropic",
    "anthropic": "anthropic",
    "zai": "glm",
    "zai-org": "glm",
    "moonshot": "kimi",
    "moonshotai": "kimi",
}

#: Model-family prefixes -> logical provider.  Matched on the final path
#: segment, at token boundaries only (so "glm" never matches "glmx").
_MODEL_FAMILIES: tuple[tuple[str, str], ...] = (
    ("gpt-", "openai"),
    ("chatgpt-", "openai"),
    ("glm", "glm"),
    ("kimi", "kimi"),
    ("deepseek", "deepseek"),
    ("qwen", "qwen"),
    ("nemotron", "nvidia"),
    ("claude", "anthropic"),
    ("llama", "meta"),
    ("mistral", "mistral"),
    ("gemini", "google"),
    ("grok", "xai"),
)
_FAMILY_BOUNDARY = "-_.0123456789"


def _resolve_family(segment: str) -> str | None:
    """Return the logical provider for a final model-id segment, if known."""
    lowered = segment.lower()
    # Bedrock-style ids embed the vendor: "anthropic.claude-3-5-sonnet".
    if "." in lowered:
        head = lowered.split(".", 1)[0]
        canonical = _PROVIDER_ALIASES.get(head)
        if canonical is not None:
            return canonical
    for prefix, provider in _MODEL_FAMILIES:
        if lowered.startswith(prefix) and (
            len(lowered) == len(prefix) or lowered[len(prefix)] in _FAMILY_BOUNDARY
        ):
            return provider
    return None


def parse_model_string(model: str) -> NormalizedModel:
    """Normalize a routing-style model string into logical identity parts.

    Strips gateway/routing prefixes (openai/, nvidia/, azure/, ...), org-style
    namespace prefixes (zai-org/, moonshotai/, meta-llama/), and tier suffixes
    (:free, :cloud), then resolves the logical provider from the model family
    or a provider alias.  provider is None when the identity is unknown;
    callers must fail closed on that.
    """
    raw = model.strip()
    if not raw:
        return NormalizedModel(provider=None, model="", namespace=None, tier=None)

    segments = [s for s in raw.split("/") if s]
    if not segments:
        return NormalizedModel(provider=None, model=raw, namespace=None, tier=None)

    # Strip routing prefixes from the left; a lone segment is never stripped.
    i = 0
    while i < len(segments) - 1 and segments[i].lower() in _ROUTING_PREFIXES:
        i += 1
    segments = segments[i:]

    # Strip a known tier suffix from the final segment.
    tier: str | None = None
    last = segments[-1]
    if ":" in last:
        stem, _, suffix = last.rpartition(":")
        if suffix.lower() in _TIER_SUFFIXES:
            tier = suffix.lower()
            last = stem
            segments[-1] = last

    namespace = "/".join(segments[:-1]) or None

    provider = _resolve_family(last)
    if provider is not None:
        # Bedrock-style dotted ids ("anthropic.claude-sonnet-4-5") embed the
        # vendor: the provider resolves from the head, and the model is the
        # remainder, so both spellings of one identity normalize identically
        # and never diverge in key derivation.
        if "." in last:
            head, _, rest = last.partition(".")
            if rest and _PROVIDER_ALIASES.get(head.lower()) is not None:
                last = rest
        return NormalizedModel(provider=provider, model=last, namespace=namespace, tier=tier)

    # Unknown model family: a leading vendor segment may still name the
    # logical provider (e.g. "claude/foo").  Otherwise fail closed (None).
    if len(segments) > 1:
        alias = _PROVIDER_ALIASES.get(segments[0].lower())
        if alias is not None:
            return NormalizedModel(
                provider=alias, model="/".join(segments[1:]), namespace=None, tier=tier
            )
    return NormalizedModel(provider=None, model=last, namespace=namespace, tier=tier)


# --- Model compatibility groups ---------------------------------------------

#: Conservative default declarations for families known to share opaque replay
#: compatibility.  Adapters register further groups explicitly, only after
#: live verification.
_DEFAULT_COMPAT_GROUPS: tuple[ModelCompatGroup, ...] = (
    ModelCompatGroup(
        name="openai-gpt-5",
        provider="openai",
        models=frozenset({"gpt-5.5", "gpt-5.6", "gpt-5.6-sol", "gpt-5-mini"}),
    ),
    ModelCompatGroup(
        name="anthropic-claude-4-5",
        provider="anthropic",
        models=frozenset(
            {"claude-sonnet-4-5", "claude-sonnet-4-5-v1", "claude-haiku-4-5-v1", "claude-opus-4-5"}
        ),
    ),
)

_COMPAT_GROUPS: dict[str, ModelCompatGroup] = {
    group.name: group for group in _DEFAULT_COMPAT_GROUPS
}


def register_compat_group(group: ModelCompatGroup) -> None:
    """Register (or explicitly override) a model compatibility group.

    Explicit registration wins over generated/default data.
    The group is stored case-normalized (model ids lowercased) so mixed-case
    registrations are reachable by the case-insensitive lookup below.
    """
    normalized = group.model_copy(update={"models": frozenset(m.lower() for m in group.models)})
    _COMPAT_GROUPS[group.name] = normalized


def compat_group_for(
    provider: str,
    model: str,
    *,
    groups: Mapping[str, ModelCompatGroup] | None = None,
) -> ModelCompatGroup | None:
    """Return the declared compat group for a provider + model, or None.

    Membership must be declared; a similar-but-undeclared model id fails
    closed.  Groups are scoped per logical provider.
    """
    registry = _COMPAT_GROUPS if groups is None else groups
    provider_key = provider.lower()
    model_key = model.lower()
    # Members are lowercased at comparison time so injected mappings match
    # regardless of how they were constructed (register_compat_group also
    # normalizes on write; this covers callers passing their own mapping).
    matches = sorted(
        (
            group
            for group in registry.values()
            if group.provider.lower() == provider_key
            and model_key in {m.lower() for m in group.models}
        ),
        key=lambda group: group.name,
    )
    return matches[0] if matches else None


# --- Opaque replay key derivation -------------------------------------------


def derive_opaque_replay_key(
    *,
    provider: str,
    api_style: str,
    model: str,
    endpoint_id: str | None = None,
    account_scope: str | None = None,
    compat_groups: Mapping[str, ModelCompatGroup] | None = None,
) -> str | None:
    """Derive the non-secret opaque replay compatibility digest.

    The key covers issuer/provider, API style, endpoint/account scope, and
    the *declared* model compat group.  The compat group is the model
    dimension of the key — raw model ids are deliberately NOT mixed in, so
    two models in the same hand-verified group are replay-compatible and
    model-id churn never breaks replay.  Identity inputs are
    normalized inside this function (tier suffixes, aliases, prefixes), so
    normalization is versioned with the key itself.

    Fail-closed: returns None (never a speculative key) when the provider
    is empty or the model has no declared compat group.  transport is
    deliberately excluded — replay compatibility is transport-neutral.
    """
    if not provider or not model:
        return None
    parsed = parse_model_string(model)
    if not parsed.model:
        return None
    # Canonicalize the caller-supplied provider through the alias map so a
    # direct caller passing "zai" or "claude" derives the same key scope as the
    # canonical "glm" / "anthropic" (the docstring promises alias stripping
    # happens here). from_model_string already supplies canonical providers;
    # this guards direct callers.
    provider_key = _PROVIDER_ALIASES.get(provider.lower(), provider.lower())
    group = compat_group_for(provider_key, parsed.model, groups=compat_groups)
    if group is None:
        return None
    payload = {
        "key_version": OPAQUE_REPLAY_KEY_VERSION,
        "provider": provider_key,
        "api_style": api_style.lower(),
        "endpoint_id": endpoint_id,
        "account_scope": account_scope,
        "compat_group": group.name,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- Reasoning capability registry -------------------------------------------

_NEUTRAL_EFFORT_LEVELS = ("minimal", "low", "medium", "high")


def _chat_reasoning_capabilities() -> ReasoningCapabilities:
    """OpenAI-compatible reasoning-content providers (GLM/Kimi/DeepSeek/...).

    Their reasoning is plain text on reasoning_content; effort levels are
    declared unsupported (None) until an adapter verifies a mapping.
    """
    return ReasoningCapabilities(
        capture_kinds=frozenset({ReasoningKind.TEXT}),
        native_replay_kinds=frozenset({ReasoningKind.TEXT}),
        effort_map=dict.fromkeys(_NEUTRAL_EFFORT_LEVELS),
        supports_reasoning_when_disabled=True,
        requires_tool_turn_reasoning_replay=False,
        replay_field="reasoning_content",
        signature_prefix_sensitive=False,
        terminal_backfill=False,
    )


#: Declared default capability catalog.  Unknown providers are absent by
#: design — lookups for them return None (fail closed, never promoted).
DEFAULT_REASONING_CAPABILITIES: dict[str, ReasoningCapabilities] = {
    "openai": ReasoningCapabilities(
        capture_kinds=frozenset(
            {ReasoningKind.OPAQUE, ReasoningKind.TEXT, ReasoningKind.CHECKPOINT}
        ),
        native_replay_kinds=frozenset({ReasoningKind.OPAQUE, ReasoningKind.CHECKPOINT}),
        effort_map={
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
        },
        supports_reasoning_when_disabled=True,
        requires_tool_turn_reasoning_replay=True,
        replay_field=None,  # Responses reasoning items, not a message field
        signature_prefix_sensitive=False,
        terminal_backfill=True,  # Azure-style terminal-only encrypted content
    ),
    "anthropic": ReasoningCapabilities(
        capture_kinds=frozenset({ReasoningKind.OPAQUE, ReasoningKind.TEXT}),
        native_replay_kinds=frozenset({ReasoningKind.OPAQUE}),
        # Effort levels are declared unsupported until a verified
        # budget_tokens-vs-effort mapping exists for these models.
        effort_map=dict.fromkeys(_NEUTRAL_EFFORT_LEVELS),
        supports_reasoning_when_disabled=True,
        requires_tool_turn_reasoning_replay=True,
        replay_field="thinking",
        signature_prefix_sensitive=True,
        terminal_backfill=False,
    ),
}
for _chat_provider in ("glm", "kimi", "deepseek", "qwen", "nvidia"):
    DEFAULT_REASONING_CAPABILITIES[_chat_provider] = _chat_reasoning_capabilities()

_REASONING_CAPABILITIES: dict[str, ReasoningCapabilities] = dict(DEFAULT_REASONING_CAPABILITIES)


def register_reasoning_capabilities(provider: str, capabilities: ReasoningCapabilities) -> None:
    """Register (or explicitly override) capabilities for a provider.

    Explicit overrides take precedence over the generated/default catalog.
    Stored under the lowercased provider key so
    mixed-case registrations stay reachable by the case-insensitive lookup.
    """
    _REASONING_CAPABILITIES[provider.lower()] = capabilities


def get_reasoning_capabilities(
    provider: str,
    *,
    catalog: Mapping[str, ReasoningCapabilities] | None = None,
) -> ReasoningCapabilities | None:
    """Return declared capabilities for a provider, or None if unknown.

    None is the fail-closed answer (unknowns are not silently promoted to
    supported); callers must treat it as "no native
    replay, no effort mapping" rather than inventing defaults.
    """
    registry = _REASONING_CAPABILITIES if catalog is None else catalog
    return registry.get(provider.lower())
