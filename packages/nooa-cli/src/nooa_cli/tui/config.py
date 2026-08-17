# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading for the NOOA TUI.

Hydra-like config: structured Pydantic models with Config.load(**overrides).

Resolution order (last wins):
    1. Model defaults
    2. Layered ``settings.yaml`` (user → project → ``NEMO_OO_SETTINGS``),
       loaded via :mod:`nooa_cli.tui.settings`
    3. Keyword overrides from CLI args (_OVERRIDES map)

Usage:
    # From the Click command
    config = Config.load(model="gpt-4o", working_dir="/tmp")

    # Programmatic
    config = Config.load()
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from nooa.unifiedllm import CompletionClient

# Default model — direct litellm-supported name. Override via config or --model.
DEFAULT_MODEL = "claude-opus-4-8"


# SummarizationConfig moved to core with the interactive-agent base;
# re-exported here so existing ``nooa_cli.tui.config`` imports keep working.
from nooa.interactive import SummarizationConfig as SummarizationConfig  # noqa: E402


class AgentConfig(BaseModel):
    """Configuration for the TUI agent's behavior."""

    # History summarization settings
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)

    # Working directory for bash commands. Stored as a string (downstream
    # always str()s it); a Path is accepted and coerced for ergonomics.
    working_dir: str = "."

    @field_validator("working_dir", mode="before")
    @classmethod
    def _coerce_working_dir(cls, v: object) -> object:
        return str(v) if isinstance(v, Path) else v


class TUIConfig(BaseModel):
    """Configuration for the TUI presentation layer."""

    # MCP servers.  Inline ``mcp_servers`` in settings.yaml is the preferred single-file
    # configuration; ``mcp_file`` remains as a compatibility bridge for VS Code / Claude
    # style .mcp.json files.
    mcp_file: Path = Path(".mcp.json")
    mcp_servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    mcp_auto_connect: list[str] = Field(default_factory=list)

    # Directories to search for skills and user-invocable commands.
    # Includes both project-local and user-global Claude/Cursor conventions.
    skills_dirs: list[Path] = Field(
        default_factory=lambda: [
            Path(".agents/skills"),
            Path(".cursor/skills"),
            Path(".claude/skills"),
            Path(".claude/commands"),
            Path.home() / ".agents" / "skills",
            Path.home() / ".claude" / "skills",
            Path.home() / ".claude" / "commands",
        ]
    )
    # Extra roots explicitly added with ``/skills add``. Unlike
    # ``skills_dirs`` (the computed runtime search path), these are persisted.
    additional_skills_dirs: list[Path] = Field(default_factory=list)
    # Skills explicitly activated with ``/skills activate``. Persisting this
    # separately from discovery lets lifecycle skills attach before the resume
    # event on the next process start.
    active_skills: list[str] = Field(default_factory=list)

    # Default LLM model (from unifiedllm registry)
    default_model: str = DEFAULT_MODEL

    # Trace output directory (None = OTLP auto-probe only; --trace writes files)
    trace_dir: Path | None = None

    # Vi keybindings in prompt_toolkit input
    vi_mode: bool = False

    # Custom agent spec: "module.path:ClassName" or "./file.py:ClassName"
    agent_spec: str | None = None

    # Show agent Python code execution panels (off by default)
    show_python: bool = False

    # Show bounded unified diffs for semantic file-edit events.
    show_diffs: bool = True

    # Audit DONE turns and continue autonomously when the configured judge
    # finds unfinished work. Disabled until explicitly enabled by the user.
    keep_going: bool = False
    keep_going_model: str | None = None

    # Long-term memory is opt-in and can be scoped to this session or shared
    # by every session rooted in the current project. Per-agent maps let custom
    # agents keep independent choices in the same settings file.
    memory: Literal["off", "session", "project"] = "off"
    memory_agents: dict[str, Literal["off", "session", "project"]] = Field(default_factory=dict)
    memory_path: Path | None = None
    memory_owner: str | None = None
    memory_owner_agents: dict[str, str] = Field(default_factory=dict)

    # Idle reflection consolidates memory between completed turns. It stays
    # off until explicitly enabled for an agent.
    reflection: bool = False
    reflection_agents: dict[str, bool] = Field(default_factory=dict)
    reflection_generative: bool = True
    reflection_debounce_s: float = 10.0
    reflection_grace_s: float = 0.5

    # Native scrollback with clear+rewrite transcript replay on resize.
    full_screen: bool = True

    # Ordered, structured toolbar providers. Third-party packages can register
    # additional providers through the ``nooa_cli.tui.toolbar_items`` group.
    toolbar_items: list[str] = Field(
        default_factory=lambda: ["time", "model", "context", "session"]
    )


class Config(BaseModel):
    """Top-level configuration. Single source of truth.

    Usage:
        config = Config.load(model="gpt-4o") # programmatic
        config = Config.load()               # pure defaults + env
    """

    tui: TUIConfig = Field(default_factory=TUIConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    # Runtime flags (not persisted)
    no_splash: bool = False
    no_trace: bool = False
    # Explicit registry YAMLs supplied by the host. These are appended after
    # the discovered chain, so a command-line path has highest precedence.
    llm_config_paths: list[Path] = Field(default_factory=list)

    # ── Declarative mapping: kwarg name → dotted config path ──────────
    # Tuple form: (path, transform_fn) for type coercion.
    # String form: path only, value passed through as-is.
    _OVERRIDES: ClassVar[dict] = {
        "model": "tui.default_model",
        "mcp_file": ("tui.mcp_file", Path),
        "mcp_servers": "tui.mcp_servers",
        "mcp_auto_connect": (
            "tui.mcp_auto_connect",
            lambda v: [str(item) for item in v] if isinstance(v, (list, tuple, set)) else [str(v)],
        ),
        "llm_config": (
            "llm_config_paths",
            lambda v: [Path(item) for item in v],
        ),
        "trace": ("tui.trace_dir", Path),
        "context_limit": "agent.summarization.max_tokens",
        "working_dir": "agent.working_dir",
        "no_splash": "no_splash",
        "no_trace": "no_trace",
        "vi": "tui.vi_mode",
        "agent": "tui.agent_spec",
        "python": "tui.show_python",
        "full_screen": "tui.full_screen",
        "memory": "tui.memory",
        "memory_agents": "tui.memory_agents",
        "memory_path": ("tui.memory_path", Path),
        "memory_owner": "tui.memory_owner",
        "memory_owner_agents": "tui.memory_owner_agents",
        "reflection": "tui.reflection",
        "reflection_agents": "tui.reflection_agents",
        "reflection_generative": "tui.reflection_generative",
        "reflection_debounce_s": "tui.reflection_debounce_s",
        "reflection_grace_s": "tui.reflection_grace_s",
    }

    # Boolean CLI flags: False means absent and must not overwrite settings.
    _FLAG_OVERRIDES: ClassVar[set[str]] = {
        "no_splash",
        "no_trace",
        "vi",
        "python",
        "full_screen",
    }

    @classmethod
    def load(cls, **overrides) -> "Config":
        """Build config: defaults → config file → overrides.

        Config file: layered ``settings.yaml`` (user → project →
        ``NEMO_OO_SETTINGS``), discovered through the shared layered-config
        helper. Accepts any keyword argument matching _OVERRIDES keys.
        Unknown keys are silently ignored so CLI adapters can pass their
        complete option mappings directly.
        """
        from .settings import load_settings

        # Layers 1-2: dataclass defaults, then layered settings.yaml.
        cfg = load_settings(cls())

        # Layer 3: explicit overrides (highest priority)
        for key, target in cls._OVERRIDES.items():
            val = overrides.get(key)
            if val is None:
                continue
            # False flag defaults mean "not provided" and must not overwrite
            # layered settings.
            if isinstance(val, bool) and not val and key in cls._FLAG_OVERRIDES:
                continue
            _set_nested(cfg, *_unpack_target(target, val))

        # ── Special-case overrides ────────────────────────────────────
        # --no-trace clears trace_dir
        if overrides.get("no_trace"):
            cfg.tui.trace_dir = None

        # Explicit skill directories take precedence over conventional paths.
        explicit: list[Path] = []

        extra_skills = overrides.get("skills_dir")
        if extra_skills:
            # Accept a single str/Path OR a list/tuple of them. The bare
            # fallback ``list(x)`` would iterate *characters* when given a
            # lone string, producing nonsense paths.
            if isinstance(extra_skills, (str, Path)):
                dirs: list = [extra_skills]
            elif isinstance(extra_skills, (list, tuple)):
                dirs = list(extra_skills)
            else:
                dirs = list(extra_skills)  # last-resort: any other iterable
            for d in dirs:
                p = Path(d)
                if p not in explicit:
                    explicit.append(p)

        persisted = [Path(d) for d in cfg.tui.additional_skills_dirs]
        ordered = explicit + persisted + cfg.tui.skills_dirs
        cfg.tui.skills_dirs = list(dict.fromkeys(ordered))

        # Ignore absent conventional locations.
        cfg.tui.skills_dirs = [d for d in cfg.tui.skills_dirs if d.exists()]

        return cfg


# ── Helpers ───────────────────────────────────────────────────────────────


def _unpack_target(target, value):
    """Unpack a target spec into (path, transformed_value)."""
    if isinstance(target, tuple):
        path, transform = target
        return path, transform(value)
    return target, value


def _set_nested(obj, path: str, value):
    """Set a dotted attribute path on a nested config model."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


# ── LLM helpers ───────────────────────────────────────────────────────────


class UnresolvedModelError(ValueError):
    """The model is neither a loaded alias nor provider-routable by LiteLLM."""

    def __init__(self, model: str):
        super().__init__(model)
        self.model = model


def get_llm_for_model(model_name: str) -> "CompletionClient":
    """Build a client after validating aliases/provider routing without noisy output."""
    from nooa.unifiedllm import MODELS, CompletionClient, get_llm_client

    if model_name in MODELS:
        return get_llm_client(model_name)

    from .health_check import _detect_provider

    if _detect_provider(model_name) is None:
        raise UnresolvedModelError(model_name)
    return CompletionClient(model=model_name)


def get_llm(config: TUIConfig | Config) -> "CompletionClient":
    """Get the LLM client based on configuration.

    Accepts either a TUIConfig or the top-level Config.
    """
    tui = config.tui if isinstance(config, Config) else config
    return get_llm_for_model(tui.default_model)


def list_models() -> list[str]:
    """List available models from the unifiedllm registry."""
    from nooa.unifiedllm import MODELS

    return sorted(MODELS.keys())


def load_agent_class(spec: str) -> type:
    """Load an agent class from a 'module:ClassName' or './file.py:ClassName' spec.

    Args:
        spec: Agent spec in the form ``module.path:ClassName`` or
              ``./path/to/file.py:ClassName`` (absolute paths also work).

    Returns:
        The agent class (uninstantiated).

    Raises:
        ValueError: If the spec format is invalid or the class is not an Agent subclass.
        FileNotFoundError: If a file-path spec points to a missing file.
        ImportError: If the module cannot be imported.
        AttributeError: If the class name is not found in the module.
    """
    import importlib
    import importlib.util
    import sys

    if ":" not in spec:
        raise ValueError(
            f"Invalid agent spec '{spec}'. "
            "Expected 'module.path:ClassName' or './path/to/file.py:ClassName'."
        )

    module_part, class_name = spec.rsplit(":", 1)
    class_name = class_name.strip()

    # File path: ends in .py OR contains a path separator OR starts with . / ~
    is_file = module_part.endswith(".py") or "/" in module_part or module_part.startswith(".")
    if is_file:
        file_path = Path(module_part).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Agent module file not found: {file_path}")

        parent_str = str(file_path.parent)
        inserted = False
        if parent_str not in sys.path:
            sys.path.insert(0, parent_str)
            inserted = True

        try:
            mod_spec = importlib.util.spec_from_file_location("_tui_custom_agent", file_path)
            if mod_spec is None or mod_spec.loader is None:
                raise ImportError(f"Cannot load module from {file_path}")
            module = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(module)  # type: ignore[union-attr]
        finally:
            if inserted:
                sys.path.remove(parent_str)
    else:
        module = importlib.import_module(module_part)

    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"Class '{class_name}' not found in '{module_part}'.")

    # Validate it's an Agent subclass
    try:
        from nooa import Agent

        if not (isinstance(cls, type) and issubclass(cls, Agent)):
            raise ValueError(
                f"'{class_name}' is not a subclass of a NOOA Agent. "
                "Make sure your class inherits from Agent."
            )
    except ImportError:
        pass  # Can't validate without nooa; proceed anyway

    return cls
