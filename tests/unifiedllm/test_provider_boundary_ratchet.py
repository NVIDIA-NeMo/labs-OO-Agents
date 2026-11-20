# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-specific code may only shrink outside ``nooa/unifiedllm/adapters``.

Counts provider references (LiteLLM imports, provider detection, provider
names, client-type checks) in ``src/nooa`` outside the adapters package and
compares the total with ``BASELINE``. Adding a reference fails. Removing one
also fails until ``BASELINE`` is lowered, so the gain is kept: set it to the
count the failure message reports, in the same change that removed the code.
Move provider code into an adapter instead of raising the baseline.
"""

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "nooa"
ADAPTERS = SRC / "unifiedllm" / "adapters"

PATTERNS = {
    "litellm import": re.compile(r"^\s*(?:import litellm\b|from litellm\b)", re.MULTILINE),
    "get_llm_provider": re.compile(r"\bget_llm_provider\b"),
    "api_style ==": re.compile(r"\bapi_style\s*[!=]="),
    '"anthropic" literal': re.compile(r"[\"']anthropic[\"']"),
    "isinstance(self, ResponsesClient)": re.compile(r"isinstance\(self, ResponsesClient\)"),
    "model-name guess": re.compile(
        r"\b_(?:is_anthropic_model|is_bedrock_model|needs_dummy_tool)\("
    ),
}

# Lower this when provider code moves into an adapter (see module docstring).
BASELINE = 116


def count_references() -> tuple[int, dict[str, int], dict[str, int]]:
    by_pattern = dict.fromkeys(PATTERNS, 0)
    by_file: dict[str, int] = {}
    for path in sorted(SRC.rglob("*.py")):
        if ADAPTERS in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in PATTERNS.items():
            found = len(pattern.findall(text))
            if found:
                by_pattern[name] += found
                rel = str(path.relative_to(SRC))
                by_file[rel] = by_file.get(rel, 0) + found
    return sum(by_pattern.values()), by_pattern, by_file


def test_provider_references_outside_adapters_do_not_grow():
    total, by_pattern, by_file = count_references()
    top = sorted(by_file.items(), key=lambda item: -item[1])[:15]
    detail = f"by pattern: {by_pattern}; top files: {top}"
    assert total <= BASELINE, (
        f"{total} provider references outside nooa/unifiedllm/adapters, baseline {BASELINE}. "
        f"Put new provider-specific code in an adapter. {detail}"
    )
    assert total == BASELINE, (
        f"Provider references dropped to {total}; lower BASELINE in this file to {total}. {detail}"
    )


def test_patterns_find_known_references():
    """Guard against a pattern silently matching nothing."""
    _, by_pattern, _ = count_references()
    assert by_pattern["litellm import"] > 0
    assert by_pattern["get_llm_provider"] > 0


def test_neutral_shell_transitive_dependencies_have_no_concrete_imports():
    """Walk first-party imports, including function-local/private dependencies.

    Package facade exports are not runtime routing dependencies: resolve explicit
    imported modules, not unrelated legacy exports from nooa.unifiedllm.__init__.
    Pure adapters.base/schema are allowed; concrete clients/providers are not.
    """
    source = SRC.parent
    pending = [
        "nooa.unifiedllm." + name
        for name in (
            "llm",
            "logical",
            "execution",
            "retry",
            "adapters._legacy_error_text",
            "_route_calibration",
            "_observations",
        )
    ]
    visited = set()
    forbidden = []
    concrete_names = {
        "CompletionClient",
        "ResponsesClient",
        "UnifiedLLM",
        "LiteLLMChatAdapter",
        "LiteLLMResponsesAdapter",
    }
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        path = source.joinpath(*module.split(".")).with_suffix(".py")
        if not path.exists():
            path = source.joinpath(*module.split("."), "__init__.py")
        if not path.exists():
            continue
        visited.add(module)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = (
                    ".".join(package.split(".")[: len(package.split(".")) - node.level + 1])
                    if node.level
                    else ""
                )
                base = ".".join(part for part in (prefix, node.module) if part)
                targets = [base] if node.module else [base + "." + a.name for a in node.names]
                for alias in node.names:
                    if alias.name in concrete_names:
                        forbidden.append((module, node.lineno, alias.name))
            for target in targets:
                if (
                    target.split(".")[0] in {"litellm", "openai", "anthropic", "google"}
                    or target == "nooa.unifiedllm.unifiedllm"
                    or target.startswith("nooa.unifiedllm.adapters.litellm")
                ):
                    forbidden.append((module, node.lineno, target))
                if target.startswith("nooa.") and target != "nooa.unifiedllm":
                    pending.append(target)
    assert "nooa.unifiedllm._route_calibration" in visited
    assert "nooa.media_interfaces" in visited
    assert not forbidden, f"Concrete imports reachable from neutral shell: {forbidden}"


def test_actor_and_pure_compat_helper_have_no_sdk_imports():
    for relative in ("runtime/actor.py", "unifiedllm/adapters/_legacy_error_text.py"):
        for node in ast.walk(ast.parse((SRC / relative).read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            assert not any(
                name.split(".")[0] in {"litellm", "openai", "anthropic"} for name in modules
            )
