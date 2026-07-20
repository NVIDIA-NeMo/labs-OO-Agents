# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live cross-model coverage for PredictStrategy structured-output return types.

Exercises every supported return type with ``PredictStrategy`` against three
provider families through the NVIDIA inference gateway:

- ``gpt-5-mini``          — OpenAI / Azure (chat-completions, strict json_schema)
- ``claude-haiku``        — Anthropic (Bedrock)
- ``nemotron-super-49b``  — NVIDIA Nemotron
- ``gpt-5.5-responses``   — OpenAI / Azure via the **Responses API** (``client_type: responses``)
- ``gpt-5.5-direct``      — OpenAI-direct via the **Responses API** (``client_type: responses``)

These tests make **real API calls** and are marked ``integration`` so the default
CI run (``-m 'not integration and not stress'``) skips them. Run locally with:

    uv run pytest tests/integration/test_predict_return_types_live.py -m integration

They require ``NVIDIA_INTERNAL_API_KEY`` (loaded from the repo ``.env``); the whole
module is skipped if it is absent.

Each case asserts the framework produces a **valid request** (no schema-rejection
error) and that the result validates to the expected Python type. Value checks are
kept lenient where model wording can legitimately vary; the regression target is
the framework's schema handling, not model accuracy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import pytest
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel

from nooa import Agent
from nooa.decorators import strategy
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import get_llm_client

load_dotenv(find_dotenv(usecwd=True), override=True)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("NVIDIA_INTERNAL_API_KEY"),
        reason="NVIDIA_INTERNAL_API_KEY not set (load repo .env to run live model tests)",
    ),
]

MODELS = [
    "gpt-5-mini",
    "claude-haiku",
    "nemotron-super-49b",
    "gpt-5.5-responses",
    "gpt-5.5-direct",
]


# ── Return-type fixtures ────────────────────────────────────────────────────
class Point(BaseModel):
    x: int
    y: int


@dataclass
class Cluster:
    theme: str


class Color(StrEnum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class AllTypesAgent(Agent):
    """One PredictStrategy generation method per supported return type."""

    @strategy(PredictStrategy())
    async def as_int(self) -> int:
        """Return the integer 42."""
        ...

    @strategy(PredictStrategy())
    async def as_float(self) -> float:
        """Return the floating point number 3.14."""
        ...

    @strategy(PredictStrategy())
    async def as_str(self) -> str:
        """Return the string: hello world."""
        ...

    @strategy(PredictStrategy())
    async def as_bool(self) -> bool:
        """Return the boolean value true."""
        ...

    @strategy(PredictStrategy())
    async def as_bare_list(self) -> list:
        """Return a list containing the numbers 1, 2 and 3."""
        ...

    @strategy(PredictStrategy())
    async def as_list_str(self) -> list[str]:
        """Return a list of three fruit names as strings."""
        ...

    @strategy(PredictStrategy())
    async def as_list_int(self) -> list[int]:
        """Return a list containing the integers 1, 2 and 3."""
        ...

    @strategy(PredictStrategy())
    async def as_list_model(self) -> list[Point]:
        """Return two points: one at x=1 y=2 and one at x=3 y=4."""
        ...

    @strategy(PredictStrategy())
    async def as_bare_dict(self) -> dict:
        """Return a dictionary mapping the key "a" to 1 and the key "b" to 2."""
        ...

    @strategy(PredictStrategy())
    async def as_dict_typed(self) -> dict[str, int]:
        """Return a dictionary mapping the key "a" to 1 and the key "b" to 2."""
        ...

    @strategy(PredictStrategy())
    async def as_tuple(self) -> tuple[int, str]:
        """Return a pair of the integer 1 and the string "one"."""
        ...

    @strategy(PredictStrategy())
    async def as_set(self) -> set[str]:
        """Return a set containing the strings "a", "b" and "c"."""
        ...

    @strategy(PredictStrategy())
    async def as_literal(self) -> Literal["red", "green", "blue"]:
        """Return the color green."""
        ...

    @strategy(PredictStrategy())
    async def as_optional(self) -> int | None:
        """Return the integer 7."""
        ...

    @strategy(PredictStrategy())
    async def as_union(self) -> int | str:
        """Return the integer 5."""
        ...

    @strategy(PredictStrategy())
    async def as_enum(self) -> Color:
        """Return the color green."""
        ...

    @strategy(PredictStrategy())
    async def as_model(self) -> Point:
        """Return a point at x=1 and y=2."""
        ...

    @strategy(PredictStrategy())
    async def as_dataclass(self) -> Cluster:
        """Return a cluster whose theme is "fruits"."""
        ...

    # issue 148 patterns
    @strategy(PredictStrategy())
    async def as_any(self) -> Any:
        """Return the JSON object {"k": [1, 2, 3]}."""
        ...

    @strategy(PredictStrategy())
    async def as_dict_union(self) -> dict[str, int | bool]:
        """Return a dictionary mapping "a" to 1 and "flag" to true."""
        ...

    @strategy(PredictStrategy())
    async def as_list_dict(self) -> list[dict]:
        """Return a list with two dictionaries: {"x": 1} and {"y": 2}."""
        ...


def _is_list_of(t):
    return lambda o: isinstance(o, list) and all(isinstance(i, t) for i in o)


# (method_name, validator) — validator returns True if the result has the right shape.
CASES = [
    ("as_int", lambda o: isinstance(o, int)),
    ("as_float", lambda o: isinstance(o, float)),
    ("as_str", lambda o: isinstance(o, str)),
    ("as_bool", lambda o: isinstance(o, bool)),
    ("as_bare_list", lambda o: isinstance(o, list)),
    ("as_list_str", _is_list_of(str)),
    ("as_list_int", _is_list_of(int)),
    ("as_list_model", lambda o: isinstance(o, list) and all(isinstance(i, Point) for i in o)),
    ("as_bare_dict", lambda o: isinstance(o, dict)),
    ("as_dict_typed", lambda o: isinstance(o, dict)),
    ("as_tuple", lambda o: isinstance(o, (tuple, list)) and len(o) == 2),
    ("as_set", lambda o: isinstance(o, (set, frozenset))),
    ("as_literal", lambda o: o in ("red", "green", "blue")),
    ("as_optional", lambda o: o is None or isinstance(o, int)),
    ("as_union", lambda o: isinstance(o, (int, str))),
    ("as_enum", lambda o: isinstance(o, Color)),
    ("as_model", lambda o: isinstance(o, Point)),
    ("as_dataclass", lambda o: isinstance(o, Cluster)),
    # issue 148 patterns — content can legitimately vary, so check type only.
    ("as_any", lambda o: o is not None),
    ("as_dict_union", lambda o: isinstance(o, dict)),
    ("as_list_dict", lambda o: isinstance(o, list) and all(isinstance(i, dict) for i in o)),
]

# Types that stress tool/response schema construction (free-form, heterogeneous,
# untyped, nested). CodeAct routes these through the return_result tool schema,
# which is the surface of issue 148.
SCHEMA_STRESS_CASES = [
    c
    for c in CASES
    if c[0]
    in {
        "as_bare_list",
        "as_list_model",
        "as_bare_dict",
        "as_dict_typed",
        "as_tuple",
        "as_set",
        "as_enum",
        "as_model",
        "as_dataclass",
        "as_any",
        "as_dict_union",
        "as_list_dict",
    }
]


@pytest.fixture(scope="module")
def agents():
    """One AllTypesAgent per model, built once for the module."""
    built = {}
    for alias in MODELS:
        built[alias] = AllTypesAgent(llm=get_llm_client(alias))
    return built


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("method_name,validator", CASES, ids=[c[0] for c in CASES])
async def test_return_type(agents, model, method_name, validator):
    """Each supported return type must produce a valid request and validating result."""
    agent = agents[model]
    result = await getattr(agent, method_name)()
    assert validator(result), f"{model}.{method_name} returned {type(result).__name__}: {result!r}"


class SchemaStressCodeActAgent(Agent):
    """Same schema-stressing return types via CodeAct (default strategy).

    CodeAct routes structured output through the ``return_result`` *tool* schema
    (``_convert_tool_to_schema``) rather than ``response_format`` — the surface of
    issue 148. This guards that the tool schema is accepted across providers for the
    return types strict mode cannot express.
    """

    async def as_bare_list(self) -> list:
        """Return a list containing the numbers 1, 2 and 3."""
        ...

    async def as_list_model(self) -> list[Point]:
        """Return two points: one at x=1 y=2 and one at x=3 y=4."""
        ...

    async def as_bare_dict(self) -> dict:
        """Return a dictionary mapping the key "a" to 1 and the key "b" to 2."""
        ...

    async def as_dict_typed(self) -> dict[str, int]:
        """Return a dictionary mapping the key "a" to 1 and the key "b" to 2."""
        ...

    async def as_tuple(self) -> tuple[int, str]:
        """Return a pair of the integer 1 and the string "one"."""
        ...

    async def as_set(self) -> set[str]:
        """Return a set containing the strings "a", "b" and "c"."""
        ...

    async def as_enum(self) -> Color:
        """Return the color green."""
        ...

    async def as_model(self) -> Point:
        """Return a point at x=1 and y=2."""
        ...

    async def as_dataclass(self) -> Cluster:
        """Return a cluster whose theme is "fruits"."""
        ...

    async def as_any(self) -> Any:
        """Return the JSON object {"k": [1, 2, 3]}."""
        ...

    async def as_dict_union(self) -> dict[str, int | bool]:
        """Return a dictionary mapping "a" to 1 and "flag" to true."""
        ...

    async def as_list_dict(self) -> list[dict]:
        """Return a list with two dictionaries: {"x": 1} and {"y": 2}."""
        ...


@pytest.fixture(scope="module")
def codeact_agents():
    """One CodeAct agent per model, built once for the module."""
    return {alias: SchemaStressCodeActAgent(llm=get_llm_client(alias)) for alias in MODELS}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "method_name,validator", SCHEMA_STRESS_CASES, ids=[c[0] for c in SCHEMA_STRESS_CASES]
)
async def test_return_type_codeact(codeact_agents, model, method_name, validator):
    """Schema-stressing return types must work via CodeAct's return_result tool (issue 148)."""
    agent = codeact_agents[model]
    result = await getattr(agent, method_name)()
    assert validator(result), f"{model}.{method_name} returned {type(result).__name__}: {result!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Complex / nested / non-serializable return types (KDD-cup feedback)
# Nested models, subtype reuse, dict[str, Model], and (CodeAct-only) numpy /
# pandas + a Pydantic model with a non-serializable nested field.
# ─────────────────────────────────────────────────────────────────────────────
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


class Address(BaseModel):
    street: str
    city: str


class Person(BaseModel):
    name: str
    address: Address  # nested model


class Route(BaseModel):
    origin: Address  # subtype reused twice
    destination: Address


class Team(BaseModel):
    lead: Person  # deeply nested + reuse (Person -> Address)
    members: list[Person]


from pydantic import ConfigDict  # noqa: E402


class Report(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    title: str
    table: pd.DataFrame  # non-JSON-serializable nested field (CodeAct only)


def _v_person(o):
    return isinstance(o, Person) and isinstance(o.address, Address)


def _v_route(o):
    return (
        isinstance(o, Route)
        and isinstance(o.origin, Address)
        and isinstance(o.destination, Address)
    )


def _v_team(o):
    return (
        isinstance(o, Team)
        and isinstance(o.lead, Person)
        and isinstance(o.lead.address, Address)
        and all(isinstance(m, Person) and isinstance(m.address, Address) for m in o.members)
    )


def _v_people(o):
    # value-typing fix: dict values must be Person, not plain dicts (empty tolerated —
    # some models return {} under a loose free-form-dict schema; that's model behavior).
    return isinstance(o, dict) and all(isinstance(v, Person) for v in o.values())


class ComplexPredictAgent(Agent):
    @strategy(PredictStrategy())
    async def person(self) -> Person:
        """Return a person named Ada living at 1 King St, London."""
        ...

    @strategy(PredictStrategy())
    async def route(self) -> Route:
        """Return a route from (Main St, Springfield) to (Elm St, Portland)."""
        ...

    @strategy(PredictStrategy())
    async def team(self) -> Team:
        """Return a team led by Ada (5 Hill Rd, Oslo) with one member Bob (6 Lake Rd, Oslo)."""
        ...

    @strategy(PredictStrategy())
    async def people_by_city(self) -> dict[str, Person]:
        """Return a dict mapping "london" to person Ada (1 King St, London)."""
        ...


class ComplexCodeActAgent(Agent):
    async def person(self) -> Person:
        """Return a person named Ada living at 1 King St, London."""
        ...

    async def route(self) -> Route:
        """Return a route from (Main St, Springfield) to (Elm St, Portland)."""
        ...

    async def team(self) -> Team:
        """Return a team led by Ada (5 Hill Rd, Oslo) with one member Bob (6 Lake Rd, Oslo)."""
        ...

    async def people_by_city(self) -> dict[str, Person]:
        """Return a dict mapping "london" to person Ada (1 King St, London)."""
        ...

    async def matrix(self) -> np.ndarray:
        """Return a 2x2 numpy array [[1, 2], [3, 4]]. Construct it in code."""
        ...

    async def frame(self) -> pd.DataFrame:
        """Return a pandas DataFrame with column "x" = [1, 2, 3]. Construct it in code."""
        ...

    async def report(self) -> Report:
        """Return a Report titled "sales" whose table is a DataFrame with column "q"=[1,2,3]. Build it in code."""
        ...


COMPLEX_PREDICT_CASES = [
    ("person", _v_person),
    ("route", _v_route),
    ("team", _v_team),
    ("people_by_city", _v_people),
]
COMPLEX_CODEACT_CASES = COMPLEX_PREDICT_CASES + [
    ("matrix", lambda o: isinstance(o, np.ndarray) and o.shape == (2, 2)),
    ("frame", lambda o: isinstance(o, pd.DataFrame) and "x" in o.columns),
    ("report", lambda o: isinstance(o, Report) and isinstance(o.table, pd.DataFrame)),
]


@pytest.fixture(scope="module")
def complex_predict_agents():
    return {alias: ComplexPredictAgent(llm=get_llm_client(alias)) for alias in MODELS}


@pytest.fixture(scope="module")
def complex_codeact_agents():
    return {alias: ComplexCodeActAgent(llm=get_llm_client(alias)) for alias in MODELS}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "method_name,validator", COMPLEX_PREDICT_CASES, ids=[c[0] for c in COMPLEX_PREDICT_CASES]
)
async def test_complex_predict(complex_predict_agents, model, method_name, validator):
    """Nested / reused / dict-of-models return types via PredictStrategy."""
    result = await getattr(complex_predict_agents[model], method_name)()
    assert validator(result), f"{model}.{method_name} returned {type(result).__name__}: {result!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "method_name,validator", COMPLEX_CODEACT_CASES, ids=[c[0] for c in COMPLEX_CODEACT_CASES]
)
async def test_complex_codeact(complex_codeact_agents, model, method_name, validator):
    """Nested / reused / dict-of-models + numpy / pandas / nested-DataFrame via CodeAct."""
    result = await getattr(complex_codeact_agents[model], method_name)()
    assert validator(result), f"{model}.{method_name} returned {type(result).__name__}: {result!r}"
