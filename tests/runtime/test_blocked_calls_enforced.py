# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every call in DEFAULT_BLOCKED_CALLS is refused before the cell runs.

The cells below would run (or hang) if they reached exec; each uses a harmless
argument so a missed refusal fails the assertion instead of blocking the suite.
"""

import ast
import asyncio
import json
import multiprocessing
import os
import threading
import time

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.runtime.code_validator import BlockingCallValidator, ValidationContext
from nooa.runtime.restrictions import DEFAULT_BLOCKED_CALLS
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

REFUSAL = "blocks the event loop and is not allowed in agent code"
HINT = "Use 'await' with an async alternative or an appropriate agent tool."

# (module, entry in DEFAULT_BLOCKED_CALLS[module], cell code)
CASES: dict[str, tuple[str, str, str]] = {
    "time.sleep": ("time", "sleep", "import time\ntime.sleep(0)"),
    "import time as t": ("time", "sleep", "import time as t\nt.sleep(0)"),
    "from time import sleep": ("time", "sleep", "from time import sleep\nsleep(0)"),
    "from time import sleep as nap": ("time", "sleep", "from time import sleep as nap\nnap(0)"),
    "alias s = time.sleep": ("time", "sleep", "import time\ns = time.sleep\ns(0)"),
    "os.system": ("os", "system", "import os\nos.system('true')"),
    "from os import system": ("os", "system", "from os import system\nsystem('true')"),
    "os.popen": ("os", "popen", "import os\nos.popen('true').read()"),
    "os.wait": (
        "os",
        "wait",
        "import os\ntry:\n    os.wait()\nexcept ChildProcessError:\n    pass",
    ),
    "os.waitpid": (
        "os",
        "waitpid",
        "import os\ntry:\n    os.waitpid(-1, os.WNOHANG)\nexcept ChildProcessError:\n    pass",
    ),
    "os.waitid": (
        "os",
        "waitid",
        "import os\ntry:\n    os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG)\n"
        "except ChildProcessError:\n    pass",
    ),
    "threading.Event().wait()": (
        "threading",
        "Event.wait",
        "import threading\nthreading.Event().wait(0)",
    ),
    "ev = threading.Event(); ev.wait()": (
        "threading",
        "Event.wait",
        "import threading\nev = threading.Event()\nev.wait(0)",
    ),
    "from threading import Event": (
        "threading",
        "Event.wait",
        "from threading import Event\nev = Event()\nev.wait(0)",
    ),
    "annotated ev: Event = threading.Event()": (
        "threading",
        "Event.wait",
        "import threading\nev: threading.Event = threading.Event()\nev.wait(0)",
    ),
    "annotated alias s: object = time.sleep": (
        "time",
        "sleep",
        "import time\ns: object = time.sleep\ns(0)",
    ),
    "lock.acquire()": (
        "threading",
        "Lock.acquire",
        "import threading\nlock = threading.Lock()\nlock.acquire()\nlock.release()",
    ),
    "Condition.wait": (
        "threading",
        "Condition.wait",
        "import threading\nc = threading.Condition()\nwith c:\n    c.wait(0)",
    ),
    "t.join() on a Thread": (
        "threading",
        "Thread.join",
        "import threading\nt = threading.Thread(target=lambda: None)\nt.start()\nt.join()",
    ),
    "Process.join": (
        "multiprocessing",
        "Process.join",
        "import multiprocessing\np = multiprocessing.Process(target=print)\n"
        "try:\n    p.join()\nexcept AssertionError:\n    pass",
    ),
    "q.put() on a multiprocessing Queue": (
        "multiprocessing",
        "Queue.put",
        "import multiprocessing\nq = multiprocessing.Queue()\nq.put(1)",
    ),
    "q.get() on a multiprocessing Queue": (
        "multiprocessing",
        "Queue.get",
        "from multiprocessing import Queue\nq = Queue()\nq.get(timeout=0.01)",
    ),
    "asyncio.run": (
        "asyncio",
        "run",
        "async def f():\n    return 1\nco = f()\ntry:\n    asyncio.run(co)\nexcept RuntimeError:\n"
        "    co.close()",
    ),
    "loop.run_until_complete": (
        "asyncio",
        "run_until_complete",
        "loop = asyncio.new_event_loop()\nco = asyncio.sleep(0)\ntry:\n"
        "    loop.run_until_complete(co)\nexcept RuntimeError:\n    co.close()\nloop.close()",
    ),
    "loop.run_forever": (
        "asyncio",
        "run_forever",
        "loop = asyncio.new_event_loop()\ntry:\n    loop.run_forever()\nexcept RuntimeError:\n"
        "    pass\nloop.close()",
    ),
    "asyncio.run_coroutine_threadsafe": (
        "asyncio",
        "run_coroutine_threadsafe",
        "co = asyncio.sleep(0)\nfut = asyncio.run_coroutine_threadsafe(co, asyncio.get_running_loop())",
    ),
    "importlib.import_module": (
        "importlib",
        "import_module",
        "import importlib\nimportlib.import_module('json')",
    ),
    "from importlib import import_module": (
        "importlib",
        "import_module",
        "from importlib import import_module\nimport_module('json')",
    ),
}

# Alternatives that must keep working (and plain lookalikes that are not in the list).
ALLOWED = {
    "await asyncio.sleep": "await asyncio.sleep(0)",
    "asyncio subprocess": (
        "proc = await asyncio.create_subprocess_exec('true')\nawait proc.wait()"
    ),
    "asyncio.to_thread(time.sleep)": "import time\nawait asyncio.to_thread(time.sleep, 0)",
    "asyncio.Event/Lock/Queue": (
        "ev = asyncio.Event()\nev.set()\nawait ev.wait()\n"
        "lock = asyncio.Lock()\nawait lock.acquire()\nlock.release()\n"
        "q = asyncio.Queue()\nawait q.put(1)\nawait q.get()"
    ),
    "os/time/threading lookalikes": (
        "import os, time, threading\nos.path.join('a', 'b')\nos.getcwd()\ntime.time()\n"
        "' '.join(['a'])\nt = threading.Thread(target=lambda: None)\nt.start()\n"
        "threading.Event().set()"
    ),
}


def _validate(code: str, exec_globals: dict | None = None) -> list:
    ctx = ValidationContext(exec_globals=exec_globals or {"asyncio": asyncio})
    return BlockingCallValidator().validate(ast.parse(code), ctx)


def test_every_blocked_call_has_a_case():
    listed = {(module, name) for module, names in DEFAULT_BLOCKED_CALLS.items() for name in names}
    covered = {(module, entry) for module, entry, _ in CASES.values()}
    assert listed <= covered, f"no refusal test for {sorted(listed - covered)}"


@pytest.mark.parametrize("name", list(CASES))
def test_validator_refuses_listed_call(name):
    module, entry, code = CASES[name]
    issues = _validate(code)
    assert issues, f"{name} was not refused"
    assert issues[0].code == "E310"
    assert f"{module}." in issues[0].message and entry.split(".")[-1] in issues[0].message
    assert REFUSAL in issues[0].message and HINT in issues[0].message


@pytest.mark.parametrize(
    "code,obj",
    [
        ("ev.wait()", threading.Event()),
        ("lock.acquire()", threading.Lock()),
        ("c.wait(0)", threading.Condition()),
        ("t.join()", threading.Thread(target=lambda: None)),
        ("q.get()", multiprocessing.Queue()),
        ("q.put(1)", multiprocessing.Queue()),
        ("p.join()", multiprocessing.Process(target=print)),
        ("s(0)", time.sleep),
        ("s('true')", os.system),
    ],
    ids=[
        "Event",
        "Lock",
        "Condition",
        "Thread",
        "Queue.get",
        "Queue.put",
        "Process",
        "sleep",
        "system",
    ],
)
def test_validator_refuses_objects_from_earlier_cells(code, obj):
    name = code.split(".")[0].split("(")[0]
    issues = _validate(code, {name: obj})
    assert issues and issues[0].code == "E310", code


@pytest.mark.parametrize(
    "code,name,obj",
    [
        ("lock = asyncio.Lock()\nawait lock.acquire()", "lock", threading.Lock()),
        ("ev = asyncio.Event()\nawait ev.wait()", "ev", threading.Event()),
        ("q: asyncio.Queue = asyncio.Queue()\nawait q.get()", "q", multiprocessing.Queue()),
        (
            "for ev in [asyncio.Event()]:\n    ev.set()\n    await ev.wait()",
            "ev",
            threading.Event(),
        ),
        ("t = threading.Thread(target=print)\nt = ' '\nt.join(['a'])", "t", None),
    ],
    ids=["Lock", "Event", "annotated Queue", "for target", "rebound local"],
)
def test_validator_allows_names_the_cell_rebinds(code, name, obj):
    """A name the cell assigns no longer means the blocking object an earlier cell bound."""
    exec_globals = {"asyncio": asyncio, "threading": threading}
    if obj is not None:
        exec_globals[name] = obj
    assert _validate(code, exec_globals) == []


@pytest.mark.parametrize(
    "code,exec_globals",
    [
        (
            "def helper():\n    ev = asyncio.Event()\nev.wait(0)",
            {"asyncio": asyncio, "ev": threading.Event()},
        ),
        (
            "import threading\nev = threading.Event()\ndef helper():\n    ev = asyncio.Event()\n"
            "ev.wait(0)",
            None,
        ),
        (
            "lock = threading.Lock()\nhelpers = [lock for lock in [asyncio.Lock()]]\nlock.acquire()",
            {"asyncio": asyncio, "threading": threading},
        ),
    ],
    ids=["earlier cell", "same cell", "comprehension"],
)
def test_validator_ignores_nested_scope_bindings(code, exec_globals):
    """A name bound inside a def/comprehension leaves the cell's blocking object in place."""
    issues = _validate(code, exec_globals)
    assert issues and issues[0].code == "E310", code


THREAD_THEN = "import threading\nt = threading.Thread(target=print)\n"


@pytest.mark.parametrize(
    "code,lines",
    [
        (THREAD_THEN + "def helper():\n    t = threading.Event()\n    t.wait()\n", [5]),
        (THREAD_THEN + "f = lambda: [t.wait() for t in [threading.Event()]]\n", []),
        (THREAD_THEN + "def helper():\n    t = threading.Event()\nt.wait()\n", []),
        (THREAD_THEN + "def helper():\n    t = threading.Event()\nt.join()\n", [5]),
        (THREAD_THEN + "def helper():\n    t = ' '\n    t.join(['a'])\nt.join()\n", [6]),
    ],
    ids=[
        "nested Event refused",
        "comprehension target",
        "outer Thread.wait allowed",
        "outer Thread.join refused",
        "nested rebind clears inside only",
    ],
)
def test_validator_tracks_nested_rebinding_inside_body_only(code, lines):
    """A nested body's own binding decides its calls; the cell's binding is back after it."""
    assert [issue.line for issue in _validate(code, None)] == lines


class _NoName:
    """A callable from a listed module without ``__name__`` (like a partial or a callable object)."""

    def __call__(self, *args):
        return None


def _no_name(module: str) -> _NoName:
    obj = _NoName()
    obj.__module__ = module  # instance attribute: the object, not its class, claims the module
    return obj


@pytest.mark.parametrize(
    "code,exec_globals",
    [
        ("sleep = asyncio.sleep\nawait sleep(0)", {"asyncio": asyncio, "sleep": time.sleep}),
        ("import time as t\nt = asyncio\nawait t.sleep(0)", None),
        (
            "def h():\n    from time import sleep\nsleep = asyncio.sleep\nawait sleep(0)",
            {"asyncio": asyncio, "sleep": time.sleep},
        ),
        ("def h():\n    from time import sleep\nawait asyncio.sleep(0)", None),
        (
            "ev = threading.Event()\nevs = [ev.wait() for ev in [asyncio.Event()]]",
            {"asyncio": asyncio, "threading": threading},
        ),
        ("def f(ev):\n    return ev.wait()", {"ev": threading.Event()}),
        ("def sleep(x):\n    return x\nsleep(0)", {"sleep": time.sleep}),
        ("import time\nfor time in [asyncio]:\n    await time.sleep(0)", None),
    ],
    ids=[
        "rebound alias",
        "rebound module",
        "import inside def",
        "import inside def only",
        "comprehension generator before element",
        "function argument",
        "def rebinds name",
        "for target rebinds module",
    ],
)
def test_validator_honours_cell_rebindings_everywhere(code, exec_globals):
    """Every resolution path sees the cell's own, scope-local bindings before exec_globals."""
    assert _validate(code, exec_globals) == []


@pytest.mark.parametrize(
    "code,exec_globals",
    [
        ("sleep(0)", {"sleep": _no_name("time")}),
        ("run(['true'])", {"run": _no_name("subprocess")}),
        ("import threading\nev, x = threading.Event(), 1\nev.wait(0)", None),
        ("import threading\n[ev, (x, lock)] = [1, (2, threading.Lock())]\nlock.acquire()", None),
        ("import threading\nev = threading.Event()\nev2 = ev\nev2.wait(0)", None),
        ("ev2 = ev\nev2.wait(0)", {"ev": threading.Event()}),
        ("ev2 = ev3 = ev\nev3.wait(0)", {"ev": threading.Event()}),
        ("sleep = asyncio.sleep\nsleep = time.sleep\nsleep(0)", {"asyncio": asyncio, "time": time}),
        ("lock = lock.acquire()", {"lock": threading.Lock()}),
        ("ev.wait(0)\nev = asyncio.Event()", {"asyncio": asyncio, "ev": threading.Event()}),
        ("def f():\n    ev.wait(0)\nimport threading\nev = threading.Event()\nf()", None),
        ("import threading\n(ev := threading.Event()).wait(0)", None),
    ],
    ids=[
        "no __name__ blocked call",
        "no __name__ blocked module",
        "tuple target",
        "nested list target",
        "instance alias same cell",
        "instance alias earlier cell",
        "chained alias",
        "later rebinding wins after it runs",
        "value before target",
        "call before rebinding",
        "function body sees final cell bindings",
        "walrus",
    ],
)
def test_validator_refuses_through_aliases_and_targets(code, exec_globals):
    issues = _validate(code, exec_globals)
    assert issues and issues[0].code == "E310", code


@pytest.mark.parametrize(
    "code,exec_globals,rendered",
    [
        (
            "import threading\nlock = threading.Lock()\nlock.acquire()",
            None,
            "threading.Lock.acquire()",
        ),
        ("lock.acquire()", {"lock": threading.Lock()}, "threading.Lock.acquire()"),
        ("import threading\nthreading.Lock().acquire()", None, "threading.Lock.acquire()"),
        (
            "import threading\nt = threading.Thread(target=print)\nt.join()",
            None,
            "threading.Thread.join()",
        ),
        ("t.join()", {"t": threading.Thread(target=print)}, "threading.Thread.join()"),
        ("loop = asyncio.new_event_loop()\nloop.run_forever()", None, "asyncio.run_forever()"),
        ("asyncio.get_event_loop().run_forever()", None, "asyncio.run_forever()"),
    ],
    ids=[
        "Lock same cell",
        "Lock earlier cell",
        "Lock constructor",
        "Thread same cell",
        "Thread earlier cell",
        "loop same cell",
        "loop chained",
    ],
)
def test_validator_renders_the_same_entry_on_every_path(code, exec_globals, rendered):
    """Hints key on the rendered name, so every path names the listed entry the same way."""
    issues = _validate(code, exec_globals)
    assert len(issues) == 1 and issues[0].message.startswith(rendered + " blocks"), issues


@pytest.mark.parametrize("name", list(ALLOWED))
def test_validator_allows_alternatives(name):
    assert _validate(ALLOWED[name]) == []


def test_rendered_restrictions_are_enforced():
    """The calls the tool description names are refused (``loop`` = an event loop)."""
    text = CodeActStrategy._restrictions_text()
    named = ["asyncio.run", "loop.run_until_complete"]
    for spelling in named:
        assert f"`{spelling}`" in text
    assert _validate("asyncio.run(main())")
    assert _validate("loop = asyncio.new_event_loop()\nloop.run_until_complete(main())")


def _response(code: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name="python_cell", arguments=json.dumps({"code": code}))],
        finish_reason="tool_calls",
    )


async def _run_cell(code: str) -> tuple[str, str]:
    llm = FakeLLMClient(
        scripted_responses=[_response(code, "cell"), _response("return_result('ok')", "done")]
    )

    class CellAgent(Agent, llm=llm):
        @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
        async def answer(self) -> str:
            """Run the cell."""
            ...

    agent = CellAgent()
    try:
        assert await asyncio.wait_for(agent.answer(), 30) == "ok"
        output = next(iter(agent.events.query(type="PythonOutput")))
        return output.error or "", output.stderr or ""
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(CASES))
async def test_codeact_v2_cell_refuses_listed_call(name):
    error, _ = await _run_cell(CASES[name][2])
    assert REFUSAL in error and HINT in error, error


@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(ALLOWED))
async def test_codeact_v2_cell_runs_alternatives(name):
    error, stderr = await _run_cell(ALLOWED[name])
    assert error == "" and stderr == "", (error, stderr)
