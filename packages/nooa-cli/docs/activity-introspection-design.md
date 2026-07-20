# Can the TUI report "where the agent is stopped" via sys.monitoring?

Short answer: **`sys.monitoring` is the wrong tool for this, and you don't need it.**
A point-in-time "where is the agent right now" probe is answered cheaply and
safely by `sys._current_frames()` + asyncio task introspection, with **zero**
changes to the core framework.

## How the TUI and agent actually run (verified)

`tui_application.py::_ensure_agent_loop` spins up a **dedicated asyncio loop on
its own daemon thread** (`nemo-tui-agent-loop`) via `loop.run_forever()`. The
agent's `handle()` runs as a task on that loop. prompt_toolkit / the UI runs on
`MainThread`. So:

- Agent work = one coroutine (`TUIAgent.handle`) on the agent loop, agent thread.
- The "asker" (a slash command / keybinding) originates on the UI thread.

## Why sys.monitoring doesn't fit

`sys.monitoring` is built for *continuous* event collection (debuggers, coverage,
profilers), not on-demand snapshots. Concretely, from experiments in this repo:

1. **Callbacks fire on the executing thread, not the asker's thread.** A `LINE`
   callback runs inline on whatever thread is running the instrumented bytecode.
   There is no thread-filtering API — you instrument the whole interpreter and
   disambiguate by reading `threading.current_thread()` *inside* the callback.
   To answer "where is the agent" you'd have to keep a running
   `last_frame_by_thread` dict and read the agent's entry from the UI thread.
   That's a continuously-maintained side channel, not a query.

2. **Cost.** Global `LINE` instrumentation measured **~3.1x** runtime on a tight
   loop (212ms vs 68ms). `PY_START` and `LINE`+`DISABLE` are ~1.0x — but `DISABLE`
   permanently de-instruments each location until `restart_events()`, so you lose
   line resolution after the first hit. There is no cheap always-on LINE feed.

3. **It tells you the *frame*, not the *await*.** The agent is async. When it's
   "stopped" it's usually suspended at an `await` (LLM call, tool, sleep). The
   running OS frame is just `BaseEventLoop._run_once`. What you want is the
   suspended coroutine's stack — which sys.monitoring does not give you.

## What works instead (prototype: `agent_locator.py`)

Two thread-safe, cheap primitives:

- **`sys._current_frames()`** — point-in-time frame for every thread. Safe to
  call from any thread, O(threads). Good for "what is the agent thread executing
  right now" when it's running CPU-bound Python.

- **asyncio task introspection** — `asyncio.all_tasks(loop)` + `Task.get_stack()`
  gives the *suspended await stack* of the agent coroutine, which is what "where
  is it stopped" usually means. **Caveat:** `asyncio.Task` is not safe to walk
  from another thread, so the UI thread must hop onto the agent loop first via
  `run_coroutine_threadsafe(locate_agent(loop), loop)` and read the result back.
  Frame data is returned as plain dataclasses, so it crosses the thread boundary
  cleanly. This works even while `handle()` is mid-turn because the snapshot
  coroutine just queues another callback on the same loop.

Verified live: the probe correctly reports the running task as
`TUIAgent.handle :: ActorRuntime._call_plan._execute_with_event` and the agent
thread's current frame.

## Recommendation

- For "where is the agent stopped / what is it doing", ship the
  `agent_locator` probe. No core changes; the TUI calls
  `locate_agent_from_other_thread(agent_loop)` from a slash command.
- The main blind spot: if the agent is blocked *inside a C call or a
  `run_in_executor` thread* (e.g. an LLM HTTP request), `get_stack()` shows the
  suspended Python frame at the `await`, which is exactly the right answer
  ("waiting on the model call"), and `_current_frames()` shows the worker thread
  if you want the lower-level detail.
- Only reach for `sys.monitoring` if you later want a *continuous timeline* of
  what the agent executed (a flight recorder), and accept the perf hit or use
  `PY_START`/sampled `LINE`+`DISABLE`+`restart_events`. That's a different
  feature from the on-demand probe you asked about.
