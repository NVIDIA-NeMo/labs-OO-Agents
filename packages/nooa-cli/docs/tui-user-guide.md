# Using the NOOA terminal UI

The NOOA terminal UI is an interactive coding workspace. You can talk to an
agent, watch tools run, inspect what happened, manage sessions and memories,
and keep editing your next message while work is in progress—all without
leaving the terminal.

This guide starts with the basics and then tours the features. For implementation
details, see [How the TUI works](tui-rendering-architecture.md).

## Start here

Install and launch the TUI from the project you want the agent to work in:

```bash
uv add nooa-cli
nooa tui
```

On first use, connect a model:

```text
/connect
```

You can also provide a service URL, for example:

```text
/connect https://api.anthropic.com
/connect https://api.openai.com/v1
/connect https://build.nvidia.com
/connect http://localhost:11434
```

`/connect` discovers models, asks for a key when needed, and stores project-local
configuration under `.nooa/`. Type `/help` at any time to see the commands
available in your installation.

## The screen at a glance

The default fullscreen interface has three areas:

- **Transcript** — your conversation, tool activity, diffs, progress, and errors.
- **Composer** — the editable input after the `❯` prompt.
- **Toolbar** — compact session information such as the model and context use,
  plus temporary confirmations and warnings.

The transcript follows new output until you scroll up. While you are reading an
older section, incoming output does not pull the view away. Click **Return to
bottom** or press `Ctrl+End` to follow the live tail again.

## Write and send a message

Type naturally and press `Enter` to submit. The composer supports normal cursor
movement and editing, multi-line drafts, history, paste, completion, and mouse
positioning.

| Action | Input |
| --- | --- |
| Submit | `Enter` |
| Insert a newline | `Shift+Enter` (terminal support varies) |
| Complete a slash command, path, or option | `Tab`; `Shift+Tab` goes backward |
| Select while editing | `Shift` + arrow keys, or mouse drag |
| Copy composer selection | `Ctrl+C` |
| Cut composer selection | `Ctrl+X` |
| Recover the most recent queued message | `Up` on an empty composer |
| Browse input history | `Up` / `Down` on an empty composer |
| Clear draft / interrupt current work | `Ctrl+C` |
| Exit immediately | `Ctrl+D` |

`Ctrl+C` is contextual. With selected composer text it copies. Otherwise its
first press clears the composer and requests interruption if the agent is
working; a second press during the confirmation window exits.

The completion menu understands slash commands and many command arguments. It
also expands `@path` mentions into Markdown links before sending, making it easy
to point the agent at project files.

## Read, scroll, select, and copy

Use the mouse wheel or `PageUp` / `PageDown` to move through the transcript.
`Ctrl+Home` jumps to the beginning and `Ctrl+End` returns to the latest output.
Scrolling works even when the pointer is over the composer or status area.

In fullscreen mode, drag across transcript text to copy it. Copying uses the
logical text, so terminal colors and visual soft wraps are not included. A
small status message confirms success or explains why the system clipboard was
unavailable.

Fullscreen mouse reporting can interfere with selection provided by your
terminal or tmux. You have three escape hatches:

- hold `Alt`/`Option` or `Shift` while dragging to request native selection;
- press `F6` to turn application mouse handling off, then press it again to
  restore application scrolling and selection;
- start with `--display-mode native` when native scrollback is the priority.

Modifier behavior ultimately depends on the terminal and multiplexer, so `F6`
is the reliable fallback.

## Understand agent activity

The transcript presents more than chat text:

- Markdown answers, tables, links, and highlighted code;
- tool calls and their results;
- Python execution panels when enabled;
- bounded file diffs;
- progress and status changes;
- warnings, rejected actions, and recoverable errors.

The agent can continue streaming while you edit. If you submit while it is
busy, the message enters its input flow rather than blocking the screen. Use
`Up` on an empty composer to withdraw the most recent input that has not yet
been consumed.

Useful visibility controls include:

```text
/show-python on     # show agent Python execution panels
/show-diffs on      # show bounded file-edit diffs
/context             # inspect context usage
/compact             # summarize older conversation context now
/reasoning            # inspect or change reasoning settings
```

## Slash commands

A slash command performs a TUI action instead of sending ordinary prose to the
agent. Start typing `/` and use `Tab` to discover commands. `/help` is the
authoritative list because installed skills and plugins can add more.

### Models and appearance

```text
/connect             # configure a model service
/models              # browse available models
/model               # show or select the active model
/reasoning           # configure model reasoning behavior
/theme               # choose the color theme
/toolbar             # inspect or configure toolbar items
```

### Sessions and history

```text
/session             # browse or manage sessions
/events              # inspect durable events from this session
/activity            # inspect activity for the current turn
/jobs                 # inspect background jobs
/todos                # inspect the agent's task list
/clear                # clear the visible conversation
```

Explorer commands open inside the fullscreen shell. Common navigation is shown
in each footer; generally arrows or `j`/`k` move, `PageUp`/`PageDown` page,
`Home`/`End` jump, `Enter` opens or acts, `/` searches where supported, and
`Esc`, `q`, or `Ctrl+C` closes. Mouse-wheel navigation is available in
fullscreen mode. Exact actions differ by explorer.

### Files, skills, and agent behavior

```text
/edit <file>          # edit a file (available when the display mode can hand off safely)
/skills               # inspect, activate, or add workflow skills
/keep-going           # audit completed turns and continue when work remains
/exit                 # close the TUI
```

Fullscreen mode intentionally refuses external-editor handoff because another
interactive program cannot safely share its alternate-screen ownership. Ask the
agent to edit the file, use a shell outside the TUI, or restart in an explicit
native display mode when you need `$EDITOR`.

### Memory and reflection

Long-term memory is opt-in:

```text
/memory on            # project memory shared by sessions
/memory local         # memory for this session only
/memory off           # detach memory from the current agent
/memories             # browse, complete, or forget memories
/reflection on        # consolidate new memories while idle
/reflection now       # run consolidation immediately
/reflection off
```

Project memory normally lives at `.nooa/memory/memory.sqlite`; local memory is
stored beside the session database. Settings are remembered per agent.

### MCP servers

The TUI can connect tools through Model Context Protocol servers:

```text
/mcp list
/mcp add docs https://docs.example.com/mcp
/mcp approve docs
/mcp connect docs
/mcp disconnect docs
/mcp remove docs
```

Adding configuration does not grant trust. Before any server transport or local
process starts, the TUI presents a secret-safe fingerprint and requires an
explicit approval code. Changing the server definition invalidates approval.
Keep secrets in environment variables rather than writing literal values into
project settings.

## Sessions

NOOA persists sessions, so closing the terminal does not make the conversation
history disappear. `/session list` opens the session explorer. From there you can
browse prior sessions and use the actions shown in its footer. Session history
and the fullscreen transcript are related but not identical: the transcript is
a bounded live working set, while durable event storage is the source for
historical inspection.

The toolbar's session item helps identify where you are. Custom agents and
project-local settings can give sessions different models, skills, memory
scopes, and tool connections.

## Events, activity, jobs, todos, and memories

These views answer different questions:

- **`/events`** — What exactly was recorded in this session?
- **`/activity`** — What did the current or recent agent turn do?
- **`/jobs`** — Which background tasks exist and what state are they in?
- **`/todos`** — What work has the agent planned or completed?
- **`/memories`** — What durable information has the memory system retained?
- **`/session list`** — Which conversation session should I inspect or resume?

They currently share the fullscreen subview host, keyboard routing, resize
behavior, and terminal safety. Their individual layouts and actions vary. A
planned follow-up will give the whole family a more consistent visual language,
mouse interaction, selection, and copy behavior.

## Keep-going mode

Keep-going mode lets a separate judge model inspect a completed turn and ask
the agent to continue if meaningful work remains:

```text
/keep-going model nemotron3-nano-30b
/keep-going on
```

It is off by default. New user input wins: it cancels an in-flight audit rather
than competing with you. Use `/keep-going off` to disable it.

## Skills and extensions

NOOA discovers skills from installed `nooa.skills` entry points and conventional
project/user directories such as `.agents/skills`, `.claude/skills`, and
`.cursor/skills`. Discovered workflow skills are available but are not exposed
to the model until you activate them or invoke them explicitly.

Packages can also contribute slash commands and toolbar items. This means the
commands shown by your TUI may be richer than the built-in list in this guide.
Use `/skills` and `/help` to inspect the running system.

A custom coding agent can be selected without forking the TUI:

```bash
nooa tui --agent your_package:YourCodingAgent
```

The custom class participates through the same interactive-agent interface as
the default local agent.

## Configuration

Project configuration lives in `.nooa/`:

| File | Purpose |
| --- | --- |
| `.nooa/settings.yaml` | TUI preferences and default model |
| `.nooa/llm_config.yaml` | saved model aliases |
| `.nooa/secrets.yaml` | API keys keyed by environment-variable name |

Settings and model configuration are loaded on the next launch. You can inspect
all discovered registry layers with `nooa config show`.

A small example:

```yaml
tui:
  display_mode: fullscreen
  vi_mode: false
  show_python: false
  show_diffs: true
  toolbar_items: [time, model, context, session]
```

Command-line options override settings for that launch. Run `nooa tui --help`
for the complete list.

## Display modes

Most users should keep the default `fullscreen` mode.

```bash
nooa tui --display-mode fullscreen
nooa tui --display-mode native
nooa tui --display-mode native-replay
```

- **`fullscreen`** gives NOOA an alternate screen, stable composer, application
  scrolling, mouse selection, responsive reflow, and embedded explorers.
- **`native`** prints inline and relies on terminal scrollback. It does not
  replay history when the terminal changes size.
- **`native-replay`** is the older inline behavior and rebuilds retained output
  after resize.

The mode is chosen at startup and cannot be switched safely in place.

## Safety and privacy

The fullscreen renderer sanitizes terminal control sequences from models,
tools, logs, and subprocesses. Styling and safe links can be displayed, but
content cannot smuggle cursor movement, screen erasure, title changes, or
clipboard writes through ordinary output.

Clipboard copying is always a user action. Locally, the TUI uses an available
platform clipboard helper. In a remote session it can use terminal clipboard
forwarding when supported. Clipboard text is validated and failures are shown
rather than silently treated as success.

MCP connections require explicit approval as described above. Model keys and
other secrets should remain in the designated secret store or environment, not
in prompts, logs, or checked-in settings.

## Troubleshooting

### I cannot select text with my terminal

The fullscreen app owns the mouse by default. Try modified drag, or press `F6`
to enable native terminal selection. Press `F6` again when you want application
mouse scrolling back.

### Copy says the clipboard is unavailable

A local session needs a supported clipboard command for the platform. A remote
session needs terminal/tmux support for clipboard forwarding. The selected text
also remains available to prompt_toolkit where possible, so the failure does
not have to destroy your selection.

### Output arrived while I was reading older text

That is expected: the transcript preserves your scroll anchor. Click **Return
to bottom** or press `Ctrl+End` when you are ready to follow live output.

### The terminal is too narrow

The interface degrades deliberately on small screens, but a wider terminal will
make Markdown, diffs, and explorer panes easier to read. Unicode source is not
modified merely because its display projection cannot fit.

### My external editor will not open

Fullscreen mode does not hand terminal ownership to `$EDITOR`. Restart with
`--display-mode native` for that workflow, edit in another terminal, or ask the
agent to make the change.

### A command is missing

Run `/help`. Some commands come from optional skills or plugins, and some
features require configuration before their command becomes active.

### The UI or terminal state looks broken

First try a normal `/exit` or `Ctrl+D`, which runs the full cleanup path. If the
process is no longer responsive, terminate it from another shell and run
`reset` in the affected terminal if necessary. When reporting a reproducible
problem, include the terminal, multiplexer, display mode, resize or handoff
steps, and whether the process was still alive; do not include secrets.

## What is coming next

The fullscreen foundation makes richer interactions possible without giving up
terminal safety. Planned work includes a unified, polished design for all
explorers; consistent mouse and copy support; semantic Copy buttons on fenced
Markdown code blocks; “Copy as Markdown” based on original source; more
keyboard discoverability; and broader PTY/tmux/SSH lifecycle testing.
