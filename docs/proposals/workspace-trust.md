<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Proposal: workspace trust for repository-supplied skills

**Status:** draft, for discussion
**Affects:** `nooa.skill_registry`, `nooa_cli.coding`, `nooa-acp`

## The problem

Creating a coding-agent session **imports Python from the workspace, before any
prompt is sent**. Opening a folder is therefore enough to execute code it
contains, as the user, in a process holding model credentials.

This was verified end to end rather than inferred. A marker-writing file placed
at `<repo>/.claude/skills/probe.py`, followed by an ACP `session/new` on that
directory:

```
skill roots discovered: ['<repo>/.claude/skills']
EXECUTED VIA new_session: True
```

The only action was opening the folder.

### Three paths execute

| Path | Trigger | Executes? |
|---|---|---|
| `.agents/skills`, `.cursor/skills`, `.claude/skills`, `.claude/commands` — any `.py` | `_register_python_skill` → `spec.loader.exec_module` | **yes**, at import, before anything checks the file defines a skill |
| Additional roots named by the repo's own `.nooa/settings.yaml` or `.nooa/config.toml` | same as above | **yes**, and the paths are not confined to the workspace |
| `<workspace>/.nooa/libs/<pkg>/` | `discover_libs` → `importlib.import_module`, directory prepended to `sys.path` for the process lifetime | **yes** |
| A directory containing `SKILL.md` | `_register_text_skill` → `TextSkill(path=...)` | **no** — read as text, and `cmd.*` is excluded from what the model sees |

The last row matters: one of the four is already inert, and it is the one that
carries most of the value of opening an unfamiliar repository.

### Why it is sharper here than in a CLI

`pytest` importing `conftest.py` and `npm install` running lifecycle scripts are
the same class of problem. The difference is the triggering action. Running a
command is a deliberate act; **opening a folder in an editor is not**, and an
ACP host makes that the entry point.

## Principle: gate on provenance, not location

The instinct is to gate on file type — markdown inert, Python gated. That breaks
immediately: agents write their own skills, and autonomous hosts load skills
programmatically with no human present.

The question is not *what is this file* but **where did this code come from**.

### Three provenance classes

**1. Host-declared — trusted, no prompt.**
`CodingAgent(skills_dirs=[...], libs_dir=...)`, or roots named in *user-level*
settings. A program passing paths is itself the trust decision; a human editing
their own config likewise. **Autonomous and embedded hosts use this path today
and are unaffected by this proposal.**

**2. Agent-authored — trusted, no prompt.**
Anything this NOOA wrote through `SkillWriting` during a session. Writing it
already required running code under the user's authority, so importing it grants
nothing new.

**3. Workspace-discovered — requires a decision.**
Conventional skill roots, roots named by the repository's own settings, and
`.nooa/libs` content that was already present when the workspace was opened.
This is the only class whose behaviour changes.

There is prior art for exactly this axis. Claude Code gates external imports in
a *project* `CLAUDE.md`, while the same imports in a user's own
`~/.claude/CLAUDE.md` are ungated — the documented reason being that you wrote
them.

### Separating classes 2 and 3, which share a directory

An ACP host points `libs_dir` at `<workspace>/.nooa/libs`, so "the host said
where to put agent-authored libraries" must not become "trust whatever is
already sitting there". The split is **temporal, not spatial**:

- On an untrusted workspace, skip `discover_libs()` and `_register_python_skill()`
  at session start — nothing already on disk is imported.
- Still allow `libs.create()` and `reload()` **within** the session, so an agent
  authors a library and uses it immediately.

Same directory, different provenance, no new machinery.

## What an untrusted workspace still gets

Everything except imported Python:

- `SKILL.md` skills load as today (inert, and already excluded from model-visible
  status)
- installed `nooa.skills` entry points — these come from the user's environment,
  not the repository
- repository instructions (`AGENTS.md`), all coding tools, sessions, MCP

Capabilities are gated individually rather than the session being degraded. This
follows Claude Code, which has no VS Code-style "restricted mode": untrusted
folders open normally and specific capabilities sit behind the gate.

## Granting trust

| Host | Mechanism |
|---|---|
| ACP | `session/request_permission` with `PermissionOption`s: *Trust this workspace* / *Just this session* / *Skip project skills* |
| Terminal | the same prompt, or an explicit `nooa trust` |
| Non-interactive | `NOOA_TRUST_WORKSPACE=1`, or a list in user-level config — CI and headless autonomous runs are unaffected |

The ACP mechanism already exists in the protocol (`RequestPermissionRequest`,
`PermissionOption`), so no protocol work is required.

### Storage and keying

- Stored **user-global**, never in the workspace. A decision stored in the
  repository could be shipped by the repository.
- Keyed to the **git repository root**, not the working directory, so opening a
  subdirectory does not re-prompt.
- **Nested repositories do not inherit** a parent's trust.
- Outside a git repository, trust is **session-only** and never persisted.

The last three are lifted directly from Claude Code's documented behaviour; each
is a case that would otherwise have been got wrong.

## Prior art

| System | Gate | Granularity | Re-prompt on change |
|---|---|---|---|
| Claude Code | trust dialog per project | git repo root | **no** — trusted hooks take effect immediately |
| VS Code | Workspace Trust | folder | no |
| JetBrains | untrusted project mode | project | no |
| `direnv` | `direnv allow` | file | **yes** — keyed to a content hash |
| Git | does not run hooks from a clone | — | — |

The incumbents converge on a one-time folder decision. `direnv` is the outlier
and the strictest: trust is bound to the bytes, so editing the file re-prompts.

This matters for phase 4 below. Path-keyed trust has a known sharp edge — a
repository trusted today can add `.claude/skills/new.py` tomorrow and it executes
silently on the next `git pull`. Claude Code has this property with hooks.
Content-hashed trust does not.

## Phasing

**1. Confine workspace-supplied paths.** A repository's own settings may only
name roots inside the workspace; user-level and env-var config keep pointing
anywhere. No UX, no trust model, no behaviour change for anyone legitimate.

**2. Provenance split, defaulting to trust.** Implement the three classes and
the temporal rule, but default class 3 to *trusted* while logging what would
have been gated. Nothing breaks; the logs show the real blast radius.

**3. Flip the default and add the prompt.** Class 3 becomes untrusted until
granted.

**4. Consider content-hashed trust.** Strictly better than path-keyed, at the
cost of re-prompting when project skills legitimately change.

Phase 1 is worth doing regardless of whether the rest proceeds.

## Open questions

1. **Should agent-authored skills survive across sessions?** The temporal rule
   means a library the agent wrote yesterday is not auto-imported today on an
   untrusted workspace. A marker written at creation time, HMAC-keyed to a
   user-global secret, would fix that — at the cost of real machinery and a
   secret to manage.
2. **Is the repository root the right granularity for a monorepo**, where one
   trust decision covers many teams' directories?
3. **What should a declined prompt do** — skip silently, or surface once per
   session what was skipped and how to enable it?

## Not in scope

MCP servers. In Claude Code they are a **separate gate** from folder trust —
per-server approval, independent of the dialog. In NOOA they arrive from the ACP
client rather than the repository, so they are a different trust question and
this proposal does not touch them.
