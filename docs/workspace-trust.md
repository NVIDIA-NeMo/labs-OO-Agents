# Workspace trust

The interactive hosts (NOOA Atom with `nooa atom`, and the coding agent behind
`nooa-acp`) **do not run code from a folder you have not trusted** when you
open it. No file in an untrusted folder is imported when the session starts.

The gate stops *automatic* execution when a folder is opened. It does not
limit what the model does during a turn: the agent can still edit files and
run commands and Python in the folder on your behalf.

## What is off in an untrusted folder

- **Code skills.** Packages in `<folder>/.nooa/skills`, and packages or `.py`
  files in `.agents/skills`, `.cursor/skills`, `.claude/skills` and
  `.claude/commands` inside the folder, are not imported or activated, and
  `.nooa/skills` is not put on `sys.path`. `SKILL.md` text skills still load.
  The agent can still write new skills; code skills load after you trust the
  folder.
- **Symlinks count as the folder.** A skill file, package or skill directory
  is workspace code when it was found in the folder *or* points into it. A
  link in the folder that points elsewhere (the rest of a clone, `/tmp`, a
  sibling checkout) is still refused. The same rule applies to a stored
  `file.py:Class` agent (Atom).
- **Skill directories and skills the folder names.** `additional_skills_dirs`
  and `active_skills` in `<folder>/.nooa/settings.yaml`, and the legacy
  `.nooa/config.toml`, are ignored, also when the directories are absolute
  paths outside the folder.
- **Atom only: model settings.** `<folder>/.nooa/llm_config.yaml` is not used,
  so a repository cannot point a model alias (and your API key) at its own
  server.
- **Atom only: stored agents.** A stored session whose agent is a
  `file.py:Class` in the folder does not load. An agent you choose with the
  `--agent` flag is not affected.

Your own skills keep loading in every folder, also when the folder contains
them (for example when the folder is your home folder): `~/.agents/skills`,
`~/.claude/skills`, `~/.claude/commands`, and absolute
`additional_skills_dirs` from your user `settings.yaml` and `NEMO_OO_SETTINGS`
files.

The `secrets.yaml` and `llm_config.yaml` that `nooa acp` and `nooa atom` read
when they start take their project layer from the launch directory's `.nooa`;
that layer is skipped when the launch directory is not trusted. `nooa-acp`
reads no per-folder model settings.

When code skills (Atom: or model settings) were skipped, the session shows
one notice, for example in `nooa-acp`:

> Code skills in this folder are off because it is not trusted. Type /trust
> to allow.

## How to trust a folder

- **When asked.** Atom asks at the first prompt in an untrusted folder, if
  the client can show forms: "Do you trust this folder?" Yes trusts it. No
  keeps this session untrusted; you are asked again next session. Stop
  cancels the prompt.
- **`/trust`.** Trusts the session's folder and loads its code skills at once
  (`.nooa/skills` skills are active; activate the others with `/skills`). In
  Atom, its model settings apply from the next session.
- **Your user settings.** List folders in `~/.config/nooa/settings.yaml`
  (`nooa.paths.get_user_dir("settings.yaml")`); a folder trusts all folders
  below it. Entries must be absolute paths (`~` is expanded); relative ones
  are ignored with a warning. `/trust` writes this list: it appends the entry
  when the file has no `trusted_folders` yet (comments are kept), and
  rewrites the file from its data otherwise (comments are lost). It refuses
  a file that is not valid YAML instead of replacing it.

  ```yaml
  trusted_folders:
    - /home/me/src/my-project
  ```

- **CI and scripts.** Put `trusted_folders` in a file named by
  `NEMO_OO_SETTINGS`.

The home folder and filesystem roots cannot be trusted with `/trust` (and
Atom does not ask there): "Open a project folder instead; trusting your home
folder would trust every repository in it."

A folder's own `.nooa/settings.yaml` is never read for `trusted_folders`, and
`atom.trust_workspace` in a settings file is ignored: a repository cannot trust
itself.

## Atom: `AtomOptions.trust_workspace`

Code that builds Atom options can set `AtomOptions.load(root,
trust_workspace=True)` (or `False`). It decides, for those options, the
`skills_dirs` and `active_skills` taken from the folder's settings and
`AtomOptions.workspace_trusted()`, which the session factory passes to the
agent as `workspace_trusted` (its code skills). Stored `file.py:Class` agent
specs and the folder's `llm_config.yaml` follow `trusted_folders` only.
`create_session_agent` loads its own `AtomOptions` from the folder, so for
sessions it builds use `trusted_folders` (for example in a `NEMO_OO_SETTINGS`
file).

## Limits

- Stored sessions: restoring a session snapshot is not covered by this gate
  (tracked separately).
- Starting a host with `python -m ...` from inside a folder puts that folder
  on `sys.path`, so a stored module spec could import from it. The installed
  `nooa`, `nooa-acp` and `nooa atom` scripts do not do this.

Plain NOOA agents and `SkillRegistry` or `SkillWriting` used directly are not
affected unless they set `SkillRegistry.code_allowed`. The API is in
`nooa.workspace_trust` (`is_trusted`, `trust`, `code_allowed_outside`).
