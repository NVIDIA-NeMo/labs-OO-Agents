# Workspace trust

The interactive hosts (NOOA Atom with `nooa atom`, and the coding agent behind
`nooa-acp`) **load no skills from a folder you have not trusted** when you
open it. No file in an untrusted folder's skill directories is imported or
read as a skill when the session starts.

The gate stops what loads *automatically* when a folder is opened. It does not
limit what the model does during a turn: the agent can still edit files and
run commands and Python in the folder on your behalf.

## What is off in an untrusted folder

- **The folder's skills, all of them.** Nothing in `<folder>/.nooa/skills`,
  `.agents/skills`, `.cursor/skills`, `.claude/skills` or `.claude/commands`
  inside the folder loads: not code skills, not `SKILL.md` text skills, not
  the Markdown slash commands they define. `.nooa/skills` is not put on
  `sys.path`. Symlinks change nothing: these directories are left out
  wherever they or their files point.
- **Skill directories and skills the folder names.** `additional_skills_dirs`
  and `active_skills` in `<folder>/.nooa/settings.yaml`, and the legacy
  `.nooa/config.toml`, are ignored, also when the directories are absolute
  paths outside the folder. Relative `additional_skills_dirs` in your own
  settings name folders in the workspace, so they are left out too.
- **Atom only: model settings.** `<folder>/.nooa/llm_config.yaml` is not used,
  so a repository cannot point a model alias (and your API key) at its own
  server.
- **Atom only: stored agents.** A stored session whose agent is a
  `file.py:Class` in the folder (or a symlink there, wherever it points) does
  not load; the user sees an error that says how to trust the folder. An
  agent you choose with the `--agent` flag is not affected.

Still on: the folder's `AGENTS.md` instructions, and skills the agent writes
in this session (`self.skill_writing.create(...)`), which load as usual.

Your own skills keep loading in every folder, also when the folder contains
them (for example when the folder is your home folder): `~/.agents/skills`,
`~/.claude/skills`, `~/.claude/commands`, and absolute
`additional_skills_dirs` from your user `settings.yaml` and `NEMO_OO_SETTINGS`
files.

The `secrets.yaml` and `llm_config.yaml` that `nooa acp` and `nooa atom` read
when they start take their project layer from the launch directory's `.nooa`;
that layer is skipped when the launch directory is not trusted. `nooa-acp`
reads no per-folder model settings.

## What the agent sees, and what you see

The agent sees nothing about trust: an untrusted folder's skills are simply
not there, and the skills status and context blocks say nothing about them.
Once you trust the folder, its skills appear for the agent like any others.

You get one message when the session starts, if the folder has skills (Atom:
or model settings) that are off:

> This folder isn't trusted, so its skills are off. Type /trust to allow.

`nooa-acp` and Atom over ACP send it as a message to the client. Headless Atom
sessions log it as a warning.

## How to trust a folder

- **When asked.** Atom asks at the first prompt in an untrusted folder, if
  the client can show forms: "Do you trust this folder?" Yes trusts it. No
  keeps this session untrusted; you are asked again next session. Stop
  cancels the prompt.
- **`/trust`.** Trusts the session's folder and loads its skills at once
  (`.nooa/skills` skills are active; in Atom, activate the others with
  `/skills`). Its reply goes to you, not to the agent. In Atom, the folder's
  model settings apply from the next session.
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

A folder's own `.nooa/settings.yaml` is never read for `trusted_folders`: a
repository cannot trust itself.

## Limits

- Stored sessions: restoring a session snapshot is not covered by this gate
  (tracked separately).
- Starting a host with `python -m ...` from inside a folder puts that folder
  on `sys.path`, so a stored module spec could import from it. The installed
  `nooa`, `nooa-acp` and `nooa atom` scripts do not do this.

## For host authors

Plain NOOA agents, `SkillRegistry` and `SkillWriting` used directly are not
affected. The hosts gate by leaving the folder's skill directories out:
`load_coding_skills_dirs` (nooa-cli) and `load_skills_dirs` (Atom) take
`trusted=`, and `SkillWriting(..., load_existing=False)` loads none of the
skills already saved while still loading the ones created in the session.
The core API is in `nooa.workspace_trust` (`is_trusted`, `trust`,
`can_trust`, `is_inside`, `untrusted_notice`).
