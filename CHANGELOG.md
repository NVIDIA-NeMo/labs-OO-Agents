# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow semantic versioning.

## [Unreleased]

- Initial public release of NVIDIA Object-Oriented Agents (NOOA).
- Text skills now load as regular `Skill` objects with documentation from
  `SKILL.md`, a stable `files: list[SkillFile]` manifest, and a skill-root-scoped
  `ShellTools` dependency. `TextSkill(path=...)` remains available as the
  compatibility constructor; migrate `read_file()` / `run_script()` calls to
  `skill.shell.read()` / `skill.shell.run()`.
- Security: MCP server configurations no longer expand host environment variables
  from `${VAR}` placeholders. Trusted caller code must resolve secrets and pass
  their values explicitly.
