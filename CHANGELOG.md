# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow semantic versioning.

## [Unreleased]

- Initial public release of NVIDIA Object-Oriented Agents (NOOA).
- Fixed generation runtimes ignoring per-parameter
  `Annotated[T, spec(max_string=..., max_length=..., max_depth=...)]` rendering
  overrides, and aligned `print_prompt()` with method-level truncation settings.
- Security: MCP server configurations no longer expand host environment variables
  from `${VAR}` placeholders. Trusted caller code must resolve secrets and pass
  their values explicitly.
