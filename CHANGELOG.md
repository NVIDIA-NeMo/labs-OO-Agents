# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow semantic versioning.

## [Unreleased]

- Initial public release of NVIDIA Object-Oriented Agents (NOOA).
- Security: MCP server configurations no longer expand host environment variables
  from `${VAR}` placeholders. Trusted caller code must resolve secrets and pass
  their values explicitly.
