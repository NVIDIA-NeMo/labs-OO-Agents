# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow semantic versioning.

## [Unreleased]

- Initial public release of NVIDIA Object-Oriented Agents (NOOA).
- Added a SQLite durable-operation ledger with request identity, renewable leases, fencing, transition history, and explicit unknown outcomes.
- Added additive SQLite schema migration, fail-closed operation decoding, authoritative database time, and explicit unknown-outcome reconciliation.
- Security: MCP server configurations no longer expand host environment variables
  from `${VAR}` placeholders. Trusted caller code must resolve secrets and pass
  their values explicitly.
