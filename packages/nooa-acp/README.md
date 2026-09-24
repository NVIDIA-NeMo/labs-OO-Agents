# nooa-acp

Superseded by [nooa-coder](../nooa-coder/README.md); this package exists so
`uv add nooa-acp` keeps installing the ACP server. The `nooa-acp` console
script is provided by nooa-coder.

This package contains no code. It depends on `nooa-coder`, which provides the
`nooa-acp` and `nooa-coder` console scripts and the `nooa acp` command. New
installations should use `uv add nooa-coder` (or `uv add "nooa[acp]"`).
