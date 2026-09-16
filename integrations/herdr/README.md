# Herdr integration for the NOOA TUI

Herdr workflow plugins can classify an agent that Herdr already recognizes, but
they cannot register a new process identity. NOOA therefore needs a native Herdr
detector plus a bundled screen-state manifest.

`0001-add-nooa-tui-detection.patch` is based on Herdr commit
`d76657f2c7fc18dcce3b9af43842c8afaba1646b`. It adds:

- strict detection for `uv run nooa tui ...` and its Python `nooa tui` child;
- rejection of non-TUI commands such as `nooa start-dev`;
- idle, working, and blocked screen-state rules for the NOOA TUI;
- `nooa` support in `herdr agent start` and the English next-version docs.

Apply it to a checkout of Herdr:

```bash
git -C /path/to/herdr apply --check \
  /path/to/labs-OO-Agents/integrations/herdr/0001-add-nooa-tui-detection.patch
git -C /path/to/herdr apply \
  /path/to/labs-OO-Agents/integrations/herdr/0001-add-nooa-tui-detection.patch
```

The focused validation commands are:

```bash
cargo fmt --check
cargo test nooa
cargo test all_bundled_manifests_parse_and_validate
```

After building or installing the patched Herdr, launch NOOA as usual:

```bash
cd labs-OO-Agents
NEMO_OO_SETTINGS=.nooa/settings.yaml uv run nooa tui \
  --working-dir /path/to/your/project
```

Herdr will identify that pane as `nooa`. For a Herdr-managed start, the
equivalent agent command is `herdr agent start <name> --kind nooa -- tui`.
