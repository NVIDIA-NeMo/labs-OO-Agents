# Install and verify Windows wheel bundles

The `windows-wheels` CI job builds `nooa`, `nooa-cli`, and `nooa-acp` with the
release build command, then installs them into a fresh Windows Python 3.12 or
3.13 environment outside the checkout. Each downloadable artifact contains:

- `dist/`: the three wheels and their source distributions.
- `acceptance/`: standalone acceptance scripts, with no checkout dependency.
- `tokenizer-cache/`: verified tokenizer data for offline startup checks.
- `frozen-dependencies.txt`: third-party dependency constraints from `uv.lock`.
- `installed-requirements.txt`: the packages installed by that CI run.
- `wheel-smoke.json`: the acceptance results, including package origins.

The bundles are CI build artifacts, not a published release. Use the artifact
from a successful run for the commit you intend to test. Dependencies still
download from the package index during installation; this is not an offline
installer.

## Install without a source checkout

Extract a bundle into an empty directory, for example `C:\nooa-wheel-bundle`.
Open PowerShell in that directory and run:

```powershell
$bundle = (Get-Location).Path
uv venv --no-config --no-project --python 3.13 .venv
if ($LASTEXITCODE -ne 0) { throw "Could not create the environment" }
$wheelPython = Join-Path $bundle ".venv\Scripts\python.exe"
$wheelFiles = @(Get-ChildItem -LiteralPath .\dist -Filter *.whl -File | Select-Object -ExpandProperty FullName)
if ($wheelFiles.Count -ne 3) { throw "Expected the three NOOA wheels" }
uv pip install --no-config --python $wheelPython --constraint .\frozen-dependencies.txt --strict @wheelFiles
if ($LASTEXITCODE -ne 0) { throw "Wheel installation failed" }
```

Use `--python 3.12` for Python 3.12. Install all three local wheels in one
command so the resolver cannot substitute an older NOOA package from the
index. Constraints pin third-party versions without preinstalling dependencies
that a wheel forgot to declare. The base install does not require the optional
Tree-sitter grammars, viewer, or data-science packages.

Prepare the tokenizer cache before running the offline check. CI bundles
already contain it; this command also supports a locally built bundle and
downloads the resource when needed. The tokenizer library verifies its pinned
content hash:

```powershell
$env:TIKTOKEN_CACHE_DIR = Join-Path $bundle "tokenizer-cache"
$env:CUSTOM_TIKTOKEN_CACHE_DIR = $env:TIKTOKEN_CACHE_DIR
uv run --no-config --no-project --no-sync --python $wheelPython python -I -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
if ($LASTEXITCODE -ne 0) { throw "Tokenizer cache preparation failed" }
```

The separate cache avoids relying on a previously warmed installation. The
offline check passes it explicitly to both the model client and its tokenizer;
network access during the acceptance workflow remains blocked by the local
test responder.

Run the bundled acceptance check with that environment's Python:

```powershell
uv run --no-config --no-project --no-sync --python $wheelPython python -I -X utf8=0 .\acceptance\smoke_installed_wheels.py --tokenizer-cache $env:TIKTOKEN_CACHE_DIR --report .\wheel-smoke-local.json
if ($LASTEXITCODE -ne 0) { throw "Installed-wheel acceptance failed; inspect wheel-smoke-local.json" }
```

The script rejects editable installs and imports outside the selected virtual
environment. It checks both `nooa-acp` and `nooa acp`, using a loopback HTTP model
responder and real coding tools. A synthetic arithmetic test must fail before
the edit and pass afterward; the acceptance tests themselves must stay
unchanged. The report also checks file-edit activity, command results, Unicode
paths, JSON-RPC stdout, normal process exit, and temporary-directory cleanup.
No model credentials are needed. This check covers protocol behavior; it does
not drive an editor UI.

## Connect Zed to the installed command

In Zed's `settings.json`, add a custom external agent. Replace the path with the
absolute location of the environment you just created, and use an existing
NOOA model alias or provider model name:

```json
{
  "agent_servers": {
    "NOOA": {
      "type": "custom",
      "command": "C:\\nooa-wheel-bundle\\.venv\\Scripts\\nooa-acp.exe",
      "args": ["--model", "my-model-alias"],
      "env": {
        "CUSTOM_TIKTOKEN_CACHE_DIR": "C:\\nooa-wheel-bundle\\tokenizer-cache",
        "TIKTOKEN_CACHE_DIR": "C:\\nooa-wheel-bundle\\tokenizer-cache"
      }
    }
  }
}
```

Open the repository to work on and choose **NOOA** in the agent panel. The
absolute command path keeps the editor on the tested wheel environment. Model
configuration and credentials follow the usual NOOA configuration rules; see
[ACP setup](../packages/nooa-acp/README.md#quick-start-zed).

For an independent editor acceptance profile, Zed supports:

```powershell
zed --new --user-data-dir C:\nooa-editor-acceptance C:\my-test-repository
```

Put that profile's agent configuration in
`C:\nooa-editor-acceptance\config\settings.json`. This keeps the acceptance
profile separate from normal Zed settings and thread history.

Verify the following in a disposable test repository:

1. Ask NOOA to fix a small function and run its tests. Confirm the source diff,
   failed baseline, successful final command, and Unicode text are visible in
   the agent's tool cards. Rerun the unchanged tests independently.
2. Start a long-running command, wait for its output, then stop the turn in
   Zed. Confirm the command's processes exit and a new prompt works in the
   same thread.
3. Close and reopen Zed. Open the earlier thread from Thread History; use
   **Import Threads** when the session is only in the agent's durable store.
   Confirm earlier messages are replayed and a follow-up uses that history.

These observations depend on the editor version as well as ACP. A passing
protocol report alone does not establish that its UI displayed the activity.

## Reproduce a bundle from the repository

In the repository root, build source distributions and wheels with the same
command used by release CI:

```powershell
foreach ($package in @("nooa", "nooa-cli", "nooa-acp")) {
    uv build --no-sources --package $package --out-dir .nooa\wheel-bundle\dist
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $package" }
}
uv export --frozen --package nooa-acp --no-dev --no-emit-workspace --no-hashes --no-annotate --output-file .nooa\wheel-bundle\frozen-dependencies.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency export failed" }
New-Item -ItemType Directory -Path .nooa\wheel-bundle\acceptance -Force | Out-Null
foreach ($script in @("smoke_installed_wheels.py", "smoke_acp_coding.py", "_acp_smoke_support.py")) {
    Copy-Item -LiteralPath (Join-Path scripts $script) -Destination .nooa\wheel-bundle\acceptance
}
```

Use a fresh output directory for each build to avoid mixing package versions.
Copy the bundle outside the checkout, then follow the installation and
acceptance commands above. The `--no-sources` build and outside-checkout install
are both necessary: an editable source installation can hide missing packaged
files or dependency declarations.
