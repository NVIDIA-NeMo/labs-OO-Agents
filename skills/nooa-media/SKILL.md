---
name: nooa-media
description: Attach images, audio, video, and PDFs to NOOA agents. Use typed parameters or CodeAct show(), configure media_inputs, and troubleshoot text-only notices, URLs, MIME types, and size limits.
compatibility: nooa package
---

# Work with NOOA media

See [the media guide](../../docs/media.md), the
[image quickstart](../../examples/quickstart/13_multimodal.py), and
[Connect](../../docs/model-connect.md) for setup.

## Check setup and authorization

- Identify the registry file, alias, model, endpoint, and API style. Preserve
  unrelated entries and credentials. Load a file explicitly with `reload_registry`
  if needed; see [model configuration](../../docs/model-configuration.md).
- Confirm consent before transmitting user data. Media bytes and URLs,
  including signed query strings, may disclose private data.
- Before configuration, network/media checks or registry writes, follow
  [nooa-connect](../nooa-connect/SKILL.md), the canonical permission-first
  headless guide. Bound synthetic samples separately from private user media;
  permission to attach media is not permission to replace an alias.
- NOOA assumes models are text-only until configured. Set `media_inputs` on the
  registry entry or client constructor: `image`, `audio`, `video`, `pdf`.
  Omitted/null/empty lists mean text-only. Python also accepts tuples; bare
  strings, duplicates, and unknown values are invalid.
- If you enable a format the provider does not accept, requests may fail.
  Choose settings from the provider's documentation.

## Attach media

Import types at module level from `nooa.media`. Prefer direct typed parameters
such as `image: Image`; leave `{image}` out of docstrings because arguments are
already shown. For extraction/classification, use `@strategy(PredictStrategy())`
with a structured return type. Predict attaches direct media arguments.

Default CodeAct shows direct media parameters during input inspection and keeps
arguments available as Python variables. With `CodeActConfig(prefill=None)` or
a custom prefill, show them yourself. For media in lists/dicts, loop over items.

In generated CodeAct cells, `show(image)` is a builtin. Hand-written helpers
import `show` from `nooa.runtime.media_capture` and must run inside the cell.
Outside it, show prints a not-captured notice. Show individual objects, not paths
or containers. Use show during execution so the next model turn receives the attachment.
Printing media displays its name and metadata.
PIL images and matplotlib figures also work with show; install them if needed.

## Construct valid inputs

```python
from pathlib import Path
from nooa.media import Image, Audio, Video, File

photo = Image.from_file("photo.png")
clip = Audio.from_bytes(Path("clip.wav").read_bytes(), media_type="audio/wav")
movie = Video.from_file("clip.mp4")
report = File.from_bytes(Path("report.pdf").read_bytes(), media_type="application/pdf")
photo_ref = Image.from_url("https://example.org/photo.png", media_type="image/png")
```

All four types support `from_file`, `from_bytes` with required `media_type`, and
`from_url`. Files are read in full; MIME is guessed from the extension. Constructors
do not validate content/codecs. WAV inference can give `audio/x-wav`; prefer bytes
with `audio/wav`. URL default MIME types are image/jpeg, audio/wav, video/mp4,
and application/pdf. URLs are stored without fetching; provider access is needed.
**Audio URLs are not downloaded automatically** and may fail; use local WAV bytes.
`File` supports PDF only, with exact MIME `application/pdf`. Extract text locally
from other documents; do not label them as PDF.

## Limit size and explain notices

Check provider byte/token/resolution/page/frame/duration/codec limits. Resize,
sample, split, or extract first. Constructors have no byte cap; base64 adds about
one third. `size_bytes` is approximate (`None` for URLs). Default CodeAct keeps
5 attachments per cell, including input inspection; later shows print a limit
notice, and repeats count again. Change the positive count through Agent
`truncation=TruncationConfig(media_capture=MediaCaptureConfig(
max_attachments_per_execution=3))`; import both from `nooa.config.truncation_config`.
Limit bytes separately. Predict's direct arguments have no per-cell cap.

Disabled formats become removal notices in requests, including earlier turns.
Notices contain no description/transcription. Stored history and media objects
stay unchanged; using that history with a configured model can send originals again.
If a saved provider-specific response raises `ReasoningReplayError`, use the
shared assistant-response format in the [replay guide](../../docs/assistant-turn-replay.md).

## Check, migrate, and report

Agents use [nooa-connect](../nooa-connect/SKILL.md) stages, not the wizard.
Within the user's authorized formats, samples and allowance,
`--stage media --test-media-inputs image,pdf --max-tokens 4096
--budget-tokens 32768` runs text and generated-sample checks. The stage does not
prompt or save. Review failures and skips; audio/video checks currently run only
for Chat. A pass confirms access, not general accuracy.

For an unsaved proposal, use `--stage plan --media-inputs image,pdf`; this
declares unverified formats. Save the reviewed entry separately when authorized.
On stage save, `--yes` permits alias replacement; it is not user authorization.

Older aliases need an explicit list; `[]` also silences the missing-list warning.
Agent method `llm=vision` overrides use the selected client's configured list.
Switching models via `existing_client.acall(model=...)` makes the request text-only.
Prefer separate configured aliases; factory model/endpoint changes clear the
list unless supplied again. Set `media_inputs` in configuration, not call kwargs.
Report modified files/aliases, checks run, and anything left unverified.
