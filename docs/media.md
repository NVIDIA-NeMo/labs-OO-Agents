# Media inputs: images, audio, video, and PDFs

Import `Image`, `Audio`, `Video`, and PDF `File` from `nooa.media` and pass them
as agent method arguments. For media created while CodeAct runs, use `show()`.

## Configure the model

NOOA assumes models are text-only until configured. Set `media_inputs` in the
model's registry entry or client constructor. Supported values are `image`,
`audio`, `video`, and `pdf`. Omitted, null, or empty lists mean text-only.
Use a list (or a tuple in Python); bare strings, duplicates, and unknown values
are rejected. If you enable a format the provider does not accept, requests may fail.

```yaml
models:
  vision:
    model_name: openai/your-vision-model
    client_type: completion
    api_base: https://gateway.example/v1
    api_key_env: MODEL_KEY
    max_tokens: 4096
    media_inputs: [image]
```

Replace placeholders with your provider's settings; keep the key in `MODEL_KEY`.
See [model configuration](model-configuration.md) and [Connect](model-connect.md).

## Run an image example

The [multimodal quickstart](../examples/quickstart/13_multimodal.py) shows
CodeAct describing an image and Predict returning structured image analysis.

Configure an image-capable model with `nooa connect`, save it as `vision`,
and enable `image`. From a source checkout, run:

```sh
uv run python examples/quickstart/13_multimodal.py --model vision
```

This sends the bundled sample images to your provider and can cost money.
For a custom configuration file, add `--config /path/to/llm_config.yaml`.

## CodeAct and `show()`

Predict and default CodeAct attach media passed directly as method arguments.
In generated CodeAct code, call `show()` for media created by code or tools:

```text
show(obj: Any) -> None
```

`obj` can be an `Image`, `Audio`, `Video`, PDF `File`, PIL image, or matplotlib
figure. `show()` attaches it for the next model turn. Show individual objects;
for a list, loop over its items.

Generated code can call `show()` without an import. Hand-written helpers use
`from nooa.runtime.media_capture import show` and run inside a CodeAct cell.
Outside execution, `show()` prints a not-captured notice. If you replace or
disable CodeAct's input inspection, show the media arguments yourself.

## Create media values

All four types support `from_file(path)`, `from_bytes(data, *, media_type)`,
and `from_url(url, *, media_type=...)`:

```python
from pathlib import Path
from nooa.media import Image, Audio, Video, File

photo = Image.from_file("photo.png")
clip = Audio.from_bytes(Path("clip.wav").read_bytes(), media_type="audio/wav")
movie = Video.from_file("clip.mp4")
report = File.from_bytes(Path("report.pdf").read_bytes(), media_type="application/pdf")
photo_ref = Image.from_url("https://example.org/photo.png", media_type="image/png")
```

`from_file` reads the whole file and guesses MIME type from its extension, with
`application/octet-stream` as the fallback. `from_bytes` requires a MIME type.
These constructors do not check the content or codecs. WAV filenames can produce
`audio/x-wav`; prefer bytes with explicit `audio/wav` for providers expecting WAV.
URL default MIME types are `image/jpeg`, `audio/wav`, `video/mp4`, and
`application/pdf`, respectively.

`from_url` stores a URL without fetching it. Image, video, and PDF URLs need
provider support and must be accessible to the provider. **Audio URLs are not
automatically downloaded** and may be rejected; prefer local WAV bytes. Fetch
remote audio in application code only with authorization and size limits.

`File` supports PDFs only, with MIME type exactly `application/pdf`. Other file
attachments are removed, even with every format enabled. Read or extract text
locally from CSV, text, or Word files and pass it as a normal argument. Use the
actual MIME type; do not label other content as PDF. Video and audio format
support, including understanding a video's audio track, varies by provider.

## Size and attachment limits

Check your provider's byte, token, resolution, page, frame, duration, and codec
limits. Resize images, sample video, split PDFs, or extract text before attaching.
Constructors have no byte-size cap. Base64 adds about one third; video adds frame
tokens. `size_bytes` is approximate (`None` for URLs). Text truncation leaves
media bytes unchanged.

Default CodeAct capture keeps the first **5 attachments per cell**, including
input inspection. Later `show()` calls print a limit notice; repeated objects
count again. Set `TruncationConfig(media_capture=MediaCaptureConfig(
max_attachments_per_execution=3))` through the agent's `truncation=` setting to
change this positive count; both classes are in `nooa.config.truncation_config`.
Limit file sizes separately. Predict's direct arguments have no per-cell cap.

## Text-only requests and replay

For formats missing from `media_inputs`, NOOA replaces attachments with text
notices when sending the request, including attachments from earlier turns:

```text
[An image was removed here because this model is not set up to receive images.]
```

The notices contain no description or transcription. Stored history and Python
media objects stay unchanged. Reusing that history with a model configured for
the formats can send the original attachments again. Some saved provider-specific
responses instead raise `ReasoningReplayError` when media is disabled. In that
case, use the shared assistant-response format described in the
[replay guide](assistant-turn-replay.md).

## Connect checks and migration

Make an offline, unsaved configuration plan:

```sh
uv run nooa connect your-model --stage plan --as vision \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY --media-inputs image,pdf --max-tokens 4096 > media-plan.json
```

Save separately with `--stage save --input media-plan.json --output llm_config.yaml`.
Load it with `reload_registry(Path("llm_config.yaml"))` or set
`NEMO_OO_LLM_CONFIG` to its absolute path. In the wizard, use explicit model/endpoint/API
style with `--media-inputs image,pdf --no-probe --no-catalogue` to stay offline;
`--no-probe` alone can still fetch catalogue metadata.

With approval to spend and send generated samples, use `--stage media` with
`--test-media-inputs image,pdf --max-tokens 4096 --budget-tokens 32768`
instead of `--media-inputs`.
This runs a text baseline and media checks, can incur charges, and has no consent
prompt or automatic save. The token budget is an estimate; failed calls count.
Wizard `--yes` approves checks and saving. Audio/video checks currently run only
for Chat. Review passed, failed, and skipped checks before saving; see
[Connect media setup](model-connect.md#choose-media-inputs-during-setup).

For existing model entries, add the formats you use. Set `media_inputs: []`
for text-only models to silence the missing-list warning.

## Use a different model for one call

Pass a configured client with `llm=`. Its media settings apply to that call:

Use `vision = get_llm_client("vision")` for an alias configured with image input,
then `await agent.method(image, llm=vision)` in your async application.
