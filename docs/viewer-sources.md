# Viewer source plugins

The viewer defaults to its SQLite store. Optional Python packages can expose
additional read sources through the `nooa.viewer.sources` entry-point group.
The viewer contains no client for any particular remote store.

Set `NOOA_VIEWER_SOURCES_CONFIG` to an absolute JSON configuration file:

```json
{
  "sources": [
    {
      "name": "example",
      "plugin": "example-source",
      "options": {}
    }
  ]
}
```

Only explicitly configured entry points are loaded. A missing or ambiguous
plugin fails startup. Keep endpoint credentials in environment variables or
private configuration, not repository files.

## Add runs without restarting

Sources may implement the optional `prepare_run(run_id)` capability. It validates
the requested run and returns updated JSON options without mutating the active
source. An authenticated caller can select a run through:

```http
POST /api/sources/example/runs
Content-Type: application/json

{"run_id": "a-run-id"}
```

The viewer preloads and validates a replacement catalog, atomically persists
options to the configured JSON file (preserving its permissions), then activates
the source. The response includes `source`, `run_id`, `added`, and `experiments`.
Repeated additions do not duplicate selections. Concurrent additions within the
viewer process are serialized. The default single-process viewer supports this
live configuration; multiple worker processes would need shared invalidation.
Persistence requires the configuration file and its directory to be writable.

Failures leave the active selection intact. Unknown sources/runs return 404;
unsupported selection returns 405; invalid input returns 400/422; unavailable
remote catalogs return 502; persistence failures return 503. This selects data
for viewing and does not upload to or change the remote store.

A package registers its factory in `pyproject.toml`:

```toml
[project.entry-points."nooa.viewer.sources"]
example-source = "my_package.source:ExampleSource"
```

The factory receives `name` and `options` keyword arguments and implements the
`TraceSource` protocol in `nooa.viewer.sources`:

- `version`: a string identifying the conversion contract.
- `list_sessions()`: session metadata without loading trace details.
- `load_session(session_id)`: portable OTLP bodies and NOOA journal records.

Catalog rows contain `id`, `name`, `experiment`, `batch_id`, `modified` (epoch
seconds), `span_count` (or null before loading), and `eval` metadata. Grading
uses `eval.passed = true | false | null`; missing grades must remain null.
Session IDs must begin with the configured source name followed by `-`, be
globally unique, and include all source attempt identities.
Plugins should cache catalog requests and bound/retry remote downloads.

Trace records use `resourceSpans` and a `session.id` resource attribute matching
the requested ID. Journal records use `nooaJournal` with matching `session_id`,
`type = blocks | call`, and the existing journal ingestion payloads. A manifest
record may also be included. Span, trace and journal-call IDs must be stable;
separate lanes must not collide.

On the first detail read, the viewer fetches and validates records, serializes
their writes with native ingestion, and caches them in local SQLite. Journal
writes are idempotent and happen before spans are published. Concurrent reads
of the same trace share one load. Subsequent reads use the local cache. Existing
rendering, journal resolution, trace exploration and local annotations therefore
work without a new frontend data format.

Catalog grades remain authoritative even for cached sessions. Local cache span
counts and recovered trace-availability metadata enrich the catalog. Source
session identities cannot overwrite existing local traces. The source interface
has no remote mutation methods: viewer deletion affects local cached data only;
the remote catalog may still list that task.

This first interface supports trace and Experiment browsing. The legacy bulk
Experiment export and history-metrics endpoints still operate on locally cached
data. A changed source conversion version is rejected instead of silently
replacing traces and annotations; cache migration needs an explicit follow-up.
Native local tracing works without a source configuration or additional packages.
