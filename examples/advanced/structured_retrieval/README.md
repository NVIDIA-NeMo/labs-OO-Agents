# Section-aware document retrieval

An optional, application-owned document index and a NOOA `PredictStrategy`
agent. A Python orchestrator selects section IDs, reads bounded excerpts, and
validates answer citations. No core framework changes, vector service, or trained
retrieval model are required.

The motivation is [IBM's STAIR paper](https://arxiv.org/abs/2609.03874v2): document
structure can inform section selection. STAIR trains a generative retriever with
table-of-contents inputs. This example is **STAIR-inspired**, not its trained
model or a replication of its reported results. No author implementation is
vendored. Selection quality and answer correctness have not been measured.

## Run

From the repository root:

```bash
uv sync
# Scripted fake responses: no credentials or model calls.
uv run python -m examples.advanced.structured_retrieval.demo
uv run pytest tests/test_structured_retrieval_example.py
# Optional: use an already-configured NOOA model alias; incurs model usage.
uv run python -m examples.advanced.structured_retrieval.demo --model my-model
```

The fixture contains production and staging sections with repeated `Rollback`
titles. Ancestor paths distinguish them. The default demonstration is a wiring
check, not evidence that a model chooses the correct section.

## Data contract and boundaries

Construct `DocumentIndex(document_id, [Section(key, title, text, parent), ...])`.
The initial scope uses explicit parent links, not a Markdown/PDF parser. A
missing parent, duplicate key, or cycle is rejected. Empty structural nodes can
have retrievable descendants; prose owned by an internal node remains searchable.
Repeated titles and Unicode text are supported.

IDs combine the canonical snapshot's SHA-256 with the source section key. They
are stable for the same snapshot; **any edit invalidates prior IDs**. Offsets are
Unicode character offsets within each original section body. Applications can
map keys to their own source files/locations. This index is separate from
`nooa-memory` and its reflection/forgetting lifecycle.

The index stays outside the agent. Only bounded outline/evidence JSON strings
are passed as generation arguments. It uses existing typed generation and
argument rendering, so it adds no context-view abstraction and does not depend
on the open context views work in #353. A future Skill wrapper can be considered
after maintainers agree on an interface.

## Selection, bounds, and validation

1. Build the outline with ancestor paths. If it cannot fit, skip model selection
   rather than present a silently incomplete catalog.
2. Ask `PredictStrategy` for section IDs, with one structured-output attempt per
   invocation. Python verifies catalog membership, deduplicates, and enforces
   `k`. Retry a failed selection once. Empty/exhausted selections use content-only
   BM25 (`k1=1.2`, `b=0.75`, Unicode word tokens, source-order ties).
3. Read at most `k` excerpts within `evidence_chars`. Budgets include serialized
   source metadata and JSON escaping. The final retained body can be truncated;
   offsets and a flag expose that fact.
4. Generate an answer. Reject unread/stale citations and quotes absent from the
   **supplied excerpt**, including the unread suffix of a selected section. Retry
   once, then return an explicit abstention if validation still fails. With no
   evidence, skip answer generation entirely.

There are at most four generation-method invocations per request. Provider
transport retries are separate. Character limits are **not token limits or total
prompt limits**: framework instructions, conversation history, and output also
consume context. Callers should use a fresh agent for independent queries and
measure actual provider usage for cost comparisons. Failure of a network call is
not treated as evidence that retrieval quality is poor.

Citation validation establishes provenance, not semantic entailment or factual
correctness. Source text is untrusted data; the prompts say so, but this example
is not a prompt-injection security boundary. Long outlines use content fallback;
bounded hierarchical navigation and learned selectors are future experiments.

## Validation

Offline tests exercise the real `DocumentAgent`/`PredictStrategy` path with
`FakeLLMClient`, as well as malformed choices, bounded retries, fallback,
abstention, stale IDs, parent prose, Unicode/escaping, and excerpt-only citations.
No live-model performance claim is made. A separate evaluation should compare
matched candidate sets and evidence budgets, with and without ancestor paths,
and report retrieval quality alongside actual tokens, calls, and latency.
