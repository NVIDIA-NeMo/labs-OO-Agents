# Release-aware structured retrieval evaluation

## Research question

What does a reproducible content baseline achieve on the public SearchTome
release, and how should a later NOOA section selector be compared fairly?
This experiment supplies pinned loading, lexical controls, raw prediction
scoring, and matched catalog rendering. It makes no claim that hierarchy or a
model selector improves retrieval. It is independently runnable from the
proposed document-retrieval example.

## Data and experiment design

[STAIR](https://arxiv.org/abs/2609.03874v2) describes a **trained** generative
retriever and an 18-book evaluation. The linked public
[SearchTome release](https://huggingface.co/datasets/meghanadh/SearchTome/tree/48d3b3545950b3857dc3657fb22e4b79b8ed8df5)
contains six source books, one per domain, with 413 sections, 4,405 train,
1,056 dev, and 6,005 test queries. Its corpus rows have title/content fields,
without explicit internal-node/parent links. This is a **public-release track**,
not an exact paper reproduction or a hierarchy experiment.

The manifest pins the revision and SHA-256 of all 24 JSONL files. Corpus title
joins must be unique and every gold title must resolve. Section IDs are
`domain:corpus_row`, valid only at the pinned revision; query IDs are
`domain:split:query_id` because raw query IDs restart across files. Gold labels
and dataset reference answers are never included in candidate catalogs.

Fixed BM25 controls use title, content, and title-plus-content, with k1=1.2,
b=0.75, casefolded Unicode word tokens, unique query terms, no stopwords or
stemming, and corpus-order tie-breaking. The three controls use identical
section units, candidates, and queries. There is no training or tuning.

For a later paired model experiment, use `catalog(..., include_ancestors=False)`
and `True` on an **explicit hierarchy** with the same section IDs/order. The
synthetic test fixture has duplicate Rollback titles under Production/Staging.
It checks the ablation's wiring, not model performance. Do not infer a complete
tree from SearchTome title strings or present flat-release results as hierarchy
gains. Fix model, candidate set, prompts, output limit, and evidence budget;
report actual usage, repeated-run uncertainty, and per-domain results. Tune only
on dev, then freeze the protocol before test. The initial lexical diagnostic
has already used test; it should not be repurposed as an untouched tuning set.

## Metrics

- Recall@1, Recall@3, and nDCG@3 (one relevant section per query), both micro and
  domain macro averages, plus per-domain results.
- Counts of invalid IDs, duplicate IDs, empty predictions, missing query rows,
  and selections exceeding k=3. **Raw rank is preserved**: filtering a bad first
  ID must not promote the second choice into a Recall@1 hit.
- Optional actual input/output tokens, model calls, and latency in milliseconds
  from prediction rows. Any missing measurement makes its aggregate `null`,
  not zero. Lexical runs have zero model calls/tokens; timing is unmeasured.

These are section-retrieval metrics. Citation provenance, answer correctness,
and end-to-end task success require a separate answering evaluation. An invalid
section ID rate is not a factual-answer hallucination rate.

## Run

From the repository root (no new dependencies):

```bash
uv run pytest tests/test_structured_retrieval_benchmark.py
# Explicit public download; preserve the dataset's attribution/license locally.
uv run python experiments/structured_retrieval/benchmark.py \
  --download --data /tmp/searchtome --split dev --output /tmp/dev-results.json
# Frozen, untuned lexical diagnostic on all released test queries.
uv run python experiments/structured_retrieval/benchmark.py \
  --data /tmp/searchtome --split test --output /tmp/test-results.json
```

CI uses tiny synthetic records and makes no downloads or model calls. Dataset
text is not committed here. The dataset card declares **CC BY-SA 4.0**; retain
the [dataset card and source-book attributions](https://huggingface.co/datasets/meghanadh/SearchTome/blob/48d3b3545950b3857dc3657fb22e4b79b8ed8df5/README.md)
when downloading or redistributing it. The independent harness is Apache-2.0.
The downloader has no automatic retries; a failed run can be restarted and
matching existing files are reused.

To score predictions from a NOOA agent or another selector, emit JSONL rows:

```json
{"query_key":"education:test:0","section_ids":["education:5"],"input_tokens":1200,"output_tokens":30,"model_calls":1,"latency_ms":500.0}
```

This row illustrates the schema, not a measured model result. Use the actual
ranked IDs, before filtering or repair. Report the selected model/prompt/settings
and matching `dataset_revision`/`split` in a run-metadata JSON file:

```bash
uv run python experiments/structured_retrieval/benchmark.py \
  --data /tmp/searchtome --split test --predictions /tmp/predictions.jsonl \
  --run-metadata /tmp/run.json --output /tmp/model-results.json
```

Duplicate prediction keys and unknown query keys are errors. Missing predictions
count as empty failures rather than being dropped. The report records the NOOA
commit and harness/manifest hashes. No live model runner is included in this
first PR; provider billing and source-budget enforcement belong to the producer
of the predictions.

## Results summary

The untuned lexical controls were run over all 6,005 released test queries on
3 November 2026. No model calls were made. Values are percentages:

| Input | Recall@1 | Recall@3 | nDCG@3 |
|---|---:|---:|---:|
| Titles | 19.70 | 33.11 | 27.46 |
| Content | 60.92 | 80.47 | 72.46 |
| Titles + content | 60.93 | 80.47 | 72.48 |

The narrow implication is to retain a content baseline/fallback. These lexical
controls do not reproduce or challenge trained STAIR results, and do not measure
NOOA answer quality. Detailed per-domain outputs are in `lexical-test.json`.
