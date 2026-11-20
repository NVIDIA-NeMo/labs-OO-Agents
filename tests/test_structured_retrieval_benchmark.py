# SPDX-FileCopyrightText: Copyright (c) 2026 Sampoorn Nagpal
# SPDX-License-Identifier: Apache-2.0
"""Synthetic, credential-free checks; CI never downloads SearchTome."""

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "experiments/structured_retrieval/benchmark.py"
_spec = importlib.util.spec_from_file_location("_structured_retrieval_benchmark", _path)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)


@pytest.fixture
def release(tmp_path):
    domain = tmp_path / "education"
    domain.mkdir()
    files = {
        "corpus": [
            {"title": "First", "content": "alpha alpha"},
            {"title": "Second", "content": "beta"},
        ],
        "dev": [{"query_id": 0, "query": "alpha", "gold_doc_title": "First"}],
        "test": [{"query_id": 0, "query": "beta", "gold_doc_title": "Second"}],
    }
    manifest = {"files": {}}
    for name, rows in files.items():
        path = domain / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        manifest["files"][f"education/{name}.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return tmp_path, manifest


def test_pinned_adapter_namespaces_split_local_ids(release):
    root, manifest = release
    docs, dev = bench.load_domain(root, "education", "dev", manifest)
    _, test = bench.load_domain(root, "education", "test", manifest)
    assert [d.section_id for d in docs] == ["education:0", "education:1"]
    assert dev[0].key == "education:dev:0"
    assert test[0].key == "education:test:0"
    assert dev[0].gold != test[0].gold


def test_changed_data_is_rejected(release):
    root, manifest = release
    (root / "education/dev.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="Hash mismatch"):
        bench.load_domain(root, "education", "dev", manifest)


@pytest.mark.parametrize("kind", ["ambiguous_title", "missing_gold", "duplicate_query"])
def test_bad_joins_and_query_identity_are_rejected(release, kind):
    root, manifest = release
    path = root / "education" / ("corpus.jsonl" if kind == "ambiguous_title" else "dev.jsonl")
    rows = bench._rows(path)
    if kind == "ambiguous_title":
        rows[1]["title"] = rows[0]["title"]
    elif kind == "missing_gold":
        rows[0]["gold_doc_title"] = "Unknown"
    else:
        rows.append(rows[0])
    path.write_text("\n".join(json.dumps(row) for row in rows))
    manifest["files"][f"education/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        bench.load_domain(root, "education", "dev", manifest)


def test_bm25_fixed_controls_and_ties():
    docs = [bench.Document("a", "Beta", "alpha alpha"), bench.Document("b", "Alpha", "beta")]
    assert bench.BM25(docs, "content").rank("ALPHA", 1) == ["a"]
    assert bench.BM25(docs, "title").rank("alpha", 1) == ["b"]
    assert bench.BM25(docs, "content").rank("no-overlap") == ["a", "b"]
    assert bench.BM25(docs, "content").rank("alpha alpha") == bench.BM25(docs, "content").rank(
        "alpha"
    )


def test_explicit_tree_fixture_catalog_ablation_preserves_candidates():
    docs = [
        bench.Document("prod", "Rollback", "production command", ("Production",)),
        bench.Document("stage", "Rollback", "staging command", ("Staging",)),
    ]
    plain = json.loads(bench.catalog(docs, include_ancestors=False))
    tree = json.loads(bench.catalog(docs, include_ancestors=True))
    assert [row["section_id"] for row in plain] == [row["section_id"] for row in tree]
    assert plain[0]["path"] == plain[1]["path"] == ["Rollback"]
    assert tree[0]["path"] == ["Production", "Rollback"]
    assert "command" not in bench.catalog(docs, include_ancestors=True)


@pytest.mark.parametrize(
    "ids,expected_r1,expected_r3,expected_ndcg",
    [
        ([], 0, 0, 0),
        (["gold"], 1, 1, 1),
        (["other", "gold"], 0, 1, 1 / math.log2(3)),
        (["bad", "other", "gold"], 0, 1, 0.5),
        (["bad", "other", "other", "gold"], 0, 0, 0),
    ],
)
def test_raw_rank_metrics(ids, expected_r1, expected_r3, expected_ndcg):
    result = bench.score(ids, "gold", {"gold", "other"})
    assert result["recall_at_1"] == expected_r1
    assert result["recall_at_3"] == expected_r3
    assert result["ndcg_at_3"] == pytest.approx(expected_ndcg)


def test_invalid_and_duplicate_ids_are_visible_without_rank_repair():
    result = bench.score(["bad", "bad", "gold", "gold"], "gold", {"gold"})
    assert result["invalid_ids"] == 2
    assert result["duplicate_ids"] == 2
    assert result["ndcg_at_3"] == 0.5
    assert result["over_k"] == 1


def test_missing_predictions_count_as_failures_and_usage_remains_unknown():
    docs = [bench.Document("a", "A", "body")]
    data = {
        "first": (docs, [bench.Query("q1", "q", "a")]),
        "second": (docs, [bench.Query("q2", "q", "a"), bench.Query("q3", "q", "a")]),
    }
    result = bench.evaluate(data, {"q1": {"section_ids": ["a"], "model_calls": 2}})
    assert result["micro"]["recall_at_1"] == pytest.approx(1 / 3)
    assert result["macro"]["recall_at_1"] == 0.5
    assert result["missing_predictions"] == 2
    assert result["micro"]["empty"] == 2
    assert result["usage_totals"]["model_calls"] is None
    with pytest.raises(ValueError, match="Unknown prediction"):
        bench.evaluate(data, {"wrong-split": {"section_ids": []}})


def test_complete_usage_totals():
    data = {"d": ([bench.Document("a", "A", "body")], [bench.Query("q", "q", "a")])}
    result = bench.evaluate(
        data,
        {
            "q": {
                "section_ids": ["a"],
                "input_tokens": 10,
                "output_tokens": 3,
                "model_calls": 1,
                "latency_ms": 12.5,
            }
        },
    )
    assert result["usage_totals"] == {
        "input_tokens": 10,
        "output_tokens": 3,
        "model_calls": 1,
        "latency_ms": 12.5,
    }


@pytest.mark.parametrize(
    "extra",
    [
        {"input_tokens": -1},
        {"model_calls": 0.5},
        {"latency_ms": float("nan")},
        {"output_tokens": True},
        {"section_ids": "a"},
    ],
)
def test_malformed_prediction_usage_or_ids_are_rejected(tmp_path, extra):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"query_key": "q", "section_ids": [], **extra}))
    with pytest.raises(ValueError):
        bench.load_predictions(path)


def test_duplicate_prediction_rows_are_rejected(tmp_path):
    path = tmp_path / "predictions.jsonl"
    row = json.dumps({"query_key": "q", "section_ids": []})
    path.write_text(row + "\n" + row)
    with pytest.raises(ValueError, match="Duplicate prediction"):
        bench.load_predictions(path)
