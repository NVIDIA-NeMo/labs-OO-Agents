# SPDX-FileCopyrightText: Copyright (c) 2026 Sampoorn Nagpal
# SPDX-License-Identifier: Apache-2.0
"""Pinned SearchTome lexical controls and raw prediction scoring; no model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

REVISION = "48d3b3545950b3857dc3657fb22e4b79b8ed8df5"
DOMAINS = ("education", "finance", "law", "medicine", "natural_sciences", "social_sciences")
MANIFEST = Path(__file__).with_name("manifest.json")


@dataclass(frozen=True)
class Document:
    section_id: str
    title: str
    content: str
    ancestors: tuple[str, ...] = ()


@dataclass(frozen=True)
class Query:
    key: str
    text: str
    gold: str


def catalog(documents: list[Document], *, include_ancestors: bool) -> str:
    """Matched catalog variants for a future paired model run; never include gold labels."""
    return json.dumps(
        [
            {
                "section_id": d.section_id,
                "path": [*d.ancestors, d.title] if include_ancestors else [d.title],
            }
            for d in documents
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_domain(
    root: Path, domain: str, split: str, manifest: dict
) -> tuple[list[Document], list[Query]]:
    """Verify file hashes, unique title joins, and split-scoped query IDs."""
    for name in ("corpus", split):
        relative = f"{domain}/{name}.jsonl"
        if (
            hashlib.sha256((root / relative).read_bytes()).hexdigest()
            != manifest["files"][relative]
        ):
            raise ValueError(f"Hash mismatch: {relative}")
    corpus = _rows(root / domain / "corpus.jsonl")
    documents = [
        Document(f"{domain}:{i}", row["title"], row["content"]) for i, row in enumerate(corpus)
    ]
    by_title = {d.title: d.section_id for d in documents}
    if not documents or len(by_title) != len(documents):
        raise ValueError(f"Empty corpus or ambiguous title joins in {domain}")
    queries = []
    for row in _rows(root / domain / f"{split}.jsonl"):
        if row["gold_doc_title"] not in by_title:
            raise ValueError(f"Missing gold title in {domain}")
        queries.append(
            Query(
                f"{domain}:{split}:{row['query_id']}", row["query"], by_title[row["gold_doc_title"]]
            )
        )
    if not queries or len({q.key for q in queries}) != len(queries):
        raise ValueError(f"Empty query split or duplicate query IDs in {domain}")
    return documents, queries


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class BM25:
    """Fixed, untuned controls; unique query terms, corpus-order ties, no stopwords/stemming."""

    def __init__(self, documents: list[Document], field: str):
        if field not in ("title", "content", "title_content"):
            raise ValueError("Unknown BM25 field")
        self.documents = documents
        texts = [
            f"{d.title}\n{d.content}" if field == "title_content" else getattr(d, field)
            for d in documents
        ]
        self.terms = [Counter(words(text)) for text in texts]
        self.lengths = [sum(t.values()) for t in self.terms]
        self.average = mean(self.lengths) if self.lengths else 0
        self.df: Counter[str] = Counter()
        for terms in self.terms:
            self.df.update(terms.keys())

    def rank(self, query: str, k: int = 3) -> list[str]:
        if k < 1:
            raise ValueError("k must be positive")
        terms = set(words(query))
        n = len(self.documents)
        scores = []
        for counts, length in zip(self.terms, self.lengths, strict=True):
            score = sum(
                math.log(1 + (n - self.df[word] + 0.5) / (self.df[word] + 0.5))
                * counts[word]
                * 2.2
                / (counts[word] + 1.2 * (0.25 + 0.75 * length / (self.average or 1)))
                for word in terms
                if counts[word]
            )
            scores.append(score)
        order = sorted(range(n), key=lambda i: (-scores[i], i))
        return [self.documents[i].section_id for i in order[:k]]


def score(raw_ids: list[str], gold: str, valid_ids: set[str]) -> dict[str, float | int]:
    """Score raw ranks: invalid IDs/duplicates are never removed to promote a later hit."""
    if gold not in valid_ids:
        raise ValueError("Gold ID is outside the corpus")
    if not isinstance(raw_ids, list) or any(not isinstance(sid, str) for sid in raw_ids):
        raise ValueError("section_ids must be a list of strings")
    rank = raw_ids.index(gold) + 1 if gold in raw_ids else None
    return {
        "recall_at_1": int(rank == 1),
        "recall_at_3": int(rank is not None and rank <= 3),
        "ndcg_at_3": 1 / math.log2(rank + 1) if rank is not None and rank <= 3 else 0,
        "invalid_ids": sum(sid not in valid_ids for sid in raw_ids),
        "duplicate_ids": len(raw_ids) - len(set(raw_ids)),
        "returned_ids": len(raw_ids),
        "empty": int(not raw_ids),
        "over_k": int(len(raw_ids) > 3),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    quality = ("recall_at_1", "recall_at_3", "ndcg_at_3")
    counts = ("invalid_ids", "duplicate_ids", "returned_ids", "empty", "over_k")
    return {
        "queries": len(rows),
        **{key: mean(row[key] for row in rows) for key in quality},
        **{key: sum(row[key] for row in rows) for key in counts},
    }


def load_predictions(path: Path) -> dict[str, dict]:
    predictions = {}
    for row in _rows(path):
        key = row["query_key"]
        if key in predictions:
            raise ValueError(f"Duplicate prediction query key: {key}")
        ids = row["section_ids"]
        if (
            not isinstance(ids, list)
            or len(ids) > 100
            or any(not isinstance(sid, str) for sid in ids)
        ):
            raise ValueError("Prediction section_ids must contain at most 100 strings")
        for metric in ("input_tokens", "output_tokens", "model_calls", "latency_ms"):
            value = row.get(metric)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (metric != "latency_ms" and not isinstance(value, int))
            ):
                raise ValueError(f"Invalid usage field: {metric}")
        predictions[key] = row
    return predictions


def evaluate(
    data: dict[str, tuple[list[Document], list[Query]]], predictions: dict[str, dict]
) -> dict:
    known_queries = {q.key for _, queries in data.values() for q in queries}
    unknown = set(predictions) - known_queries
    if unknown:
        raise ValueError(f"Unknown prediction query keys: {sorted(unknown)[:3]}")
    domain_results, all_rows = {}, []
    for domain, (documents, queries) in data.items():
        valid = {d.section_id for d in documents}
        rows = [
            score(predictions.get(q.key, {}).get("section_ids", []), q.gold, valid) for q in queries
        ]
        domain_results[domain] = summarize(rows)
        all_rows.extend(rows)
    usage = {}
    for metric in ("input_tokens", "output_tokens", "model_calls", "latency_ms"):
        values = [predictions.get(key, {}).get(metric) for key in known_queries]
        usage[metric] = sum(values) if all(value is not None for value in values) else None
    return {
        "micro": summarize(all_rows),
        "macro": {
            key: mean(r[key] for r in domain_results.values())
            for key in ("recall_at_1", "recall_at_3", "ndcg_at_3")
        },
        "domains": domain_results,
        "missing_predictions": len(known_queries - set(predictions)),
        "usage_totals": usage,
    }


def download(root: Path, manifest: dict) -> None:
    """Explicit opt-in download; no text is bundled with this experiment."""
    import urllib.request

    for relative, expected in manifest["files"].items():
        target = root / relative
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        url = f"https://huggingface.co/datasets/meghanadh/SearchTome/resolve/{REVISION}/{relative}"
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Download hash mismatch: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--download", action="store_true", help="Fetch the pinned public dataset")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--run-metadata",
        type=Path,
        help="Required with predictions: revision, split, model, prompts, settings",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if manifest["revision"] != REVISION:
        raise ValueError("Unexpected manifest revision")
    if args.download:
        download(args.data, manifest)
    data = {domain: load_domain(args.data, domain, args.split, manifest) for domain in DOMAINS}
    results = {}
    run_metadata = None
    if args.predictions:
        if not args.run_metadata:
            parser.error("--predictions requires --run-metadata")
        run_metadata = json.loads(args.run_metadata.read_text())
        if (
            run_metadata.get("dataset_revision") != REVISION
            or run_metadata.get("split") != args.split
        ):
            raise ValueError("Prediction metadata does not match this dataset/split")
        for key in ("model", "prompts", "settings"):
            if key not in run_metadata:
                raise ValueError(f"Missing run metadata: {key}")
        results["predictions"] = evaluate(data, load_predictions(args.predictions))
    else:
        for field in ("title", "content", "title_content"):
            predictions = {}
            for documents, queries in data.values():
                ranker = BM25(documents, field)
                for query in queries:
                    predictions[query.key] = {
                        "section_ids": ranker.rank(query.text),
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "model_calls": 0,
                    }
            results[field] = evaluate(data, predictions)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    output = {
        "dataset_revision": REVISION,
        "split": args.split,
        "nooa_commit": commit,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest()
        if args.predictions
        else None,
        "protocol": {
            "k1": 1.2,
            "b": 0.75,
            "tokenizer": "casefold Unicode words",
            "tuning": "none",
            "ties": "corpus order",
        },
        "prediction_run": run_metadata,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value["micro"] for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
