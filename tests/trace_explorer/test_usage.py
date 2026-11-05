# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accounting contracts and CLI tests for recorded, completed-call usage."""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from dataclasses import asdict
from decimal import localcontext
from pathlib import Path

import pytest

from nooa.trace_explorer import load_usage_report
from nooa.trace_explorer.explorer import _async_main, _load_spans
from nooa.tracing._otlp_serialize import attrs_to_otlp


def _span(number=1, *, start=10, end=11, kind="LLM", model="small", attrs=None, trace=1):
    """Build a current-format span; None attribute overrides omit that field."""
    values = {
        "openinference.span.kind": kind,
        "llm.model_name": model,
        "llm.token_count.prompt": 10,
        "llm.token_count.completion": 5,
    }
    values.update(attrs or {})
    return {
        "traceId": f"{trace:032x}",
        "spanId": f"{number:016x}",
        "name": "synthetic",
        "startTimeUnixNano": str(int(start * 1e9)),
        "endTimeUnixNano": str(int(end * 1e9)),
        "attributes": attrs_to_otlp({k: v for k, v in values.items() if v is not None}),
    }


def _trace(tmp_path, spans, extra_lines=()):
    """Write generated OTLP data, optionally followed by deliberately bad records."""
    path = tmp_path / "trace.jsonl"
    body = {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}
    path.write_text("\n".join([json.dumps(body), *extra_lines]) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "until,calls,inflight",
    [
        (0, 0, 1),
        (0.999999999, 0, 1),
        (1, 1, 0),
        (1.5, 1, 1),
        (2, 2, 0),
        (7200, 2, 0),
        (None, 2, 0),
    ],
)
def test_inclusive_cutoff_and_pending_boundaries(tmp_path, until, calls, inflight):
    """Only completed calls count, including those ending exactly at the cutoff."""
    path = _trace(tmp_path, [_span(1), _span(2, start=11.5, end=12, model="large")])
    report = load_usage_report(path, until_seconds=until)
    assert report.totals.completed_calls == calls
    assert report.totals.known_total_tokens == 15 * calls
    assert report.in_flight_calls == inflight
    assert json.loads(json.dumps(report.to_dict())) == report.to_dict()


def test_decimal_cutoff_does_not_drift_at_subsecond_boundary(tmp_path):
    """A user-supplied decimal 0.3 includes a call ending 300 million ns later."""
    path = _trace(tmp_path, [_span(end=10.3)])
    assert load_usage_report(path, until_seconds=0.3).totals.completed_calls == 1


def test_parent_origin_rollups_models_and_order(tmp_path):
    """An earlier parent defines time zero, but its rollup is not another call."""
    first = _span(1, attrs={"llm.token_count.prompt_details.cache_read": 8})
    second = _span(2, start=11, end=12, model="large", attrs={"llm.token_count.prompt": 20})
    parent = _span(3, start=8, end=13, kind="CHAIN", attrs={"llm.token_count.total": 999})
    report = load_usage_report(_trace(tmp_path, [second, parent, first]))
    assert report.start_time_unix_nano == "8000000000"
    assert report.recorded_duration_seconds == 5
    assert [p.elapsed_seconds for p in report.timeline] == [3, 4]
    assert report.by_model["small"].known_total_tokens == 15
    assert report.by_model["large"].known_total_tokens == 25
    assert report.totals.known_total_tokens == 40
    assert report.totals.known_cached_input_tokens == 8
    assert report.timeline[0].cumulative.known_total_tokens == 15
    assert report.timeline[1].cumulative.known_total_tokens == 40
    assert (
        load_usage_report(
            _trace(tmp_path, [second, parent, first]), until_seconds=2
        ).totals.completed_calls
        == 0
    )


@pytest.mark.parametrize(
    "prompt,completion,total,expected,missing",
    [
        (10, 5, None, 15, 0),
        (0, 0, None, 0, 0),
        (10, None, None, 0, 1),
        (None, 5, None, 0, 1),
        (None, None, 99, 99, 1),
        (None, None, None, 0, 1),
        (10, 5, 17, 17, 0),
    ],
)
def test_missing_and_partial_usage(tmp_path, prompt, completion, total, expected, missing):
    """Missing fields remain null, and a reported total takes precedence."""
    path = _trace(
        tmp_path,
        [
            _span(
                attrs={
                    "llm.token_count.prompt": prompt,
                    "llm.token_count.completion": completion,
                    "llm.token_count.total": total,
                }
            )
        ],
    )
    report = load_usage_report(path)
    assert report.timeline[0].input_tokens == prompt
    assert report.timeline[0].output_tokens == completion
    assert report.totals.known_total_tokens == expected
    assert report.totals.missing_usage_calls == missing
    assert report.totals.missing_cached_input_calls == 1
    rows = list(csv.DictReader(io.StringIO(report.to_csv())))
    assert rows[0]["input_tokens"] == ("" if prompt is None else str(prompt))
    assert rows[-1]["missing_usage_calls"] == str(missing)


def test_duplicate_identity_includes_trace_and_conflicts_are_not_silent(tmp_path):
    """Repeated exports are idempotent, but equal span IDs in distinct traces are not."""
    first = _span()
    report = load_usage_report(_trace(tmp_path, [first, first, _span(trace=2)]))
    assert report.totals.completed_calls == 2
    assert report.deduplicated_spans == 1
    assert report.totals.known_total_tokens == 30
    with pytest.raises(ValueError, match="Conflicting LLM accounting"):
        load_usage_report(_trace(tmp_path, [first, _span(end=12)]))


@pytest.mark.parametrize(
    "attrs",
    [
        {"llm.token_count.prompt": 20},
        {"llm.token_count.completion": None},
        {"llm.token_count.prompt_details.cache_read": 2},
    ],
)
def test_conflicting_duplicate_usage_is_rejected(tmp_path, attrs):
    """A duplicate with a different accounting payload cannot select a winner silently."""
    with pytest.raises(ValueError, match="Conflicting LLM accounting"):
        load_usage_report(_trace(tmp_path, [_span(), _span(attrs=attrs)]))


def test_equal_time_order_is_stable_and_snapshot_is_detached(tmp_path):
    """Ties use trace/span identity, and cumulative snapshots do not alias final totals."""
    forward = load_usage_report(_trace(tmp_path, [_span(1), _span(2)]))
    reverse = load_usage_report(_trace(tmp_path, [_span(2), _span(1)]))
    assert forward.to_dict() == reverse.to_dict()
    assert forward.timeline[0].cumulative.completed_calls == 1
    assert forward.timeline[1].cumulative.completed_calls == 2


@pytest.mark.parametrize("until", [-1, float("nan"), float("inf"), -float("inf"), True, "1"])
def test_invalid_cutoff_is_rejected(tmp_path, until):
    """The public API rejects ambiguous or non-finite cutoffs before reading data."""
    with pytest.raises(ValueError, match="finite non-negative"):
        load_usage_report(_trace(tmp_path, [_span()]), until_seconds=until)


@pytest.mark.parametrize(
    "key",
    [
        "llm.token_count.prompt",
        "llm.token_count.completion",
        "llm.token_count.total",
        "llm.token_count.prompt_details.cache_read",
    ],
)
@pytest.mark.parametrize("value", [-1, 1.5, True, "3", float("nan"), float("inf")])
def test_invalid_token_counts_are_not_coerced(tmp_path, key, value):
    """Counts are non-negative integers; booleans, fractional and string values are errors."""
    with pytest.raises(ValueError, match="non-negative integer"):
        load_usage_report(_trace(tmp_path, [_span(attrs={key: value})]))


@pytest.mark.parametrize("start,end", [(0, 1), (-1, 1), (10, 9), (10, -1)])
def test_invalid_llm_timing_is_visible(tmp_path, start, end):
    """Unplaceable calls cannot disappear into an apparently complete report."""
    with pytest.raises(ValueError, match="timestamp|timing"):
        load_usage_report(_trace(tmp_path, [_span(start=start, end=end)]))


@pytest.mark.parametrize("key", ["traceId", "spanId"])
def test_missing_call_identity_is_visible(tmp_path, key):
    """Missing IDs cannot collapse unrelated calls into one deduplication key."""
    call = _span()
    call.pop(key)
    with pytest.raises(ValueError, match="traceId and spanId"):
        load_usage_report(_trace(tmp_path, [call]))


def test_non_llm_and_unfinished_files_have_summary_rows(tmp_path):
    """A zero-completed-call CSV still distinguishes pending work from no LLM work."""
    for spans, inflight in [([_span(kind="AGENT")], 0), ([_span(end=0)], 1)]:
        report = load_usage_report(_trace(tmp_path, spans))
        assert not report.timeline
        assert report.in_flight_calls == inflight
        rows = list(csv.DictReader(io.StringIO(report.to_csv())))
        assert len(rows) == 1
        assert rows[0]["record_type"] == "summary"
        assert rows[0]["in_flight_calls"] == str(inflight)


@pytest.mark.parametrize(
    "model",
    [
        "small,large",
        'model"quoted',
        "a\nb",
        "a\rb",
        "模型",
        "=1+1",
        "+cmd",
        "-cmd",
        "@cmd",
        "  =cmd",
        "\ttext",
    ],
)
def test_csv_round_trip_and_spreadsheet_text_safety(tmp_path, model):
    """CSV quotes delimiters/newlines and neutralizes formulas without changing JSON."""
    report = load_usage_report(_trace(tmp_path, [_span(model=model)]))
    assert report.to_dict()["timeline"][0]["model"] == model
    row = list(csv.DictReader(io.StringIO(report.to_csv())))[0]
    dangerous = model.lstrip().startswith(("=", "+", "-", "@")) or model.startswith(
        ("\t", "\r", "\n")
    )
    assert row["model"] == ("'" + model if dangerous else model)


def test_csv_uses_lf_record_terminators(tmp_path):
    """Return LF records so callers can apply text-mode newline conversion once."""
    report = load_usage_report(_trace(tmp_path, [_span()]))
    output = report.to_csv()
    assert "\r" not in output
    assert output.count("\n") == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("until,record_types", [(0, ["summary"]), (1, ["call", "summary"])])
async def test_cli_csv_survives_stdout_newline_translation(
    tmp_path, monkeypatch, newline, until, record_types
):
    """POSIX and Windows-style stdout write one record per line, including empty reports."""
    path = _trace(tmp_path, [_span()])
    raw = io.BytesIO()
    with io.TextIOWrapper(raw, encoding="utf-8", newline=newline) as output:
        with monkeypatch.context() as patch:
            patch.setattr(sys, "stdout", output)
            patch.setattr(
                sys,
                "argv",
                [
                    "trace-explorer",
                    str(path),
                    "--usage",
                    "--usage-format",
                    "csv",
                    "--until-seconds",
                    str(until),
                ],
            )
            await _async_main()
            output.flush()
            captured = raw.getvalue()
    assert b"\r\r\n" not in captured
    rows = list(csv.reader(io.StringIO(captured.decode("utf-8"), newline="")))
    assert len(rows) == 1 + len(record_types)
    assert [row[0] for row in rows[1:]] == record_types


@pytest.mark.parametrize(
    "record",
    [
        "{",
        "[]",
        "null",
        "{}",
        '{"nooaJournal":{}}',
        '{"resourceSpans":null}',
        '{"resourceSpans":[null]}',
        '{"resourceSpans":[{"scopeSpans":null}]}',
        '{"resourceSpans":[{"scopeSpans":[{"spans":[null]}]}]}',
        '{"resourceSpans":{}}',
        '{"resourceSpans":[{"scopeSpans":{}}]}',
        '{"resourceSpans":[{"scopeSpans":[{"spans":{}}]}]}',
    ],
)
def test_bad_records_after_valid_data_are_not_skipped(tmp_path, record):
    """Strict accounting reports the bad line, even after usable earlier calls."""
    with pytest.raises(ValueError, match="Invalid trace record.*:2"):
        load_usage_report(_trace(tmp_path, [_span()], [record]))


@pytest.mark.parametrize("contents", ["", " \n\n", '{"resourceSpans":[]}\n'])
def test_empty_trace_is_not_a_complete_zero_report(tmp_path, contents):
    """An empty capture cannot be mistaken for a measured no-token run."""
    path = tmp_path / "empty.jsonl"
    path.write_text(contents)
    with pytest.raises(ValueError, match="no spans"):
        load_usage_report(path)


def test_existing_loader_retains_tolerant_default(tmp_path, capsys):
    """Strict reporting must not change ordinary Explorer's crash-tolerant behavior."""
    path = _trace(tmp_path, [_span()], ["{"])
    assert len(_load_spans(path)) == 1
    assert "Parse error" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["json", "csv"])
async def test_cli_exports_without_constructing_sessions(tmp_path, monkeypatch, capsys, mode):
    """Usage works on LLM-only traces; it does not require an AGENT/session tree."""
    path = _trace(tmp_path, [_span()])
    monkeypatch.setattr(
        sys,
        "argv",
        ["trace-explorer", str(path), "--usage", "--usage-format", mode, "--until-seconds", "1"],
    )
    await _async_main()
    captured = capsys.readouterr()
    assert not captured.err
    if mode == "json":
        report = json.loads(captured.out)
        assert report["totals"]["known_total_tokens"] == 15
    else:
        rows = list(csv.DictReader(io.StringIO(captured.out)))
        assert rows[-1]["known_total_tokens"] == "15"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        ["--usage", "--usage-format", "csv", "--json"],
        ["--usage-format", "csv"],
        ["--until-seconds", "1"],
        ["--usage", "--viewer", "http://unused.invalid", "--session-id", "x"],
        ["--usage", "--root-generation", "0"],
        ["--usage", "--turn", "0"],
        ["--usage", "--session", "x"],
        ["--usage", "--errors"],
        ["--usage", "--timeline"],
        ["--usage", "--search", "x"],
    ],
)
async def test_cli_rejects_ignored_or_conflicting_options(tmp_path, monkeypatch, capsys, options):
    """A reporting request must not silently ignore another view or source selection."""
    path = _trace(tmp_path, [_span()])
    monkeypatch.setattr(sys, "argv", ["trace-explorer", str(path), *options])
    with pytest.raises(SystemExit) as caught:
        await _async_main()
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out and "error:" in captured.err


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing", "empty", "malformed", "directory", "nan"])
async def test_cli_errors_have_nonzero_status_and_no_partial_output(
    tmp_path, monkeypatch, capsys, case
):
    """Bad input does not print an apparently valid partial accounting document."""
    path = tmp_path / "trace.jsonl"
    if case == "empty":
        path.touch()
    elif case == "malformed":
        path.write_text("{\n")
    elif case == "directory":
        path.mkdir()
    else:
        path = _trace(tmp_path, [_span()]) if case == "nan" else path
    argv = ["trace-explorer", str(path), "--usage"]
    if case == "nan":
        argv += ["--until-seconds", "nan"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        await _async_main()
    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert not captured.out and captured.err.startswith("Error:")


def test_no_content_or_provider_state_is_exported(tmp_path):
    """The report contains only accounting metadata, not model-visible/private payloads."""
    path = _trace(
        tmp_path,
        [
            _span(
                attrs={
                    "input.value": "PRIVATE PROMPT",
                    "output.value": "PRIVATE RESPONSE",
                    "llm.reasoning": "PRIVATE REASONING",
                }
            )
        ],
    )
    report = load_usage_report(path)
    assert "PRIVATE" not in json.dumps(report.to_dict())
    assert "PRIVATE" not in report.to_csv()
    assert asdict(report.totals)["known_total_tokens"] == 15


@pytest.mark.parametrize("key", ["traceId", "spanId"])
@pytest.mark.parametrize("value", [True, 12, [], {}])
def test_wrong_identity_types_are_clear_errors(tmp_path, key, value):
    """An invalid identity is not a hashability failure or a CSV serialization error."""
    call = _span()
    call[key] = value
    with pytest.raises(ValueError, match="traceId and spanId"):
        load_usage_report(_trace(tmp_path, [call]))


@pytest.mark.parametrize("value", [False, 0, 42, [], {}])
def test_non_string_models_are_clear_errors(tmp_path, value):
    """Falsey non-string route names must not be quietly relabeled unknown."""
    with pytest.raises(ValueError, match="model_name must be a string"):
        load_usage_report(_trace(tmp_path, [_span(model=value)]))


@pytest.mark.parametrize("key", ["startTimeUnixNano", "endTimeUnixNano"])
@pytest.mark.parametrize("value", [True, 1.5, None, "1.5"])
def test_timestamp_types_are_not_truncated(tmp_path, key, value):
    """Strict accounting does not turn malformed timestamp values into integers."""
    call = _span()
    call[key] = value
    with pytest.raises(ValueError, match="integer timestamp"):
        load_usage_report(_trace(tmp_path, [call]))


@pytest.mark.parametrize("value", [True, 1.5, None, "1.5"])
def test_malformed_otlp_int_values_are_not_truncated(tmp_path, value):
    """OTLP integer encoding is validated before the existing normalizer coerces it."""
    call = _span()
    call["attributes"].append({"key": "llm.token_count.total", "value": {"intValue": value}})
    with pytest.raises(ValueError, match="intValue must be an integer"):
        load_usage_report(_trace(tmp_path, [call]))


def test_duplicate_attribute_keys_are_not_last_value_wins(tmp_path):
    """Ambiguous OTLP attributes cannot silently override the accounting payload."""
    call = _span()
    call["attributes"].append(call["attributes"][-1])
    with pytest.raises(ValueError, match="Duplicate span attribute"):
        load_usage_report(_trace(tmp_path, [call]))


@pytest.mark.parametrize("model", [None, ""])
def test_unavailable_model_name_is_explicit(tmp_path, model):
    """A missing model is grouped under unknown rather than dropping the call."""
    report = load_usage_report(_trace(tmp_path, [_span(model=model)]))
    assert report.by_model["unknown"].completed_calls == 1


@pytest.mark.parametrize("value", [-1, 1 << 64])
def test_out_of_range_timestamp_is_not_float_overflow(tmp_path, value):
    """Untrusted timestamps outside OTLP's range fail clearly before duration math."""
    call = _span()
    call["startTimeUnixNano"] = str(value)
    with pytest.raises(ValueError, match="timestamp.*uint64"):
        load_usage_report(_trace(tmp_path, [call]))


def test_empty_span_after_good_data_is_not_silently_ignored(tmp_path):
    """An empty span is malformed, not evidence that the file had no more LLM work."""
    with pytest.raises(ValueError, match="traceId and spanId"):
        load_usage_report(_trace(tmp_path, [_span(), {}]))


def test_non_object_any_value_is_rejected(tmp_path):
    """A malformed typed-attribute wrapper cannot masquerade as missing telemetry."""
    call = _span()
    call["attributes"][0]["value"] = []
    with pytest.raises(ValueError, match="OTLP AnyValue object"):
        load_usage_report(_trace(tmp_path, [call]))


def test_ambiguous_any_value_does_not_pick_an_arbitrary_count(tmp_path):
    """A malformed oneof cannot choose one conflicting token count silently."""
    call = _span()
    call["attributes"][-1]["value"] = {"intValue": "5", "doubleValue": 99}
    with pytest.raises(ValueError, match="at most one typed value"):
        load_usage_report(_trace(tmp_path, [call]))


@pytest.fixture(scope="module")
def recorded_workload(tmp_path_factory):
    """Capture a real framework workflow with mocked inference in a fresh process."""
    directory = tmp_path_factory.mktemp("recorded-usage")
    fixture = Path(__file__).parent / "fixtures" / "usage_workload.py"
    env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True", "OPENAI_API_KEY": "synthetic"}
    result = subprocess.run(
        [sys.executable, str(fixture), str(directory)],
        capture_output=True,
        text=True,
        env=env,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    path = directory / "usage-workload.jsonl"
    assert path.exists()
    return path


@pytest.mark.parametrize("mode", ["json", "csv"])
def test_saved_workload_through_actual_command(recorded_workload, mode):
    """Child agents -> CompletionClient mock -> instrumentation -> exporter -> CLI."""
    env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
    command = [
        sys.executable,
        "-m",
        "nooa.trace_explorer",
        str(recorded_workload),
        "--usage",
        "--usage-format",
        mode,
        "--until-seconds",
        "7200",
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    if mode == "json":
        report = json.loads(result.stdout)
        assert report["totals"]["completed_calls"] == 2
        assert report["totals"]["known_input_tokens"] == 20
        assert report["totals"]["known_output_tokens"] == 40
        assert report["totals"]["known_total_tokens"] == 60
        assert set(report["by_model"]) == {"openai/gpt-4o-mini", "openai/gpt-4o"}
        assert report["in_flight_calls"] == 0
    else:
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert [row["record_type"] for row in rows] == ["call", "call", "summary"]
        assert rows[-1]["known_total_tokens"] == "60"
        assert rows[-1]["missing_usage_calls"] == "0"


def test_saved_workload_cutoff_before_and_after_completions(recorded_workload):
    """Real exported timings partition a recorded concurrent workload consistently."""
    full = load_usage_report(recorded_workload)
    first_completion = full.timeline[0].elapsed_seconds
    before = load_usage_report(recorded_workload, until_seconds=first_completion - 0.000001)
    assert before.totals.completed_calls == 0
    assert before.in_flight_calls >= 1
    assert full.totals.completed_calls == 2


def test_cutoff_ignores_ambient_decimal_precision(tmp_path):
    """A host application's decimal settings do not round a requested cutoff."""
    path = _trace(tmp_path, [_span(end=10.31)])
    with localcontext() as context:
        context.prec = 1
        assert load_usage_report(path, until_seconds=0.31).totals.completed_calls == 1


@pytest.mark.asyncio
async def test_usage_flag_before_file_is_unambiguous(tmp_path, monkeypatch, capsys):
    """The boolean report switch accepts normal flags-before-positional usage."""
    path = _trace(tmp_path, [_span()])
    monkeypatch.setattr(sys, "argv", ["trace-explorer", "--usage", str(path)])
    await _async_main()
    assert json.loads(capsys.readouterr().out)["totals"]["completed_calls"] == 1
