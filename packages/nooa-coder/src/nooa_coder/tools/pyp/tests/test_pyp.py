# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pysh library — method-chaining, async-native."""

from pathlib import Path

import pytest

from nooa_cli.tools.pyp.errors import PipeError, Result, make_pipe_error
from nooa_cli.tools.pyp.sources import cat, empty, find, glob, items, lines, run, seq
from nooa_cli.tools.pyp.stream import Stream

# ─── Stream basics ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_from_iterable():
    s = Stream(iter(["a", "b", "c"]))
    assert await s.collect() == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_stream_chaining():
    s = items(["hello", "world", "help"]).grep("hel")
    assert await s.collect() == ["hello", "help"]


@pytest.mark.asyncio
async def test_stream_pipe_with_callable():
    async def upper(ait):
        async for x in ait:
            yield x.upper()

    s = items(["a", "b", "c"]) | upper
    assert await s.collect() == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_stream_repr():
    s = items(["x"]).grep("x").head(1)
    r = repr(s)
    assert "grep" in r
    assert "head" in r


# ─── Error types ─────────────────────────────────────────────────────


def test_result_ok():
    r = Result(lines=["a", "b"], returncode=0)
    assert r.ok
    assert len(r) == 2
    assert r.text == "a\nb"
    assert bool(r) is True


def test_result_failure():
    r = Result(lines=[], returncode=1, stderr="oops")
    assert not r.ok
    assert bool(r) is False


def test_pipe_error_fields():
    e = make_pipe_error("failed", cmd="ls", returncode=2, stderr="no such file")
    assert e.cmd == "ls"
    assert e.returncode == 2
    assert "no such file" in e.stderr


# ─── Sources ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cat(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line1\nline2\nline3\n")
    result = await cat(str(f)).collect()
    assert result == ["line1", "line2", "line3"]


@pytest.mark.asyncio
async def test_cat_multiple(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("a\n")
    f2.write_text("b\n")
    result = await cat(str(f1), str(f2)).collect()
    assert result == ["a", "b"]


@pytest.mark.asyncio
async def test_run_success():
    result = await run("echo hello").collect()
    assert result == ["hello"]


@pytest.mark.asyncio
async def test_run_failure():
    with pytest.raises(PipeError):
        await run("false").collect()


@pytest.mark.asyncio
async def test_run_check_false():
    r = await run("false", check=False).result()
    assert not r.ok


@pytest.mark.asyncio
async def test_find_files(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("")
    result = await find(str(tmp_path), name="*.py").collect()
    assert len(result) == 2
    assert all(".py" in r for r in result)


@pytest.mark.asyncio
async def test_glob_files(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    result = await glob("*.py", root=str(tmp_path)).collect()
    assert len(result) == 2


@pytest.mark.asyncio
async def test_lines_source():
    text = "foo\nbar\nbaz"
    assert await lines(text).collect() == ["foo", "bar", "baz"]


@pytest.mark.asyncio
async def test_items_source():
    assert await items([1, 2, 3]).collect() == [1, 2, 3]


@pytest.mark.asyncio
async def test_empty_source():
    assert await empty().collect() == []


@pytest.mark.asyncio
async def test_seq():
    assert await seq(5).collect() == ["1", "2", "3", "4", "5"]
    assert await seq(2, 10, step=3).collect() == ["2", "5", "8"]


# ─── Transforms (method chaining) ───────────────────────────────────


@pytest.mark.asyncio
async def test_grep_basic():
    result = await items(["apple", "banana", "apricot"]).grep("ap").collect()
    assert result == ["apple", "apricot"]


@pytest.mark.asyncio
async def test_grep_invert():
    result = await items(["apple", "banana", "apricot"]).grep("ap", invert=True).collect()
    assert result == ["banana"]


@pytest.mark.asyncio
async def test_grep_ignore_case():
    result = await items(["Apple", "BANANA"]).grep("apple", ignore_case=True).collect()
    assert result == ["Apple"]


@pytest.mark.asyncio
async def test_grep_fixed():
    result = await items(["a.b", "axb", "a*b"]).grep("a.b", fixed=True).collect()
    assert result == ["a.b"]


@pytest.mark.asyncio
async def test_head():
    result = await items(range(100)).head(3).collect()
    assert result == [0, 1, 2]


@pytest.mark.asyncio
async def test_tail():
    result = await items(range(100)).tail(3).collect()
    assert result == [97, 98, 99]


@pytest.mark.asyncio
async def test_sort_basic():
    result = await items(["c", "a", "b"]).sort().collect()
    assert result == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_sort_reverse():
    result = await items(["a", "b", "c"]).sort(reverse=True).collect()
    assert result == ["c", "b", "a"]


@pytest.mark.asyncio
async def test_sort_numeric():
    result = await items(["10", "2", "1", "20"]).sort(numeric=True).collect()
    assert result == ["1", "2", "10", "20"]


@pytest.mark.asyncio
async def test_uniq_consecutive():
    result = await items(["a", "a", "b", "b", "a"]).uniq().collect()
    assert result == ["a", "b", "a"]


@pytest.mark.asyncio
async def test_uniq_all():
    result = await items(["a", "b", "a", "c", "b"]).uniq(all_unique=True).collect()
    assert result == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_cut_basic():
    data = ["Alice 30 NYC", "Bob 25 LA"]
    result = await items(data).cut(fields=[0, 2]).collect()
    assert result == ["Alice\tNYC", "Bob\tLA"]


@pytest.mark.asyncio
async def test_cut_with_sep():
    data = ["a,b,c", "d,e,f"]
    result = await items(data).cut(fields=[0, 2], sep=",", out_sep=",").collect()
    assert result == ["a,c", "d,f"]


@pytest.mark.asyncio
async def test_sed_basic():
    result = await items(["hello world", "hello python"]).sed("hello", "hi").collect()
    assert result == ["hi world", "hi python"]


@pytest.mark.asyncio
async def test_sed_regex():
    result = await items(["foo123bar"]).sed(r"(\d+)", "NUM").collect()
    assert result == ["fooNUMbar"]


@pytest.mark.asyncio
async def test_map():
    result = await items(["a", "b", "c"]).map(str.upper).collect()
    assert result == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_filter():
    result = (
        await items(["short", "a very long line indeed"]).filter(lambda x: len(x) > 10).collect()
    )
    assert result == ["a very long line indeed"]


@pytest.mark.asyncio
async def test_wc():
    result = await items(["a", "b", "c"]).wc().collect()
    assert result == ["3"]


@pytest.mark.asyncio
async def test_wc_full():
    result = await items(["hello world", "foo"]).wc(lines_only=False).collect()
    parts = result[0].split()
    assert parts[0] == "2"


@pytest.mark.asyncio
async def test_tee(tmp_path):
    outfile = str(tmp_path / "tee_out.txt")
    result = await items(["a", "b", "c"]).tee(outfile).collect()
    assert result == ["a", "b", "c"]
    content = Path(outfile).read_text()
    assert "a\n" in content
    assert "c\n" in content


@pytest.mark.asyncio
async def test_skip():
    result = await items(["header", "row1", "row2"]).skip(1).collect()
    assert result == ["row1", "row2"]


@pytest.mark.asyncio
async def test_take_while():
    result = await items([1, 2, 3, 10, 4]).take_while(lambda x: x < 5).collect()
    assert result == [1, 2, 3]


@pytest.mark.asyncio
async def test_drop_while():
    result = await items([1, 2, 3, 10, 4]).drop_while(lambda x: x < 5).collect()
    assert result == [10, 4]


@pytest.mark.asyncio
async def test_flatten():
    result = await items(["a,b,c", "d,e"]).flatten(sep=",").collect()
    assert result == ["a", "b", "c", "d", "e"]


@pytest.mark.asyncio
async def test_strip():
    result = await items(["  hello  ", "  world  "]).strip().collect()
    assert result == ["hello", "world"]


# ─── Composition ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chained_pipeline():
    result = (
        await items(["banana", "apple", "avocado", "cherry"]).grep("a").sort().head(2).collect()
    )
    assert result == ["apple", "avocado"]


@pytest.mark.asyncio
async def test_full_pipeline(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("ERROR: disk full\nINFO: ok\nERROR: timeout\nDEBUG: trace\n")
    result = await cat(str(f)).grep("ERROR").sed("ERROR: ", "").sort().collect()
    assert result == ["disk full", "timeout"]


@pytest.mark.asyncio
async def test_pipeline_with_run():
    result = await run("echo -e 'foo\\nbar\\nbaz'").grep("ba").sort().collect()
    assert "bar" in result
    assert "baz" in result


# ─── Sinks ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first():
    assert await items(["a", "b", "c"]).first() == "a"
    assert await empty().first() is None


@pytest.mark.asyncio
async def test_last():
    assert await items(["a", "b", "c"]).last() == "c"
    assert await empty().last() is None


@pytest.mark.asyncio
async def test_count():
    assert await items(range(42)).count() == 42
    assert await empty().count() == 0


@pytest.mark.asyncio
async def test_write(tmp_path):
    out = tmp_path / "out.txt"
    n = await items(["x", "y", "z"]).write(str(out))
    assert n == 3
    assert out.read_text() == "x\ny\nz\n"


@pytest.mark.asyncio
async def test_to_set():
    assert await items(["a", "b", "a"]).to_set() == {"a", "b"}


@pytest.mark.asyncio
async def test_to_dict():
    assert await items(["k1=v1", "k2=v2"]).to_dict() == {"k1": "v1", "k2": "v2"}


@pytest.mark.asyncio
async def test_json():
    data = await items(['{"a": 1, "b": 2}']).json()
    assert data == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_text():
    assert await items(["hello", "world"]).text() == "hello\nworld"


@pytest.mark.asyncio
async def test_table():
    table = await items(["Alice 30 NYC", "Bob 25 LA"]).table()
    assert "Alice" in table
    assert "Bob" in table
