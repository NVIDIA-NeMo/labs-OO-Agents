# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A subclass that overrides only ``replace()`` guards every in-place edit.

Downstream read-only guards (e.g. a vulnerability-analysis agent that must not
modify the code under review) override ``replace()`` alone. ShellTools must not
offer a second public method that rewrites a file's content in place, or the
guard is silently bypassed.
"""

import inspect

import pytest

from nooa.tools.shell_tools import Match, ShellTools


class ReadOnlyShellTools(ShellTools):
    async def replace(self, target, old_or_new="", new=None):
        raise PermissionError("read-only: replace() is disabled")


@pytest.fixture
async def guarded(tmp_path):
    tool = ReadOnlyShellTools(cwd=str(tmp_path))
    yield tool
    await tool.close()


async def test_replace_override_refuses_both_forms(guarded, tmp_path):
    path = tmp_path / "f.py"
    path.write_text("a = 1\nb = 2\n")
    match = await guarded.read("f.py", (1, 1))
    with pytest.raises(PermissionError):
        await guarded.replace("f.py", "a = 1", "a = 2")
    with pytest.raises(PermissionError):
        await guarded.replace(match, "a = 2")
    with pytest.raises(PermissionError):
        await guarded.replace(target=match, old_or_new="a = 2")
    assert path.read_text() == "a = 1\nb = 2\n"


def test_replace_is_the_only_public_in_place_edit_method():
    """Every public coroutine that takes a Match or old text is ``replace``.

    ``write_file`` creates/overwrites whole files and ``run`` executes commands;
    neither edits content in place, so a ``replace()`` override covers edits.
    """
    assert not hasattr(ShellTools, "replace_match")
    editing = []
    for name, member in inspect.getmembers(ShellTools, inspect.iscoroutinefunction):
        if name.startswith("_"):
            continue
        params = inspect.signature(member).parameters
        if {"old", "old_or_new", "match", "target"} & set(params):
            editing.append(name)
    assert editing == ["replace"]


async def test_guard_holds_for_every_public_editing_method(guarded, tmp_path):
    path = tmp_path / "f.py"
    path.write_text("x\n")
    match = await guarded.read("f.py")
    assert isinstance(match, Match)
    for name in ("replace",):
        with pytest.raises(PermissionError):
            await getattr(guarded, name)(match, "y\n")
    assert not hasattr(guarded, "replace_match")
    assert path.read_text() == "x\n"


def test_replace_signature_matches_main():
    params = list(inspect.signature(ShellTools.replace).parameters)
    assert params == ["self", "target", "old_or_new", "new"]
    sig = inspect.signature(ShellTools.replace)
    assert sig.parameters["old_or_new"].default == ""
    assert sig.parameters["new"].default is None
