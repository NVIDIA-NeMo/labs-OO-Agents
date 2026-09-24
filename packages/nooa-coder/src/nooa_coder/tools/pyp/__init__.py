# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async-native composable shell piping in Python with method-chaining streams."""

from nooa.skill import Skill
from nooa_cli.tools.pyp.errors import PipeError as PipeError
from nooa_cli.tools.pyp.errors import Result as Result
from nooa_cli.tools.pyp.errors import make_pipe_error as make_pipe_error
from nooa_cli.tools.pyp.sources import arun as arun
from nooa_cli.tools.pyp.sources import cat as cat
from nooa_cli.tools.pyp.sources import empty as empty
from nooa_cli.tools.pyp.sources import find as find
from nooa_cli.tools.pyp.sources import glob as glob
from nooa_cli.tools.pyp.sources import items as items
from nooa_cli.tools.pyp.sources import lines as lines
from nooa_cli.tools.pyp.sources import rg as rg
from nooa_cli.tools.pyp.sources import run as run
from nooa_cli.tools.pyp.sources import seq as seq
from nooa_cli.tools.pyp.sources import stdin as stdin
from nooa_cli.tools.pyp.stream import Stream as Stream


class Pyp(Skill):
    """Async-native shell piping — source().transforms().sink().

    Build pipelines: pick a source, chain transforms, await a sink.
    All streaming, all async, all non-blocking.

    Examples::

        # Search logs
        errors = await self.pyp.cat("app.log").grep("ERROR").head(10).collect()

        # Find files
        files = await self.pyp.find("src", name="*.py").sort().collect()

        # Ripgrep → count
        n = await self.pyp.rg("TODO", type_filter="py").count()

        # Shell command → table
        table = await self.pyp.run("ps aux").head(5).table()

        # Apply function to each line
        out = await self.pyp.run("ls").xargs(process).collect()

        # Pipeline → file
        n = await self.pyp.cat("data.csv").grep("error").write("errors.csv")

        # Stream via ShellTools
        fails = await self.pyp.arun(self.shell, "make test").grep("FAIL").collect()

    Transforms (on any Stream): .grep() .head() .tail() .sort() .uniq()
    .cut() .sed() .map() .filter() .xargs() .wc() .skip() .strip()
    .tee() .flatten() .take_while() .drop_while() .pipe()

    Sinks: .collect() .text() .first() .last() .count() .table()
    .json() .result() .write() .to_set() .to_dict()
    """

    # Expose all source functions as instance methods
    cat = staticmethod(cat)
    run = staticmethod(run)
    arun = staticmethod(arun)
    find = staticmethod(find)
    glob = staticmethod(glob)
    rg = staticmethod(rg)
    seq = staticmethod(seq)
    lines = staticmethod(lines)
    items = staticmethod(items)
    empty = staticmethod(empty)
    stdin = staticmethod(stdin)
