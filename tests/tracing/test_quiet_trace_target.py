# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""quiet_trace_target() hides only the "OTel tracing enabled" banner."""

import nooa.tracing as tracing


class _Exporter:
    def describe(self):
        return "journal:http://viewer.test:5001"


def test_banner_prints_by_default(capsys):
    tracing._print_trace_target([_Exporter()], None)
    assert "OTel tracing enabled: journal:http://viewer.test:5001" in capsys.readouterr().out


def test_banner_is_silent_inside_quiet_block_and_restored_after(capsys):
    with tracing.quiet_trace_target():
        tracing._print_trace_target([_Exporter()], "exp")
        with tracing.quiet_trace_target():
            pass
        tracing._print_trace_target([_Exporter()], None)
    captured = capsys.readouterr()
    assert "OTel tracing" not in captured.out + captured.err
    tracing._print_trace_target([_Exporter()], None)
    assert "OTel tracing enabled" in capsys.readouterr().out


def test_quiet_block_restores_state_after_an_error(capsys):
    try:
        with tracing.quiet_trace_target():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    tracing._print_trace_target([_Exporter()], None)
    assert "OTel tracing enabled" in capsys.readouterr().out


def test_quiet_blocks_in_other_threads_do_not_leak(capsys):
    """Overlapping quiet blocks in two threads: each restores only its own state."""
    import threading

    first_in, second_in, first_out = threading.Event(), threading.Event(), threading.Event()
    printed = {}

    def first():
        with tracing.quiet_trace_target():
            first_in.set()
            second_in.wait(5)
        first_out.set()  # leaves while the second thread is still quiet

    def second():
        first_in.wait(5)
        with tracing.quiet_trace_target():
            second_in.set()
            first_out.wait(5)
            printed["inside"] = tracing._announce_target.get()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert printed["inside"] is False  # still quiet after the first thread left
    tracing._print_trace_target([_Exporter()], None)  # this context was never quiet
    assert "OTel tracing enabled" in capsys.readouterr().out
