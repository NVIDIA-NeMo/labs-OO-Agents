# SPDX-License-Identifier: Apache-2.0
"""Regression test for native tracing lifecycle isolation."""

from nooa.tracing._llm_journal import _SINKS, MessageJournalCallback, register_sink


def test_conftest_reset_clears_native_journal_sinks():
    from tests.integration.conftest import _reset_tracing_module_state

    sink = register_sink(MessageJournalCallback("http://meta-test.invalid"))
    assert sink in _SINKS
    _reset_tracing_module_state()
    assert sink not in _SINKS
