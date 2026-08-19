# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nooa_cli.tui.resize_reflow import TranscriptResizeState


def test_first_observation_establishes_replayed_width_without_rebuild() -> None:
    state = TranscriptResizeState()

    observation = state.observe((120, 40))

    assert observation.changed is True
    assert observation.should_debounce is False
    assert state.replayed_width == 120
    assert state.has_pending_replay is False


def test_height_only_resize_does_not_schedule_transcript_replay() -> None:
    state = TranscriptResizeState()
    state.observe((120, 40))

    observation = state.observe((120, 24))

    assert observation.changed is True
    assert observation.should_debounce is False
    assert state.prepare_replay() is None


def test_explicit_recovery_replays_at_the_same_width() -> None:
    state = TranscriptResizeState()
    state.observe((120, 40))
    state.observe((120, 4))
    state.observe((120, 40))

    assert state.request_replay() is True
    request = state.prepare_replay()

    assert request is not None
    assert request.width == 120
    assert request.required is True
    assert state.is_current(request) is True

    state.mark_replayed(request)
    assert state.has_pending_replay is False
    assert state.replay_required is False


def test_height_change_extends_an_already_pending_width_resize() -> None:
    state = TranscriptResizeState()
    state.observe((120, 40))
    state.observe((80, 40))

    observation = state.observe((80, 24))

    assert observation.should_debounce is True
    request = state.prepare_replay()
    assert request is not None
    assert request.width == 80


def test_transient_width_that_returns_to_replayed_width_needs_no_rebuild() -> None:
    state = TranscriptResizeState()
    state.observe((120, 40))
    state.observe((80, 24))

    observation = state.observe((120, 40))

    assert observation.should_debounce is True
    assert state.prepare_replay() is None


def test_queued_replay_becomes_stale_when_geometry_changes_again() -> None:
    state = TranscriptResizeState()
    state.observe((120, 40))
    state.observe((80, 24))
    request = state.prepare_replay()
    assert request is not None

    state.observe((90, 24))

    assert state.is_current(request) is False


def test_row_change_while_replay_is_in_flight_preserves_width_repair() -> None:
    state = TranscriptResizeState()
    state.observe((120, 40))
    state.observe((80, 24))
    first_request = state.prepare_replay()
    assert first_request is not None

    observation = state.observe((80, 20))

    assert observation.should_debounce is True
    assert state.is_current(first_request) is False
    replacement_request = state.prepare_replay()
    assert replacement_request is not None
    assert replacement_request.width == 80


def test_mark_replayed_commits_the_width_that_was_physically_written() -> None:
    state = TranscriptResizeState()
    state.observe((120, 40))
    state.observe((80, 24))
    request = state.prepare_replay()
    assert request is not None

    state.mark_replayed(request)

    assert state.replayed_width == 80
    assert state.has_pending_replay is False
    assert state.is_current(request) is False


def test_successful_stale_replay_records_physical_width_and_repairs_latest_width() -> None:
    state = TranscriptResizeState()
    state.observe((120, 40))
    state.observe((80, 24))
    request = state.prepare_replay()
    assert request is not None

    state.observe((120, 40))
    state.mark_replayed(request)

    assert state.replayed_width == 80
    replacement_request = state.prepare_replay()
    assert replacement_request is not None
    assert replacement_request.width == 120


def test_row_change_during_successful_replay_does_not_require_second_rebuild() -> None:
    state = TranscriptResizeState()
    state.observe((120, 40))
    state.observe((80, 24))
    request = state.prepare_replay()
    assert request is not None

    state.observe((80, 20))
    state.mark_replayed(request)

    assert state.replayed_width == 80
    assert state.has_pending_replay is False
