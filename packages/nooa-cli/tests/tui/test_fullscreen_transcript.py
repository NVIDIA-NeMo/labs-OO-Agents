¶»§q«^v‹­¦лNєЪn¶Ъ(•Єа{mјг]њ…ЄмJ0ЉxЉ»-Чќ4С©Э•«-Чќ4СИZ®Дб{_|г]њ…ЄмЉ{azhќvWљ­йи¶·њўч«i№^# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ownership tests for the alternate-screen transcript renderer."""

from __future__ import annotations

import asyncio
import io
from unittest.mock import MagicMock

import pytest
from nooa_cli.tui.config import DisplayMode
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput

from .tui_app_harness import TUIHarness


def _make_fullscreen_app():
    from nooa_cli.tui.tui_application import TUIApplication

    with create_app_session(input=DummyInput(), output=DummyOutput()):
        return TUIApplication(display_mode=DisplayMode.FULLSCREEN)


def test_resize_application_prompt_toolkit_private_api_contract() -> None:
    """Fail clearly if a prompt-toolkit upgrade changes private resize hooks."""
    import inspect

    from nooa_cli.tui.tui_application import _ResizeAwareApplication
    from prompt_toolkit.application import Application

    on_resize = inspect.signature(Application._on_resize)
    assert tuple(on_resize.parameters) == ("self",)

    redraw = inspect.signature(Application._redraw)
    assert tuple(redraw.parameters) == ("self", "render_as_done")
    assert redraw.parameters["render_as_done"].default is False

    request_cursor = inspect.signature(Application._request_absolute_cursor_position)
    assert tuple(request_cursor.parameters) == ("self",)

    assert tuple(inspect.signature(_ResizeAwareApplication._on_resize).parameters) == ("self",)
    assert tuple(inspect.signature(_ResizeAwareApplication._redraw).parameters) == (
        "self",
        "render_as_done",
    )

    app = _make_fullscreen_app()
    assert isinstance(app._app._running_in_terminal, bool)


def test_fullscreen_shell_owns_alternate_screen_and_transcript_window() -> None:
    app = _make_fullscreen_app()

    assert app.display_mode is DisplayMode.FULLSCREEN
    assert app._app.full_screen is True
    assert app.full_screen is False  # legacy destructive native replay flag
    assert app._output_window is not None


def test_fullscreen_copy_and_return_actions_share_status_region() -> None:
    from prompt_toolkit.layout import VSplit

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(30)))
    app._transcript_viewport_size = lambda: (20, 4)
    app._show_transient_status("Copied 5 characters", style="class:return-to-tail")
    app._scroll_fullscreen_transcript(-4)

    status_container = app._status_region_container
    assert status_container is not None
    status_region = status_container.content
    assert isinstance(status_region, VSplit)
    assert app._transient_status_container in status_region.children
    assert app._return_to_tail_container in status_region.children
    assert bool(status_container.filter())
    assert bool(app._transient_status_container.filter())
    assert bool(app._return_to_tail_container.filter())


def test_fullscreen_agent_message_notice_appears_only_while_scrolled_up() -> None:
    from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(30)))
    app._transcript_viewport_size = lambda: (20, 4)
    app._scroll_fullscreen_transcript(-4)

    app.emit_block("tool activity\n")
    text = fragment_list_to_text(to_formatted_text(app._return_to_tail_control.text()))
    assert "New agent message" not in text
    return_position = text.index("Return to bottom")

    app.emit_block("agent reply\n", agent_message=True)
    fragments = to_formatted_text(app._return_to_tail_control.text())
    text = fragment_list_to_text(fragments)
    assert text.index("New agent message") < text.index("Return to bottom")
    assert text.index("Return to bottom") == return_position
    notice_style = next(style for style, value, *_ in fragments if "New agent message" in value)
    return_style = next(style for style, value, *_ in fragments if "Return to bottom" in value)
    assert notice_style == return_style
    assert app._has_unseen_agent_message

    app._jump_fullscreen_to_tail()
    assert not app._has_unseen_agent_message
    assert not bool(app._return_to_tail_container.filter())


def test_fullscreen_clear_removes_unseen_agent_message_notice() -> None:
    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(30)))
    app._transcript_viewport_size = lambda: (20, 4)
    app._scroll_fullscreen_transcript(-4)
    app.emit_block("agent reply\n", agent_message=True)
    assert app._has_unseen_agent_message

    app.clear_transcript()

    assert not app._has_unseen_agent_message
    assert app._fullscreen_transcript.viewport.follows_tail


def test_fullscreen_agent_message_at_tail_does_not_create_notice() -> None:
    from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text

    app = _make_fullscreen_app()
    app.emit_block("agent reply\n", agent_message=True)

    assert app._fullscreen_transcript.viewport.follows_tail
    assert not app._has_unseen_agent_message
    assert "New agent message" not in fragment_list_to_text(
        to_formatted_text(app._return_to_tail_control.text())
    )


@pytest.mark.asyncio
async def test_fullscreen_input_composer_is_capped_at_ten_rows() -> None:
    from prompt_toolkit.application.current import set_app

    app = _make_fullscreen_app()
    app.input_buffer.text = "\n".join(f"line {index}" for index in range(20))

    with set_app(app._app):
        dimension = app._input_window.preferred_height(80, 100)

    assert dimension.min == 1
    assert dimension.max == 10
    assert dimension.preferred == 10


def test_fullscreen_status_removes_visual_transcript_gap() -> None:
    from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text

    app = _make_fullscreen_app()
    app.emit_block("answer\n")

    assert not bool(app._status_region_container.filter())
    assert (
        fragment_list_to_text(
            to_formatted_text(app._fullscreen_transcript.formatted_text(width=20, height=2))
        )
        == "answer\n"
    )

    app._command_status_text = "thinking..."
    app._before_render(app._app)

    assert bool(app._status_region_container.filter())
    assert app._fullscreen_transcript.text == "answer\n"
    assert (
        fragment_list_to_text(
            to_formatted_text(app._fullscreen_transcript.formatted_text(width=20, height=1))
        )
        == "answer"
    )


def test_native_hyperlink_marker_vocabulary_is_bounded() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append(
        " ".join(
            f"\x1b]8;;https://example.test/{index}\x1b\\x\x1b]8;;\x1b\\" for index in range(64)
        )
    )

    def marker_classes(render_counter: int) -> set[str]:
        return {
            token
            for style, _text, *_ in model.formatted_text(
                width=4_096,
                height=1,
                render_counter=render_counter,
            )
            for token in style.split()
            if token.startswith("class:native-hyperlink-")
        }

    assert marker_classes(0) == {"class:native-hyperlink-0"}
    assert marker_classes(1) == {"class:native-hyperlink-1"}
    model._formatted_cache.clear()
    assert marker_classes(2) == {"class:native-hyperlink-0"}


def test_fullscreen_status_occupancy_is_sampled_once_per_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_fullscreen_app()
    app.emit_block("answer\n")
    calls = 0

    def status_rows(*, include_transient: bool = True):
        del include_transient
        nonlocal calls
        calls += 1
        return [[("class:status", "busy")]]

    monkeypatch.setattr(app, "_status_rows", status_rows)
    app._before_render(app._app)

    assert calls == 1
    app._fullscreen_transcript.formatted_text(width=20, height=2)
    app._fullscreen_transcript.cursor_position(width=20, height=2)
    assert bool(app._status_region_container.filter())
    assert calls == 1


def test_fullscreen_bootstrap_output_only_mutates_renderer_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_fullscreen_app()
    stdout = io.StringIO()
    real_stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.__stdout__", real_stdout)

    app.emit_block("hello \x1b[31mred\x1b[0m\n")

    assert app.output_buffer.text == ""
    assert app._fullscreen_transcript.text == "hello red\n"
    assert stdout.getvalue() == ""
    assert real_stdout.getvalue() == ""


@pytest.mark.asyncio
async def test_fullscreen_live_output_never_suspends_renderer_or_writes_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def forbidden_run_in_terminal(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("fullscreen transcript must remain application-owned")

    stdout = io.StringIO()
    monkeypatch.setattr("prompt_toolkit.application.run_in_terminal", forbidden_run_in_terminal)
    monkeypatch.setattr("sys.__stdout__", stdout)

    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN) as harness:
        app = harness.app
        assert app is not None
        app.emit_block("first\n")
        await asyncio.sleep(0)
        app.clear_transcript()
        app.emit_block("second\n")
        if app._block_queue is not None:
            await app._block_queue.join()

        assert app.output_buffer.text == ""
        assert app._fullscreen_transcript.text == "second\n"

    assert calls == 0
    assert stdout.getvalue() == ""


def test_fullscreen_drops_accepted_off_thread_callback_after_queue_retirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callback accepted before teardown cannot write after screen restoration."""
    app = _make_fullscreen_app()
    callbacks: list[tuple[object, tuple[object, ...]]] = []

    class DelayedLoop:
        def call_soon_threadsafe(self, callback, *args) -> None:
            callbacks.append((callback, args))

    stdout = io.StringIO()
    monkeypatch.setattr("sys.__stdout__", stdout)
    app._loop = DelayedLoop()  # type: ignore[assignment]
    app._block_queue = asyncio.Queue()

    app.emit_block("late fullscreen output\n")
    assert len(callbacks) == 1

    app._block_queue = None
    callback, args = callbacks.pop()
    callback(*args)  # type: ignore[operator]

    assert "late fullscreen output" not in app.output_buffer.text
    assert "late fullscreen output" not in app._fullscreen_transcript.text
    assert stdout.getvalue() == ""


@pytest.mark.asyncio
async def test_fullscreen_resize_reflows_without_destructive_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .tui_app_harness import MutableRecordingOutput

    async def forbidden_run_in_terminal(*_args, **_kwargs):
        raise AssertionError("fullscreen resize must remain renderer-owned")

    stdout = io.StringIO()
    monkeypatch.setattr("prompt_toolkit.application.run_in_terminaЧЯ8Чg!j»'ўЫ!Ј	Ь—ЫЭ]]ЭЪ[™ЭИ\И›Э›Ы™B€ЫЫќ›ЫH\—ЫЭ]]ЭЪ[™ЭЛЫЫќ[ќ€ЫЫќ›ЫЬ™X]WШЫЫќ[ќ
ЊЊ
B€HHШЫЬWЫX™[ЬЬЪ][ЫЉ\—Щќ[ШЬ™Y[—Э[њШЬљ\ЪYMЊZYЪLЊ
B‚€ЫЫќ›Ы›[Э\ЩWЪ[™\ЉЫ[Э\ЩWЩ]™[ќ
[Э\ЩQ]™[ќ\K“SХTСWСХУ‹^O^JJB€ЫЫќ›Ы›[Э\ЩWЪ[™\ЉЫ[Э\ЩWЩ]™[ќ
[Э\ЩQ]™[ќ\K“SХTСWУSХ‘KLO^JJB€ЫЫќ›Ы›[Э\ЩWЪ[™\ЉЫ[Э\ЩWЩ]™[ќ
[Э\ЩQ]™[ќ\K“SХTСWХT^O^JJB‚€\ЬЩ\ќЫЬYYOHЧB‚‚\Ю[ИY€\ЭЩќ[ШЬ™Y[—Э[YWЬ™Yњ™\ЪЬ™XЫЫЬњЧЫZ^YЬ™]Z[™YЪ\ЭЬћWЩ›Ь—Ш[Э[Y\К
HO€›Ы™N‚€њ›ЫH\\И[\ЬќЪ[\S[Y\ЬXЩB‚€њ›ЫH›ЫШWШЫKќZH[\Ьќ[YB€њ›ЫH›ЫШWШЫKќZK™њ›Ыќ[™[\Ьќ\›Z[[њ›Ыќ[™€њ›ЫH›ЫШWШЫKќZK›Э]][\Ьќ
€YЩ[ќY\ЬШYЩK€[Э]]€\ЭЬћT™\^K€\ЭЬћU\›‹€Э\ќ\[™›Л€X›SЭ]]€^Э]]€
B€њ›ЫH›ЫШWШЫKќZKњЩ\ЬЪ[Ы€[\ЬќС[Z]Э™X[B€њ›ЫHљXЪЫЫњЫЫH[\ЬќЫЫњЫЫB‚€ЬљYЪ[[Э[YHH[YK™Щ]Э[YJ
B€ћN‚€[YKњЩ]Э[YJ›[ШЪHЉB€\HЫXZЩWЩќ[ШЬ™Y[—Ш\

B€њ›Ыќ[™H\›Z[[њ›Ыќ[™
Ъ[\S[Y\ЬXЩJZOTЪ[\S[Y\ЬXЩJ[YOH›[ШЪHЉJJB€њ›Ыќ[™љ[™Ш\
\
B€Э™X[HHС[Z]Э™X[J€\™[Z]Ш›ШЪЛ€™\^WЭЪY[[X™N€М‹€^[Э]ЭЪY[[X™N€М‹€Э\ЬќЧШЫЩWШЫЬWШXЭ[ЫњПUќYK€
B€њ›Ыќ[™—ШЫЫњЫЫKњ™\XЩWШЫЫњЫЫJ€ЫЫњЫЫJ€љ[O\Э™X[K€›ЬЩWЭ\›Z[[UќYK€ЫЫЬ—ЬЮ\Э[OHЊЌM€‹€ЪYMМ‹€[YO][YKЬ™X]WЭ[YJ
K€
B€
B€\™[Z]Ш›ШЪКњ]И™]Z[™YЭ]]€‹]™[ќЪYHњ]И‹YЬП^Ињ]ИџKЩY\UќYJB€Э]]ИHВ€
Љ€^Э]]
€њ™]Z[™YЫ]™[H‹]™[
B€›Ь€]™[[€
љ[™›И‹™\њ›Ь€‹ќШ\›љ[™И‹њЭXШЩ\ЬИ‹њЭ]\ИЉB€
K€X›SЭ]]
ИљЪ[™‹ќ[YH—KЦИќ[YH‹›]™H—WK]OHњ™]Z[™YX›HЉK€[Э]]
И‹Э[YHЋ€”ЭЪ]Ъ[YHџJK€YЩ[ќY\ЬШYЩJ”™]Z[™Y›ЬЩHЪ][›[™HЫЩX[™——]Ы—њљ[ќ
	Ы]™IКWЉK€Э\ќ\[™›К€[Щ[Hњ›ЭљY\‹Ы[Щ[‹€ЪЬќЫ[Щ[H›[Щ[‹€ЫЬљЪ[™ЧЩ\ЏH‹ЭЫЬљИ‹€љWЫ[ЩOQ[ЩK€
K€\ЭЬћT™\^J€\›њПVТ\ЭЬћU\›Љ›ЫOHYЩ[ќ‹ЫЫќ[ќH‘X\›Y\€™]Z[™Y™\ЬЫњЩHЪ]ЫЩXЉWK€Щ\ЬЪ[Ы—ЪYHќ[YKZ\ЭЬћH‹€ЪЭЧЪXY\ЏUќYK€ЪЭЧЩ›ЫЭ\ЏUќYK€
K€B€Ъ]њ›Ыќ[™]ЪЬ™[™\Љ
N‚€›Ь€Э]][€Э]]О‚€]ШZ]њ›Ыќ[™њ™[™\ЉЭ]]
B‚€\ЬЩ\ќ[Љ\—Э[њШЬљ\Ш›ШЪЬКHOHВ€]ЧШ›ШЪЛ]ЪYШ›ШЪЛ\ЭЬћWШ›ШЪИH\—Э[њШЬљ\Ш›ШЪЬВ€\ЬЩ\ќ]ЧШ›ШЪЛњ™\^H\И›Ы™B€\ЬЩ\ќ]ЧШ›ШЪЛ™]™[ќЪYOHњ]И‚€\ЬЩ\ќ]ЧШ›ШЪЛќYЬИOHњ›Ю™[њЩ]
Ињ]ИџJB€\ЬЩ\ќ]ЧШ›ШЪЛљЩY\\ИќYB€\ЬЩ\ќ]ЪYШ›ШЪЛњ™\^H\И›Э›Ы™B€\ЬЩ\ќ\ЭЬћWШ›ШЪЛњ™\^H\И›Э›Ы™B€™XЫЬ™ЪYИHШ›ШЪЛќ[њШЬљ\Ь™XЫЬ™ЪY›Ь€›ШЪИ[€\—Э[њШЬљ\Ш›ШЪЬЧB€ЫЬWШXЭ[ЫњИHЩXЭ
›ШЪЛЫЩWШЫЬWШXЭ[ЫњКH›Ь€›ШЪИ[€\—Э[њШЬљ\Ш›ШЪЬЧB€]ЧЬ™[™\™YH]ЧШ›ШЪЛ™ќ[ШЬ™Y[—Ь™[™\™Y€™[™\™YШћWЭ[YHHЯB€›Ь€[YH[€[YK•SQTО‚€[YKњЩ]Э[YJ[YJB€њ›Ыќ[™њ™Yњ™\ЪЭ[YJ
B€™[™\™YШћWЭ[YVЫ[YWHH€‹љ›Ъ[Љ€›ШЪЛ™ќ[ШЬ™Y[—Ь™[™\™YЬ€€€›Ь€›ШЪИ[€\—Э[њШЬљ\Ш›ШЪЬВ€
B€Z[€H\—Щќ[ШЬ™Y[—Э[њШЬљ\ќ^€\ЬЩ\ќ]ЧШ›ШЪЛ™ќ[ШЬ™Y[—Ь™[™\™YOH]ЧЬ™[™\™Y€\ЬЩ\ќШ›ШЪЛќ[њШЬљ\Ь™XЫЬ™ЪY›Ь€›ШЪИ[€\—Э[њШЬљ\Ш›ШЪЬЧHOH™XЫЬ™ЪYВ€\ЬЩ\ќШ›ШЪЛЫЩWШЫЬWШXЭ[ЫњИ›Ь€›ШЪИ[€\—Э[њШЬљ\Ш›ШЪЬЧHOHЫЬWШXЭ[ЫњВ€›Ь€^XЭY[€
€њ]И™]Z[™YЭ]]‹€
Љ€€њ™]Z[™YЫ]™[H‚€›Ь€]™[[€
љ[™›И‹™\њ›Ь€‹ќШ\›љ[™И‹њЭXШЩ\ЬИ‹њЭ]\ИЉB€
K€њ™]Z[™YX›H‹€‹Э[YH‹€”™]Z[™Y›ЬЩH‹€““УРH™XYH‹€‘X\›Y\€™]Z[™Y™\ЬЫњЩH‹€
N‚€\ЬЩ\ќZ[‹ЫЭ[ќ
^XЭY
HOHB‚€\ЬЩ\ќ[ЉЩ]
™[™\™YШћWЭ[YKќ[Y\К
JJHOH[Љ[YK•SQTКB€љ[[N‚€[YKњЩ]Э[YJЬљYЪ[[Э[YJB‚‚™Y€\ЭЩќ[ШЬ™Y[—Э[YWЬ™Yњ™\ЪЬ™XЫЫЬњЧЬ™]Z[™YШYЩ[ќЩ]™[ќЬЮ[ќ^

HO€›Ы™N‚€њ›ЫH›ЫШWШЫKќZH[\Ьќ[YB€њ›ЫH›ЫШWШЫKќZKњЩ\ЬЪ[Ы€[\ЬќЩ\ЬЪ[Ы‚€њ›ЫH›ЫШWШЫKќZKќ[YH[\Ьќ[YTЮ[ќ^‚€ЬљYЪ[[Э[YHH[YK™Щ]Э[YJ
B€ћN‚€[YKњЩ]Э[YJ›[ШЪHЉB€\HЫXZЩWЩќ[ШЬ™Y[—Ш\

B€Щ\ЬЪ[Ы€HЩ\ЬЪ[Ы‹—ЧЫ™]ЧЧКЩ\ЬЪ[ЫЉB€Щ\ЬЪ[Ы‹—Ш\H\€Ю[ќ^H[YTЮ[ќ^
€™Y€Ь™Y]
[YN€ЭЉHO€ЭЋ—€™]\›€[YH‹€њ]Ы€‹€XЪЩЬ›Э[™ШЫЫЬЏH™Y][‹€
B€Щ\ЬЪ[Ы‹—Щ[Z]Э^
Ю[ќ^
B€™Y›Ь™HH\—Э[њШЬљ\Ш›ШЪЬЦМK™ќ[ШЬ™Y[—Ь™[™\™Y€\ЬЩ\ќ™Y›Ь™H\И›Э›Ы™B‚€[YKњЩ]Э[YJќњЫYЪЉB€\њ™Yњ™\ЪЬЭ[J
B€Yќ\€H\—Э[њШЬљ\Ш›ШЪЬЦМK™ќ[ШЬ™Y[—Ь™[™\™Y‚€\ЬЩ\ќYќ\€\И›Э›Ы™B€\ЬЩ\ќYќ\€OH™Y›Ь™B€\ЬЩ\ќ\—Щќ[ШЬ™Y[—Э[њШЬљ\ќ^ЫЭ[ќ
™Y€Ь™Y]ЉHOHB€љ[[N‚€[YKњЩ]Э[YJЬљYЪ[[Э[YJB‚‚ђ]\Э›X\љЛ\Ю[Ъ[В\Ю[ИY€\ЭЫ]]™WЬ™\^WЭ[YWЬ™Yњ™\ЪЬ™\]Y\ЭЧЬШ[YWЭЪYЬ™\^J
HO€›Ы™N‚€њ›ЫH›ЫШWШЫKќZKќZWШ\XШ][Ы€[\Ьќ
€[њШЬљ\›ШЪЛ€RP\XШ][Ы‹€Ф™\Ъ^™T™\^T]Y]YR][K€
B‚€њ›ЫHќZWШ\Ъ\›™\ЬИ[\Ьќ]]X›T™XЫЬ™[™УЭ]]‚€Э]]H]]X›T™XЫЬ™[™УЭ]]
ЫЫ[[њПN›ЭЬПLЌ
B€Ъ]Ь™X]WШ\ЬЩ\ЬЪ[ЫЉ[њ]Q[[^R[њ]

KЭ]][Э]]
N‚€\HRP\XШ][ЫЉ\Ь^WЫ[ЩOQ\Ь^S[ЩK“ђUU‘WФ‘TVJB€\—ЫЫЬH\Ю[Ъ[Л™Щ]Ьќ[›љ[™ЧЫЫЬ

B€\—Ш›ШЪЧЬ]Y]YHH\Ю[Ъ[Л”]Y]YJ
B€\—Ь™\Ъ^™WЬ™\^\ЧЩ[X›YHќYB€\—Ь™\Ъ^™WЬ™Y›ЭЛ›ШњЩ\ќ™J
Ќ
JB€\™[Z]Ш›ШЪК›Ы[YW€‹™\^O[[X™N€›™]И[YW€ЉB‚€\њ™Yњ™\ЪЬЭ[J
B‚€\ЬЩ\ќ\—Ь™\Ъ^™WЬ™Y›ЭЛњ™\^WЬ™\]Z\™Y\ИќYB€\ЬЩ\ќ\—Ь™\Ъ^™WЬ™Y›ЭЛњ[™[™ЧЭЪYOH€]ШZ]\Ю[Ъ[ЛњЫY\
ЊJB€]Y]YYH\—Ш›ШЪЧЬ]Y]YK™Щ]Ы›ЭШZ]

B€\ЬЩ\ќ\Ъ[њЭ[ЩJ]Y]YY[њШЬљ\›ШЪКB€]Y]YYH\—Ш›ШЪЧЬ]Y]YK™Щ]Ы›ЭШZ]

B€\ЬЩ\ќ\Ъ[њЭ[ЩJ]Y]YYФ™\Ъ^™T™\^T]Y]YR][JB€\ЬЩ\ќ]Y]YYњ™\]Y\Эњ™\]Z\™Y\ИќYB€\ЬЩ\ќ]Y]YYњ™\]Y\ЭќЪYOH€\—ШШ[Щ[Ь™\Ъ^™WЬ™\^WЭЫЬљК
B‚‚™Y€\ЭЭ[њШЬљ\ЬЩX\ЪЪYЪYЪЧШ[ЫX]Ъ\ЧШ[™Ь™]™X[ЧЩљ\њЭ

HO€›Ы™N‚€њ›ЫH›ЫШWШЫKќZK™ќ[ШЬ™Y[—Э[њШЬљ\[\Ьќќ[ШЬ™Y[•[њШЬљ\[Щ[‚€[Щ[Hќ[ШЬ™Y[•[њШЬљ\[Щ[
ЪЭЧЭZ[[™ЧШ›[љПQ[ЩJB€[Щ[\[™
ћ™\›Ч›Ы™W“™YYHљ\њЭќ™YW™›Э\—›™YYHЩXЫЫ™њЪ^ЉB€[Щ[њЩ]ЬЩX\Ъ
“‘QQH‹ЪYLЊZYЪLЉB‚€И™]™X[ИЩ[ќ\€HX]Ъ™\ќXШ[HЪ[€ЫЫќ^^\ЭИX›Э™H]€И[њЭXYЩ€[›љ[™И]ИHљY]ЬЬќ	ЬИљ\њЭ›ЭЛ‚€\ЬЩ\ќ[Щ[њЩX\ЪЬЬЪ][Ы€OH
KЉB€\ЬЩ\ќ[Щ[ќЬЬ›ЭКЪYLЊZYЪLЉHOHB€њYЫY[ќИH[Щ[™›Ь›X]YЭ^
ЪYLЊZYЪLЉB€\ЬЩ\ќ“™YYH€[€€‹љ›Ъ[Љ^›Ь€ЬЭ[K^[€њYЫY[ќКB€\ЬЩ\ќ[ћJќ[њШЬљ\\ЩX\ЪXЭ\њ™[ќ€[€Э[H›Ь€Э[K^[€њYЫY[ќИY€^њЭљ\

JB‚€\ЬЩ\ќ[Щ[›[Э™WЬЩX\ЪЫX]Ъ
KЪYLЊZYЪLЉB€\ЬЩ\ќ[Щ[њЩX\ЪЬЬЪ][Ы€OH
‹ЉB€\ЬЩ\ќ[Щ[ќЬЬ›ЭКЪYLЊZYЪLЉHOH€\ЬЩ\ќ[Щ[›[Э™WЬЩX\ЪЫX]Ъ
KЪYLЊZYЪLЉB€\ЬЩ\ќ[Щ[њЩX\ЪЬЬЪ][Ы€OH
KЉB‚‚™Y€\ЭЭ[њШЬљ\ЬЩX\ЪЬ™]™X[ШЩ[ќ\њЧЫX]ЪЭ™\ќXШ[J
HO€›Ы™N‚€€€ђH™]™X[YX]ЪЪЭЬИ™Y›Ь™KШYќ\€ЫЫќ^›Эќ\ЭHX]Ъ›ЭЛ€€€‚€њ›ЫH›ЫШWШЫKќZK™ќ[ШЬ™Y[—Э[њШЬљ\[\Ьќќ[ШЬ™Y[•[њШЬљ\[Щ[‚€[Щ[Hќ[ШЬ™Y[•[њШЬљ\[Щ[
ЪЭЧЭZ[[™ЧШ›[љПQ[ЩJB€[Щ[\[™
—€‹љ›Ъ[Љ€›[™HЪ_H€›Ь€H[€[™ЩJLЉJH
И—›™YYW€€
И—€‹љ›Ъ[Љ€ќZ[Ъ_H€›Ь€H[€[™ЩJLЉJJB€[Щ[њЩ]ЬЩX\Ъ
›™YYH‹ЪYLМZYЪMКB‚€ЬH[Щ[ќЬЬ›ЭКЪYLМZYЪMКB€ИHX]Ъ›ЭИ
LЉHЪ]И[€HZYHЩ€HЛ\›ЭИЪ[™ЭЛ‚€\ЬЩ\ќЬOHB€љ\ЪX›HH[Щ[™›Ь›X]YЭ^
ЪYLМZYЪMКB€^H€‹љ›Ъ[ЉњYЦМWH›Ь€њYИ[€љ\ЪX›JB€\ЬЩ\ќ›™YYH€[€^€\ЬЩ\ќ›[™HL€[€^[™›[™HLH€[€^ИЫЫќ^X›Э™B€\ЬЩ\ќќZ[€[€^[™ќZ[H€[€^ИЫЫќ^™[ЭВ‚‚™Y€\ЭЭ[њШЬљ\ЬЩX\ЪЩ\Э[™ЭZ\Ъ\ЧШЭ\њ™[ќШ[™ЫЭ\—ЫX]Ъ\К
HO€›Ы™N‚€њ›ЫH›ЫШWШЫKќZK™ќ[ШЬ™Y[—Э[њШЬљ\[\Ьќќ[ШЬ™Y[•[њШЬљ\[Щ[‚€[Щ[Hќ[ШЬ™Y[•[њШЬљ\[Щ[
ЪЭЧЭZ[[™ЧШ›[љПQ[ЩJB€[Щ[\[™
›™YYH[™™YYHЉB€[Щ[њЩ]ЬЩX\Ъ
›™YYH‹ЪYMZYЪLЉB€њYЫY[ќИH[Щ[™›Ь›X]YЭ^
ЪYMZYЪLЉB‚€\ЬЩ\ќ[ћJќ[њШЬљ\\ЩX\ЪXЭ\њ™[ќ€[€Э[H›Ь€Э[K^[€њYЫY[ќИY€›€€[€^
B€\ЬЩ\ќ[ћJќ[њШЬљ\\ЩX\Ъ[X]Ъ€[€Э[H›Ь€Э[K^[€њYЫY[ќИY€›€€[€^
B‚‚™Y€\ЭЭ[њШЬљ\ЬЩX\ЪШЫX\—Ь™]\›њЧЭЧЭZ[

HO€›Ы™N‚€њ›ЫH›ЫШWШЫKќZK™ќ[ШЬ™Y[—Э[њШЬљ\[\Ьќќ[ШЬ™Y[•[њШЬљ\[Щ[‚€[Щ[Hќ[ШЬ™Y[•[њШЬљ\[Щ[
ЪЭЧЭZ[[™ЧШ›[љПQ[ЩJB€[Щ[\[™
›™YYW›[™W›[™WќZ[ЉB€[Щ[њЩ]ЬЩX\Ъ
›™YYH‹ЪYLLZYЪLЉB€\ЬЩ\ќ›Э[Щ[ќљY]ЬЬќ™›ЫЭЬЧЭZ[‚€[Щ[њЩ]ЬЩX\Ъ
€‹ЪYLLZYЪLЉB‚€\ЬЩ\ќ[Щ[њЩX\ЪЬЬЪ][Ы€OH

B€\ЬЩ\ќ[Щ[ќљY]ЬЬќ™›ЫЭЬЧЭZ[€\ЬЩ\ќ›Э[ћJ€ќ[њШЬљ\\ЩX\Ъ€[€Э[H›Ь€Э[KЭ^[€[Щ[™›Ь›X]YЭ^
ЪYLLZYЪLЉB€
B‚‚‚™Y€\ЭЬ™\[™ШЫЫњЭ[Y\ЧЩЩ[™\]YЬ™XЫЬ™ЪYК
HO€›Ы™N‚€€€’[\XЪ]™\[™QИЭ^H[љ\]YH[™›ЫЭИ^XЪ]QЛ€€€‚€њ›ЫH›ЫШWШЫKќZK™ќ[ШЬ™Y[—Э[њШЬљ\[\Ьќќ[ШЬ™Y[•[њШЬљ\[Щ[‚€[Щ[Hќ[ШЬ™Y[•[њШЬљ\[Щ[
ЪЭЧЭZ[[™ЧШ›[љПQ[ЩJB€[Щ[њ™\[™
›™]Щ\ЭЉB€[Щ[њ™\[™
›ZYHЉB€[Щ[њ™\[™
›Ы\Э‹™XЫЬ™ЪYLL
B€[Щ[њ™\[™
™љ\њЭЉB‚€\ЬЩ\ќЬ™XЫЬ™њ™XЫЬ™ЪY›Ь€™XЫЬ™[€[Щ[—Ь™XЫЬ™ЧHOHМLKLKB€\ЬЩ\ќ[Љ[Щ[—Ь™XЫЬ™Ъ[™^\К
VМJHOH‚‚™Y€\ЭЬ™\[™Ь™\Щ\ќ™\ЧЭљ\ЪX›WЭZ[Ш[™Ь™XЫЬ™Ш[ЪЬЉ
HO€›Ы™N‚€€€“Ы\€\ЭЬћH\X\њИX›Э™HЪ]Э][Эљ[™И[™XYK]љ\ЪX›H™]Щ\€[™\Л€€€‚€њ›ЫH›ЫШWШЫKќZK™ќ[ШЬ™Y[—Э[њШЬљ\[\Ьќќ[ШЬ™Y[•[њШЬљ\[Щ[‚€[Щ[Hќ[ШЬ™Y[•[њШЬљ\[Щ[
ЪЭЧЭZ[[™ЧШ›[љПQ[ЩJB€[Щ[\[™
›™]И›™]ИW›™]И—€‹™XЫЬ™ЪYLЊ
B€™Y›Ь™HH[Щ[™›Ь›X]YЭ^
ЪYLЊZYЪLЉB‚€[Щ[њ™\[™
›Ы›ЫW€‹™XЫЬ™ЪYLL
B‚€\ЬЩ\ќ[Щ[™›Ь›X]YЭ^
ЪYLЊZYЪLЉHOH™Y›Ь™B€\ЬЩ\ќ[Щ[ќ^OH›Ы›ЫW›™]И›™]ИW›™]И—€‚‚€[Щ[њШЬ›ЫЭљ\ЭX[Ы[™\КLKЪYLЊZYЪLЉB€[ЪЬ€H[Щ[ќљY]ЬЬќ[ЪЬ‚€\ЬЩ\ќ[ЪЬ€\И›Э›Ы™H[™[ЪЬ‹њ™XЫЬ™ЪYOHЊ€[ЪЬ™YШ™Y›Ь™HH[Щ[™›Ь›X]YЭ^
ЪYLЊZYЪLЉB€[Щ[њ™\[™
›Ы\Э€‹™XЫЬ™ЪYMJB€\ЬЩ\ќ[Щ[ќљY]ЬЬќ[ЪЬ€OH[ЪЬ‚€\ЬЩ\ќ[Щ[™›Ь›X]YЭ^
ЪYLЊZYЪLЉHOH[ЪЬ™YШ™Y›Ь™BѓыkєwµзhєЪn