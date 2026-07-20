# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_resolve_image`` -- the inverse of ``_encode_image``.

The encode/decode pair is "string pass-through; otherwise canonical
JSON".  The decode side has subtle branches the round-trip integration
tests don't exercise (string URLs that happen to be valid JSON, raw
non-JSON content, lists, mixed elements).  These unit tests pin each
branch directly.
"""

from __future__ import annotations

from nooa.tracing._journal_builder import _encode_image
from nooa.viewer.otlp_store import _resolve_image


class TestResolveImage:
    def test_string_url_round_trips_unchanged(self):
        url = "https://example.com/cat.png"
        encoded = _encode_image(url)
        assert encoded == url, "encode pass-through"
        assert _resolve_image(encoded) == url

    def test_dict_round_trips_to_dict(self):
        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
        encoded = _encode_image(img)
        assert encoded.startswith("{")
        assert _resolve_image(encoded) == img

    def test_dict_with_unicode_keys_and_values(self):
        img = {"alt": "café ☕", "url": "ünïcødé"}
        encoded = _encode_image(img)
        assert _resolve_image(encoded) == img

    def test_list_round_trips_to_list(self):
        img = ["a", "b", "c"]
        encoded = _encode_image(img)
        assert encoded.startswith("[")
        assert _resolve_image(encoded) == img

    def test_json_scalar_strings_kept_as_strings(self):
        """A user-supplied string image whose content happens to be
        valid JSON for a *scalar* (``"null"``, ``"42"``, ``"true"``,
        ``'"hello"'``) round-trips as the original string -- only
        dict/list shapes are accepted as round-tripped non-strings."""
        for s in ["null", "42", "true", "false", "3.14", '"hello"']:
            encoded = _encode_image(s)  # pass-through
            assert encoded == s
            assert _resolve_image(encoded) == s, (
                f"scalar-shaped string {s!r} should round-trip as the "
                f"original string, not the parsed value"
            )

    def test_string_that_parses_as_dict_or_list_returns_parsed(self):
        """A pathological string image that happens to be JSON for a
        dict or list returns the parsed value -- the inverse can't
        disambiguate vs. an actual dict/list input.  This is documented
        as the boundary, not the desired behaviour for arbitrary inputs;
        in practice ``_encode_image`` of a string always passes through
        and only dicts/lists trigger the canonical-JSON path."""
        for s in ['{"a": 1}', "[1, 2, 3]"]:
            encoded = _encode_image(s)
            assert encoded == s
            resolved = _resolve_image(encoded)
            assert isinstance(resolved, (dict, list))

    def test_malformed_json_stays_a_string(self):
        bad = "{not valid json"
        encoded = _encode_image(bad)
        assert encoded == bad  # pass-through
        # _resolve_image tries json.loads, gets JSONDecodeError, returns
        # the original string -- consumers see a string, not an exception.
        assert _resolve_image(bad) == bad

    def test_non_string_input_returned_unchanged(self):
        """``_resolve_image`` is defensive: if a caller hands it
        something that already isn't a string (shouldn't happen via the
        DB read path but is cheap to handle), pass it through."""
        assert _resolve_image(42) == 42  # type: ignore[arg-type]
        d = {"already": "a dict"}
        assert _resolve_image(d) is d  # type: ignore[arg-type]

    def test_round_trip_for_dict_with_nested_list(self):
        img = {"parts": [{"a": 1}, {"a": 2}], "tag": "x"}
        encoded = _encode_image(img)
        assert _resolve_image(encoded) == img

    def test_round_trip_preserves_dict_key_order_via_canonical_form(self):
        """``_encode_image`` uses ``sort_keys=True`` so two dicts with
        different insertion orders hash identically.  ``_resolve_image``
        returns Python's natural dict (insertion-ordered), so equality
        with the original dict still holds after the round trip."""
        img = {"z": 1, "a": 2, "m": 3}
        encoded = _encode_image(img)
        assert encoded == '{"a":2,"m":3,"z":1}'
        assert _resolve_image(encoded) == img
