# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn results refuse unknown fields instead of dropping them."""

import pytest
from pydantic import ValidationError

from nooa.interactive import Done, NeedInput, Waiting


@pytest.mark.parametrize(
    "make",
    [
        lambda: Done(explanation="done", value="42"),
        lambda: Waiting(explanation="tests", on=["tests"], reply="later"),
        lambda: NeedInput(question="which?", answer="a"),
    ],
)
def test_an_unknown_field_is_an_error(make):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        make()


def test_known_fields_still_work():
    assert Done(explanation="done", message="hi").message == "hi"
    assert Waiting(explanation="tests", on=["tests"], message="soon").on == ["tests"]


def test_the_json_kind_tag_is_accepted_only_for_its_own_type():
    assert Done.model_validate({"kind": "done", "explanation": "ok"}) == Done(explanation="ok")
    assert NeedInput.model_validate({"kind": "need_input", "question": "q"}).question == "q"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Done.model_validate({"kind": "waiting", "explanation": "ok"})
