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
