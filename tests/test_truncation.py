"""Our own cuts have to announce themselves.

These formatters build prose for a model to read rather than dicts, so there is
no sibling boolean to carry the flag — the marker has to live in the string,
and it has to stay short enough to repeat twenty times in a listing.

On 2026-08-18 a bare slice in a sibling package cut a 1,699 character post to
1,500; the agent reading it correctly saw the text stop mid-sentence and
reported in public that the *author* had posted it that way. It was truthful
about the bytes it received.

Every arm is paired with a control.
"""

from __future__ import annotations

from langchain_colony.tools import _cut

LONG = "x" * 1699
SHORT = "a complete short body."


class TestCut:
    def test_long_text_is_cut_and_says_so(self) -> None:
        out = _cut(LONG, 200)
        assert out.startswith("x" * 200)
        assert "+1499 chars cut by us, not the author" in out

    def test_short_text_is_byte_identical(self) -> None:
        """The control. A marker on everything carries no information."""
        assert _cut(SHORT, 200) == SHORT
        assert "cut by us" not in _cut(SHORT, 200)

    def test_exactly_at_the_limit_is_untouched(self) -> None:
        exact = "y" * 200
        assert _cut(exact, 200) == exact

    def test_one_over_is_cut(self) -> None:
        assert "+1 chars cut by us" in _cut("y" * 201, 200)

    def test_empty_is_untouched(self) -> None:
        assert _cut("", 200) == ""

    def test_it_is_not_the_bare_slice(self) -> None:
        """Mutation arm: reverting to ``text[:limit]`` fails here."""
        assert _cut(LONG, 200) != LONG[:200]

    def test_the_marker_names_us_not_the_author(self) -> None:
        assert "not the author" in _cut(LONG, 200)

    def test_the_marker_stays_cheap_enough_to_repeat(self) -> None:
        """A listing gives each item ~200 chars; the note must not swamp it."""
        overhead = len(_cut(LONG, 200)) - 200
        assert overhead < 50, f"marker is {overhead} chars, too heavy for a listing"

    def test_the_reported_remainder_is_exact(self) -> None:
        for limit in (10, 200, 1698):
            assert f"+{1699 - limit} chars" in _cut(LONG, limit)
