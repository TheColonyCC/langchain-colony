"""The assertions whose absence let the async poller return nothing for releases.

The bug: before colony-sdk 1.30.0, ``AsyncColonyClient`` wrapped bare-array
bodies as ``{"data": [...]}``. Every call site here unwrapped with a *guessed*
key (``notifications``, ``colonies``, ``webhooks``, ``items``), none of which is
``data`` — so ``poll_once_async()`` returned ``[]`` on every call and the async event
stream never fired.

Why no existing test caught it: the async tests asserted the poller *ran without
raising*, and `[]` is a completely plausible answer to "any new notifications?".
So the tests below assert on the **returned rows**, not on the absence of an
exception. That is the only assertion that can catch this class.

Each case is paired with a control, because a test that only ever sees the fixed
shape cannot tell a working unwrapper from one that returns everything.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from langchain_colony._response import as_list
from langchain_colony.events import ColonyEventPoller
from langchain_colony.tools import _format_colonies, _format_notifications, _format_webhooks

NOTIF = {
    "id": "n1",
    "type": "reply",
    "post_id": "p1",
    "comment_id": "c1",
    "created_at": "2026-07-25T10:00:00Z",
    "is_read": False,
}


# ── the helper, shape by shape ────────────────────────────────────────


class TestAsList:
    def test_a_bare_list_passes_through(self) -> None:
        """The shape every measured list endpoint actually returns."""
        assert as_list([NOTIF], "get_notifications") == [NOTIF]

    def test_the_data_envelope_is_unwrapped(self) -> None:
        """colony-sdk < 1.30.0's AsyncColonyClient wrapping — the actual bug.

        This is the case that returned [] for several releases.
        """
        assert as_list({"data": [NOTIF]}, "get_notifications") == [NOTIF]

    def test_the_items_envelope_is_unwrapped(self) -> None:
        """`get_comments()` really does paginate under `items` — measured
        against the live API, not assumed. Removing this key would break
        comment enrichment."""
        assert as_list({"items": [NOTIF], "total": 1, "page": 1}, "get_comments") == [NOTIF]

    def test_an_empty_list_stays_empty(self) -> None:
        """CONTROL: the fix must not invent rows. 'Nothing there' is a real
        answer and has to survive."""
        assert as_list([], "get_notifications") == []
        assert as_list({"data": []}, "get_notifications") == []

    def test_an_unrecognised_dict_is_empty_AND_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """The half that matters more than the unwrapping.

        Returning [] silently is what made the original bug invisible: an
        empty feed is indistinguishable from a shape nobody parsed. A future
        envelope must now leave a trace.
        """
        with caplog.at_level(logging.WARNING, logger="langchain_colony"):
            assert as_list({"unexpected": [NOTIF]}, "get_notifications") == []
        assert "get_notifications" in caplog.text
        assert "no recognised list" in caplog.text

    def test_a_non_collection_is_empty_AND_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="langchain_colony"):
            assert as_list("nope", "get_colonies") == []
        assert "get_colonies" in caplog.text

    def test_the_warning_is_silent_on_the_happy_path(self, caplog: pytest.LogCaptureFixture) -> None:
        """CONTROL for the two tests above: a helper that warned on every call
        would satisfy them while making the logs useless."""
        with caplog.at_level(logging.WARNING, logger="langchain_colony"):
            as_list([NOTIF], "get_notifications")
            as_list({"data": [NOTIF]}, "get_notifications")
            as_list({"items": [NOTIF]}, "get_comments")
        assert caplog.text == ""


# ── the poller, end to end, on both transports ───────────────────────


class _FakeAsyncClient:
    """Returns whatever shape the test asks for, from an async method — so
    `asyncio.iscoroutinefunction` sends the poller down its native-async
    branch, which is where the bug lived."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload

    async def get_notifications(self, unread_only: bool = False) -> Any:
        return self.payload


class _FakeSyncClient:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def get_notifications(self, unread_only: bool = False) -> Any:
        return self.payload


class TestPollerReturnsRows:
    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param([NOTIF], id="bare-list (sdk >= 1.30.0)"),
            pytest.param({"data": [NOTIF]}, id="data-envelope (sdk < 1.30.0 async)"),
        ],
    )
    def test_poll_once_async_yields_the_notification(self, payload: Any) -> None:
        """THE regression test. Pre-fix, the data-envelope case returned []."""
        poller = ColonyEventPoller(client=_FakeAsyncClient(payload), enrich=False)  # type: ignore[arg-type]
        got = asyncio.run(poller.poll_once_async())
        assert [n.id for n in got] == ["n1"], f"expected the row through, got {got!r}"

    def test_poll_once_yields_the_notification(self) -> None:
        """The sync client was always correct; pin that it stays so."""
        poller = ColonyEventPoller(client=_FakeSyncClient([NOTIF]), enrich=False)  # type: ignore[arg-type]
        assert [n.id for n in poller.poll_once()] == ["n1"]

    def test_sync_and_async_agree(self) -> None:
        """The assertion that would have caught this from the start: the two
        transports must return the same thing for the same logical response.
        Each client's own tests passed because each was self-consistent."""
        s = ColonyEventPoller(client=_FakeSyncClient([NOTIF]), enrich=False)  # type: ignore[arg-type]
        a = ColonyEventPoller(client=_FakeAsyncClient({"data": [NOTIF]}), enrich=False)  # type: ignore[arg-type]
        assert [n.id for n in s.poll_once()] == [n.id for n in asyncio.run(a.poll_once_async())]

    def test_an_empty_feed_is_still_empty(self) -> None:
        """CONTROL: the fix must not turn 'no notifications' into a phantom."""
        poller = ColonyEventPoller(client=_FakeAsyncClient([]), enrich=False)  # type: ignore[arg-type]
        assert asyncio.run(poller.poll_once_async()) == []


# ── the formatters, which had the same guessed keys ───────────────────


class TestFormattersAcceptBothShapes:
    @pytest.mark.parametrize(
        "payload",
        [[NOTIF], {"data": [NOTIF]}, {"notifications": [NOTIF]}],
    )
    def test_notifications(self, payload: Any) -> None:
        assert "No notifications." not in _format_notifications(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            [{"name": "general", "description": "d", "member_count": 1}],
            {"data": [{"name": "general", "description": "d", "member_count": 1}]},
        ],
    )
    def test_colonies(self, payload: Any) -> None:
        assert "No colonies found." not in _format_colonies(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            [{"id": "w1", "url": "https://e.test", "events": ["post"]}],
            {"data": [{"id": "w1", "url": "https://e.test", "events": ["post"]}]},
        ],
    )
    def test_webhooks(self, payload: Any) -> None:
        assert "No webhooks registered." not in _format_webhooks(payload)

    def test_empty_still_reads_as_empty(self) -> None:
        """CONTROL: 'nothing here' must keep saying so on every formatter."""
        assert _format_notifications([]) == "No notifications."
        assert _format_colonies([]) == "No colonies found."
        assert _format_webhooks([]) == "No webhooks registered."
