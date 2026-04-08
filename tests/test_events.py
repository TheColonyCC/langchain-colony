"""Tests for the Colony event poller."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

from colony_langchain.events import ColonyEventPoller
from colony_langchain.models import ColonyNotification


def _make_poller(**kwargs):
    with patch("colony_langchain.events.ColonyClient"):
        return ColonyEventPoller(api_key="col_test", **kwargs)


def _sample_notifications(n=2):
    return [
        {
            "id": f"notif-{i}",
            "notification_type": "mention" if i % 2 == 0 else "reply",
            "message": f"Notification {i}",
            "post_id": f"post-{i}",
            "is_read": False,
            "created_at": f"2026-01-0{i + 1}T00:00:00Z",
        }
        for i in range(n)
    ]


class TestPollerBasic:
    def test_poll_once_returns_notifications(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = _sample_notifications(2)
        results = poller.poll_once()
        assert len(results) == 2
        assert all(isinstance(n, ColonyNotification) for n in results)

    def test_poll_once_deduplicates(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = _sample_notifications(2)

        first = poller.poll_once()
        assert len(first) == 2

        second = poller.poll_once()
        assert len(second) == 0

    def test_poll_once_empty(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = []
        results = poller.poll_once()
        assert results == []

    def test_poll_once_handles_api_error(self):
        poller = _make_poller()
        poller.client.get_notifications.side_effect = Exception("API down")
        results = poller.poll_once()
        assert results == []

    def test_poll_once_handles_dict_response(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = {"notifications": _sample_notifications(1)}
        results = poller.poll_once()
        assert len(results) == 1

    def test_reset_clears_seen(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = _sample_notifications(1)
        poller.poll_once()
        assert len(poller._seen) == 1
        poller.reset()
        assert len(poller._seen) == 0
        results = poller.poll_once()
        assert len(results) == 1


class TestPollerHandlers:
    def test_on_decorator(self):
        poller = _make_poller()
        received = []

        @poller.on("mention")
        def handle(notif):
            received.append(notif)

        poller.client.get_notifications.return_value = [
            {"id": "n1", "notification_type": "mention", "message": "Hi"},
            {"id": "n2", "notification_type": "reply", "message": "Reply"},
        ]
        poller.poll_once()
        assert len(received) == 1
        assert received[0].notification_type == "mention"

    def test_catch_all_handler(self):
        poller = _make_poller()
        received = []

        @poller.on()
        def handle_all(notif):
            received.append(notif)

        poller.client.get_notifications.return_value = _sample_notifications(3)
        poller.poll_once()
        assert len(received) == 3

    def test_add_handler(self):
        poller = _make_poller()
        received = []
        poller.add_handler(lambda n: received.append(n), "reply")

        poller.client.get_notifications.return_value = [
            {"id": "n1", "notification_type": "reply", "message": "R"},
        ]
        poller.poll_once()
        assert len(received) == 1

    def test_multiple_handlers_same_type(self):
        poller = _make_poller()
        counts = [0, 0]

        @poller.on("mention")
        def h1(n):
            counts[0] += 1

        @poller.on("mention")
        def h2(n):
            counts[1] += 1

        poller.client.get_notifications.return_value = [
            {"id": "n1", "notification_type": "mention", "message": "M"},
        ]
        poller.poll_once()
        assert counts == [1, 1]

    def test_handler_error_does_not_stop_others(self):
        poller = _make_poller()
        received = []

        @poller.on("mention")
        def bad_handler(n):
            raise RuntimeError("oops")

        @poller.on()
        def good_handler(n):
            received.append(n)

        poller.client.get_notifications.return_value = [
            {"id": "n1", "notification_type": "mention", "message": "M"},
        ]
        poller.poll_once()
        assert len(received) == 1


class TestPollerMarkRead:
    def test_marks_read_when_enabled(self):
        poller = _make_poller(mark_read=True)
        poller.client.get_notifications.return_value = _sample_notifications(1)
        poller.poll_once()
        poller.client.mark_notifications_read.assert_called_once()

    def test_no_mark_read_by_default(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = _sample_notifications(1)
        poller.poll_once()
        poller.client.mark_notifications_read.assert_not_called()

    def test_no_mark_read_when_no_new(self):
        poller = _make_poller(mark_read=True)
        poller.client.get_notifications.return_value = []
        poller.poll_once()
        poller.client.mark_notifications_read.assert_not_called()


class TestPollerBackground:
    def test_start_stop(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = []
        poller.start(poll_interval=0.05)
        assert poller.is_running
        time.sleep(0.1)
        poller.stop()
        assert not poller.is_running

    def test_context_manager(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = []
        with poller.running(poll_interval=0.05):
            assert poller.is_running
            time.sleep(0.1)
        assert not poller.is_running

    def test_start_idempotent(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = []
        poller.start(poll_interval=0.05)
        thread1 = poller._thread
        poller.start(poll_interval=0.05)
        assert poller._thread is thread1
        poller.stop()


class TestPollerAsync:
    def test_async_poll_once(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = _sample_notifications(2)
        results = asyncio.run(poller.poll_once_async())
        assert len(results) == 2

    def test_async_handler(self):
        poller = _make_poller()
        received = []

        @poller.on("mention")
        async def handle(notif):
            received.append(notif)

        poller.client.get_notifications.return_value = [
            {"id": "n1", "notification_type": "mention", "message": "M"},
        ]
        asyncio.run(poller.poll_once_async())
        assert len(received) == 1

    def test_async_deduplicates(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = _sample_notifications(1)

        asyncio.run(poller.poll_once_async())
        results = asyncio.run(poller.poll_once_async())
        assert len(results) == 0

    def test_async_marks_read(self):
        poller = _make_poller(mark_read=True)
        poller.client.get_notifications.return_value = _sample_notifications(1)
        asyncio.run(poller.poll_once_async())
        poller.client.mark_notifications_read.assert_called_once()
