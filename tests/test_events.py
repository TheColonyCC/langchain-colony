"""Tests for the Colony event poller."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

from langchain_colony.events import ColonyEventPoller
from langchain_colony.models import ColonyNotification


def _make_poller(**kwargs):
    with patch("langchain_colony.events.ColonyClient"):
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


# ──────────────────────────────────────────────────────────────────────
# Notification enrichment (sender_*, body)
# ──────────────────────────────────────────────────────────────────────

_BASE_TS = "2026-04-26T16:38:48.432087Z"
# Conversation timestamps drift a few ms behind the matching notification.
_CONV_TS_MATCH = "2026-04-26T16:38:48.427682Z"


def _dm_notification(notif_id: str = "n1", created_at: str = _BASE_TS) -> dict:
    return {
        "id": notif_id,
        "notification_type": "direct_message",
        "message": "ColonistOne sent you a message",
        "post_id": None,
        "comment_id": None,
        "is_read": False,
        "created_at": created_at,
    }


def _conversation(
    username: str = "colonist-one",
    display_name: str = "ColonistOne",
    user_id: str = "u-cone",
    last_message_at: str = _CONV_TS_MATCH,
    preview: str = "Hi there",
    unread: int = 1,
) -> dict:
    return {
        "id": "conv-1",
        "other_user": {
            "id": user_id,
            "username": username,
            "display_name": display_name,
        },
        "last_message_at": last_message_at,
        "unread_count": unread,
        "last_message_preview": preview,
        "is_archived": False,
    }


def _mention_notification(
    notif_id: str = "n1",
    post_id: str = "post-1",
    comment_id: str | None = "c-1",
    type_: str = "mention",
) -> dict:
    return {
        "id": notif_id,
        "notification_type": type_,
        "message": "ColonistOne mentioned you",
        "post_id": post_id,
        "comment_id": comment_id,
        "is_read": False,
        "created_at": _BASE_TS,
    }


def _post(post_id: str = "post-1", author_username: str = "post-author") -> dict:
    return {
        "id": post_id,
        "title": "A Post",
        "body": "Post body",
        "author": {
            "id": "u-post-author",
            "username": author_username,
            "display_name": "Post Author",
        },
    }


def _comment_list(comment_id: str = "c-1", author_username: str = "comment-author") -> dict:
    return {
        "items": [
            {
                "id": comment_id,
                "body": "Comment body @langford",
                "author": {
                    "id": "u-comment-author",
                    "username": author_username,
                    "display_name": "Comment Author",
                },
            }
        ]
    }


class TestEnrichDirectMessage:
    def test_dm_populates_sender_and_body(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = [_dm_notification()]
        poller.client.list_conversations.return_value = {"items": [_conversation()]}
        results = poller.poll_once()
        assert len(results) == 1
        n = results[0]
        assert n.sender_username == "colonist-one"
        assert n.sender_display_name == "ColonistOne"
        assert n.sender_id == "u-cone"
        assert n.body == "Hi there"

    def test_dm_no_match_when_timestamps_far_apart(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = [_dm_notification()]
        poller.client.list_conversations.return_value = {
            # Way outside the 5-minute tolerance.
            "items": [_conversation(last_message_at="2025-01-01T00:00:00Z")]
        }
        n = poller.poll_once()[0]
        assert n.sender_username is None
        assert n.body is None

    def test_dm_picks_closest_match_within_tolerance(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = [_dm_notification()]
        poller.client.list_conversations.return_value = {
            "items": [
                _conversation(
                    username="other-user",
                    last_message_at="2026-04-26T16:38:00.000000Z",
                    preview="far",
                ),
                _conversation(
                    username="colonist-one",
                    last_message_at=_CONV_TS_MATCH,
                    preview="close",
                ),
            ]
        }
        n = poller.poll_once()[0]
        assert n.sender_username == "colonist-one"
        assert n.body == "close"

    def test_dm_no_conversations(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = [_dm_notification()]
        poller.client.list_conversations.return_value = {"items": []}
        n = poller.poll_once()[0]
        assert n.sender_username is None

    def test_dm_lists_conversations_once_per_cycle(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = [
            _dm_notification(notif_id="n1"),
            _dm_notification(notif_id="n2"),
        ]
        poller.client.list_conversations.return_value = {"items": [_conversation()]}
        poller.poll_once()
        assert poller.client.list_conversations.call_count == 1

    def test_dm_skips_when_created_at_unparseable(self):
        poller = _make_poller()
        notif = _dm_notification(created_at="not-a-date")
        poller.client.get_notifications.return_value = [notif]
        poller.client.list_conversations.return_value = {"items": [_conversation()]}
        n = poller.poll_once()[0]
        assert n.sender_username is None

    def test_dm_handles_list_form_response(self):
        # list_conversations might also return a bare list.
        poller = _make_poller()
        poller.client.get_notifications.return_value = [_dm_notification()]
        poller.client.list_conversations.return_value = [_conversation()]
        n = poller.poll_once()[0]
        assert n.sender_username == "colonist-one"


class TestEnrichComment:
    def test_mention_with_comment_id_uses_comment_author(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = [_mention_notification()]
        poller.client.get_post.return_value = _post()
        poller.client.get_comments.return_value = _comment_list()
        n = poller.poll_once()[0]
        assert n.sender_username == "comment-author"
        assert n.body == "Comment body @langford"

    def test_mention_without_comment_id_falls_back_to_post_author(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = [
            _mention_notification(comment_id=None)
        ]
        poller.client.get_post.return_value = _post(author_username="po")
        n = poller.poll_once()[0]
        assert n.sender_username == "po"
        assert n.body in {"Post body", "A Post"}
        # get_comments should not be called when there's no comment_id.
        poller.client.get_comments.assert_not_called()

    def test_mention_unmatched_comment_id_falls_back_to_post_author(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = [
            _mention_notification(comment_id="nope")
        ]
        poller.client.get_post.return_value = _post(author_username="po")
        poller.client.get_comments.return_value = _comment_list(comment_id="other")
        n = poller.poll_once()[0]
        assert n.sender_username == "po"

    def test_reply_type_is_enriched(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = [
            _mention_notification(type_="reply")
        ]
        poller.client.get_post.return_value = _post()
        poller.client.get_comments.return_value = _comment_list()
        n = poller.poll_once()[0]
        assert n.sender_username == "comment-author"

    def test_get_post_cached_per_cycle(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = [
            _mention_notification(notif_id="n1", comment_id=None),
            _mention_notification(notif_id="n2", comment_id=None),
        ]
        poller.client.get_post.return_value = _post()
        poller.poll_once()
        assert poller.client.get_post.call_count == 1

    def test_no_post_id_short_circuits(self):
        poller = _make_poller()
        notif = _mention_notification()
        notif["post_id"] = None
        poller.client.get_notifications.return_value = [notif]
        n = poller.poll_once()[0]
        assert n.sender_username is None
        poller.client.get_post.assert_not_called()


class TestEnrichToggle:
    def test_enrich_disabled_skips_extra_calls(self):
        poller = _make_poller(enrich=False)
        poller.client.get_notifications.return_value = [
            _dm_notification(),
            _mention_notification(),
        ]
        results = poller.poll_once()
        assert all(n.sender_username is None for n in results)
        poller.client.list_conversations.assert_not_called()
        poller.client.get_post.assert_not_called()

    def test_enrichment_failure_does_not_break_dispatch(self):
        poller = _make_poller()
        received: list = []

        @poller.on()
        def handle(notif):
            received.append(notif)

        poller.client.get_notifications.return_value = [_dm_notification()]
        poller.client.list_conversations.side_effect = RuntimeError("boom")

        results = poller.poll_once()
        # Dispatch still happened despite enrichment failure.
        assert len(received) == 1
        assert results[0].sender_username is None

    def test_unknown_type_is_not_enriched(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = [
            {
                "id": "v1",
                "notification_type": "vote",
                "message": "Someone upvoted you",
                "post_id": "p1",
                "is_read": False,
                "created_at": _BASE_TS,
            }
        ]
        n = poller.poll_once()[0]
        assert n.sender_username is None
        poller.client.list_conversations.assert_not_called()
        poller.client.get_post.assert_not_called()


class TestEnrichAsync:
    def test_async_dm_enrichment(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = [_dm_notification()]
        poller.client.list_conversations.return_value = {"items": [_conversation()]}
        results = asyncio.run(poller.poll_once_async())
        assert results[0].sender_username == "colonist-one"

    def test_async_mention_enrichment(self):
        poller = _make_poller()
        poller.client.get_notifications.return_value = [_mention_notification()]
        poller.client.get_post.return_value = _post()
        poller.client.get_comments.return_value = _comment_list()
        results = asyncio.run(poller.poll_once_async())
        assert results[0].sender_username == "comment-author"

    def test_async_enrich_disabled(self):
        poller = _make_poller(enrich=False)
        poller.client.get_notifications.return_value = [_dm_notification()]
        results = asyncio.run(poller.poll_once_async())
        assert results[0].sender_username is None
        poller.client.list_conversations.assert_not_called()
