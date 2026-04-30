"""Tests for the SDK 1.4.0 / 1.5.0 tools added in v0.6.0:

- Social graph: ColonyFollowUser, ColonyUnfollowUser
- Reactions: ColonyReactToPost, ColonyReactToComment
- Polls: ColonyGetPoll, ColonyVotePoll
- Membership: ColonyJoinColony, ColonyLeaveColony
- Webhooks: ColonyCreateWebhook, ColonyGetWebhooks, ColonyDeleteWebhook
- Webhook signature verification: ColonyVerifyWebhook (standalone)
- ``verify_webhook`` re-export from colony_sdk

These mirror the same shape as the existing tool tests — patch
``langchain_colony.toolkit.ColonyClient`` and exercise both ``invoke``
and ``ainvoke`` paths.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from unittest.mock import patch

from langchain_colony import (
    ColonyCreateWebhook,
    ColonyDeleteWebhook,
    ColonyFollowUser,
    ColonyGetPoll,
    ColonyGetWebhooks,
    ColonyJoinColony,
    ColonyLeaveColony,
    ColonyReactToComment,
    ColonyReactToPost,
    ColonyToolkit,
    ColonyUnfollowUser,
    ColonyVerifyWebhook,
    ColonyVotePoll,
    verify_webhook,
)


def _toolkit():
    """Build a toolkit with a mocked ColonyClient and return the (tools dict, mock client) pair."""
    with patch("langchain_colony.toolkit.ColonyClient") as MockClient:
        toolkit = ColonyToolkit(api_key="col_test")
        return {t.name: t for t in toolkit.get_tools()}, MockClient.return_value


# ── Social graph ───────────────────────────────────────────────────


class TestFollowUnfollow:
    def test_follow(self):
        tools, client = _toolkit()
        client.follow.return_value = {"id": "follow-1", "status": "ok"}
        result = tools["colony_follow_user"].invoke({"user_id": "u-1"})
        client.follow.assert_called_once_with("u-1")
        assert "OK" in result or "Followed" in result

    def test_unfollow(self):
        tools, client = _toolkit()
        client.unfollow.return_value = {"status": "ok"}
        result = tools["colony_unfollow_user"].invoke({"user_id": "u-1"})
        client.unfollow.assert_called_once_with("u-1")
        assert "OK" in result or "Unfollowed" in result

    def test_follow_async(self):
        tools, client = _toolkit()
        client.follow.return_value = {"id": "f-1"}
        result = asyncio.run(tools["colony_follow_user"].ainvoke({"user_id": "u-1"}))
        assert "OK" in result or "Followed" in result

    def test_unfollow_uses_distinct_method(self):
        """Regression: crewai-colony 1.4.0 had a bug where unfollow() called
        the wrong HTTP method. Make sure we call ``client.unfollow``, not
        ``client.follow``."""
        tools, client = _toolkit()
        tools["colony_unfollow_user"].invoke({"user_id": "u-1"})
        client.follow.assert_not_called()
        client.unfollow.assert_called_once()


# ── Reactions ──────────────────────────────────────────────────────


class TestReactions:
    def test_react_to_post(self):
        tools, client = _toolkit()
        client.react_post.return_value = {"reaction": "thumbs_up"}
        result = tools["colony_react_to_post"].invoke({"post_id": "p-1", "emoji": "thumbs_up"})
        client.react_post.assert_called_once_with("p-1", "thumbs_up")
        assert "OK" in result or "thumbs_up" in result

    def test_react_to_comment(self):
        tools, client = _toolkit()
        client.react_comment.return_value = {"reaction": "fire"}
        result = tools["colony_react_to_comment"].invoke({"comment_id": "c-1", "emoji": "fire"})
        client.react_comment.assert_called_once_with("c-1", "fire")
        assert "OK" in result or "fire" in result

    def test_react_async(self):
        tools, client = _toolkit()
        client.react_post.return_value = {"ok": True}
        asyncio.run(tools["colony_react_to_post"].ainvoke({"post_id": "p-1", "emoji": "heart"}))
        client.react_post.assert_called_once_with("p-1", "heart")


# ── Polls ──────────────────────────────────────────────────────────


class TestPolls:
    def test_get_poll(self):
        tools, client = _toolkit()
        client.get_poll.return_value = {
            "options": [
                {"id": "opt-a", "text": "Option A", "votes": 10},
                {"id": "opt-b", "text": "Option B", "votes": 5},
            ],
            "total_votes": 15,
        }
        result = tools["colony_get_poll"].invoke({"post_id": "p-1"})
        client.get_poll.assert_called_once_with("p-1")
        assert "Option A" in result
        assert "10 votes" in result
        assert "opt-a" in result
        assert "15 total votes" in result

    def test_get_poll_options_label_fallback(self):
        """Some poll responses use ``label`` instead of ``text``."""
        tools, client = _toolkit()
        client.get_poll.return_value = {
            "options": [{"id": "x", "label": "Labelled", "votes": 1}],
        }
        result = tools["colony_get_poll"].invoke({"post_id": "p-1"})
        assert "Labelled" in result

    def test_vote_poll(self):
        tools, client = _toolkit()
        client.vote_poll.return_value = {"status": "ok"}
        result = tools["colony_vote_poll"].invoke({"post_id": "p-1", "option_id": "opt-a"})
        client.vote_poll.assert_called_once_with("p-1", "opt-a")
        assert "OK" in result or "opt-a" in result

    def test_vote_poll_async(self):
        tools, client = _toolkit()
        client.vote_poll.return_value = {}
        asyncio.run(tools["colony_vote_poll"].ainvoke({"post_id": "p-1", "option_id": "x"}))
        client.vote_poll.assert_called_once_with("p-1", "x")


# ── Colony membership ──────────────────────────────────────────────


class TestColonyMembership:
    def test_join(self):
        tools, client = _toolkit()
        client.join_colony.return_value = {"status": "joined"}
        result = tools["colony_join_colony"].invoke({"colony": "findings"})
        client.join_colony.assert_called_once_with("findings")
        assert "OK" in result or "Joined" in result

    def test_leave(self):
        tools, client = _toolkit()
        client.leave_colony.return_value = {"status": "left"}
        result = tools["colony_leave_colony"].invoke({"colony": "art"})
        client.leave_colony.assert_called_once_with("art")
        assert "OK" in result or "Left" in result

    def test_join_async(self):
        tools, client = _toolkit()
        client.join_colony.return_value = {}
        asyncio.run(tools["colony_join_colony"].ainvoke({"colony": "crypto"}))
        client.join_colony.assert_called_once_with("crypto")


# ── Webhooks ───────────────────────────────────────────────────────


class TestWebhookTools:
    def test_create_webhook(self):
        tools, client = _toolkit()
        client.create_webhook.return_value = {"id": "wh-1", "url": "https://example.com"}
        result = tools["colony_create_webhook"].invoke(
            {
                "url": "https://example.com/hook",
                "events": ["post_created", "comment_created"],
                "secret": "very-secret-string-123",
            }
        )
        client.create_webhook.assert_called_once_with(
            "https://example.com/hook",
            ["post_created", "comment_created"],
            "very-secret-string-123",
        )
        assert "wh-1" in result

    def test_get_webhooks_empty(self):
        tools, client = _toolkit()
        client.get_webhooks.return_value = {"webhooks": []}
        result = tools["colony_get_webhooks"].invoke({})
        assert "No webhooks" in result

    def test_get_webhooks_listed(self):
        tools, client = _toolkit()
        client.get_webhooks.return_value = {
            "webhooks": [
                {
                    "id": "wh-1",
                    "url": "https://example.com/hook",
                    "events": ["post_created"],
                }
            ]
        }
        result = tools["colony_get_webhooks"].invoke({})
        assert "wh-1" in result
        assert "https://example.com/hook" in result
        assert "post_created" in result

    def test_get_webhooks_list_response(self):
        """Some endpoints return a bare list instead of {"webhooks": [...]}."""
        tools, client = _toolkit()
        client.get_webhooks.return_value = [{"id": "wh-2", "url": "https://x", "events": ["mention"]}]
        result = tools["colony_get_webhooks"].invoke({})
        assert "wh-2" in result

    def test_delete_webhook(self):
        tools, client = _toolkit()
        client.delete_webhook.return_value = {"status": "deleted"}
        result = tools["colony_delete_webhook"].invoke({"webhook_id": "wh-1"})
        client.delete_webhook.assert_called_once_with("wh-1")
        assert "OK" in result or "Deleted" in result

    def test_create_webhook_async(self):
        tools, client = _toolkit()
        client.create_webhook.return_value = {"id": "wh-async"}
        result = asyncio.run(
            tools["colony_create_webhook"].ainvoke(
                {
                    "url": "https://example.com/hook",
                    "events": ["mention"],
                    "secret": "another-secret-key-456",
                }
            )
        )
        assert "wh-async" in result


# ── verify_webhook re-export + ColonyVerifyWebhook tool ────────────


SECRET = "shh-this-is-a-shared-secret"
PAYLOAD = b'{"event":"post_created","post":{"id":"p1","title":"Hello"}}'


def _sign(payload: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


class TestVerifyWebhookReExport:
    def test_is_sdk_function(self):
        """``langchain_colony.verify_webhook`` *is* the SDK function — no
        wrapper. We re-export rather than re-implement so callers
        automatically pick up SDK security fixes."""
        from colony_sdk import verify_webhook as sdk_fn

        assert verify_webhook is sdk_fn

    def test_valid_signature(self):
        sig = _sign(PAYLOAD, SECRET)
        assert verify_webhook(PAYLOAD, sig, SECRET) is True

    def test_invalid_signature(self):
        assert verify_webhook(PAYLOAD, "deadbeef" * 8, SECRET) is False

    def test_signature_with_sha256_prefix(self):
        sig = _sign(PAYLOAD, SECRET)
        assert verify_webhook(PAYLOAD, f"sha256={sig}", SECRET) is True

    def test_str_payload(self):
        body = '{"event":"post_created"}'
        sig = _sign(body.encode(), SECRET)
        assert verify_webhook(body, sig, SECRET) is True


class TestColonyVerifyWebhookTool:
    def test_not_in_default_toolkit(self):
        """Verification doesn't need an authenticated client, so it's a
        standalone tool — instantiate directly when you need it. Same
        pattern as ``ColonyRegister`` in crewai-colony."""
        with patch("langchain_colony.toolkit.ColonyClient"):
            toolkit = ColonyToolkit(api_key="col_test")
            names = {t.name for t in toolkit.get_tools()}
            assert "colony_verify_webhook" not in names

    def test_run_valid(self):
        sig = _sign(PAYLOAD, SECRET)
        tool = ColonyVerifyWebhook()
        result = tool.invoke({"payload": PAYLOAD.decode(), "signature": sig, "secret": SECRET})
        assert "valid" in result.lower()
        assert result.startswith("OK")

    def test_run_invalid(self):
        tool = ColonyVerifyWebhook()
        result = tool.invoke({"payload": PAYLOAD.decode(), "signature": "deadbeef" * 8, "secret": SECRET})
        assert "invalid" in result.lower()
        assert result.startswith("Error")

    def test_run_with_sha256_prefix(self):
        sig = _sign(PAYLOAD, SECRET)
        tool = ColonyVerifyWebhook()
        result = tool.invoke({"payload": PAYLOAD.decode(), "signature": f"sha256={sig}", "secret": SECRET})
        assert result.startswith("OK")

    def test_run_handles_unexpected_error(self):
        """If the underlying ``verify_webhook`` raises (e.g. exotic input),
        the tool catches it and formats the message rather than crashing
        the agent run."""
        tool = ColonyVerifyWebhook()
        with patch("langchain_colony.tools.verify_webhook", side_effect=ValueError("bad payload")):
            result = tool.invoke({"payload": "x", "signature": "y", "secret": "z"})
        assert "Error" in result
        assert "bad payload" in result

    def test_arun_valid(self):
        sig = _sign(PAYLOAD, SECRET)
        tool = ColonyVerifyWebhook()
        result = asyncio.run(tool.ainvoke({"payload": PAYLOAD.decode(), "signature": sig, "secret": SECRET}))
        assert result.startswith("OK")

    def test_arun_invalid(self):
        tool = ColonyVerifyWebhook()
        result = asyncio.run(tool.ainvoke({"payload": PAYLOAD.decode(), "signature": "0" * 64, "secret": SECRET}))
        assert result.startswith("Error")


# ── Direct constructibility (without toolkit) ──────────────────────


class TestDirectConstruction:
    """The new tools should also be importable from the package and
    constructible directly with a custom client (e.g. for stateless usage
    in a webhook handler)."""

    def test_import_all_new_tools(self):
        # Just verifying the package surface compiles. ColonyVerifyWebhook
        # has a default ``client=None`` since it doesn't need one.
        assert ColonyFollowUser is not None
        assert ColonyUnfollowUser is not None
        assert ColonyReactToPost is not None
        assert ColonyReactToComment is not None
        assert ColonyGetPoll is not None
        assert ColonyVotePoll is not None
        assert ColonyJoinColony is not None
        assert ColonyLeaveColony is not None
        assert ColonyCreateWebhook is not None
        assert ColonyGetWebhooks is not None
        assert ColonyDeleteWebhook is not None
        assert ColonyVerifyWebhook is not None
