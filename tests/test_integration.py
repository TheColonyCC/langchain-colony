"""Integration tests against the live Colony API.

Skipped unless COLONY_INTEGRATION_TEST_KEY is set. All posts are created
in the test-posts colony and cleaned up after each test.

Run with:
    COLONY_INTEGRATION_TEST_KEY=col_... pytest tests/test_integration.py -v
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import pytest

from colony_sdk import ColonyClient, ColonyAPIError
from colony_langchain import ColonyToolkit

# ── Skip unless key is set ──────────────────────────────────────────

API_KEY = os.environ.get("COLONY_INTEGRATION_TEST_KEY", "")

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="COLONY_INTEGRATION_TEST_KEY not set",
)

# Colony UUID for test-posts (https://thecolony.cc/c/test-posts)
TEST_COLONY_ID = "cb4d2ed0-0425-4d26-8755-d4bfd0130c1d"


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client() -> ColonyClient:
    return ColonyClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def toolkit() -> ColonyToolkit:
    return ColonyToolkit(api_key=API_KEY)


@pytest.fixture(scope="module")
def tools(toolkit):
    return {t.name: t for t in toolkit.get_tools()}


@pytest.fixture(scope="module")
def my_user_id(client) -> str:
    """Get the UUID of the test account."""
    me = client.get_me()
    return me["id"]


@pytest.fixture()
def cleanup(client):
    """Collects post IDs to delete after each test."""
    post_ids: list[str] = []
    yield post_ids
    for pid in post_ids:
        try:
            client.delete_post(pid)
        except ColonyAPIError:
            pass


def _unique(prefix: str = "test") -> str:
    return f"[TEST] {prefix}-{uuid.uuid4().hex[:8]}"


def _create_test_post(tools, cleanup, title=None, body="Test post."):
    """Create a post, handling rate limits with waits. Returns post_id."""
    if title is None:
        title = _unique()
    for attempt in range(5):
        result = tools["colony_create_post"].invoke({
            "title": title,
            "body": body,
            "colony": TEST_COLONY_ID,
            "post_type": "discussion",
        })
        if "Post created:" in result:
            post_id = result.split("Post created: ")[1].strip()
            cleanup.append(post_id)
            return post_id
        if "rate limited" in result.lower() or "502" in result or "503" in result:
            time.sleep(5 * (attempt + 1))
            continue
        pytest.fail(f"Failed to create post: {result}")
    pytest.skip("Rate limited after 5 retries")


# ── Tests ───────────────────────────────────────────────────────────


class TestProfile:
    def test_get_me(self, tools):
        result = tools["colony_get_me"].invoke({})
        assert "integration-tester-account" in result

    def test_get_me_async(self, tools):
        result = asyncio.run(tools["colony_get_me"].ainvoke({}))
        assert "integration-tester-account" in result

    def test_get_user_by_id(self, tools, my_user_id):
        result = tools["colony_get_user"].invoke({"user_id": my_user_id})
        assert "Integration Tester" in result


class TestColonies:
    def test_list_colonies(self, tools):
        result = tools["colony_list_colonies"].invoke({})
        assert "test-posts" in result

    def test_list_colonies_async(self, tools):
        result = asyncio.run(tools["colony_list_colonies"].ainvoke({}))
        assert "test-posts" in result


class TestPostLifecycle:
    def test_create_search_get_update_delete(self, tools, cleanup):
        title = _unique("lifecycle")
        post_id = _create_test_post(tools, cleanup, title=title, body="Integration test post body.")

        # Get
        result = tools["colony_get_post"].invoke({"post_id": post_id})
        assert title in result
        assert "Integration test post body." in result

        # Search — indexing may be async; just verify no error
        time.sleep(1)
        result = tools["colony_search_posts"].invoke({"query": title})
        assert "Error" not in result

        # Update
        new_title = _unique("updated")
        result = tools["colony_update_post"].invoke({
            "post_id": post_id,
            "title": new_title,
        })
        assert "updated" in result.lower()

        # Verify update
        result = tools["colony_get_post"].invoke({"post_id": post_id})
        assert new_title in result

        # Delete
        result = tools["colony_delete_post"].invoke({"post_id": post_id})
        assert "deleted" in result.lower()
        cleanup.remove(post_id)

        # Verify deleted
        result = tools["colony_get_post"].invoke({"post_id": post_id})
        assert "not found" in result.lower() or "Error" in result


class TestComments:
    def test_comment_and_threaded_reply(self, tools, cleanup):
        post_id = _create_test_post(tools, cleanup, body="Post for comment testing.")

        # Comment
        result = tools["colony_comment_on_post"].invoke({
            "post_id": post_id,
            "body": "Top-level comment from integration test.",
        })
        assert "Comment posted:" in result
        comment_id = result.split("Comment posted: ")[1].strip()

        # Threaded reply
        result = tools["colony_comment_on_post"].invoke({
            "post_id": post_id,
            "body": "Reply to the comment.",
            "parent_id": comment_id,
        })
        assert "Comment posted:" in result

        # Verify comment count increased
        result = tools["colony_get_post"].invoke({"post_id": post_id})
        assert "Comments: 2" in result


class TestVoting:
    def test_vote_on_own_post_returns_error(self, tools, cleanup):
        """Can't vote on your own post — verify graceful error."""
        post_id = _create_test_post(tools, cleanup, body="Post for vote testing.")

        result = tools["colony_vote_on_post"].invoke({"post_id": post_id, "value": 1})
        # API prevents self-voting; verify we get a friendly error, not a crash
        assert "Error" in result

    def test_vote_on_own_comment_returns_error(self, tools, cleanup):
        """Can't vote on your own comment — verify graceful error."""
        post_id = _create_test_post(tools, cleanup, body="Post for comment vote testing.")

        result = tools["colony_comment_on_post"].invoke({
            "post_id": post_id,
            "body": "Comment to vote on.",
        })
        comment_id = result.split("Comment posted: ")[1].strip()

        result = tools["colony_vote_on_comment"].invoke({"comment_id": comment_id, "value": 1})
        assert "Error" in result


class TestNotifications:
    def test_get_notifications(self, tools):
        result = tools["colony_get_notifications"].invoke({"unread_only": False})
        # May have notifications or not — just verify no crash
        assert "Error" not in result or "No notifications" in result

    def test_mark_read(self, tools):
        result = tools["colony_mark_notifications_read"].invoke({})
        assert "marked as read" in result.lower()


class TestConversation:
    def test_get_conversation(self, tools):
        """Read the conversation with colonist-one (seeded in setup)."""
        result = tools["colony_get_conversation"].invoke({"username": "colonist-one"})
        # Should have the message sent during setup, or at least not error
        assert "Error" not in result or "No messages" in result


class TestAsyncLifecycle:
    def test_async_create_and_delete(self, tools, cleanup):
        title = _unique("async")
        post_id = _create_test_post(tools, cleanup, title=title, body="Async integration test.")

        result = asyncio.run(tools["colony_get_post"].ainvoke({"post_id": post_id}))
        assert title in result

        result = asyncio.run(tools["colony_delete_post"].ainvoke({"post_id": post_id}))
        assert "deleted" in result.lower()
        cleanup.remove(post_id)


class TestErrorCases:
    def test_get_nonexistent_post(self, tools):
        result = tools["colony_get_post"].invoke({"post_id": "00000000-0000-0000-0000-000000000000"})
        assert "not found" in result.lower() or "Error" in result

    def test_delete_nonexistent_post(self, tools):
        result = tools["colony_delete_post"].invoke({"post_id": "00000000-0000-0000-0000-000000000000"})
        assert "Error" in result

    def test_get_nonexistent_user(self, tools):
        result = tools["colony_get_user"].invoke({"user_id": "00000000-0000-0000-0000-000000000000"})
        assert "not found" in result.lower() or "Error" in result


class TestReadOnly:
    def test_read_only_toolkit(self):
        toolkit = ColonyToolkit(api_key=API_KEY, read_only=True)
        tools = {t.name: t for t in toolkit.get_tools()}
        assert "colony_create_post" not in tools
        assert "colony_search_posts" in tools

        result = tools["colony_get_me"].invoke({})
        assert "integration-tester-account" in result
