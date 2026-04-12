"""Tests for the Colony LangChain toolkit.

These tests inject a :class:`colony_sdk.testing.MockColonyClient` via
``ColonyToolkit(client=...)`` instead of patching ``ColonyClient`` at
import time. The mock records every method call (including kwargs) on
``mock.calls`` so assertions stay simple.
"""

from __future__ import annotations

import asyncio
from typing import Any

from colony_sdk.testing import MockColonyClient

from langchain_colony import ColonyToolkit
from langchain_colony.tools import (
    _format_colonies,
    _format_conversation,
    _format_notifications,
    _format_post,
    _format_posts,
    _format_user,
)


def _make_toolkit(**kwargs: Any) -> ColonyToolkit:
    """Build a toolkit backed by a default MockColonyClient.

    Use this when the test only cares about the toolkit's tool registry
    (names, schemas, tags, etc.) and doesn't actually invoke any tool.
    """
    return ColonyToolkit(client=MockColonyClient(), **kwargs)


def _toolkit_with(responses: dict[str, Any], **kwargs: Any) -> tuple[ColonyToolkit, MockColonyClient]:
    """Build a toolkit + a configured MockColonyClient and return both.

    Pass a ``responses`` dict mapping method names to canned responses
    (mirrors :class:`MockColonyClient`'s ``responses=`` argument).
    Returns ``(toolkit, mock_client)`` so the test can both invoke tools
    and inspect ``mock_client.calls`` for call assertions.
    """
    mock = MockColonyClient(responses=responses)
    return ColonyToolkit(client=mock, **kwargs), mock


def _tools_by_name() -> tuple[dict[str, Any], MockColonyClient]:
    mock = MockColonyClient()
    toolkit = ColonyToolkit(client=mock)
    return {t.name: t for t in toolkit.get_tools()}, mock


# ── Toolkit ─────────────────────────────────────────────────────────


class TestToolkit:
    def test_get_tools_returns_all(self):
        """Toolkit ships 27 tools across the SDK 1.5.0 surface — 9 read +
        18 write. ColonyVerifyWebhook is intentionally NOT in the registry
        (instantiate directly when you need it, like ColonyRegister)."""
        toolkit = _make_toolkit()
        tools = toolkit.get_tools()
        assert len(tools) == 27
        names = {t.name for t in tools}
        assert names == {
            # Read (9)
            "colony_search_posts",
            "colony_get_post",
            "colony_get_notifications",
            "colony_get_me",
            "colony_get_user",
            "colony_list_colonies",
            "colony_get_conversation",
            "colony_get_poll",
            "colony_get_webhooks",
            # Write (18)
            "colony_create_post",
            "colony_comment_on_post",
            "colony_vote_on_post",
            "colony_send_message",
            "colony_update_post",
            "colony_delete_post",
            "colony_vote_on_comment",
            "colony_mark_notifications_read",
            "colony_update_profile",
            "colony_follow_user",
            "colony_unfollow_user",
            "colony_react_to_post",
            "colony_react_to_comment",
            "colony_vote_poll",
            "colony_join_colony",
            "colony_leave_colony",
            "colony_create_webhook",
            "colony_delete_webhook",
        }

    def test_verify_webhook_not_in_toolkit(self):
        """``ColonyVerifyWebhook`` is a standalone tool — not in ALL_TOOLS,
        same pattern as ``ColonyRegister`` in crewai-colony. Webhook
        verification is done in handler code, not by an LLM agent loop."""
        toolkit = _make_toolkit()
        names = {t.name for t in toolkit.get_tools()}
        assert "colony_verify_webhook" not in names

    def test_read_only_returns_nine(self):
        toolkit = _make_toolkit(read_only=True)
        tools = toolkit.get_tools()
        assert len(tools) == 9
        names = {t.name for t in tools}
        assert names == {
            "colony_search_posts",
            "colony_get_post",
            "colony_get_notifications",
            "colony_get_me",
            "colony_get_user",
            "colony_list_colonies",
            "colony_get_conversation",
            "colony_get_poll",
            "colony_get_webhooks",
        }

    def test_include_filter(self):
        toolkit = _make_toolkit()
        tools = toolkit.get_tools(include=["colony_search_posts", "colony_get_post"])
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"colony_search_posts", "colony_get_post"}

    def test_exclude_filter(self):
        toolkit = _make_toolkit()
        tools = toolkit.get_tools(exclude=["colony_delete_post", "colony_update_profile"])
        names = {t.name for t in tools}
        assert "colony_delete_post" not in names
        assert "colony_update_profile" not in names
        assert len(tools) == 25

    def test_include_and_exclude_raises(self):
        toolkit = _make_toolkit()
        raised = False
        try:
            toolkit.get_tools(include=["colony_get_post"], exclude=["colony_delete_post"])
        except ValueError as exc:
            raised = True
            assert "Cannot specify both" in str(exc)
        assert raised, "Should have raised ValueError"

    def test_include_with_read_only(self):
        toolkit = _make_toolkit(read_only=True)
        tools = toolkit.get_tools(include=["colony_search_posts", "colony_create_post"])
        # colony_create_post is a write tool, not available in read_only
        names = {t.name for t in tools}
        assert names == {"colony_search_posts"}

    def test_exclude_with_read_only(self):
        toolkit = _make_toolkit(read_only=True)
        tools = toolkit.get_tools(exclude=["colony_get_me"])
        assert len(tools) == 8
        assert "colony_get_me" not in {t.name for t in tools}

    def test_include_empty_list(self):
        toolkit = _make_toolkit()
        tools = toolkit.get_tools(include=[])
        assert tools == []

    def test_exclude_empty_list(self):
        toolkit = _make_toolkit()
        tools = toolkit.get_tools(exclude=[])
        assert len(tools) == 27

    def test_include_nonexistent_name(self):
        toolkit = _make_toolkit()
        tools = toolkit.get_tools(include=["colony_does_not_exist"])
        assert tools == []

    def test_tools_have_descriptions(self):
        toolkit = _make_toolkit()
        for tool in toolkit.get_tools():
            assert tool.description, f"{tool.name} has no description"

    def test_tools_have_metadata(self):
        toolkit = _make_toolkit()
        for tool in toolkit.get_tools():
            assert tool.metadata is not None, f"{tool.name} has no metadata"
            assert tool.metadata["provider"] == "thecolony.cc"
            assert "category" in tool.metadata
            assert "operation" in tool.metadata

    def test_tools_have_tags(self):
        toolkit = _make_toolkit()
        for tool in toolkit.get_tools():
            assert tool.tags is not None, f"{tool.name} has no tags"
            assert "colony" in tool.tags
            assert "read" in tool.tags or "write" in tool.tags

    def test_write_tools_tagged_write(self):
        toolkit = _make_toolkit()
        write_names = {
            "colony_create_post",
            "colony_comment_on_post",
            "colony_vote_on_post",
            "colony_vote_on_comment",
            "colony_send_message",
            "colony_update_post",
            "colony_delete_post",
            "colony_mark_notifications_read",
            "colony_update_profile",
            "colony_follow_user",
            "colony_unfollow_user",
            "colony_react_to_post",
            "colony_react_to_comment",
            "colony_vote_poll",
            "colony_join_colony",
            "colony_leave_colony",
            "colony_create_webhook",
            "colony_delete_webhook",
        }
        for tool in toolkit.get_tools():
            if tool.name in write_names:
                assert "write" in tool.tags, f"{tool.name} should be tagged 'write'"
            else:
                assert "read" in tool.tags, f"{tool.name} should be tagged 'read'"

    def test_tools_have_args_schema(self):
        # Tools that take no arguments have args_schema=None
        no_args_tools = {
            "colony_get_me",
            "colony_mark_notifications_read",
            "colony_get_webhooks",
        }
        toolkit = _make_toolkit()
        for tool in toolkit.get_tools():
            if tool.name in no_args_tools:
                assert tool.args_schema is None, f"{tool.name} should have no args_schema"
            else:
                assert tool.args_schema is not None, f"{tool.name} has no args_schema"


# ── Construction (api_key vs client= injection) ─────────────────────


class TestColonyToolkitConstruction:
    def test_accepts_injected_client(self):
        """Passing a pre-built client uses it directly without constructing one."""
        mock = MockColonyClient()
        toolkit = ColonyToolkit(client=mock)
        assert toolkit.client is mock

    def test_injected_client_overrides_api_key(self):
        """When client= is set, api_key/base_url/retry are ignored."""
        mock = MockColonyClient()
        toolkit = ColonyToolkit(api_key="col_ignored", base_url="https://ignored", client=mock)
        assert toolkit.client is mock

    def test_no_api_key_or_client_raises(self):
        """Either api_key or client must be provided."""
        import pytest

        with pytest.raises(ValueError, match="api_key or client"):
            ColonyToolkit()

    def test_api_key_path_constructs_client(self):
        """The legacy api_key= path still wraps a real ColonyClient."""
        from colony_sdk import ColonyClient

        toolkit = ColonyToolkit(api_key="col_test")
        assert isinstance(toolkit.client, ColonyClient)


class TestAsyncColonyToolkitConstruction:
    def test_accepts_injected_client(self):
        """AsyncColonyToolkit also accepts client= for injection."""
        from langchain_colony import AsyncColonyToolkit

        mock = MockColonyClient()
        toolkit = AsyncColonyToolkit(client=mock)
        assert toolkit.client is mock

    def test_injected_client_overrides_api_key(self):
        from langchain_colony import AsyncColonyToolkit

        mock = MockColonyClient()
        toolkit = AsyncColonyToolkit(api_key="col_ignored", client=mock)
        assert toolkit.client is mock

    def test_no_api_key_or_client_raises(self):
        import pytest

        from langchain_colony import AsyncColonyToolkit

        with pytest.raises(ValueError, match="api_key or client"):
            AsyncColonyToolkit()


# ── Formatters ──────────────────────────────────────────────────────


class TestFormatPosts:
    def test_empty(self):
        assert _format_posts({"posts": []}) == "No posts found."
        assert _format_posts({}) == "No posts found."

    def test_single_post(self):
        result = _format_posts(
            {
                "posts": [
                    {
                        "id": "abc",
                        "title": "Hello",
                        "post_type": "discussion",
                        "score": 5,
                        "comment_count": 2,
                        "author": {"username": "agent-x"},
                        "colony": {"name": "general"},
                    }
                ]
            }
        )
        assert "Hello" in result
        assert "agent-x" in result
        assert "general" in result
        assert "abc" in result
        assert "score: 5" in result

    def test_multiple_posts(self):
        result = _format_posts(
            {
                "posts": [
                    {
                        "id": "1",
                        "title": "First",
                        "post_type": "finding",
                        "score": 1,
                        "comment_count": 0,
                        "author": {"username": "a"},
                        "colony": {"name": "c1"},
                    },
                    {
                        "id": "2",
                        "title": "Second",
                        "post_type": "question",
                        "score": 3,
                        "comment_count": 1,
                        "author": {"username": "b"},
                        "colony": {"name": "c2"},
                    },
                ]
            }
        )
        assert "First" in result
        assert "Second" in result

    def test_missing_fields_fallback(self):
        result = _format_posts({"posts": [{"id": "x", "title": "T", "post_type": "discussion"}]})
        assert "?" in result  # missing author/colony fall back to ?
        assert "score: 0" in result  # missing score defaults to 0


class TestFormatPost:
    def test_basic_post(self):
        result = _format_post(
            {
                "title": "My Post",
                "post_type": "analysis",
                "score": 10,
                "comment_count": 3,
                "author": {"username": "researcher"},
                "colony": {"name": "findings"},
                "id": "post-1",
                "body": "This is the body.",
            }
        )
        assert "My Post" in result
        assert "analysis" in result
        assert "researcher" in result
        assert "This is the body." in result

    def test_nested_post_wrapper(self):
        result = _format_post(
            {
                "post": {
                    "title": "Wrapped",
                    "post_type": "discussion",
                    "score": 0,
                    "comment_count": 0,
                    "author": {"username": "bot"},
                    "colony": {"name": "general"},
                    "id": "w-1",
                    "body": "wrapped body",
                }
            }
        )
        assert "Wrapped" in result
        assert "wrapped body" in result

    def test_with_comments(self):
        result = _format_post(
            {
                "title": "Post",
                "post_type": "discussion",
                "score": 0,
                "comment_count": 2,
                "author": {"username": "op"},
                "colony": {"name": "general"},
                "id": "p-1",
                "body": "",
                "comments": [
                    {"author": {"username": "commenter1"}, "body": "Great post!"},
                    {"author": {"username": "commenter2"}, "body": "I disagree."},
                ],
            }
        )
        assert "Top comments:" in result
        assert "commenter1" in result
        assert "Great post!" in result
        assert "commenter2" in result

    def test_comments_truncated_to_ten(self):
        comments = [{"author": {"username": f"user{i}"}, "body": f"Comment {i}"} for i in range(15)]
        result = _format_post(
            {
                "title": "P",
                "post_type": "d",
                "score": 0,
                "comment_count": 15,
                "author": {"username": "x"},
                "colony": {"name": "y"},
                "id": "z",
                "body": "",
                "comments": comments,
            }
        )
        assert "user9" in result
        assert "user10" not in result

    def test_missing_fields(self):
        result = _format_post({})
        assert "?" in result  # fallback for missing title, author, etc.


class TestFormatNotifications:
    def test_empty(self):
        assert _format_notifications({"notifications": []}) == "No notifications."
        assert _format_notifications({}) == "No notifications."

    def test_with_notifications(self):
        result = _format_notifications(
            {
                "notifications": [
                    {"type": "reply", "actor": {"username": "agent-a"}, "preview": "Thanks for sharing"},
                    {"type": "mention", "actor": {"username": "agent-b"}, "body": "Check out @you"},
                ]
            }
        )
        assert "[reply]" in result
        assert "agent-a" in result
        assert "Thanks for sharing" in result
        assert "[mention]" in result
        assert "Check out @you" in result

    def test_long_preview_truncated(self):
        result = _format_notifications(
            {
                "notifications": [
                    {"type": "dm", "actor": {"username": "x"}, "preview": "A" * 200},
                ]
            }
        )
        # preview is truncated to 100 chars
        assert len(result.split(": ", 1)[1]) == 100


# ── Tool invocations ────────────────────────────────────────────────


class TestSearchPosts:
    def test_formats_results(self):
        toolkit, _ = _toolkit_with(
            {
                "get_posts": {
                    "posts": [
                        {
                            "id": "abc-123",
                            "title": "Test Post",
                            "post_type": "discussion",
                            "score": 5,
                            "comment_count": 2,
                            "author": {"username": "test-agent"},
                            "colony": {"name": "general"},
                        }
                    ]
                }
            }
        )
        tool = toolkit.get_tools()[0]
        assert tool.name == "colony_search_posts"

        result = tool.invoke({"query": "test"})
        assert "Test Post" in result
        assert "test-agent" in result

    def test_no_results(self):
        toolkit, _ = _toolkit_with({"get_posts": {"posts": []}})
        tool = toolkit.get_tools()[0]
        result = tool.invoke({"query": "nonexistent"})
        assert "No posts found" in result

    def test_passes_all_params(self):
        toolkit, mock = _toolkit_with({"get_posts": {"posts": []}})
        tool = toolkit.get_tools()[0]
        tool.invoke({"query": "ai", "colony": "findings", "sort": "top", "limit": 5})
        # MockColonyClient captures the kwargs handed to get_posts.
        method, kwargs = mock.calls[-1]
        assert method == "get_posts"
        assert kwargs["colony"] == "findings"
        assert kwargs["sort"] == "top"
        assert kwargs["limit"] == 5

    def test_async_formats_results(self):
        toolkit, _ = _toolkit_with(
            {
                "get_posts": {
                    "posts": [
                        {
                            "id": "abc-123",
                            "title": "Async Post",
                            "post_type": "finding",
                            "score": 3,
                            "comment_count": 1,
                            "author": {"username": "async-agent"},
                            "colony": {"name": "findings"},
                        }
                    ]
                }
            }
        )
        tool = toolkit.get_tools()[0]
        result = asyncio.run(tool.ainvoke({"query": "async"}))
        assert "Async Post" in result
        assert "async-agent" in result


class TestGetPost:
    def test_returns_formatted_post(self):
        toolkit, _ = _toolkit_with(
            {
                "get_post": {
                    "post": {
                        "title": "Deep Dive",
                        "post_type": "analysis",
                        "score": 12,
                        "comment_count": 4,
                        "author": {"username": "analyst"},
                        "colony": {"name": "findings"},
                        "id": "post-99",
                        "body": "Detailed analysis here.",
                        "comments": [
                            {"author": {"username": "reader"}, "body": "Very insightful!"},
                        ],
                    }
                }
            }
        )
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_get_post"].invoke({"post_id": "post-99"})
        assert "Deep Dive" in result
        assert "Detailed analysis here." in result
        assert "reader" in result
        assert "Very insightful!" in result

    def test_async_returns_formatted_post(self):
        toolkit, _ = _toolkit_with(
            {
                "get_post": {
                    "title": "Simple",
                    "post_type": "discussion",
                    "score": 0,
                    "comment_count": 0,
                    "author": {"username": "bot"},
                    "colony": {"name": "general"},
                    "id": "p-1",
                    "body": "Hello world",
                }
            }
        )
        tools = {t.name: t for t in toolkit.get_tools()}
        result = asyncio.run(tools["colony_get_post"].ainvoke({"post_id": "p-1"}))
        assert "Simple" in result
        assert "Hello world" in result


class TestCreatePost:
    def test_returns_post_id(self):
        toolkit, _ = _toolkit_with({"create_post": {"id": "new-post-123"}})
        tools_by_name = {t.name: t for t in toolkit.get_tools()}
        tool = tools_by_name["colony_create_post"]
        result = tool.invoke({"title": "Hello", "body": "World"})
        assert "new-post-123" in result

    def test_nested_post_id(self):
        toolkit, _ = _toolkit_with({"create_post": {"post": {"id": "nested-789"}}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_create_post"].invoke({"title": "T", "body": "B"})
        assert "nested-789" in result

    def test_unknown_id_fallback(self):
        toolkit, _ = _toolkit_with({"create_post": {"status": "ok"}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_create_post"].invoke({"title": "T", "body": "B"})
        assert "unknown" in result

    def test_passes_all_params(self):
        toolkit, mock = _toolkit_with({"create_post": {"id": "x"}})
        tools = {t.name: t for t in toolkit.get_tools()}
        tools["colony_create_post"].invoke({"title": "T", "body": "B", "colony": "crypto", "post_type": "finding"})
        assert mock.calls[-1] == (
            "create_post",
            {"title": "T", "body": "B", "colony": "crypto", "post_type": "finding"},
        )

    def test_async_returns_post_id(self):
        toolkit, _ = _toolkit_with({"create_post": {"id": "async-post-456"}})
        tools_by_name = {t.name: t for t in toolkit.get_tools()}
        tool = tools_by_name["colony_create_post"]
        result = asyncio.run(tool.ainvoke({"title": "Async", "body": "Post"}))
        assert "async-post-456" in result


class TestCommentOnPost:
    def test_returns_comment_id(self):
        toolkit, mock = _toolkit_with({"create_comment": {"id": "comment-1"}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_comment_on_post"].invoke({"post_id": "p-1", "body": "Nice!"})
        assert "comment-1" in result
        assert mock.calls[-1] == ("create_comment", {"post_id": "p-1", "body": "Nice!", "parent_id": None})

    def test_threaded_reply(self):
        toolkit, mock = _toolkit_with({"create_comment": {"comment": {"id": "reply-2"}}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_comment_on_post"].invoke({"post_id": "p-1", "body": "Reply", "parent_id": "comment-1"})
        assert "reply-2" in result
        assert mock.calls[-1] == ("create_comment", {"post_id": "p-1", "body": "Reply", "parent_id": "comment-1"})

    def test_async_returns_comment_id(self):
        toolkit, _ = _toolkit_with({"create_comment": {"id": "async-c"}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = asyncio.run(tools["colony_comment_on_post"].ainvoke({"post_id": "p-1", "body": "Async!"}))
        assert "async-c" in result


class TestVoteOnPost:
    def test_upvote(self):
        toolkit, _ = _toolkit_with({"vote_post": {}})
        tools_by_name = {t.name: t for t in toolkit.get_tools()}
        tool = tools_by_name["colony_vote_on_post"]
        result = tool.invoke({"post_id": "abc-123", "value": 1})
        assert "Upvoted" in result

    def test_downvote(self):
        toolkit, _ = _toolkit_with({"vote_post": {}})
        tools_by_name = {t.name: t for t in toolkit.get_tools()}
        tool = tools_by_name["colony_vote_on_post"]
        result = tool.invoke({"post_id": "abc-123", "value": -1})
        assert "Downvoted" in result

    def test_async_upvote(self):
        toolkit, _ = _toolkit_with({"vote_post": {}})
        tools_by_name = {t.name: t for t in toolkit.get_tools()}
        tool = tools_by_name["colony_vote_on_post"]
        result = asyncio.run(tool.ainvoke({"post_id": "abc-123", "value": 1}))
        assert "Upvoted" in result


class TestSendMessage:
    def test_sends_message(self):
        toolkit, mock = _toolkit_with({"send_message": {}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_send_message"].invoke({"username": "agent-b", "body": "Hello!"})
        assert "agent-b" in result
        assert mock.calls[-1] == ("send_message", {"username": "agent-b", "body": "Hello!"})

    def test_async_sends_message(self):
        toolkit, _ = _toolkit_with({"send_message": {}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = asyncio.run(tools["colony_send_message"].ainvoke({"username": "bot-z", "body": "Hi"}))
        assert "bot-z" in result


class TestGetNotifications:
    def test_no_notifications(self):
        toolkit, _ = _toolkit_with({"get_notifications": {"notifications": []}})
        tools_by_name = {t.name: t for t in toolkit.get_tools()}
        tool = tools_by_name["colony_get_notifications"]
        result = tool.invoke({"unread_only": True})
        assert "No notifications" in result

    def test_with_notifications(self):
        toolkit, mock = _toolkit_with(
            {
                "get_notifications": {
                    "notifications": [
                        {"type": "reply", "actor": {"username": "responder"}, "preview": "Good point"},
                        {"type": "dm", "actor": {"username": "friend"}, "body": "Hey there"},
                    ]
                }
            }
        )
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_get_notifications"].invoke({"unread_only": False})
        assert "responder" in result
        assert "Good point" in result
        assert "Hey there" in result
        # Toolkit defaults to limit=50; check method + the params we care about.
        method, kwargs = mock.calls[-1]
        assert method == "get_notifications"
        assert kwargs["unread_only"] is False

    def test_async_no_notifications(self):
        toolkit, _ = _toolkit_with({"get_notifications": {"notifications": []}})
        tools_by_name = {t.name: t for t in toolkit.get_tools()}
        tool = tools_by_name["colony_get_notifications"]
        result = asyncio.run(tool.ainvoke({"unread_only": True}))
        assert "No notifications" in result


# ── New formatter tests ─────────────────────────────────────────────


class TestFormatUser:
    def test_basic_user(self):
        result = _format_user(
            {
                "username": "agent-x",
                "display_name": "Agent X",
                "bio": "I research things",
                "post_count": 10,
                "comment_count": 25,
                "score": 42,
                "created_at": "2025-01-01",
            }
        )
        assert "agent-x" in result
        assert "Agent X" in result
        assert "I research things" in result
        assert "Posts: 10" in result
        assert "Score: 42" in result
        assert "2025-01-01" in result

    def test_nested_user_wrapper(self):
        result = _format_user({"user": {"username": "wrapped", "display_name": "W"}})
        assert "wrapped" in result

    def test_minimal_user(self):
        result = _format_user({})
        assert "?" in result

    def test_no_bio_omitted(self):
        result = _format_user({"username": "bot", "display_name": "Bot"})
        assert "Bio" not in result


class TestFormatColonies:
    def test_empty(self):
        assert _format_colonies({"colonies": []}) == "No colonies found."
        assert _format_colonies({}) == "No colonies found."

    def test_with_colonies(self):
        result = _format_colonies(
            {
                "colonies": [
                    {"name": "general", "description": "General discussion", "post_count": 100},
                    {"name": "findings", "description": "Research findings", "post_count": 50},
                ]
            }
        )
        assert "general" in result
        assert "General discussion" in result
        assert "100 posts" in result
        assert "findings" in result

    def test_no_description(self):
        result = _format_colonies({"colonies": [{"name": "empty", "post_count": 0}]})
        assert "empty" in result
        assert "0 posts" in result


class TestFormatConversation:
    def test_empty(self):
        assert _format_conversation({"messages": []}) == "No messages in conversation."
        assert _format_conversation({}) == "No messages in conversation."

    def test_with_messages(self):
        result = _format_conversation(
            {
                "messages": [
                    {"sender": {"username": "alice"}, "body": "Hey there"},
                    {"sender": {"username": "bob"}, "body": "Hi!"},
                ]
            }
        )
        assert "alice" in result
        assert "Hey there" in result
        assert "bob" in result

    def test_fallback_from_field(self):
        result = _format_conversation({"messages": [{"from": "legacy-user", "body": "old format"}]})
        assert "legacy-user" in result


# ── New tool invocation tests ───────────────────────────────────────


class TestGetMe:
    def test_returns_profile(self):
        toolkit, _ = _toolkit_with(
            {
                "get_me": {
                    "username": "my-agent",
                    "display_name": "My Agent",
                    "bio": "I do things",
                    "post_count": 5,
                    "comment_count": 10,
                    "score": 15,
                }
            }
        )
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_get_me"].invoke({})
        assert "my-agent" in result
        assert "I do things" in result

    def test_async_returns_profile(self):
        toolkit, _ = _toolkit_with({"get_me": {"username": "async-me"}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = asyncio.run(tools["colony_get_me"].ainvoke({}))
        assert "async-me" in result


class TestGetUser:
    def test_returns_user(self):
        toolkit, mock = _toolkit_with(
            {"get_user": {"user": {"username": "other-agent", "display_name": "Other", "bio": "Explorer"}}}
        )
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_get_user"].invoke({"user_id": "other-agent"})
        assert "other-agent" in result
        assert "Explorer" in result
        assert mock.calls[-1] == ("get_user", {"user_id": "other-agent"})

    def test_async_returns_user(self):
        toolkit, _ = _toolkit_with({"get_user": {"username": "u2"}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = asyncio.run(tools["colony_get_user"].ainvoke({"user_id": "u2"}))
        assert "u2" in result


class TestListColonies:
    def test_returns_colonies(self):
        toolkit, mock = _toolkit_with(
            {
                "get_colonies": {
                    "colonies": [
                        {"name": "general", "description": "Main forum", "post_count": 200},
                    ]
                }
            }
        )
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_list_colonies"].invoke({})
        assert "general" in result
        assert "200 posts" in result
        assert mock.calls[-1] == ("get_colonies", {"limit": 50})

    def test_async_returns_colonies(self):
        toolkit, _ = _toolkit_with({"get_colonies": {"colonies": []}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = asyncio.run(tools["colony_list_colonies"].ainvoke({"limit": 10}))
        assert "No colonies found" in result


class TestGetConversation:
    def test_returns_messages(self):
        toolkit, mock = _toolkit_with(
            {
                "get_conversation": {
                    "messages": [
                        {"sender": {"username": "me"}, "body": "Hi"},
                        {"sender": {"username": "them"}, "body": "Hello!"},
                    ]
                }
            }
        )
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_get_conversation"].invoke({"username": "them"})
        assert "me" in result
        assert "Hello!" in result
        assert mock.calls[-1] == ("get_conversation", {"username": "them"})

    def test_async_empty_conversation(self):
        toolkit, _ = _toolkit_with({"get_conversation": {"messages": []}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = asyncio.run(tools["colony_get_conversation"].ainvoke({"username": "nobody"}))
        assert "No messages" in result


class TestUpdatePost:
    def test_updates_post(self):
        toolkit, mock = _toolkit_with({"update_post": {}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_update_post"].invoke({"post_id": "p-1", "title": "New Title"})
        assert "updated" in result.lower()
        assert "p-1" in result
        assert mock.calls[-1] == ("update_post", {"post_id": "p-1", "title": "New Title", "body": None})

    def test_async_updates_post(self):
        toolkit, _ = _toolkit_with({"update_post": {}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = asyncio.run(tools["colony_update_post"].ainvoke({"post_id": "p-2", "body": "Updated body"}))
        assert "p-2" in result


class TestDeletePost:
    def test_deletes_post(self):
        toolkit, mock = _toolkit_with({"delete_post": {}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_delete_post"].invoke({"post_id": "p-1"})
        assert "deleted" in result.lower()
        assert "p-1" in result
        assert mock.calls[-1] == ("delete_post", {"post_id": "p-1"})

    def test_async_deletes_post(self):
        toolkit, _ = _toolkit_with({"delete_post": {}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = asyncio.run(tools["colony_delete_post"].ainvoke({"post_id": "p-3"}))
        assert "p-3" in result


class TestVoteOnComment:
    def test_upvote(self):
        toolkit, mock = _toolkit_with({"vote_comment": {}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_vote_on_comment"].invoke({"comment_id": "c-1", "value": 1})
        assert "Upvoted" in result
        assert "c-1" in result
        assert mock.calls[-1] == ("vote_comment", {"comment_id": "c-1", "value": 1})

    def test_downvote(self):
        toolkit, _ = _toolkit_with({"vote_comment": {}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_vote_on_comment"].invoke({"comment_id": "c-2", "value": -1})
        assert "Downvoted" in result

    def test_async_upvote(self):
        toolkit, _ = _toolkit_with({"vote_comment": {}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = asyncio.run(tools["colony_vote_on_comment"].ainvoke({"comment_id": "c-3"}))
        assert "Upvoted" in result


class TestMarkNotificationsRead:
    def test_marks_read(self):
        toolkit, mock = _toolkit_with({})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_mark_notifications_read"].invoke({})
        assert "marked as read" in result.lower()
        assert mock.calls[-1] == ("mark_notifications_read", {})

    def test_async_marks_read(self):
        toolkit, _ = _toolkit_with({})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = asyncio.run(tools["colony_mark_notifications_read"].ainvoke({}))
        assert "marked as read" in result.lower()


class TestUpdateProfile:
    def test_updates_display_name(self):
        toolkit, mock = _toolkit_with({"update_profile": {}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_update_profile"].invoke({"display_name": "New Name"})
        assert "updated" in result.lower()
        assert "display_name" in result
        assert mock.calls[-1] == ("update_profile", {"display_name": "New Name"})

    def test_updates_both_fields(self):
        toolkit, _ = _toolkit_with({"update_profile": {}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_update_profile"].invoke({"display_name": "X", "bio": "New bio"})
        assert "display_name" in result
        assert "bio" in result

    def test_no_fields_provided(self):
        toolkit, mock = _toolkit_with({})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_update_profile"].invoke({})
        assert "No fields" in result
        # update_profile should NOT have been called.
        assert all(call[0] != "update_profile" for call in mock.calls)

    def test_async_updates_profile(self):
        toolkit, _ = _toolkit_with({"update_profile": {}})
        tools = {t.name: t for t in toolkit.get_tools()}
        result = asyncio.run(tools["colony_update_profile"].ainvoke({"bio": "Async bio"}))
        assert "bio" in result
