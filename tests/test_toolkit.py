"""Tests for the Colony LangChain toolkit."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from colony_langchain import ColonyToolkit
from colony_langchain.tools import (
    ColonyCommentOnPost,
    ColonyCreatePost,
    ColonyGetNotifications,
    ColonyGetPost,
    ColonySearchPosts,
    ColonySendMessage,
    ColonyVoteOnPost,
    _format_notifications,
    _format_post,
    _format_posts,
)


def _make_toolkit(**kwargs):
    with patch("colony_langchain.toolkit.ColonyClient"):
        return ColonyToolkit(api_key="col_test", **kwargs)


def _tools_by_name():
    with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
        toolkit = ColonyToolkit(api_key="col_test")
        return {t.name: t for t in toolkit.get_tools()}, MockClient.return_value


# ── Toolkit ─────────────────────────────────────────────────────────


class TestToolkit:
    def test_get_tools_returns_all_seven(self):
        toolkit = _make_toolkit()
        tools = toolkit.get_tools()
        assert len(tools) == 7
        names = {t.name for t in tools}
        assert names == {
            "colony_search_posts",
            "colony_get_post",
            "colony_create_post",
            "colony_comment_on_post",
            "colony_vote_on_post",
            "colony_send_message",
            "colony_get_notifications",
        }

    def test_read_only_returns_three(self):
        toolkit = _make_toolkit(read_only=True)
        tools = toolkit.get_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {
            "colony_search_posts",
            "colony_get_post",
            "colony_get_notifications",
        }

    def test_tools_have_descriptions(self):
        toolkit = _make_toolkit()
        for tool in toolkit.get_tools():
            assert tool.description, f"{tool.name} has no description"

    def test_tools_have_args_schema(self):
        toolkit = _make_toolkit()
        for tool in toolkit.get_tools():
            assert tool.args_schema is not None, f"{tool.name} has no args_schema"


# ── Formatters ──────────────────────────────────────────────────────


class TestFormatPosts:
    def test_empty(self):
        assert _format_posts({"posts": []}) == "No posts found."
        assert _format_posts({}) == "No posts found."

    def test_single_post(self):
        result = _format_posts({
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
        })
        assert "Hello" in result
        assert "agent-x" in result
        assert "general" in result
        assert "abc" in result
        assert "score: 5" in result

    def test_multiple_posts(self):
        result = _format_posts({
            "posts": [
                {"id": "1", "title": "First", "post_type": "finding", "score": 1, "comment_count": 0, "author": {"username": "a"}, "colony": {"name": "c1"}},
                {"id": "2", "title": "Second", "post_type": "question", "score": 3, "comment_count": 1, "author": {"username": "b"}, "colony": {"name": "c2"}},
            ]
        })
        assert "First" in result
        assert "Second" in result

    def test_missing_fields_fallback(self):
        result = _format_posts({
            "posts": [{"id": "x", "title": "T", "post_type": "discussion"}]
        })
        assert "?" in result  # missing author/colony fall back to ?
        assert "score: 0" in result  # missing score defaults to 0


class TestFormatPost:
    def test_basic_post(self):
        result = _format_post({
            "title": "My Post",
            "post_type": "analysis",
            "score": 10,
            "comment_count": 3,
            "author": {"username": "researcher"},
            "colony": {"name": "findings"},
            "id": "post-1",
            "body": "This is the body.",
        })
        assert "My Post" in result
        assert "analysis" in result
        assert "researcher" in result
        assert "This is the body." in result

    def test_nested_post_wrapper(self):
        result = _format_post({
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
        })
        assert "Wrapped" in result
        assert "wrapped body" in result

    def test_with_comments(self):
        result = _format_post({
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
        })
        assert "Top comments:" in result
        assert "commenter1" in result
        assert "Great post!" in result
        assert "commenter2" in result

    def test_comments_truncated_to_ten(self):
        comments = [
            {"author": {"username": f"user{i}"}, "body": f"Comment {i}"}
            for i in range(15)
        ]
        result = _format_post({
            "title": "P", "post_type": "d", "score": 0, "comment_count": 15,
            "author": {"username": "x"}, "colony": {"name": "y"}, "id": "z",
            "body": "", "comments": comments,
        })
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
        result = _format_notifications({
            "notifications": [
                {"type": "reply", "actor": {"username": "agent-a"}, "preview": "Thanks for sharing"},
                {"type": "mention", "actor": {"username": "agent-b"}, "body": "Check out @you"},
            ]
        })
        assert "[reply]" in result
        assert "agent-a" in result
        assert "Thanks for sharing" in result
        assert "[mention]" in result
        assert "Check out @you" in result

    def test_long_preview_truncated(self):
        result = _format_notifications({
            "notifications": [
                {"type": "dm", "actor": {"username": "x"}, "preview": "A" * 200},
            ]
        })
        # preview is truncated to 100 chars
        assert len(result.split(": ", 1)[1]) == 100


# ── Tool invocations ────────────────────────────────────────────────


class TestSearchPosts:
    def test_formats_results(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_posts.return_value = {
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
            toolkit = ColonyToolkit(api_key="col_test")
            tool = toolkit.get_tools()[0]
            assert tool.name == "colony_search_posts"

            result = tool.invoke({"query": "test"})
            assert "Test Post" in result
            assert "test-agent" in result

    def test_no_results(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_posts.return_value = {"posts": []}
            toolkit = ColonyToolkit(api_key="col_test")
            tool = toolkit.get_tools()[0]
            result = tool.invoke({"query": "nonexistent"})
            assert "No posts found" in result

    def test_passes_all_params(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_posts.return_value = {"posts": []}
            toolkit = ColonyToolkit(api_key="col_test")
            tool = toolkit.get_tools()[0]
            tool.invoke({"query": "ai", "colony": "findings", "sort": "top", "limit": 5})
            mock_client.get_posts.assert_called_once_with(search="ai", colony="findings", sort="top", limit=5)

    def test_async_formats_results(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_posts.return_value = {
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
            toolkit = ColonyToolkit(api_key="col_test")
            tool = toolkit.get_tools()[0]
            result = asyncio.run(tool.ainvoke({"query": "async"}))
            assert "Async Post" in result
            assert "async-agent" in result


class TestGetPost:
    def test_returns_formatted_post(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_post.return_value = {
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
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_get_post"].invoke({"post_id": "post-99"})
            assert "Deep Dive" in result
            assert "Detailed analysis here." in result
            assert "reader" in result
            assert "Very insightful!" in result

    def test_async_returns_formatted_post(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_post.return_value = {
                "title": "Simple",
                "post_type": "discussion",
                "score": 0,
                "comment_count": 0,
                "author": {"username": "bot"},
                "colony": {"name": "general"},
                "id": "p-1",
                "body": "Hello world",
            }
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = asyncio.run(tools["colony_get_post"].ainvoke({"post_id": "p-1"}))
            assert "Simple" in result
            assert "Hello world" in result


class TestCreatePost:
    def test_returns_post_id(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.create_post.return_value = {"id": "new-post-123"}
            toolkit = ColonyToolkit(api_key="col_test")
            tools_by_name = {t.name: t for t in toolkit.get_tools()}
            tool = tools_by_name["colony_create_post"]
            result = tool.invoke({"title": "Hello", "body": "World"})
            assert "new-post-123" in result

    def test_nested_post_id(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.create_post.return_value = {"post": {"id": "nested-789"}}
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_create_post"].invoke({"title": "T", "body": "B"})
            assert "nested-789" in result

    def test_unknown_id_fallback(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.create_post.return_value = {"status": "ok"}
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_create_post"].invoke({"title": "T", "body": "B"})
            assert "unknown" in result

    def test_passes_all_params(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.create_post.return_value = {"id": "x"}
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            tools["colony_create_post"].invoke({
                "title": "T", "body": "B", "colony": "crypto", "post_type": "finding"
            })
            mock_client.create_post.assert_called_once_with(
                title="T", body="B", colony="crypto", post_type="finding"
            )

    def test_async_returns_post_id(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.create_post.return_value = {"id": "async-post-456"}
            toolkit = ColonyToolkit(api_key="col_test")
            tools_by_name = {t.name: t for t in toolkit.get_tools()}
            tool = tools_by_name["colony_create_post"]
            result = asyncio.run(tool.ainvoke({"title": "Async", "body": "Post"}))
            assert "async-post-456" in result


class TestCommentOnPost:
    def test_returns_comment_id(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.create_comment.return_value = {"id": "comment-1"}
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_comment_on_post"].invoke({"post_id": "p-1", "body": "Nice!"})
            assert "comment-1" in result
            mock_client.create_comment.assert_called_once_with(post_id="p-1", body="Nice!", parent_id=None)

    def test_threaded_reply(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.create_comment.return_value = {"comment": {"id": "reply-2"}}
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_comment_on_post"].invoke({
                "post_id": "p-1", "body": "Reply", "parent_id": "comment-1"
            })
            assert "reply-2" in result
            mock_client.create_comment.assert_called_once_with(post_id="p-1", body="Reply", parent_id="comment-1")

    def test_async_returns_comment_id(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.create_comment.return_value = {"id": "async-c"}
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = asyncio.run(tools["colony_comment_on_post"].ainvoke({"post_id": "p-1", "body": "Async!"}))
            assert "async-c" in result


class TestVoteOnPost:
    def test_upvote(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.vote_post.return_value = {}
            toolkit = ColonyToolkit(api_key="col_test")
            tools_by_name = {t.name: t for t in toolkit.get_tools()}
            tool = tools_by_name["colony_vote_on_post"]
            result = tool.invoke({"post_id": "abc-123", "value": 1})
            assert "Upvoted" in result

    def test_downvote(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.vote_post.return_value = {}
            toolkit = ColonyToolkit(api_key="col_test")
            tools_by_name = {t.name: t for t in toolkit.get_tools()}
            tool = tools_by_name["colony_vote_on_post"]
            result = tool.invoke({"post_id": "abc-123", "value": -1})
            assert "Downvoted" in result

    def test_async_upvote(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.vote_post.return_value = {}
            toolkit = ColonyToolkit(api_key="col_test")
            tools_by_name = {t.name: t for t in toolkit.get_tools()}
            tool = tools_by_name["colony_vote_on_post"]
            result = asyncio.run(tool.ainvoke({"post_id": "abc-123", "value": 1}))
            assert "Upvoted" in result


class TestSendMessage:
    def test_sends_message(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.send_message.return_value = {}
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_send_message"].invoke({"username": "agent-b", "body": "Hello!"})
            assert "agent-b" in result
            mock_client.send_message.assert_called_once_with(username="agent-b", body="Hello!")

    def test_async_sends_message(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.send_message.return_value = {}
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = asyncio.run(tools["colony_send_message"].ainvoke({"username": "bot-z", "body": "Hi"}))
            assert "bot-z" in result


class TestGetNotifications:
    def test_no_notifications(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_notifications.return_value = {"notifications": []}
            toolkit = ColonyToolkit(api_key="col_test")
            tools_by_name = {t.name: t for t in toolkit.get_tools()}
            tool = tools_by_name["colony_get_notifications"]
            result = tool.invoke({"unread_only": True})
            assert "No notifications" in result

    def test_with_notifications(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_notifications.return_value = {
                "notifications": [
                    {"type": "reply", "actor": {"username": "responder"}, "preview": "Good point"},
                    {"type": "dm", "actor": {"username": "friend"}, "body": "Hey there"},
                ]
            }
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_get_notifications"].invoke({"unread_only": False})
            assert "responder" in result
            assert "Good point" in result
            assert "Hey there" in result
            mock_client.get_notifications.assert_called_once_with(unread_only=False)

    def test_async_no_notifications(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_notifications.return_value = {"notifications": []}
            toolkit = ColonyToolkit(api_key="col_test")
            tools_by_name = {t.name: t for t in toolkit.get_tools()}
            tool = tools_by_name["colony_get_notifications"]
            result = asyncio.run(tool.ainvoke({"unread_only": True}))
            assert "No notifications" in result
