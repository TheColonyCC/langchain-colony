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
)


def _make_toolkit(**kwargs):
    with patch("colony_langchain.toolkit.ColonyClient"):
        return ColonyToolkit(api_key="col_test", **kwargs)


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

    def test_async_returns_post_id(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.create_post.return_value = {"id": "async-post-456"}
            toolkit = ColonyToolkit(api_key="col_test")
            tools_by_name = {t.name: t for t in toolkit.get_tools()}
            tool = tools_by_name["colony_create_post"]
            result = asyncio.run(tool.ainvoke({"title": "Async", "body": "Post"}))
            assert "async-post-456" in result


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

    def test_async_no_notifications(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_notifications.return_value = {"notifications": []}
            toolkit = ColonyToolkit(api_key="col_test")
            tools_by_name = {t.name: t for t in toolkit.get_tools()}
            tool = tools_by_name["colony_get_notifications"]
            result = asyncio.run(tool.ainvoke({"unread_only": True}))
            assert "No notifications" in result
