"""Native-async tests — verifies AsyncColonyToolkit, ColonyRetriever (with
AsyncColonyClient), and ColonyEventPoller (with AsyncColonyClient) dispatch
through native ``await`` rather than ``asyncio.to_thread``.

Uses ``httpx.MockTransport`` so we exercise the full SDK 1.5.0 async stack
without hitting the network."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest
from colony_sdk import AsyncColonyClient, RetryConfig

from langchain_colony import (
    AsyncColonyToolkit,
    ColonyEventPoller,
    ColonyGetMe,
    ColonyGetPost,
    ColonyMarkNotificationsRead,
    ColonyRetriever,
    ColonySearchPosts,
)

# ── Dispatcher behaviour ───────────────────────────────────────────


class TestDispatcher:
    """``_ColonyBaseTool._aapi`` should ``await`` async client methods natively
    and only fall back to ``to_thread`` for sync methods."""

    async def test_native_await_for_coroutine_function(self) -> None:
        """When the bound method is a coroutine function, no thread is used."""

        async def fake_method(post_id: str) -> dict:
            return {
                "id": post_id,
                "title": "ok",
                "post_type": "discussion",
                "score": 0,
                "comment_count": 0,
                "author": {"username": "u"},
                "colony": {"name": "g"},
                "body": "",
            }

        client = MagicMock()
        client.get_post = fake_method
        tool = ColonyGetPost(client=client)

        with patch("asyncio.to_thread") as mock_to_thread:
            result = await tool._arun(post_id="p1")
            mock_to_thread.assert_not_called()
        assert "ok" in result

    async def test_to_thread_fallback_for_sync_function(self) -> None:
        """A plain (non-coroutine) callable goes through ``asyncio.to_thread``
        so it can't block the event loop."""
        client = MagicMock()
        client.get_post = MagicMock(
            return_value={
                "id": "p1",
                "title": "sync",
                "post_type": "discussion",
                "score": 0,
                "comment_count": 0,
                "author": {"username": "u"},
                "colony": {"name": "g"},
                "body": "",
            }
        )
        tool = ColonyGetPost(client=client)

        with patch("asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
            await tool._arun(post_id="p1")
            mock_to_thread.assert_called_once()

    async def test_native_await_propagates_sdk_error(self) -> None:
        """Errors raised by the awaited coroutine are formatted, not bubbled."""
        from colony_sdk import ColonyNotFoundError

        async def fake_method(post_id: str) -> dict:
            raise ColonyNotFoundError(
                "get_post failed: not found (not found — the resource doesn't exist or has been deleted)",
                status=404,
            )

        client = MagicMock()
        client.get_post = fake_method
        tool = ColonyGetPost(client=client)
        result = await tool._arun(post_id="p1")
        assert "Error" in result
        assert "404" in result
        assert "not found" in result.lower()

    async def test_native_await_catches_unexpected_exception(self) -> None:
        async def fake_method(post_id: str) -> dict:
            raise RuntimeError("unexpected")

        client = MagicMock()
        client.get_post = fake_method
        tool = ColonyGetPost(client=client)
        result = await tool._arun(post_id="p1")
        assert "Error" in result
        assert "unexpected" in result


# ── AsyncColonyToolkit construction ────────────────────────────────


class TestAsyncToolkit:
    def test_constructs_async_client(self) -> None:
        toolkit = AsyncColonyToolkit(api_key="col_test")
        assert isinstance(toolkit.client, AsyncColonyClient)

    def test_passes_retry_to_client(self) -> None:
        with patch("colony_sdk.AsyncColonyClient") as MockCls:
            retry = RetryConfig(max_retries=5, base_delay=0.1)
            AsyncColonyToolkit(api_key="col_test", retry=retry)
            kwargs = MockCls.call_args.kwargs
            assert kwargs["retry"] is retry

    def test_omits_retry_when_unset(self) -> None:
        with patch("colony_sdk.AsyncColonyClient") as MockCls:
            AsyncColonyToolkit(api_key="col_test")
            kwargs = MockCls.call_args.kwargs
            assert "retry" not in kwargs

    def test_get_tools_returns_all(self) -> None:
        toolkit = AsyncColonyToolkit(api_key="col_test")
        tools = toolkit.get_tools()
        assert len(tools) == 16
        names = {t.name for t in tools}
        assert "colony_create_post" in names
        assert "colony_search_posts" in names

    def test_get_tools_read_only(self) -> None:
        toolkit = AsyncColonyToolkit(api_key="col_test", read_only=True)
        tools = toolkit.get_tools()
        assert len(tools) == 7
        names = {t.name for t in tools}
        assert "colony_create_post" not in names

    def test_get_tools_include(self) -> None:
        toolkit = AsyncColonyToolkit(api_key="col_test")
        tools = toolkit.get_tools(include=["colony_get_me"])
        assert len(tools) == 1

    def test_get_tools_exclude(self) -> None:
        toolkit = AsyncColonyToolkit(api_key="col_test")
        tools = toolkit.get_tools(exclude=["colony_create_post"])
        assert len(tools) == 15
        names = {t.name for t in tools}
        assert "colony_create_post" not in names

    def test_get_tools_include_and_exclude_raises(self) -> None:
        toolkit = AsyncColonyToolkit(api_key="col_test")
        with pytest.raises(ValueError, match="both"):
            toolkit.get_tools(include=["x"], exclude=["y"])

    def test_remembers_retry_config(self) -> None:
        retry = RetryConfig(max_retries=4)
        toolkit = AsyncColonyToolkit(api_key="col_test", retry=retry)
        assert toolkit.retry_config is retry

    async def test_async_context_manager(self) -> None:
        async with AsyncColonyToolkit(api_key="col_test") as toolkit:
            tools = toolkit.get_tools()
            assert len(tools) == 16

    async def test_aclose(self) -> None:
        toolkit = AsyncColonyToolkit(api_key="col_test")
        await toolkit.aclose()


# ── End-to-end via httpx.MockTransport ─────────────────────────────


def _mock_transport(responses: dict[str, dict]) -> httpx.MockTransport:
    """Build an httpx.MockTransport that returns the given JSON for any
    request matching a path key (longest match wins)."""

    def handler(request: httpx.Request) -> httpx.Response:
        # longest path key first so /posts/p1 wins over /posts
        for path in sorted(responses.keys(), key=len, reverse=True):
            if request.url.path.endswith(path):
                return httpx.Response(200, json=responses[path])
        return httpx.Response(404, json={"detail": f"no mock for {request.url.path}"})

    return httpx.MockTransport(handler)


@pytest.fixture
def mock_async_client() -> AsyncColonyClient:
    transport = _mock_transport(
        {
            "/auth/token": {"access_token": "jwt.fake", "expires_in": 3600},
            "/posts": {
                "posts": [
                    {
                        "id": "p1",
                        "title": "Hello",
                        "post_type": "discussion",
                        "author": {"username": "bot"},
                        "score": 5,
                        "comment_count": 2,
                        "colony": {"name": "general"},
                        "body": "test body",
                    }
                ]
            },
            "/posts/p1": {
                "id": "p1",
                "title": "Hello",
                "post_type": "discussion",
                "author": {"username": "bot"},
                "score": 5,
                "comment_count": 0,
                "colony": {"name": "general"},
                "body": "full body",
                "comments": [],
            },
            "/users/me": {
                "username": "colonist-one",
                "display_name": "Colonist One",
                "bio": "the AI agent CMO",
                "karma": 99,
            },
            "/notifications": {"notifications": []},
            "/notifications/read-all": {"status": "ok"},
        }
    )
    httpx_client = httpx.AsyncClient(transport=transport, base_url="https://thecolony.cc/api/v1")
    return AsyncColonyClient("col_test", client=httpx_client)


class TestEndToEnd:
    """Tools wired to a real ``AsyncColonyClient`` with a mocked transport."""

    async def test_search_posts_native(self, mock_async_client: AsyncColonyClient) -> None:
        tool = ColonySearchPosts(client=mock_async_client)
        result = await tool._arun(query="hello")
        assert "Hello" in result
        assert "bot" in result
        await mock_async_client.aclose()

    async def test_get_post_native(self, mock_async_client: AsyncColonyClient) -> None:
        tool = ColonyGetPost(client=mock_async_client)
        result = await tool._arun(post_id="p1")
        assert "Hello" in result
        assert "full body" in result
        await mock_async_client.aclose()

    async def test_get_me_native(self, mock_async_client: AsyncColonyClient) -> None:
        tool = ColonyGetMe(client=mock_async_client)
        result = await tool._arun()
        assert "colonist-one" in result
        await mock_async_client.aclose()

    async def test_concurrent_fan_out(self, mock_async_client: AsyncColonyClient) -> None:
        """The whole point of native async — many tool calls in parallel
        on a single event loop, no thread pool."""
        tool = ColonySearchPosts(client=mock_async_client)
        results = await asyncio.gather(*[tool._arun(query=f"q{i}") for i in range(10)])
        assert len(results) == 10
        assert all("Hello" in r for r in results)
        await mock_async_client.aclose()


# ── ColonyRetriever native async ───────────────────────────────────


class TestRetrieverNativeAsync:
    async def test_retriever_with_async_client(self, mock_async_client: AsyncColonyClient) -> None:
        """Passing an AsyncColonyClient to ColonyRetriever should give native
        ``await`` on ``ainvoke()`` — no ``asyncio.to_thread`` involved."""
        retriever = ColonyRetriever(client=mock_async_client)
        with patch("asyncio.to_thread") as mock_to_thread:
            docs = await retriever.ainvoke("hello")
            mock_to_thread.assert_not_called()
        assert len(docs) == 1
        assert docs[0].metadata["title"] == "Hello"
        assert docs[0].metadata["source"] == "thecolony.cc"
        await mock_async_client.aclose()

    async def test_retriever_with_sync_client_uses_thread(self) -> None:
        """Passing a sync ``ColonyClient`` (or a MagicMock) — ``ainvoke``
        falls back to ``to_thread`` so it doesn't block the event loop."""
        sync_client = MagicMock()
        sync_client.get_posts = MagicMock(
            return_value={
                "posts": [
                    {
                        "id": "p1",
                        "title": "Hello",
                        "post_type": "discussion",
                        "author": {"username": "bot"},
                        "score": 1,
                        "comment_count": 0,
                        "colony": {"name": "g"},
                        "body": "x",
                    }
                ]
            }
        )
        retriever = ColonyRetriever(client=sync_client)
        with patch("asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
            docs = await retriever.ainvoke("hello")
            mock_to_thread.assert_called_once()
        assert len(docs) == 1

    async def test_retriever_include_comments_native(self, mock_async_client: AsyncColonyClient) -> None:
        retriever = ColonyRetriever(client=mock_async_client, include_comments=True)
        docs = await retriever.ainvoke("hello")
        assert len(docs) == 1
        # Mock has empty comments — no Comments section appended.
        assert "## Comments" not in docs[0].page_content
        await mock_async_client.aclose()

    def test_retriever_requires_api_key_or_client(self) -> None:
        with pytest.raises(ValueError, match="api_key or client"):
            ColonyRetriever()

    def test_retriever_with_only_api_key_still_works(self) -> None:
        retriever = ColonyRetriever(api_key="col_test")
        from colony_sdk import ColonyClient

        assert isinstance(retriever.client, ColonyClient)


# ── ColonyEventPoller native async ─────────────────────────────────


class TestEventPollerNativeAsync:
    async def test_poller_with_async_client_native(self, mock_async_client: AsyncColonyClient) -> None:
        """Passing an AsyncColonyClient to ColonyEventPoller should give
        native ``await`` on ``poll_once_async()`` — no thread pool."""
        poller = ColonyEventPoller(client=mock_async_client)
        with patch("asyncio.to_thread") as mock_to_thread:
            results = await poller.poll_once_async()
            mock_to_thread.assert_not_called()
        # Mock returns empty notification list — that's still a successful poll.
        assert results == []
        await mock_async_client.aclose()

    async def test_poller_with_sync_client_uses_thread(self) -> None:
        sync_client = MagicMock()
        sync_client.get_notifications = MagicMock(return_value={"notifications": []})
        poller = ColonyEventPoller(client=sync_client)
        with patch("asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
            await poller.poll_once_async()
            mock_to_thread.assert_called_once()

    async def test_poller_mark_read_native(self) -> None:
        """``mark_read=True`` with an async client must use native await
        for ``mark_notifications_read``, not ``to_thread``."""

        class FakeAsyncClient:
            def __init__(self) -> None:
                self.marked = False

            async def get_notifications(self, unread_only: bool = True) -> dict:
                return {
                    "notifications": [
                        {
                            "id": "n1",
                            "type": "mention",
                            "message": "hi",
                            "read": False,
                        }
                    ]
                }

            async def mark_notifications_read(self) -> None:
                self.marked = True

        client = FakeAsyncClient()
        poller = ColonyEventPoller(client=client, mark_read=True)
        with patch("asyncio.to_thread") as mock_to_thread:
            results = await poller.poll_once_async()
            mock_to_thread.assert_not_called()
        assert len(results) == 1
        assert client.marked is True

    def test_poller_requires_api_key_or_client(self) -> None:
        with pytest.raises(ValueError, match="api_key or client"):
            ColonyEventPoller()

    def test_poller_with_only_api_key_still_works(self) -> None:
        poller = ColonyEventPoller(api_key="col_test")
        from colony_sdk import ColonyClient

        assert isinstance(poller.client, ColonyClient)


class TestMarkNotificationsReadNativeAsync:
    """The ``ColonyMarkNotificationsRead`` tool — same dispatcher trick on the
    tool path."""

    async def test_native_await(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.called = False

            async def mark_notifications_read(self) -> None:
                self.called = True

        client = FakeClient()
        tool = ColonyMarkNotificationsRead(client=client)
        with patch("asyncio.to_thread") as mock_to_thread:
            result = await tool._arun()
            mock_to_thread.assert_not_called()
        assert client.called is True
        assert "Error" not in result or "OK" in result

    async def test_sync_fallback(self) -> None:
        client = MagicMock()
        client.mark_notifications_read = MagicMock(return_value=None)
        tool = ColonyMarkNotificationsRead(client=client)
        with patch("asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
            await tool._arun()
            mock_to_thread.assert_called_once()
