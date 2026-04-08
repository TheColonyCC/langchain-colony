"""Tests for error handling and retry logic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

from colony_sdk import ColonyAPIError

from colony_langchain import ColonyToolkit
from colony_langchain.tools import (
    _friendly_error,
    _retry_api_call,
    _async_retry_api_call,
    _MAX_RETRIES,
)


# ── Friendly error messages ─────────────────────────────────────────


class TestFriendlyError:
    def test_auth_error(self):
        err = ColonyAPIError("bad token", status=401, code="AUTH_INVALID_TOKEN")
        msg = _friendly_error(err)
        assert "authentication failed" in msg

    def test_forbidden(self):
        err = ColonyAPIError("nope", status=403, code="FORBIDDEN")
        msg = _friendly_error(err)
        assert "permission" in msg

    def test_not_found(self):
        err = ColonyAPIError("missing", status=404, code="NOT_FOUND")
        msg = _friendly_error(err)
        assert "not found" in msg

    def test_conflict(self):
        err = ColonyAPIError("already voted", status=409, code="CONFLICT")
        msg = _friendly_error(err)
        assert "conflict" in msg.lower()

    def test_validation(self):
        err = ColonyAPIError("title too short", status=422, code="VALIDATION_ERROR")
        msg = _friendly_error(err)
        assert "invalid input" in msg

    def test_rate_limit(self):
        err = ColonyAPIError("slow down", status=429, code="RATE_LIMIT_VOTE_HOURLY")
        msg = _friendly_error(err)
        assert "rate limited" in msg

    def test_rate_limit_by_code_only(self):
        err = ColonyAPIError("slow down", status=400, code="RATE_LIMIT_POST")
        msg = _friendly_error(err)
        assert "rate limited" in msg

    def test_server_error_fallback(self):
        err = ColonyAPIError("oops", status=500)
        msg = _friendly_error(err)
        assert "500" in msg

    def test_auth_by_code(self):
        err = ColonyAPIError("expired", status=400, code="AUTH_TOKEN_EXPIRED")
        msg = _friendly_error(err)
        assert "authentication failed" in msg


# ── Retry logic ─────────────────────────────────────────────────────


class TestRetryApiCall:
    def test_succeeds_first_try(self):
        fn = MagicMock(return_value={"ok": True})
        result = _retry_api_call(fn, "arg1", key="val")
        assert result == {"ok": True}
        fn.assert_called_once_with("arg1", key="val")

    @patch("colony_langchain.tools.time.sleep")
    def test_retries_on_429(self, mock_sleep):
        fn = MagicMock(side_effect=[
            ColonyAPIError("rate limit", status=429),
            {"ok": True},
        ])
        result = _retry_api_call(fn)
        assert result == {"ok": True}
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    @patch("colony_langchain.tools.time.sleep")
    def test_retries_on_500(self, mock_sleep):
        fn = MagicMock(side_effect=[
            ColonyAPIError("server error", status=500),
            ColonyAPIError("server error", status=502),
            {"recovered": True},
        ])
        result = _retry_api_call(fn)
        assert result == {"recovered": True}
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("colony_langchain.tools.time.sleep")
    def test_retries_on_network_error(self, mock_sleep):
        fn = MagicMock(side_effect=[
            ConnectionError("refused"),
            {"ok": True},
        ])
        result = _retry_api_call(fn)
        assert result == {"ok": True}
        assert fn.call_count == 2

    def test_no_retry_on_4xx(self):
        fn = MagicMock(side_effect=ColonyAPIError("not found", status=404))
        try:
            _retry_api_call(fn)
        except ColonyAPIError as exc:
            assert exc.status == 404
        fn.assert_called_once()

    @patch("colony_langchain.tools.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        fn = MagicMock(side_effect=ColonyAPIError("overloaded", status=503))
        try:
            _retry_api_call(fn)
        except ColonyAPIError as exc:
            assert exc.status == 503
        assert fn.call_count == _MAX_RETRIES

    @patch("colony_langchain.tools.time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        fn = MagicMock(side_effect=[
            ColonyAPIError("rate limit", status=429),
            ColonyAPIError("rate limit", status=429),
            {"ok": True},
        ])
        _retry_api_call(fn)
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays[0] < delays[1]  # exponential


class TestAsyncRetryApiCall:
    def test_retries_on_429(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ColonyAPIError("rate limit", status=429)
            return {"ok": True}

        async def run():
            with patch("colony_langchain.tools.asyncio.sleep", new_callable=AsyncMock):
                return await _async_retry_api_call(fn)

        result = asyncio.run(run())
        assert result == {"ok": True}
        assert call_count == 2

    def test_no_retry_on_4xx(self):
        fn = MagicMock(side_effect=ColonyAPIError("forbidden", status=403))
        try:
            asyncio.run(_async_retry_api_call(fn))
        except ColonyAPIError as exc:
            assert exc.status == 403
        fn.assert_called_once()


# ── Tool-level error handling ───────────────────────────────────────


class TestToolErrorHandling:
    def test_search_returns_friendly_error(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_posts.side_effect = ColonyAPIError("bad token", status=401, code="AUTH_INVALID_TOKEN")
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_search_posts"].invoke({"query": "test"})
            assert "authentication failed" in result
            assert "Error" in result

    def test_create_post_returns_friendly_error(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.create_post.side_effect = ColonyAPIError("title too short", status=422, code="VALIDATION_ERROR")
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_create_post"].invoke({"title": "X", "body": "Y"})
            assert "invalid input" in result

    def test_get_post_not_found(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_post.side_effect = ColonyAPIError("gone", status=404)
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_get_post"].invoke({"post_id": "nonexistent"})
            assert "not found" in result

    def test_vote_rate_limited(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.vote_post.side_effect = ColonyAPIError("too fast", status=429, code="RATE_LIMIT_VOTE_HOURLY")
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            # Rate limit will be retried and then return friendly error
            with patch("colony_langchain.tools.time.sleep"):
                result = tools["colony_vote_on_post"].invoke({"post_id": "p-1", "value": 1})
            assert "rate limited" in result

    def test_delete_forbidden(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.delete_post.side_effect = ColonyAPIError("not yours", status=403, code="FORBIDDEN")
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_delete_post"].invoke({"post_id": "p-1"})
            assert "permission" in result

    def test_async_error_handling(self):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_me.side_effect = ColonyAPIError("expired", status=401, code="AUTH_TOKEN_EXPIRED")
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = asyncio.run(tools["colony_get_me"].ainvoke({}))
            assert "authentication failed" in result

    @patch("colony_langchain.tools.time.sleep")
    def test_retry_then_succeed(self, mock_sleep):
        with patch("colony_langchain.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_posts.side_effect = [
                ColonyAPIError("overloaded", status=503),
                {"posts": [{"id": "1", "title": "Recovered", "post_type": "discussion", "score": 0, "comment_count": 0, "author": {"username": "a"}, "colony": {"name": "b"}}]},
            ]
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_search_posts"].invoke({"query": "test"})
            assert "Recovered" in result
            assert mock_client.get_posts.call_count == 2
