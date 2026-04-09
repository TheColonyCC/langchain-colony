"""Tests for error handling and retry delegation.

Retry semantics now live inside ``colony_sdk.ColonyClient`` — this layer just
catches whatever the SDK ultimately raises and formats it as an LLM-friendly
string. These tests cover:

* ``_friendly_error`` formatting against the SDK's typed exception classes
* ``_ColonyBaseTool._api`` / ``_aapi`` catching and formatting errors at the
  tool boundary (so a failed call returns a string instead of crashing the
  agent)
* ``ColonyToolkit`` handing the ``RetryConfig`` straight to ``ColonyClient``
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from colony_sdk import (
    ColonyAPIError,
    ColonyAuthError,
    ColonyConflictError,
    ColonyNetworkError,
    ColonyNotFoundError,
    ColonyRateLimitError,
    ColonyServerError,
    ColonyValidationError,
    RetryConfig,
)

from langchain_colony import ColonyToolkit
from langchain_colony.tools import RetryConfig as ToolsRetryConfig
from langchain_colony.tools import _friendly_error

# ── Friendly error messages ─────────────────────────────────────────


class TestFriendlyError:
    """``_friendly_error`` should lean on ``str(exc)`` from SDK typed errors —
    those exception messages already contain the human-readable hint and the
    server's ``detail`` field, so all we add is the ``Error (status) [code]``
    prefix that LLMs and LangSmith traces can grep on."""

    def test_auth_error(self):
        err = ColonyAuthError(
            "get_me failed: bad token (unauthorized — check your API key)",
            status=401,
            code="AUTH_INVALID_TOKEN",
        )
        msg = _friendly_error(err)
        assert "Error" in msg
        assert "401" in msg
        assert "AUTH_INVALID_TOKEN" in msg
        assert "unauthorized" in msg.lower()

    def test_forbidden(self):
        err = ColonyAPIError(
            "delete_post failed: nope (forbidden — your account lacks permission for this operation)",
            status=403,
            code="FORBIDDEN",
        )
        msg = _friendly_error(err)
        assert "403" in msg
        assert "permission" in msg.lower()

    def test_not_found(self):
        err = ColonyNotFoundError(
            "get_post failed: missing (not found — the resource doesn't exist or has been deleted)",
            status=404,
        )
        msg = _friendly_error(err)
        assert "404" in msg
        assert "not found" in msg.lower()

    def test_conflict(self):
        err = ColonyConflictError(
            "vote_post failed: already voted (conflict — already done, or state mismatch)",
            status=409,
            code="CONFLICT",
        )
        msg = _friendly_error(err)
        assert "409" in msg
        assert "conflict" in msg.lower()

    def test_validation(self):
        err = ColonyValidationError(
            "create_post failed: title too short (validation failed — check field requirements)",
            status=422,
            code="VALIDATION_ERROR",
        )
        msg = _friendly_error(err)
        assert "422" in msg
        assert "validation failed" in msg.lower()

    def test_rate_limit(self):
        err = ColonyRateLimitError(
            "vote_post failed: slow down (rate limited — slow down and retry after the backoff window)",
            status=429,
            code="RATE_LIMIT_VOTE_HOURLY",
            retry_after=7,
        )
        msg = _friendly_error(err)
        assert "429" in msg
        assert "rate limited" in msg.lower()
        assert "RATE_LIMIT_VOTE_HOURLY" in msg
        # retry_after stays accessible for higher-level backoff logic
        assert err.retry_after == 7

    def test_server_error(self):
        err = ColonyServerError(
            "get_posts failed: oops (server error — Colony API failure, usually transient)",
            status=500,
        )
        msg = _friendly_error(err)
        assert "500" in msg
        assert "server error" in msg.lower()

    def test_network_error_suppresses_status_zero(self):
        """Network errors carry status=0 — that's an internal sentinel, not a
        real HTTP code. We must NOT surface a misleading ``(0)`` to LLMs."""
        err = ColonyNetworkError(
            "Colony API network error: Connection refused",
            status=0,
            response={},
        )
        msg = _friendly_error(err)
        assert "(0)" not in msg
        assert "Connection refused" in msg

    def test_plain_exception(self):
        """Anything that escapes the SDK's typed-error layer (e.g. a bug in
        post-processing) still gets caught at the tool boundary."""
        msg = _friendly_error(ValueError("unexpected"))
        assert "Error" in msg
        assert "unexpected" in msg


# ── Tool-level error handling ───────────────────────────────────────


class TestToolErrorHandling:
    """End-to-end: a tool whose underlying client raises a typed error should
    return the formatted string instead of bubbling the exception up to the
    agent."""

    def test_search_returns_friendly_error(self):
        with patch("langchain_colony.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_posts.side_effect = ColonyAuthError(
                "get_posts failed: bad token (unauthorized — check your API key)",
                status=401,
                code="AUTH_INVALID_TOKEN",
            )
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_search_posts"].invoke({"query": "test"})
            assert "Error" in result
            assert "401" in result
            assert "unauthorized" in result.lower()

    def test_create_post_returns_friendly_error(self):
        with patch("langchain_colony.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.create_post.side_effect = ColonyValidationError(
                "create_post failed: title too short (validation failed — check field requirements)",
                status=422,
                code="VALIDATION_ERROR",
            )
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_create_post"].invoke({"title": "X", "body": "Y"})
            assert "422" in result
            assert "validation failed" in result.lower()

    def test_get_post_not_found(self):
        with patch("langchain_colony.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_post.side_effect = ColonyNotFoundError(
                "get_post failed: gone (not found — the resource doesn't exist or has been deleted)",
                status=404,
            )
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_get_post"].invoke({"post_id": "nonexistent"})
            assert "not found" in result.lower()
            assert "404" in result

    def test_vote_rate_limited(self):
        with patch("langchain_colony.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.vote_post.side_effect = ColonyRateLimitError(
                "vote_post failed: too fast (rate limited — slow down and retry after the backoff window)",
                status=429,
                code="RATE_LIMIT_VOTE_HOURLY",
                retry_after=5,
            )
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_vote_on_post"].invoke({"post_id": "p-1", "value": 1})
            assert "rate limited" in result.lower()
            assert "RATE_LIMIT_VOTE_HOURLY" in result

    def test_delete_forbidden(self):
        with patch("langchain_colony.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.delete_post.side_effect = ColonyAPIError(
                "delete_post failed: not yours (forbidden — your account lacks permission for this operation)",
                status=403,
                code="FORBIDDEN",
            )
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_delete_post"].invoke({"post_id": "p-1"})
            assert "permission" in result.lower()
            assert "403" in result

    def test_async_error_handling(self):
        with patch("langchain_colony.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_me.side_effect = ColonyAuthError(
                "get_me failed: expired (unauthorized — check your API key)",
                status=401,
                code="AUTH_TOKEN_EXPIRED",
            )
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = asyncio.run(tools["colony_get_me"].ainvoke({}))
            assert "401" in result
            assert "unauthorized" in result.lower()

    def test_unexpected_exception_caught(self):
        """A non-SDK exception that escapes the client (e.g. a JSON-decode bug
        in post-processing) still becomes a formatted string at the tool
        boundary, not a crash."""
        with patch("langchain_colony.toolkit.ColonyClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_posts.side_effect = ValueError("unexpected")
            toolkit = ColonyToolkit(api_key="col_test")
            tools = {t.name: t for t in toolkit.get_tools()}
            result = tools["colony_search_posts"].invoke({"query": "test"})
            assert "Error" in result
            assert "unexpected" in result


# ── Configurable retry ──────────────────────────────────────────────


class TestRetryConfig:
    """``RetryConfig`` is now re-exported straight from ``colony_sdk`` —
    the SDK enforces the policy inside ``ColonyClient``."""

    def test_retry_config_is_sdk_class(self):
        """``langchain_colony.tools.RetryConfig`` is the same object as
        ``colony_sdk.RetryConfig`` — we re-export rather than re-wrap so
        retries stay in lockstep with the SDK."""
        assert ToolsRetryConfig is RetryConfig

    def test_defaults(self):
        cfg = RetryConfig()
        # SDK defaults: 2 retries (3 total attempts), 1s base, 10s cap,
        # retries on 429 + 5xx gateway errors.
        assert cfg.max_retries == 2
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 10.0
        assert 429 in cfg.retry_on
        assert 502 in cfg.retry_on

    def test_custom_values(self):
        cfg = RetryConfig(max_retries=5, base_delay=0.5, max_delay=30.0)
        assert cfg.max_retries == 5
        assert cfg.base_delay == 0.5
        assert cfg.max_delay == 30.0

    def test_disable_retry(self):
        cfg = RetryConfig(max_retries=0)
        assert cfg.max_retries == 0

    def test_toolkit_passes_retry_to_client(self):
        """The toolkit must hand the ``RetryConfig`` down to ``ColonyClient``
        because retry semantics now live inside the SDK."""
        with patch("langchain_colony.toolkit.ColonyClient") as MockClient:
            cfg = RetryConfig(max_retries=7, base_delay=0.1)
            ColonyToolkit(api_key="col_test", retry=cfg)
            kwargs = MockClient.call_args.kwargs
            assert kwargs["retry"] is cfg

    def test_toolkit_omits_retry_when_unset(self):
        """When the caller doesn't specify retry, we don't override the
        SDK's default — we just don't pass the kwarg."""
        with patch("langchain_colony.toolkit.ColonyClient") as MockClient:
            ColonyToolkit(api_key="col_test")
            kwargs = MockClient.call_args.kwargs
            assert "retry" not in kwargs

    def test_tools_no_longer_have_retry_config_attribute(self):
        """The per-tool ``retry_config`` field has been removed — retry now
        lives entirely at the client layer."""
        with patch("langchain_colony.toolkit.ColonyClient"):
            toolkit = ColonyToolkit(api_key="col_test")
            tool = toolkit.get_tools()[0]
            assert not hasattr(tool, "retry_config") or tool.retry_config is None  # type: ignore[attr-defined]

    def test_toolkit_remembers_retry_config(self):
        """Backwards-compat introspection: ``toolkit.retry_config`` still
        reflects what was passed in (even though tools no longer use it)."""
        with patch("langchain_colony.toolkit.ColonyClient"):
            cfg = RetryConfig(max_retries=4)
            toolkit = ColonyToolkit(api_key="col_test", retry=cfg)
            assert toolkit.retry_config is cfg

    def test_toolkit_default_retry_config_is_none(self):
        with patch("langchain_colony.toolkit.ColonyClient"):
            toolkit = ColonyToolkit(api_key="col_test")
            assert toolkit.retry_config is None
