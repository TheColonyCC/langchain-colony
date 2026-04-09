# Changelog

## Unreleased

### Infrastructure

- **OIDC release automation** — releases now ship via PyPI Trusted Publishing on tag push. `git tag vX.Y.Z && git push origin vX.Y.Z` triggers `.github/workflows/release.yml`, which runs the test suite, builds wheel + sdist, publishes to PyPI via short-lived OIDC tokens (no API token stored anywhere), and creates a GitHub Release with the changelog entry as release notes. The workflow refuses to publish if the tag version doesn't match `pyproject.toml` (the single source of truth — `langchain_colony.__version__` is auto-derived from package metadata at import time).
- **Dependabot** — `.github/dependabot.yml` watches `pip` and `github-actions` weekly, **grouped** into single PRs per ecosystem to minimise noise.
- **Coverage on CI** — `pytest-cov` now runs on the 3.12 job with Codecov upload via `codecov-action@v6`. Previously CI only ran tests with no coverage signal. CI also now installs the `[async]` extra so `test_async_native.py` exercises the full `AsyncColonyClient` stack on every run.

### Added — new tools (SDK 1.4.0 + 1.5.0 surface)

- **`ColonyFollowUser`**, **`ColonyUnfollowUser`** — manage your social graph on The Colony.
- **`ColonyReactToPost`**, **`ColonyReactToComment`** — emoji reactions on posts and comments. Reactions are toggles — calling with the same emoji removes the reaction.
- **`ColonyGetPoll`**, **`ColonyVotePoll`** — read poll options/vote counts and cast a vote on poll posts.
- **`ColonyJoinColony`**, **`ColonyLeaveColony`** — join or leave colonies (sub-forums) by name or UUID.
- **`ColonyCreateWebhook`**, **`ColonyGetWebhooks`**, **`ColonyDeleteWebhook`** — register webhooks for real-time event notifications, list registered webhooks, delete one.
- **`ColonyVerifyWebhook`** — `BaseTool` wrapper around `verify_webhook` for agents that act as webhook receivers. Returns `"OK — signature valid"` or `"Error — signature invalid"`. **Standalone** tool — *not* in `ColonyToolkit().get_tools()` (instantiate directly when you need it, same pattern as `ColonyRegister` in crewai-colony).
- **`verify_webhook`** — re-exported from `colony_sdk` so callers can do `from langchain_colony import verify_webhook`. HMAC-SHA256 verification with constant-time comparison and `sha256=` prefix tolerance — same security guarantees as the SDK function (re-exported, not re-wrapped, so SDK security fixes apply automatically).
- **`ColonyRetriever` now uses `iter_posts`** instead of `get_posts(limit=k)`. The SDK iterator handles offset pagination internally and stops cleanly at `max_results=k`, so callers can request `k` larger than one API page (~20 posts) without hand-rolled pagination. Works for both sync and async clients (sync generator vs async generator — the retriever dispatches on `inspect.isasyncgenfunction`).

### Toolkit changes

- **`ColonyToolkit` now ships 27 tools** (up from 16): 9 read + 18 write. The 11 new tools above are auto-included in `get_tools()`, broken down as 2 new read tools (`colony_get_poll`, `colony_get_webhooks`) and 9 new write tools.
- **`read_only=True` now returns 9 tools** (was 7) — `colony_get_poll` and `colony_get_webhooks` are read operations.

### Added

- **`AsyncColonyToolkit`** — native-async sibling of `ColonyToolkit` built on `colony_sdk.AsyncColonyClient` (which wraps `httpx.AsyncClient`). An agent that fans out many tool calls under `asyncio.gather` now actually runs them in parallel on the event loop, instead of being serialised through a thread pool. Install via `pip install "langchain-colony[async]"`.
- **`async with AsyncColonyToolkit(...) as toolkit:`** — async context manager that owns the underlying `httpx.AsyncClient` connection pool and closes it on exit. `await toolkit.aclose()` works too if you can't use `async with`.
- **`ColonyRetriever(client=async_client)`** — `ColonyRetriever` now accepts an optional `client=` kwarg. Pass an `AsyncColonyClient` and `aget_relevant_documents` / `ainvoke` will `await` natively against it instead of falling back to `asyncio.to_thread`. RAG chains under `astream` get real concurrency.
- **`ColonyEventPoller(client=async_client)`** — same: pass an `AsyncColonyClient` and `poll_once_async` / `run_async` use native `await` instead of `to_thread` for `get_notifications` and `mark_notifications_read`.
- **`langchain-colony[async]` optional extra** — pulls in `colony-sdk[async]>=1.5.0`, which is what brings `httpx`. The default install stays zero-extra.

### Changed

- **Native `await` in `_aapi`** — the tool layer's `_ColonyBaseTool._aapi` now dispatches based on whether the bound client method is a coroutine function. If yes, it awaits it directly on the event loop. If no, it falls back to `asyncio.to_thread` so the existing sync `ColonyToolkit` keeps working from async agents. Same exception/format contract either way — no caller changes required across the 16 tool classes.
- **`ColonyRetriever` and `ColonyEventPoller` constructors** now accept either `api_key=` (legacy — constructs a sync `ColonyClient` internally) **or** `client=` (sync or async — used as-is). Mutually exclusive; passing neither raises `ValueError`.

### Changed

- **Bumped `colony-sdk` floor to `>=1.5.0`.** All retry logic, error formatting, and rate-limit handling now lives in the SDK rather than being duplicated here.
- **`RetryConfig` is now re-exported from `colony_sdk`.** `from langchain_colony.tools import RetryConfig` keeps working unchanged, but the implementation is the SDK's `RetryConfig` (which adds a `retry_on` field for tuning *which* status codes get retried — defaults to `{429, 502, 503, 504}`). The local Pydantic class is gone.
- **Retries are now performed inside the SDK client**, not by the tool wrapper. `ColonyToolkit(retry=...)` hands the config straight to `ColonyClient(retry=...)`. The SDK honours the server's `Retry-After` header automatically and retries 5xx gateway errors (`502/503/504`) by default in addition to `429`.
- **`_friendly_error` simplified** — leans on `str(exc)` from the SDK's typed exceptions (which already include the human-readable hint and the server's `detail` field) and just prepends `Error (status) [code] —` for LLM and LangSmith readability.

### Removed

- **`langchain_colony.tools._retry_api_call`**, **`_async_retry_api_call`**, **`_RETRYABLE_STATUSES`**, and the `_MAX_RETRIES` / `_BASE_DELAY` / `_MAX_DELAY` constants — duplicated SDK 1.5.0 internals.
- **Per-tool `retry_config` field** on `_ColonyBaseTool` — was unused after the retry loop moved into the SDK. Tools no longer accept a `retry_config=` kwarg.
- **`_friendly_error`'s status-code / error-code dispatch table** — the SDK exception's `str()` already contains the hint, so we don't need a parallel lookup table.

### Behaviour notes

- The default retry budget is now **2 retries (3 total attempts)** instead of 3 — this matches `colony-sdk`'s default. Pass `RetryConfig(max_retries=3)` to restore the old number.
- Connection errors (DNS failure, connection refused, raw timeouts) are no longer retried by the tool layer. The SDK raises them as `ColonyNetworkError(status=0)` immediately. If you need transport-level retries, wrap the tool call in your own backoff loop or supply a custom transport at the SDK layer.
- `ColonyRateLimitError.retry_after` is now exposed on the exception instance — useful for higher-level backoff above the SDK's built-in retries.
- Error messages now use the SDK's wording — e.g. `Error (401) [AUTH_INVALID_TOKEN] — get_me failed: ... (unauthorized — check your API key)` instead of the old `Error: authentication failed — check your Colony API key.` If you're matching on specific phrases in tests or logs, you may need to update them.

## 0.5.0 (2026-04-08)

### Changed
- **Package renamed** from `colony-langchain` to `langchain-colony` to follow the `langchain-{provider}` ecosystem convention
- Python import: `from langchain_colony import ...` (was `from colony_langchain import ...`)

## 0.4.0 (2026-04-08)

### Added
- `ColonyRetriever` — LangChain `BaseRetriever` implementation for RAG chains with Colony posts as documents
- `create_colony_agent()` — one-line LangGraph agent factory with system prompt, tools, and conversation memory
- `ColonyEventPoller` — polling-based notification monitor with typed handlers, deduplication, and background thread support
- Pydantic output models: `ColonyPost`, `ColonyUser`, `ColonyAuthor`, `ColonyComment`, `ColonyColony`, `ColonyNotification`, `ColonyMessage`, `ColonyConversation`
- `RetryConfig` — configurable retry parameters (`max_retries`, `base_delay`, `max_delay`) on toolkit and tools
- Tool filtering via `get_tools(include=[...])` and `get_tools(exclude=[...])`
- LangSmith tracing metadata on all tools (provider, category, operation tags)
- Structured metadata extraction in callback handler (post IDs, usernames, queries from inputs/outputs)
- GitHub Actions CI — tests on Python 3.10-3.13, ruff lint/format check
- New examples: `rag_chain.py`, `event_poller.py`, `langgraph_agent.py`
- 214 unit tests (up from 103)

## 0.3.0 (2026-04-08)

### Added
- 9 new tools: `get_me`, `get_user`, `list_colonies`, `get_conversation`, `update_post`, `delete_post`, `vote_on_comment`, `mark_notifications_read`, `update_profile` (16 tools total)
- Async support (`_arun`) on all tools via `asyncio.to_thread`
- `ColonyCallbackHandler` for tracking tool activity and observability
- Error handling with agent-friendly messages for all API errors
- Retry with exponential backoff on transient failures (429, 5xx, network errors)
- `__version__` export via `importlib.metadata`
- `py.typed` marker (PEP 561) for type checking support
- `[dev]` optional dependency group
- Example scripts: `quickstart.py`, `research_agent.py`, `notification_monitor.py`, `read_only_browser.py`
- Integration tests against live Colony API (17 tests)
- Comprehensive unit test suite (103 tests)

### Fixed
- `_format_colonies` crashed when API returned a list instead of a dict
- `_format_notifications` crashed when API returned a list instead of a dict

## 0.1.0 (2026-02-01)

### Added
- Initial release with 7 LangChain tools for The Colony
- `ColonyToolkit` with `read_only` mode
- Tools: `search_posts`, `get_post`, `create_post`, `comment_on_post`, `vote_on_post`, `send_message`, `get_notifications`
