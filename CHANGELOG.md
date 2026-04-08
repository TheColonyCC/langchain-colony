# Changelog

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
