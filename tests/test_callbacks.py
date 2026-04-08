"""Tests for the Colony callback handler."""

from __future__ import annotations

import uuid

from colony_langchain.callbacks import ColonyCallbackHandler


def _run_id() -> str:
    return str(uuid.uuid4())


class TestColonyCallbackHandler:
    def test_tracks_tool_start_and_end(self):
        handler = ColonyCallbackHandler(log_level=None)
        rid = _run_id()
        handler.on_tool_start({"name": "colony_search_posts"}, "", run_id=rid)
        handler.on_tool_end("3 posts found", run_id=rid)

        assert len(handler.actions) == 1
        assert handler.actions[0]["tool"] == "colony_search_posts"
        assert handler.actions[0]["output"] == "3 posts found"
        assert handler.actions[0]["error"] is None
        assert handler.actions[0]["is_write"] is False

    def test_tracks_write_tools(self):
        handler = ColonyCallbackHandler(log_level=None)
        rid = _run_id()
        handler.on_tool_start({"name": "colony_create_post"}, "", run_id=rid)
        handler.on_tool_end("Post created: abc-123", run_id=rid)

        assert handler.actions[0]["is_write"] is True

    def test_tracks_errors(self):
        handler = ColonyCallbackHandler(log_level=None)
        rid = _run_id()
        handler.on_tool_start({"name": "colony_vote_on_post"}, "", run_id=rid)
        handler.on_tool_error(RuntimeError("API timeout"), run_id=rid)

        assert len(handler.actions) == 1
        assert handler.actions[0]["error"] == "API timeout"
        assert handler.actions[0]["output"] is None

    def test_ignores_non_colony_tools(self):
        handler = ColonyCallbackHandler(log_level=None)
        rid = _run_id()
        handler.on_tool_start({"name": "google_search"}, "", run_id=rid)
        handler.on_tool_end("results", run_id=rid)

        assert len(handler.actions) == 0

    def test_summary_no_actions(self):
        handler = ColonyCallbackHandler(log_level=None)
        assert handler.summary() == "No Colony actions recorded."

    def test_summary_with_actions(self):
        handler = ColonyCallbackHandler(log_level=None)

        # A read
        rid1 = _run_id()
        handler.on_tool_start({"name": "colony_search_posts"}, "", run_id=rid1)
        handler.on_tool_end("results", run_id=rid1)

        # A write
        rid2 = _run_id()
        handler.on_tool_start({"name": "colony_create_post"}, "", run_id=rid2)
        handler.on_tool_end("Post created: x", run_id=rid2)

        summary = handler.summary()
        assert "2 actions" in summary
        assert "1 reads" in summary
        assert "1 writes" in summary
        assert "colony_create_post" in summary

    def test_summary_with_errors(self):
        handler = ColonyCallbackHandler(log_level=None)
        rid = _run_id()
        handler.on_tool_start({"name": "colony_send_message"}, "", run_id=rid)
        handler.on_tool_error(RuntimeError("fail"), run_id=rid)

        summary = handler.summary()
        assert "Errors: 1" in summary
        assert "FAILED" in summary

    def test_reset_clears_state(self):
        handler = ColonyCallbackHandler(log_level=None)
        rid = _run_id()
        handler.on_tool_start({"name": "colony_search_posts"}, "", run_id=rid)
        handler.on_tool_end("results", run_id=rid)

        assert len(handler.actions) == 1
        handler.reset()
        assert len(handler.actions) == 0
        assert handler.summary() == "No Colony actions recorded."

    def test_multiple_concurrent_tools(self):
        handler = ColonyCallbackHandler(log_level=None)
        rid1 = _run_id()
        rid2 = _run_id()

        handler.on_tool_start({"name": "colony_search_posts"}, "", run_id=rid1)
        handler.on_tool_start({"name": "colony_get_post"}, "", run_id=rid2)
        handler.on_tool_end("post detail", run_id=rid2)
        handler.on_tool_end("search results", run_id=rid1)

        assert len(handler.actions) == 2
        assert handler.actions[0]["tool"] == "colony_get_post"
        assert handler.actions[1]["tool"] == "colony_search_posts"
