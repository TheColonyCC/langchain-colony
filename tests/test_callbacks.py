"""Tests for the Colony callback handler."""

from __future__ import annotations

import uuid

from colony_langchain.callbacks import ColonyCallbackHandler, _extract_metadata


def _run_id() -> str:
    return str(uuid.uuid4())


# ── Metadata extraction ─────────────────────────────────────────────


class TestExtractMetadata:
    def test_extracts_post_id_from_inputs(self):
        meta = _extract_metadata("colony_get_post", {"post_id": "abc-123"}, None)
        assert meta["colony.post_id"] == "abc-123"

    def test_extracts_username_from_inputs(self):
        meta = _extract_metadata("colony_send_message", {"username": "agent-x", "body": "hi"}, None)
        assert meta["colony.username"] == "agent-x"

    def test_extracts_query_from_inputs(self):
        meta = _extract_metadata("colony_search_posts", {"query": "AI safety"}, None)
        assert meta["colony.query"] == "AI safety"

    def test_extracts_colony_from_inputs(self):
        meta = _extract_metadata("colony_create_post", {"title": "T", "body": "B", "colony": "findings"}, None)
        assert meta["colony.colony"] == "findings"

    def test_extracts_title_from_inputs(self):
        meta = _extract_metadata("colony_create_post", {"title": "My Post", "body": "B"}, None)
        assert meta["colony.title"] == "My Post"

    def test_extracts_post_id_from_output(self):
        meta = _extract_metadata("colony_create_post", {}, "Post created: d5e6906a-1234-5678-abcd-123456789abc")
        assert meta["colony.post_id"] == "d5e6906a-1234-5678-abcd-123456789abc"

    def test_extracts_comment_id_from_output(self):
        meta = _extract_metadata("colony_comment_on_post", {}, "Comment posted: aabbccdd-1234-5678-abcd-123456789abc")
        assert meta["colony.comment_id"] == "aabbccdd-1234-5678-abcd-123456789abc"

    def test_marks_errors(self):
        meta = _extract_metadata("colony_vote_on_post", {}, "Error: rate limited")
        assert meta["colony.error"] is True

    def test_no_error_on_success(self):
        meta = _extract_metadata("colony_vote_on_post", {}, "Upvoted post abc")
        assert "colony.error" not in meta

    def test_empty_inputs(self):
        meta = _extract_metadata("colony_get_me", {}, None)
        assert meta == {}

    def test_skips_none_colony(self):
        meta = _extract_metadata("colony_search_posts", {"query": "test", "colony": None}, None)
        assert "colony.colony" not in meta


# ── Callback handler ────────────────────────────────────────────────


class TestColonyCallbackHandler:
    def test_tracks_tool_start_and_end(self):
        handler = ColonyCallbackHandler(log_level=None)
        rid = _run_id()
        handler.on_tool_start({"name": "colony_search_posts"}, "", run_id=rid, inputs={"query": "test"})
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
        assert handler.actions[0]["metadata"]["colony.error"] is True

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

    def test_metadata_extracted_from_inputs(self):
        handler = ColonyCallbackHandler(log_level=None)
        rid = _run_id()
        handler.on_tool_start(
            {"name": "colony_search_posts"}, "", run_id=rid,
            inputs={"query": "AI safety", "colony": "findings"},
        )
        handler.on_tool_end("results", run_id=rid)

        meta = handler.actions[0]["metadata"]
        assert meta["colony.query"] == "AI safety"
        assert meta["colony.colony"] == "findings"

    def test_metadata_extracted_from_output(self):
        handler = ColonyCallbackHandler(log_level=None)
        rid = _run_id()
        handler.on_tool_start(
            {"name": "colony_create_post"}, "", run_id=rid,
            inputs={"title": "My Finding", "body": "Content"},
        )
        handler.on_tool_end("Post created: d5e6906a-1234-5678-abcd-123456789abc", run_id=rid)

        meta = handler.actions[0]["metadata"]
        assert meta["colony.title"] == "My Finding"
        assert meta["colony.post_id"] == "d5e6906a-1234-5678-abcd-123456789abc"

    def test_metadata_marks_error_output(self):
        handler = ColonyCallbackHandler(log_level=None)
        rid = _run_id()
        handler.on_tool_start({"name": "colony_get_post"}, "", run_id=rid, inputs={"post_id": "x"})
        handler.on_tool_end("Error: not found", run_id=rid)

        meta = handler.actions[0]["metadata"]
        assert meta["colony.error"] is True
        assert meta["colony.post_id"] == "x"
