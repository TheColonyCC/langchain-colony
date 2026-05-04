"""Tests for the Colony callback handler."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

from langchain_colony.callbacks import (
    ColonyCallbackHandler,
    FinishReasonCallback,
    _extract_finish_reasons,
    _extract_metadata,
)


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
            {"name": "colony_search_posts"},
            "",
            run_id=rid,
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
            {"name": "colony_create_post"},
            "",
            run_id=rid,
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


# ── FinishReasonCallback ────────────────────────────────────────────


def _chat_response(reason: str | None) -> SimpleNamespace:
    """Build a minimal LLMResult-like object with a ChatGeneration whose
    message carries response_metadata['finish_reason']=reason.
    """
    meta = {"finish_reason": reason} if reason is not None else {}
    message = SimpleNamespace(response_metadata=meta, usage_metadata=None)
    chat_gen = SimpleNamespace(message=message, generation_info=None)
    return SimpleNamespace(generations=[[chat_gen]])


def _completion_response(reason: str | None) -> SimpleNamespace:
    """Build a completion-shape LLMResult: Generation with generation_info."""
    info = {"finish_reason": reason} if reason is not None else {}
    gen = SimpleNamespace(message=None, generation_info=info)
    return SimpleNamespace(generations=[[gen]])


class TestExtractFinishReasons:
    def test_chat_path(self):
        assert _extract_finish_reasons(_chat_response("stop")) == ["stop"]

    def test_completion_path(self):
        assert _extract_finish_reasons(_completion_response("length")) == ["length"]

    def test_missing_metadata_returns_empty(self):
        assert _extract_finish_reasons(_chat_response(None)) == []

    def test_no_generations_returns_empty(self):
        assert _extract_finish_reasons(SimpleNamespace(generations=[])) == []

    def test_object_without_generations_attribute_returns_empty(self):
        assert _extract_finish_reasons(SimpleNamespace()) == []

    def test_multiple_generations_in_one_batch(self):
        gens = [
            SimpleNamespace(
                message=SimpleNamespace(response_metadata={"finish_reason": "stop"}, usage_metadata=None),
                generation_info=None,
            ),
            SimpleNamespace(
                message=SimpleNamespace(response_metadata={"finish_reason": "length"}, usage_metadata=None),
                generation_info=None,
            ),
        ]
        assert _extract_finish_reasons(SimpleNamespace(generations=[gens])) == ["stop", "length"]

    def test_falls_back_to_stop_reason_alias(self):
        message = SimpleNamespace(response_metadata={"stop_reason": "length"}, usage_metadata=None)
        gen = SimpleNamespace(message=message, generation_info=None)
        assert _extract_finish_reasons(SimpleNamespace(generations=[[gen]])) == ["length"]


class TestFinishReasonCallback:
    def test_initial_state(self):
        cb = FinishReasonCallback()
        assert cb.last_finish_reason is None
        assert cb.length_count == 0
        assert cb.total_count == 0

    def test_stop_increments_total_only(self):
        cb = FinishReasonCallback(log_level=None)
        cb.on_llm_end(_chat_response("stop"))
        assert cb.last_finish_reason == "stop"
        assert cb.length_count == 0
        assert cb.total_count == 1

    def test_length_increments_both_counters(self):
        cb = FinishReasonCallback(log_level=None)
        cb.on_llm_end(_chat_response("length"))
        assert cb.last_finish_reason == "length"
        assert cb.length_count == 1
        assert cb.total_count == 1

    def test_warning_emitted_on_length(self, caplog):
        cb = FinishReasonCallback()
        with caplog.at_level("WARNING", logger="langchain_colony"):
            cb.on_llm_end(_chat_response("length"))
        assert any("finish_reason=length" in record.message for record in caplog.records)

    def test_no_warning_emitted_on_stop(self, caplog):
        cb = FinishReasonCallback()
        with caplog.at_level("WARNING", logger="langchain_colony"):
            cb.on_llm_end(_chat_response("stop"))
        assert not any("finish_reason=length" in record.message for record in caplog.records)

    def test_warning_silenced_when_log_level_none(self, caplog):
        cb = FinishReasonCallback(log_level=None)
        with caplog.at_level("WARNING", logger="langchain_colony"):
            cb.on_llm_end(_chat_response("length"))
        assert cb.length_count == 1
        assert not any("finish_reason=length" in record.message for record in caplog.records)

    def test_missing_finish_reason_is_silent(self, caplog):
        cb = FinishReasonCallback()
        with caplog.at_level("WARNING", logger="langchain_colony"):
            cb.on_llm_end(_chat_response(None))
        assert cb.last_finish_reason is None
        assert cb.total_count == 0
        assert cb.length_count == 0

    def test_reset_clears_state(self):
        cb = FinishReasonCallback(log_level=None)
        cb.on_llm_end(_chat_response("length"))
        cb.on_llm_end(_chat_response("stop"))
        cb.reset()
        assert cb.last_finish_reason is None
        assert cb.length_count == 0
        assert cb.total_count == 0

    def test_multiple_calls_track_last(self):
        cb = FinishReasonCallback(log_level=None)
        cb.on_llm_end(_chat_response("stop"))
        cb.on_llm_end(_chat_response("length"))
        cb.on_llm_end(_chat_response("stop"))
        assert cb.last_finish_reason == "stop"
        assert cb.length_count == 1
        assert cb.total_count == 3
