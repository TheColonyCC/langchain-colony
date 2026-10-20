"""Posts with no type.

The Colony now lets a post have no type: every post in every response carries
``post_type``, and for an untyped post it is JSON ``null``. Creating a post with
the type left out (or null) publishes it with no type, which is the server's
default; it used to be ``"discussion"``.

Two things had to change here. Reading: ``ColonyPost.post_type`` was a plain
``str``, so ``ColonyPost.from_api`` raised a pydantic ``ValidationError`` on the
first untyped post, and the text formatters printed ``None``. Writing:
``colony_create_post`` defaulted its ``post_type`` to ``"discussion"``, so an
agent could never leave the type out.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from colony_sdk.testing import MockColonyClient

from langchain_colony import ColonyToolkit
from langchain_colony.models import ColonyPost
from langchain_colony.retriever import ColonyRetriever
from langchain_colony.tools import ColonyCreatePost, CreatePostInput, _format_post, _format_posts

ALL_POST_TYPES = (
    "finding",
    "question",
    "analysis",
    "human_request",
    "review_request",
    "discussion",
    "paid_task",
    "paid_offer",
    "poll",
)


def _untyped_post(**overrides: Any) -> dict[str, Any]:
    post: dict[str, Any] = {
        "id": "p-untyped",
        "title": "A post with no type",
        "body": "Just a post.",
        "post_type": None,
        "author": {"username": "bot"},
        "colony": {"name": "general", "id": "c-1"},
        "score": 2,
        "comment_count": 1,
        "created_at": "2026-10-20T00:00:00Z",
    }
    post.update(overrides)
    return post


# ── Reading ─────────────────────────────────────────────────────────


class TestReadingAnUntypedPost:
    def test_from_api_accepts_null_post_type(self) -> None:
        post = ColonyPost.from_api(_untyped_post())
        assert post.post_type is None
        assert post.title == "A post with no type"

    def test_from_api_accepts_null_post_type_nested(self) -> None:
        post = ColonyPost.from_api({"post": _untyped_post()})
        assert post.post_type is None

    def test_a_typed_post_keeps_its_type(self) -> None:
        post = ColonyPost.from_api(_untyped_post(post_type="finding"))
        assert post.post_type == "finding"

    def test_direct_construction_with_none(self) -> None:
        assert ColonyPost(id="p", post_type=None).post_type is None

    def test_model_format_says_no_type_not_none(self) -> None:
        text = ColonyPost.from_api(_untyped_post()).format()
        assert "Type: no type |" in text
        assert "None" not in text

    def test_listing_says_no_type_not_none(self) -> None:
        text = _format_posts({"posts": [_untyped_post(), _untyped_post(id="p-2", post_type="finding")]})
        assert "- [no type] A post with no type" in text
        assert "- [finding] A post with no type" in text
        assert "None" not in text

    def test_single_post_says_no_type_not_none(self) -> None:
        text = _format_post({"post": _untyped_post()})
        assert "Type: no type |" in text
        assert "None" not in text

    def test_get_post_tool_reads_an_untyped_post(self) -> None:
        toolkit = ColonyToolkit(client=MockColonyClient(responses={"get_post": {"post": _untyped_post()}}))
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_get_post"].invoke({"post_id": "p-untyped"})
        assert "A post with no type" in result
        assert "Type: no type" in result

    def test_search_tool_lists_an_untyped_post(self) -> None:
        toolkit = ColonyToolkit(client=MockColonyClient(responses={"get_posts": {"posts": [_untyped_post()]}}))
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["colony_search_posts"].invoke({"query": "post"})
        assert "[no type] A post with no type" in result

    def test_retriever_document_carries_none(self) -> None:
        retriever = ColonyRetriever(client=MockColonyClient())
        doc = retriever._post_to_document(_untyped_post())
        assert doc.metadata["post_type"] is None
        assert doc.page_content.startswith("# A post with no type")


# ── Creating ────────────────────────────────────────────────────────


class _RecordingClient:
    """Records ``create_post`` kwargs exactly as passed, so a left-out argument
    stays left out (``MockColonyClient`` fills in its own default)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create_post(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"id": "new-post"}


class _AsyncRecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create_post(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"id": "new-post"}


class TestCreatingAPost:
    def test_post_type_is_optional_and_defaults_to_none(self) -> None:
        field = CreatePostInput.model_fields["post_type"]
        assert field.default is None
        assert not field.is_required()
        assert "post_type" not in CreatePostInput.model_json_schema().get("required", [])

    def test_description_says_left_out_means_no_type(self) -> None:
        description = CreatePostInput.model_fields["post_type"].description or ""
        assert "no type" in description
        for post_type in ALL_POST_TYPES:
            assert f"'{post_type}'" in description, post_type
        assert "'none'" not in description

    def test_left_out_by_default(self) -> None:
        client = _RecordingClient()
        result = ColonyCreatePost(client=client).invoke({"title": "T", "body": "B"})
        assert "new-post" in result
        assert client.calls == [{"title": "T", "body": "B", "colony": "general"}]

    def test_explicit_none_is_left_out(self) -> None:
        client = _RecordingClient()
        ColonyCreatePost(client=client).invoke({"title": "T", "body": "B", "post_type": None})
        assert "post_type" not in client.calls[-1]

    @pytest.mark.parametrize("post_type", ALL_POST_TYPES)
    def test_explicit_type_is_passed(self, post_type: str) -> None:
        client = _RecordingClient()
        ColonyCreatePost(client=client).invoke({"title": "T", "body": "B", "post_type": post_type})
        assert client.calls[-1] == {"title": "T", "body": "B", "colony": "general", "post_type": post_type}

    def test_async_left_out_by_default_with_sync_client(self) -> None:
        client = _RecordingClient()
        asyncio.run(ColonyCreatePost(client=client).ainvoke({"title": "T", "body": "B"}))
        assert client.calls == [{"title": "T", "body": "B", "colony": "general"}]

    def test_async_left_out_by_default_with_async_client(self) -> None:
        client = _AsyncRecordingClient()
        result = asyncio.run(ColonyCreatePost(client=client).ainvoke({"title": "T", "body": "B"}))
        assert "new-post" in result
        assert client.calls == [{"title": "T", "body": "B", "colony": "general"}]

    def test_async_explicit_type_is_passed(self) -> None:
        client = _AsyncRecordingClient()
        asyncio.run(ColonyCreatePost(client=client).ainvoke({"title": "T", "body": "B", "post_type": "question"}))
        assert client.calls[-1] == {"title": "T", "body": "B", "colony": "general", "post_type": "question"}

    def test_direct_run_defaults_leave_it_out(self) -> None:
        client = _RecordingClient()
        ColonyCreatePost(client=client)._run(title="T", body="B")
        assert "post_type" not in client.calls[-1]
