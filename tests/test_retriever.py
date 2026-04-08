"""Tests for the Colony retriever."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from colony_langchain.retriever import ColonyRetriever


def _make_retriever(**kwargs):
    with patch("colony_langchain.retriever.ColonyClient"):
        return ColonyRetriever(api_key="col_test", **kwargs)


def _sample_posts(n=3):
    return {
        "posts": [
            {
                "id": f"post-{i}",
                "title": f"Post {i}",
                "body": f"Body of post {i} with some content.",
                "post_type": "discussion",
                "score": 10 - i,
                "comment_count": i,
                "author": {"username": f"agent-{i}"},
                "colony": {"name": "general"},
                "created_at": f"2026-01-0{i + 1}T00:00:00Z",
            }
            for i in range(n)
        ]
    }


class TestRetrieverBasic:
    def test_returns_documents(self):
        retriever = _make_retriever()
        retriever.client.get_posts.return_value = _sample_posts(3)
        docs = retriever.invoke("test query")
        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)

    def test_document_content(self):
        retriever = _make_retriever()
        retriever.client.get_posts.return_value = _sample_posts(1)
        docs = retriever.invoke("test")
        assert "# Post 0" in docs[0].page_content
        assert "Body of post 0" in docs[0].page_content

    def test_document_metadata(self):
        retriever = _make_retriever()
        retriever.client.get_posts.return_value = _sample_posts(1)
        docs = retriever.invoke("test")
        meta = docs[0].metadata
        assert meta["post_id"] == "post-0"
        assert meta["title"] == "Post 0"
        assert meta["author"] == "agent-0"
        assert meta["colony"] == "general"
        assert meta["post_type"] == "discussion"
        assert meta["score"] == 10
        assert meta["url"] == "https://thecolony.cc/post/post-0"
        assert meta["source"] == "thecolony.cc"

    def test_document_id_set(self):
        retriever = _make_retriever()
        retriever.client.get_posts.return_value = _sample_posts(1)
        docs = retriever.invoke("test")
        assert docs[0].id == "post-0"

    def test_empty_results(self):
        retriever = _make_retriever()
        retriever.client.get_posts.return_value = {"posts": []}
        docs = retriever.invoke("nonexistent")
        assert docs == []

    def test_list_response_format(self):
        """API may return a plain list instead of {"posts": [...]}."""
        retriever = _make_retriever()
        retriever.client.get_posts.return_value = [
            {
                "id": "p-1",
                "title": "List Post",
                "body": "Content.",
                "post_type": "finding",
                "score": 5,
                "comment_count": 0,
                "author": {"username": "bot"},
                "colony": {"name": "findings"},
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]
        docs = retriever.invoke("test")
        assert len(docs) == 1
        assert "List Post" in docs[0].page_content


class TestRetrieverParams:
    def test_k_limits_results(self):
        retriever = _make_retriever(k=2)
        retriever.client.get_posts.return_value = _sample_posts(5)
        docs = retriever.invoke("test")
        assert len(docs) == 2

    def test_passes_colony_filter(self):
        retriever = _make_retriever(colony="findings")
        retriever.client.get_posts.return_value = {"posts": []}
        retriever.invoke("test")
        retriever.client.get_posts.assert_called_once_with(
            search="test", colony="findings", post_type=None, sort="top", limit=5
        )

    def test_passes_post_type_filter(self):
        retriever = _make_retriever(post_type="analysis")
        retriever.client.get_posts.return_value = {"posts": []}
        retriever.invoke("test")
        retriever.client.get_posts.assert_called_once_with(
            search="test", colony=None, post_type="analysis", sort="top", limit=5
        )

    def test_passes_sort(self):
        retriever = _make_retriever(sort="new")
        retriever.client.get_posts.return_value = {"posts": []}
        retriever.invoke("test")
        call_kwargs = retriever.client.get_posts.call_args.kwargs
        assert call_kwargs["sort"] == "new"

    def test_passes_k_as_limit(self):
        retriever = _make_retriever(k=10)
        retriever.client.get_posts.return_value = {"posts": []}
        retriever.invoke("test")
        call_kwargs = retriever.client.get_posts.call_args.kwargs
        assert call_kwargs["limit"] == 10


class TestRetrieverComments:
    def test_include_comments(self):
        retriever = _make_retriever(include_comments=True)
        retriever.client.get_posts.return_value = _sample_posts(1)
        retriever.client.get_post.return_value = {
            "id": "post-0",
            "title": "Post 0",
            "body": "Body.",
            "comments": [
                {"author": {"username": "commenter"}, "body": "Great post!"},
                {"author": {"username": "other"}, "body": "I agree."},
            ],
        }
        docs = retriever.invoke("test")
        assert "## Comments" in docs[0].page_content
        assert "commenter" in docs[0].page_content
        assert "Great post!" in docs[0].page_content

    def test_no_comments_by_default(self):
        retriever = _make_retriever()
        retriever.client.get_posts.return_value = _sample_posts(1)
        docs = retriever.invoke("test")
        assert "## Comments" not in docs[0].page_content
        retriever.client.get_post.assert_not_called()

    def test_comments_error_does_not_fail(self):
        retriever = _make_retriever(include_comments=True)
        retriever.client.get_posts.return_value = _sample_posts(1)
        retriever.client.get_post.side_effect = Exception("API error")
        docs = retriever.invoke("test")
        assert len(docs) == 1  # still returns the doc without comments


class TestRetrieverAsync:
    def test_async_returns_documents(self):
        retriever = _make_retriever()
        retriever.client.get_posts.return_value = _sample_posts(2)
        docs = asyncio.run(retriever.ainvoke("async query"))
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

    def test_async_with_comments(self):
        retriever = _make_retriever(include_comments=True)
        retriever.client.get_posts.return_value = _sample_posts(1)
        retriever.client.get_post.return_value = {
            "id": "post-0",
            "comments": [{"author": {"username": "async-commenter"}, "body": "Async!"}],
        }
        docs = asyncio.run(retriever.ainvoke("test"))
        assert "async-commenter" in docs[0].page_content

    def test_async_empty_results(self):
        retriever = _make_retriever()
        retriever.client.get_posts.return_value = {"posts": []}
        docs = asyncio.run(retriever.ainvoke("nothing"))
        assert docs == []


class TestRetrieverEdgeCases:
    def test_missing_body_uses_safe_text(self):
        retriever = _make_retriever()
        retriever.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p-1",
                    "title": "Safe",
                    "safe_text": "Safe text content.",
                    "post_type": "discussion",
                    "score": 0,
                    "comment_count": 0,
                    "author": {"username": "bot"},
                    "colony": {"name": "general"},
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ]
        }
        docs = retriever.invoke("test")
        assert "Safe text content." in docs[0].page_content

    def test_missing_author_fallback(self):
        retriever = _make_retriever()
        retriever.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p-1",
                    "title": "No Author",
                    "body": "Content.",
                    "post_type": "discussion",
                    "score": 0,
                    "comment_count": 0,
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ]
        }
        docs = retriever.invoke("test")
        assert docs[0].metadata["author"] == "?"

    def test_string_colony_in_metadata(self):
        """When colony is a string ID instead of a dict."""
        retriever = _make_retriever()
        retriever.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p-1",
                    "title": "T",
                    "body": "B",
                    "post_type": "discussion",
                    "score": 0,
                    "comment_count": 0,
                    "author": {"username": "a"},
                    "colony": "some-uuid",
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ]
        }
        docs = retriever.invoke("test")
        assert docs[0].metadata["colony"] == "some-uuid"
