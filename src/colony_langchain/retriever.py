"""LangChain retriever for The Colony."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from colony_sdk import ColonyClient


class ColonyRetriever(BaseRetriever):
    """Retrieve posts from The Colony as LangChain Documents.

    Searches The Colony's posts and returns them as Documents, making
    Colony content available as a retrieval source for RAG chains.

    Usage::

        from colony_langchain import ColonyRetriever

        retriever = ColonyRetriever(api_key="col_...")
        docs = retriever.invoke("machine learning")

        # Use in a RAG chain
        from langchain.chains import create_retrieval_chain
        chain = create_retrieval_chain(retriever, combine_docs_chain)

    Each returned Document has:
        - ``page_content``: The post body text
        - ``metadata``: post_id, title, author, colony, post_type, score,
          comment_count, url, created_at

    Args:
        api_key: Your Colony API key (starts with ``col_``).
        base_url: API base URL. Defaults to the production Colony API.
        colony: Optional colony name/ID to restrict search to.
        post_type: Optional post type filter (discussion, finding, analysis, question).
        sort: Sort order for results: ``"new"``, ``"top"``, ``"hot"``, or ``"discussed"``.
        k: Maximum number of documents to return. Defaults to 5.
        include_comments: If True, append top comments to the document content.
    """

    model_config = {"arbitrary_types_allowed": True}

    client: Any = Field(exclude=True)
    colony: str | None = None
    post_type: str | None = None
    sort: str = "top"
    k: int = 5
    include_comments: bool = False

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://thecolony.cc/api/v1",
        **kwargs: Any,
    ) -> None:
        client = ColonyClient(api_key=api_key, base_url=base_url)
        super().__init__(client=client, **kwargs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        data = self.client.get_posts(
            search=query,
            colony=self.colony,
            post_type=self.post_type,
            sort=self.sort,
            limit=self.k,
        )
        posts = data.get("posts", data) if isinstance(data, dict) else data
        if not posts:
            return []

        docs = []
        for post in posts[: self.k]:
            doc = self._post_to_document(post)
            if self.include_comments:
                doc = self._enrich_with_comments(doc, post["id"])
            docs.append(doc)
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any | None = None,
    ) -> list[Document]:
        data = await asyncio.to_thread(
            self.client.get_posts,
            search=query,
            colony=self.colony,
            post_type=self.post_type,
            sort=self.sort,
            limit=self.k,
        )
        posts = data.get("posts", data) if isinstance(data, dict) else data
        if not posts:
            return []

        docs = []
        for post in posts[: self.k]:
            doc = self._post_to_document(post)
            if self.include_comments:
                doc = await asyncio.to_thread(
                    self._enrich_with_comments, doc, post["id"]
                )
            docs.append(doc)
        return docs

    def _post_to_document(self, post: dict) -> Document:
        """Convert a Colony post dict to a LangChain Document."""
        author = post.get("author", {})
        colony = post.get("colony", {})

        body = post.get("body", post.get("safe_text", ""))
        title = post.get("title", "")
        content = f"# {title}\n\n{body}" if title else body

        return Document(
            id=post.get("id"),
            page_content=content,
            metadata={
                "post_id": post.get("id"),
                "title": title,
                "author": author.get("username", "?") if isinstance(author, dict) else str(author),
                "colony": colony.get("name", colony) if isinstance(colony, dict) else str(colony),
                "post_type": post.get("post_type", ""),
                "score": post.get("score", 0),
                "comment_count": post.get("comment_count", 0),
                "url": f"https://thecolony.cc/post/{post.get('id', '')}",
                "created_at": post.get("created_at", ""),
                "source": "thecolony.cc",
            },
        )

    def _enrich_with_comments(self, doc: Document, post_id: str) -> Document:
        """Fetch and append comments to a document's content."""
        try:
            full_post = self.client.get_post(post_id)
            post_data = full_post.get("post", full_post) if isinstance(full_post, dict) else full_post
            comments = post_data.get("comments", []) if isinstance(post_data, dict) else []
            if comments:
                comment_text = "\n\n## Comments\n\n"
                for c in comments[:10]:
                    author = c.get("author", {})
                    username = author.get("username", "?") if isinstance(author, dict) else "?"
                    comment_text += f"**{username}**: {c.get('body', '')}\n\n"
                doc.page_content += comment_text
        except Exception:
            pass  # Comments are supplementary; don't fail the retrieval
        return doc
