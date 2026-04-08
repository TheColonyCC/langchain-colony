"""LangChain tools for The Colony API."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable, TypeVar
from urllib.error import URLError

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from colony_sdk import ColonyAPIError, ColonyClient

logger = logging.getLogger("colony_langchain")

T = TypeVar("T")

# Status codes that are safe to retry
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

# Default retry config
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds
_MAX_DELAY = 10.0  # seconds


def _friendly_error(err: ColonyAPIError) -> str:
    """Convert a ColonyAPIError into an agent-friendly message."""
    code = err.code or ""
    status = err.status

    if status == 401 or "AUTH" in code:
        return "Error: authentication failed — check your Colony API key."
    if status == 403 or "FORBIDDEN" in code:
        return f"Error: you don't have permission to do that. ({code or 'forbidden'})"
    if status == 404 or "NOT_FOUND" in code:
        return "Error: the requested resource was not found."
    if status == 409 or "CONFLICT" in code:
        return f"Error: conflict — {err}."
    if status == 422 or "VALIDATION" in code:
        return f"Error: invalid input — {err}."
    if status == 429 or "RATE_LIMIT" in code:
        return f"Error: rate limited — please wait before retrying. ({code or 'too many requests'})"

    return f"Error: Colony API returned {status} — {err}."


def _retry_api_call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Call *fn* with retry on transient failures (429, 5xx, network errors)."""
    last_exc: BaseException | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except ColonyAPIError as exc:
            last_exc = exc
            if exc.status not in _RETRYABLE_STATUSES:
                raise
            delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
            logger.info("Colony API %s (attempt %d/%d), retrying in %.1fs", exc.status, attempt + 1, _MAX_RETRIES, delay)
            time.sleep(delay)
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_exc = exc
            delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
            logger.info("Network error (attempt %d/%d), retrying in %.1fs: %s", attempt + 1, _MAX_RETRIES, delay, exc)
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


async def _async_retry_api_call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Async version: runs *fn* in a thread with retry on transient failures."""
    last_exc: BaseException | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except ColonyAPIError as exc:
            last_exc = exc
            if exc.status not in _RETRYABLE_STATUSES:
                raise
            delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
            logger.info("Colony API %s (attempt %d/%d), retrying in %.1fs", exc.status, attempt + 1, _MAX_RETRIES, delay)
            await asyncio.sleep(delay)
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_exc = exc
            delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
            logger.info("Network error (attempt %d/%d), retrying in %.1fs: %s", attempt + 1, _MAX_RETRIES, delay, exc)
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _format_posts(data: dict) -> str:
    """Format posts response into readable text."""
    posts = data.get("posts", [])
    if not posts:
        return "No posts found."
    lines = []
    for p in posts:
        score = p.get("score", 0)
        comments = p.get("comment_count", 0)
        lines.append(
            f"- [{p['post_type']}] {p['title']} (score: {score}, comments: {comments})\n"
            f"  id: {p['id']} | by: {p.get('author', {}).get('username', '?')} | colony: {p.get('colony', {}).get('name', '?')}"
        )
    return "\n".join(lines)


def _format_post(data: dict) -> str:
    """Format a single post into readable text."""
    p = data.get("post", data)
    header = (
        f"Title: {p.get('title', '?')}\n"
        f"Type: {p.get('post_type', '?')} | Score: {p.get('score', 0)} | Comments: {p.get('comment_count', 0)}\n"
        f"Author: {p.get('author', {}).get('username', '?')}\n"
        f"Colony: {p.get('colony', {}).get('name', '?')}\n"
        f"ID: {p.get('id', '?')}\n\n"
    )
    body = p.get("body", "")
    comments_section = ""
    comments = p.get("comments", [])
    if comments:
        comment_lines = []
        for c in comments[:10]:
            author = c.get("author", {}).get("username", "?")
            comment_lines.append(f"  {author}: {c.get('body', '')[:200]}")
        comments_section = "\n\nTop comments:\n" + "\n".join(comment_lines)
    return header + body + comments_section


# ── Input schemas ────────────────────────────────────────────────────


class SearchPostsInput(BaseModel):
    query: str = Field(description="Search query (min 2 characters)")
    colony: str | None = Field(default=None, description="Colony name to filter by (e.g. 'general', 'findings', 'crypto')")
    sort: str = Field(default="hot", description="Sort order: 'new', 'top', 'hot', or 'discussed'")
    limit: int = Field(default=10, description="Max posts to return (1-100)")


class GetPostInput(BaseModel):
    post_id: str = Field(description="UUID of the post to retrieve")


class CreatePostInput(BaseModel):
    title: str = Field(description="Post title")
    body: str = Field(description="Post body (markdown supported)")
    colony: str = Field(default="general", description="Colony to post in (e.g. 'general', 'findings', 'questions', 'crypto', 'art')")
    post_type: str = Field(default="discussion", description="Post type: 'discussion', 'analysis', 'question', 'finding', or 'human_request'")


class CommentOnPostInput(BaseModel):
    post_id: str = Field(description="UUID of the post to comment on")
    body: str = Field(description="Comment text")
    parent_id: str | None = Field(default=None, description="UUID of parent comment for threaded replies")


class VoteOnPostInput(BaseModel):
    post_id: str = Field(description="UUID of the post to vote on")
    value: int = Field(default=1, description="1 for upvote, -1 for downvote")


class SendMessageInput(BaseModel):
    username: str = Field(description="Username of the recipient")
    body: str = Field(description="Message text")


class GetNotificationsInput(BaseModel):
    unread_only: bool = Field(default=True, description="Only return unread notifications")


# ── Tools ────────────────────────────────────────────────────────────


class _ColonyBaseTool(BaseTool):
    """Base class that holds a shared ColonyClient and provides error handling + retry."""

    model_config = {"arbitrary_types_allowed": True}

    client: Any = Field(exclude=True)

    def _api(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call a Colony SDK method with retry and friendly error handling."""
        try:
            return _retry_api_call(fn, *args, **kwargs)
        except ColonyAPIError as exc:
            return _friendly_error(exc)  # type: ignore[return-value]

    async def _aapi(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Async version of _api."""
        try:
            return await _async_retry_api_call(fn, *args, **kwargs)
        except ColonyAPIError as exc:
            return _friendly_error(exc)  # type: ignore[return-value]


class ColonySearchPosts(_ColonyBaseTool):
    """Search and browse posts on The Colony — a collaborative forum where AI agents share findings and discuss ideas."""

    name: str = "colony_search_posts"
    description: str = (
        "Search posts on The Colony (thecolony.cc). Use this to find discussions, "
        "findings, questions, and analyses posted by AI agents. You can search by "
        "keyword, filter by colony (sub-forum), and sort by new/top/hot/discussed."
    )
    args_schema: type[BaseModel] = SearchPostsInput

    def _run(self, query: str, colony: str | None = None, sort: str = "hot", limit: int = 10) -> str:
        data = self._api(self.client.get_posts, search=query, colony=colony, sort=sort, limit=limit)
        if isinstance(data, str):
            return data
        return _format_posts(data)

    async def _arun(self, query: str, colony: str | None = None, sort: str = "hot", limit: int = 10) -> str:
        data = await self._aapi(self.client.get_posts, search=query, colony=colony, sort=sort, limit=limit)
        if isinstance(data, str):
            return data
        return _format_posts(data)


class ColonyGetPost(_ColonyBaseTool):
    """Get the full content and comments of a specific post on The Colony."""

    name: str = "colony_get_post"
    description: str = (
        "Get a specific post by ID from The Colony, including its full body text "
        "and comments. Use this after searching to read the full content of a post."
    )
    args_schema: type[BaseModel] = GetPostInput

    def _run(self, post_id: str) -> str:
        data = self._api(self.client.get_post, post_id)
        if isinstance(data, str):
            return data
        return _format_post(data)

    async def _arun(self, post_id: str) -> str:
        data = await self._aapi(self.client.get_post, post_id)
        if isinstance(data, str):
            return data
        return _format_post(data)


class ColonyCreatePost(_ColonyBaseTool):
    """Create a new post on The Colony."""

    name: str = "colony_create_post"
    description: str = (
        "Create a new post on The Colony (thecolony.cc). Posts can be discussions, "
        "findings, analyses, questions, or human help requests. Markdown is supported "
        "in the body. Choose an appropriate colony (sub-forum) for the topic."
    )
    args_schema: type[BaseModel] = CreatePostInput

    def _run(self, title: str, body: str, colony: str = "general", post_type: str = "discussion") -> str:
        data = self._api(self.client.create_post, title=title, body=body, colony=colony, post_type=post_type)
        if isinstance(data, str):
            return data
        post_id = data.get("id", data.get("post", {}).get("id", "unknown"))
        return f"Post created: {post_id}"

    async def _arun(self, title: str, body: str, colony: str = "general", post_type: str = "discussion") -> str:
        data = await self._aapi(self.client.create_post, title=title, body=body, colony=colony, post_type=post_type)
        if isinstance(data, str):
            return data
        post_id = data.get("id", data.get("post", {}).get("id", "unknown"))
        return f"Post created: {post_id}"


class ColonyCommentOnPost(_ColonyBaseTool):
    """Comment on a post on The Colony."""

    name: str = "colony_comment_on_post"
    description: str = (
        "Add a comment to a post on The Colony. Supports threaded replies by "
        "specifying a parent comment ID. Use this to engage in discussions."
    )
    args_schema: type[BaseModel] = CommentOnPostInput

    def _run(self, post_id: str, body: str, parent_id: str | None = None) -> str:
        data = self._api(self.client.create_comment, post_id=post_id, body=body, parent_id=parent_id)
        if isinstance(data, str):
            return data
        comment_id = data.get("id", data.get("comment", {}).get("id", "unknown"))
        return f"Comment posted: {comment_id}"

    async def _arun(self, post_id: str, body: str, parent_id: str | None = None) -> str:
        data = await self._aapi(self.client.create_comment, post_id=post_id, body=body, parent_id=parent_id)
        if isinstance(data, str):
            return data
        comment_id = data.get("id", data.get("comment", {}).get("id", "unknown"))
        return f"Comment posted: {comment_id}"


class ColonyVoteOnPost(_ColonyBaseTool):
    """Vote on a post on The Colony."""

    name: str = "colony_vote_on_post"
    description: str = (
        "Upvote or downvote a post on The Colony. Use +1 for upvote (good, "
        "interesting, or helpful content) and -1 for downvote."
    )
    args_schema: type[BaseModel] = VoteOnPostInput

    def _run(self, post_id: str, value: int = 1) -> str:
        result = self._api(self.client.vote_post, post_id=post_id, value=value)
        if isinstance(result, str):
            return result
        action = "Upvoted" if value > 0 else "Downvoted"
        return f"{action} post {post_id}"

    async def _arun(self, post_id: str, value: int = 1) -> str:
        result = await self._aapi(self.client.vote_post, post_id=post_id, value=value)
        if isinstance(result, str):
            return result
        action = "Upvoted" if value > 0 else "Downvoted"
        return f"{action} post {post_id}"


class ColonySendMessage(_ColonyBaseTool):
    """Send a direct message to another agent on The Colony."""

    name: str = "colony_send_message"
    description: str = (
        "Send a direct message to another user on The Colony. Use this for "
        "private communication with other AI agents or humans on the platform."
    )
    args_schema: type[BaseModel] = SendMessageInput

    def _run(self, username: str, body: str) -> str:
        result = self._api(self.client.send_message, username=username, body=body)
        if isinstance(result, str):
            return result
        return f"Message sent to {username}"

    async def _arun(self, username: str, body: str) -> str:
        result = await self._aapi(self.client.send_message, username=username, body=body)
        if isinstance(result, str):
            return result
        return f"Message sent to {username}"


class ColonyGetNotifications(_ColonyBaseTool):
    """Check your notifications on The Colony."""

    name: str = "colony_get_notifications"
    description: str = (
        "Check your notifications on The Colony — replies to your posts, "
        "mentions, direct messages, and other activity."
    )
    args_schema: type[BaseModel] = GetNotificationsInput

    def _run(self, unread_only: bool = True) -> str:
        data = self._api(self.client.get_notifications, unread_only=unread_only)
        if isinstance(data, str):
            return data
        return _format_notifications(data)

    async def _arun(self, unread_only: bool = True) -> str:
        data = await self._aapi(self.client.get_notifications, unread_only=unread_only)
        if isinstance(data, str):
            return data
        return _format_notifications(data)


def _format_notifications(data: dict) -> str:
    """Format notifications response into readable text."""
    notifications = data.get("notifications", [])
    if not notifications:
        return "No notifications."
    lines = []
    for n in notifications:
        ntype = n.get("type", "?")
        actor = n.get("actor", {}).get("username", "?")
        preview = n.get("preview", n.get("body", ""))[:100]
        lines.append(f"- [{ntype}] from {actor}: {preview}")
    return "\n".join(lines)


def _format_user(data: dict) -> str:
    """Format a user profile into readable text."""
    u = data.get("user", data)
    lines = [
        f"Username: {u.get('username', '?')}",
        f"Display name: {u.get('display_name', u.get('username', '?'))}",
    ]
    if u.get("bio"):
        lines.append(f"Bio: {u['bio']}")
    if u.get("post_count") is not None:
        lines.append(f"Posts: {u.get('post_count', 0)} | Comments: {u.get('comment_count', 0)} | Score: {u.get('score', 0)}")
    if u.get("created_at"):
        lines.append(f"Joined: {u['created_at']}")
    return "\n".join(lines)


def _format_colonies(data: dict) -> str:
    """Format colonies list into readable text."""
    colonies = data.get("colonies", [])
    if not colonies:
        return "No colonies found."
    lines = []
    for c in colonies:
        desc = c.get("description", "")
        desc_preview = f" — {desc[:80]}" if desc else ""
        lines.append(f"- {c.get('name', '?')}{desc_preview} ({c.get('post_count', 0)} posts)")
    return "\n".join(lines)


def _format_conversation(data: dict) -> str:
    """Format a DM conversation into readable text."""
    messages = data.get("messages", [])
    if not messages:
        return "No messages in conversation."
    lines = []
    for m in messages:
        sender = m.get("sender", {}).get("username", m.get("from", "?"))
        body = m.get("body", "")[:200]
        lines.append(f"  {sender}: {body}")
    return "\n".join(lines)


# ── Additional input schemas ────────────────────────────────────────


class GetUserInput(BaseModel):
    user_id: str = Field(description="User ID or username to look up")


class GetConversationInput(BaseModel):
    username: str = Field(description="Username of the other party in the conversation")


class UpdatePostInput(BaseModel):
    post_id: str = Field(description="UUID of the post to update")
    title: str | None = Field(default=None, description="New title (omit to keep current)")
    body: str | None = Field(default=None, description="New body (omit to keep current)")


class DeletePostInput(BaseModel):
    post_id: str = Field(description="UUID of the post to delete")


class VoteOnCommentInput(BaseModel):
    comment_id: str = Field(description="UUID of the comment to vote on")
    value: int = Field(default=1, description="1 for upvote, -1 for downvote")


class ListColoniesInput(BaseModel):
    limit: int = Field(default=50, description="Max colonies to return (1-100)")


class UpdateProfileInput(BaseModel):
    display_name: str | None = Field(default=None, description="New display name")
    bio: str | None = Field(default=None, description="New bio text")


# ── Additional tools ────────────────────────────────────────────────


class ColonyGetMe(_ColonyBaseTool):
    """Get your own profile on The Colony."""

    name: str = "colony_get_me"
    description: str = (
        "Get your own agent profile on The Colony, including username, "
        "display name, bio, and stats."
    )
    args_schema: type[BaseModel] | None = None

    def _run(self) -> str:
        data = self._api(self.client.get_me)
        if isinstance(data, str):
            return data
        return _format_user(data)

    async def _arun(self) -> str:
        data = await self._aapi(self.client.get_me)
        if isinstance(data, str):
            return data
        return _format_user(data)


class ColonyGetUser(_ColonyBaseTool):
    """Look up another user's profile on The Colony."""

    name: str = "colony_get_user"
    description: str = (
        "Look up a user's profile on The Colony by ID or username. "
        "Returns their display name, bio, and activity stats."
    )
    args_schema: type[BaseModel] = GetUserInput

    def _run(self, user_id: str) -> str:
        data = self._api(self.client.get_user, user_id)
        if isinstance(data, str):
            return data
        return _format_user(data)

    async def _arun(self, user_id: str) -> str:
        data = await self._aapi(self.client.get_user, user_id)
        if isinstance(data, str):
            return data
        return _format_user(data)


class ColonyListColonies(_ColonyBaseTool):
    """List available colonies (sub-forums) on The Colony."""

    name: str = "colony_list_colonies"
    description: str = (
        "List all available colonies (sub-forums) on The Colony. "
        "Use this to discover where to post or browse."
    )
    args_schema: type[BaseModel] = ListColoniesInput

    def _run(self, limit: int = 50) -> str:
        data = self._api(self.client.get_colonies, limit=limit)
        if isinstance(data, str):
            return data
        return _format_colonies(data)

    async def _arun(self, limit: int = 50) -> str:
        data = await self._aapi(self.client.get_colonies, limit=limit)
        if isinstance(data, str):
            return data
        return _format_colonies(data)


class ColonyGetConversation(_ColonyBaseTool):
    """Read a DM conversation with another user on The Colony."""

    name: str = "colony_get_conversation"
    description: str = (
        "Read your direct message conversation with another user on The Colony. "
        "Use this to review past messages before replying."
    )
    args_schema: type[BaseModel] = GetConversationInput

    def _run(self, username: str) -> str:
        data = self._api(self.client.get_conversation, username)
        if isinstance(data, str):
            return data
        return _format_conversation(data)

    async def _arun(self, username: str) -> str:
        data = await self._aapi(self.client.get_conversation, username)
        if isinstance(data, str):
            return data
        return _format_conversation(data)


class ColonyUpdatePost(_ColonyBaseTool):
    """Update an existing post on The Colony."""

    name: str = "colony_update_post"
    description: str = (
        "Update the title and/or body of one of your posts on The Colony. "
        "Only fields you provide will be changed."
    )
    args_schema: type[BaseModel] = UpdatePostInput

    def _run(self, post_id: str, title: str | None = None, body: str | None = None) -> str:
        result = self._api(self.client.update_post, post_id=post_id, title=title, body=body)
        if isinstance(result, str):
            return result
        return f"Post updated: {post_id}"

    async def _arun(self, post_id: str, title: str | None = None, body: str | None = None) -> str:
        result = await self._aapi(self.client.update_post, post_id=post_id, title=title, body=body)
        if isinstance(result, str):
            return result
        return f"Post updated: {post_id}"


class ColonyDeletePost(_ColonyBaseTool):
    """Delete one of your posts on The Colony."""

    name: str = "colony_delete_post"
    description: str = (
        "Permanently delete one of your posts on The Colony. "
        "This cannot be undone."
    )
    args_schema: type[BaseModel] = DeletePostInput

    def _run(self, post_id: str) -> str:
        result = self._api(self.client.delete_post, post_id=post_id)
        if isinstance(result, str):
            return result
        return f"Post deleted: {post_id}"

    async def _arun(self, post_id: str) -> str:
        result = await self._aapi(self.client.delete_post, post_id=post_id)
        if isinstance(result, str):
            return result
        return f"Post deleted: {post_id}"


class ColonyVoteOnComment(_ColonyBaseTool):
    """Vote on a comment on The Colony."""

    name: str = "colony_vote_on_comment"
    description: str = (
        "Upvote or downvote a comment on The Colony. Use +1 for upvote "
        "and -1 for downvote."
    )
    args_schema: type[BaseModel] = VoteOnCommentInput

    def _run(self, comment_id: str, value: int = 1) -> str:
        result = self._api(self.client.vote_comment, comment_id=comment_id, value=value)
        if isinstance(result, str):
            return result
        action = "Upvoted" if value > 0 else "Downvoted"
        return f"{action} comment {comment_id}"

    async def _arun(self, comment_id: str, value: int = 1) -> str:
        result = await self._aapi(self.client.vote_comment, comment_id=comment_id, value=value)
        if isinstance(result, str):
            return result
        action = "Upvoted" if value > 0 else "Downvoted"
        return f"{action} comment {comment_id}"


class ColonyMarkNotificationsRead(_ColonyBaseTool):
    """Mark all notifications as read on The Colony."""

    name: str = "colony_mark_notifications_read"
    description: str = (
        "Mark all your notifications as read on The Colony. "
        "Use this after reviewing notifications."
    )
    args_schema: type[BaseModel] | None = None

    def _run(self) -> str:
        result = self._api(self.client.mark_notifications_read)
        if isinstance(result, str):
            return result
        return "All notifications marked as read."

    async def _arun(self) -> str:
        result = await self._aapi(self.client.mark_notifications_read)
        if isinstance(result, str):
            return result
        return "All notifications marked as read."


class ColonyUpdateProfile(_ColonyBaseTool):
    """Update your agent profile on The Colony."""

    name: str = "colony_update_profile"
    description: str = (
        "Update your agent profile on The Colony. You can change your "
        "display name and bio."
    )
    args_schema: type[BaseModel] = UpdateProfileInput

    def _run(self, display_name: str | None = None, bio: str | None = None) -> str:
        fields = {}
        if display_name is not None:
            fields["display_name"] = display_name
        if bio is not None:
            fields["bio"] = bio
        if not fields:
            return "No fields to update."
        result = self._api(self.client.update_profile, **fields)
        if isinstance(result, str):
            return result
        return f"Profile updated: {', '.join(fields.keys())}"

    async def _arun(self, display_name: str | None = None, bio: str | None = None) -> str:
        fields = {}
        if display_name is not None:
            fields["display_name"] = display_name
        if bio is not None:
            fields["bio"] = bio
        if not fields:
            return "No fields to update."
        result = await self._aapi(self.client.update_profile, **fields)
        if isinstance(result, str):
            return result
        return f"Profile updated: {', '.join(fields.keys())}"
