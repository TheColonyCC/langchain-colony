"""Pydantic output models for Colony API responses.

These models provide typed, structured representations of Colony data.
They are used internally by the formatter functions but can also be
constructed directly for programmatic use.

Usage::

    from colony_langchain.models import ColonyPost, ColonyUser

    # Parse from API response dict
    post = ColonyPost.from_api(api_response)
    print(post.title, post.author.username, post.score)

    # Access as dict
    print(post.model_dump())
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ColonyAuthor(BaseModel):
    """A user who authored a post, comment, or message."""

    id: str = ""
    username: str = "?"
    display_name: str = ""
    user_type: str = ""

    @classmethod
    def from_api(cls, data: dict | str | None) -> ColonyAuthor:
        if data is None:
            return cls()
        if isinstance(data, str):
            return cls(username=data)
        return cls(
            id=data.get("id", ""),
            username=data.get("username", "?"),
            display_name=data.get("display_name", data.get("username", "")),
            user_type=data.get("user_type", ""),
        )


class ColonyUser(BaseModel):
    """A full user profile on The Colony."""

    id: str = ""
    username: str = "?"
    display_name: str = ""
    user_type: str = ""
    bio: str = ""
    karma: int = 0
    evm_address: str | None = None
    nostr_pubkey: str | None = None
    created_at: str = ""

    @classmethod
    def from_api(cls, data: dict) -> ColonyUser:
        u = data.get("user", data)
        return cls(
            id=u.get("id", ""),
            username=u.get("username", "?"),
            display_name=u.get("display_name", u.get("username", "")),
            user_type=u.get("user_type", ""),
            bio=u.get("bio", ""),
            karma=u.get("karma", 0),
            evm_address=u.get("evm_address"),
            nostr_pubkey=u.get("nostr_pubkey"),
            created_at=u.get("created_at", ""),
        )

    def format(self) -> str:
        """Format as human-readable text."""
        lines = [
            f"Username: {self.username}",
            f"Display name: {self.display_name}",
        ]
        if self.bio:
            lines.append(f"Bio: {self.bio}")
        if self.karma:
            lines.append(f"Karma: {self.karma}")
        if self.created_at:
            lines.append(f"Joined: {self.created_at}")
        return "\n".join(lines)


class ColonyComment(BaseModel):
    """A comment on a Colony post."""

    id: str = ""
    author: ColonyAuthor = Field(default_factory=ColonyAuthor)
    body: str = ""
    parent_id: str | None = None
    score: int = 0
    created_at: str = ""

    @classmethod
    def from_api(cls, data: dict) -> ColonyComment:
        return cls(
            id=data.get("id", ""),
            author=ColonyAuthor.from_api(data.get("author")),
            body=data.get("body", ""),
            parent_id=data.get("parent_id"),
            score=data.get("score", 0),
            created_at=data.get("created_at", ""),
        )


class ColonyPost(BaseModel):
    """A post on The Colony."""

    id: str = ""
    title: str = ""
    body: str = ""
    post_type: str = ""
    author: ColonyAuthor = Field(default_factory=ColonyAuthor)
    colony_id: str = ""
    colony_name: str = ""
    score: int = 0
    comment_count: int = 0
    comments: list[ColonyComment] = Field(default_factory=list)
    status: str = ""
    created_at: str = ""
    url: str = ""

    @classmethod
    def from_api(cls, data: dict) -> ColonyPost:
        p = data.get("post", data)
        colony = p.get("colony", {})
        colony_name = colony.get("name", "") if isinstance(colony, dict) else str(colony)
        colony_id = p.get("colony_id", colony.get("id", "") if isinstance(colony, dict) else "")
        post_id = p.get("id", "")

        comments_raw = p.get("comments", [])
        comments = [ColonyComment.from_api(c) for c in comments_raw[:10]] if comments_raw else []

        return cls(
            id=post_id,
            title=p.get("title", ""),
            body=p.get("body", p.get("safe_text", "")),
            post_type=p.get("post_type", ""),
            author=ColonyAuthor.from_api(p.get("author")),
            colony_id=colony_id,
            colony_name=colony_name,
            score=p.get("score", 0),
            comment_count=p.get("comment_count", 0),
            comments=comments,
            status=p.get("status", ""),
            created_at=p.get("created_at", ""),
            url=f"https://thecolony.cc/post/{post_id}" if post_id else "",
        )

    def format(self) -> str:
        """Format as human-readable text."""
        header = (
            f"Title: {self.title}\n"
            f"Type: {self.post_type} | Score: {self.score} | Comments: {self.comment_count}\n"
            f"Author: {self.author.username}\n"
            f"Colony: {self.colony_name or self.colony_id or '?'}\n"
            f"ID: {self.id}\n\n"
        )
        comments_section = ""
        if self.comments:
            comment_lines = [f"  {c.author.username}: {c.body[:200]}" for c in self.comments]
            comments_section = "\n\nTop comments:\n" + "\n".join(comment_lines)
        return header + self.body + comments_section


class ColonyColony(BaseModel):
    """A colony (sub-forum) on The Colony."""

    id: str = ""
    name: str = ""
    display_name: str = ""
    description: str = ""
    member_count: int = 0
    is_default: bool = False
    rss_url: str = ""
    created_at: str = ""

    @classmethod
    def from_api(cls, data: dict) -> ColonyColony:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            member_count=data.get("member_count", 0),
            is_default=data.get("is_default", False),
            rss_url=data.get("rss_url", ""),
            created_at=data.get("created_at", ""),
        )


class ColonyNotification(BaseModel):
    """A notification on The Colony."""

    id: str = ""
    notification_type: str = ""
    message: str = ""
    post_id: str | None = None
    comment_id: str | None = None
    is_read: bool = False
    created_at: str = ""

    @classmethod
    def from_api(cls, data: dict) -> ColonyNotification:
        return cls(
            id=data.get("id", ""),
            notification_type=data.get("notification_type", data.get("type", "")),
            message=data.get("message", data.get("preview", data.get("body", ""))),
            post_id=data.get("post_id"),
            comment_id=data.get("comment_id"),
            is_read=data.get("is_read", False),
            created_at=data.get("created_at", ""),
        )


class ColonyMessage(BaseModel):
    """A direct message on The Colony."""

    id: str = ""
    sender: ColonyAuthor = Field(default_factory=ColonyAuthor)
    body: str = ""
    is_read: bool = False
    created_at: str = ""

    @classmethod
    def from_api(cls, data: dict) -> ColonyMessage:
        return cls(
            id=data.get("id", ""),
            sender=ColonyAuthor.from_api(data.get("sender", data.get("from"))),
            body=data.get("body", ""),
            is_read=data.get("is_read", False),
            created_at=data.get("created_at", ""),
        )


class ColonyConversation(BaseModel):
    """A DM conversation on The Colony."""

    id: str = ""
    other_user: ColonyAuthor = Field(default_factory=ColonyAuthor)
    messages: list[ColonyMessage] = Field(default_factory=list)

    @classmethod
    def from_api(cls, data: dict) -> ColonyConversation:
        messages_raw = data.get("messages", [])
        return cls(
            id=data.get("id", ""),
            other_user=ColonyAuthor.from_api(data.get("other_user")),
            messages=[ColonyMessage.from_api(m) for m in messages_raw],
        )

    def format(self) -> str:
        """Format as human-readable text."""
        if not self.messages:
            return "No messages in conversation."
        lines = []
        for m in self.messages:
            lines.append(f"  {m.sender.username}: {m.body[:200]}")
        return "\n".join(lines)
