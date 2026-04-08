"""Tests for Pydantic output models."""

from __future__ import annotations

from colony_langchain.models import (
    ColonyAuthor,
    ColonyColony,
    ColonyComment,
    ColonyConversation,
    ColonyMessage,
    ColonyNotification,
    ColonyPost,
    ColonyUser,
)


class TestColonyAuthor:
    def test_from_api_dict(self):
        author = ColonyAuthor.from_api({"id": "a1", "username": "bot", "display_name": "Bot", "user_type": "agent"})
        assert author.id == "a1"
        assert author.username == "bot"
        assert author.display_name == "Bot"

    def test_from_api_string(self):
        author = ColonyAuthor.from_api("legacy-user")
        assert author.username == "legacy-user"

    def test_from_api_none(self):
        author = ColonyAuthor.from_api(None)
        assert author.username == "?"

    def test_defaults(self):
        author = ColonyAuthor()
        assert author.username == "?"


class TestColonyUser:
    def test_from_api(self):
        user = ColonyUser.from_api({
            "id": "u1",
            "username": "agent-x",
            "display_name": "Agent X",
            "user_type": "agent",
            "bio": "I research things",
            "karma": 42,
            "evm_address": "0x123",
            "created_at": "2026-01-01T00:00:00Z",
        })
        assert user.username == "agent-x"
        assert user.bio == "I research things"
        assert user.karma == 42
        assert user.evm_address == "0x123"

    def test_from_api_nested_user(self):
        user = ColonyUser.from_api({"user": {"id": "u2", "username": "nested"}})
        assert user.username == "nested"

    def test_format(self):
        user = ColonyUser.from_api({
            "username": "bot",
            "display_name": "Bot",
            "bio": "A bot",
            "karma": 10,
            "created_at": "2026-01-01",
        })
        text = user.format()
        assert "bot" in text
        assert "A bot" in text
        assert "Karma: 10" in text

    def test_format_no_bio(self):
        user = ColonyUser.from_api({"username": "minimal"})
        text = user.format()
        assert "Bio" not in text

    def test_model_dump(self):
        user = ColonyUser.from_api({"username": "bot", "karma": 5})
        d = user.model_dump()
        assert d["username"] == "bot"
        assert d["karma"] == 5


class TestColonyPost:
    def test_from_api(self):
        post = ColonyPost.from_api({
            "id": "p1",
            "title": "My Post",
            "body": "Content here.",
            "post_type": "finding",
            "author": {"username": "researcher"},
            "colony_id": "c1",
            "colony": {"name": "findings", "id": "c1"},
            "score": 10,
            "comment_count": 3,
            "status": "open",
            "created_at": "2026-01-01T00:00:00Z",
        })
        assert post.id == "p1"
        assert post.title == "My Post"
        assert post.author.username == "researcher"
        assert post.colony_name == "findings"
        assert post.score == 10
        assert post.url == "https://thecolony.cc/post/p1"

    def test_from_api_nested_post(self):
        post = ColonyPost.from_api({"post": {"id": "p2", "title": "Nested", "body": "B"}})
        assert post.title == "Nested"

    def test_from_api_with_comments(self):
        post = ColonyPost.from_api({
            "id": "p3",
            "title": "T",
            "body": "B",
            "comments": [
                {"id": "c1", "author": {"username": "commenter"}, "body": "Great!"},
                {"id": "c2", "author": {"username": "other"}, "body": "Agree."},
            ],
        })
        assert len(post.comments) == 2
        assert post.comments[0].author.username == "commenter"
        assert post.comments[0].body == "Great!"

    def test_from_api_string_colony(self):
        post = ColonyPost.from_api({"id": "p4", "title": "T", "colony": "some-uuid"})
        assert post.colony_name == "some-uuid"

    def test_from_api_safe_text_fallback(self):
        post = ColonyPost.from_api({"id": "p5", "title": "T", "safe_text": "Safe content"})
        assert post.body == "Safe content"

    def test_format(self):
        post = ColonyPost.from_api({
            "id": "p6",
            "title": "Formatted Post",
            "body": "The body text.",
            "post_type": "discussion",
            "author": {"username": "author"},
            "colony": {"name": "general"},
            "score": 5,
            "comment_count": 2,
        })
        text = post.format()
        assert "Formatted Post" in text
        assert "The body text." in text
        assert "author" in text
        assert "Score: 5" in text

    def test_format_with_comments(self):
        post = ColonyPost.from_api({
            "id": "p7",
            "title": "T",
            "body": "B",
            "comments": [{"author": {"username": "c1"}, "body": "Hello"}],
        })
        text = post.format()
        assert "Top comments:" in text
        assert "c1" in text

    def test_model_dump(self):
        post = ColonyPost.from_api({"id": "p8", "title": "T", "body": "B", "score": 3})
        d = post.model_dump()
        assert d["id"] == "p8"
        assert d["score"] == 3
        assert isinstance(d["author"], dict)


class TestColonyComment:
    def test_from_api(self):
        comment = ColonyComment.from_api({
            "id": "c1",
            "author": {"username": "commenter"},
            "body": "Nice post!",
            "parent_id": None,
            "score": 2,
        })
        assert comment.id == "c1"
        assert comment.author.username == "commenter"
        assert comment.body == "Nice post!"
        assert comment.parent_id is None

    def test_threaded_reply(self):
        comment = ColonyComment.from_api({"id": "c2", "body": "Reply", "parent_id": "c1"})
        assert comment.parent_id == "c1"


class TestColonyColony:
    def test_from_api(self):
        colony = ColonyColony.from_api({
            "id": "col1",
            "name": "findings",
            "display_name": "Findings",
            "description": "Research findings",
            "member_count": 44,
            "is_default": True,
            "rss_url": "https://thecolony.cc/c/findings/feed.rss",
        })
        assert colony.name == "findings"
        assert colony.member_count == 44
        assert colony.is_default is True


class TestColonyNotification:
    def test_from_api_new_format(self):
        notif = ColonyNotification.from_api({
            "id": "n1",
            "notification_type": "mention",
            "message": "Someone mentioned you",
            "post_id": "p1",
            "comment_id": "c1",
            "is_read": False,
        })
        assert notif.notification_type == "mention"
        assert notif.message == "Someone mentioned you"
        assert notif.post_id == "p1"

    def test_from_api_legacy_format(self):
        notif = ColonyNotification.from_api({
            "id": "n2",
            "type": "reply",
            "preview": "Thanks for sharing",
        })
        assert notif.notification_type == "reply"
        assert notif.message == "Thanks for sharing"


class TestColonyMessage:
    def test_from_api(self):
        msg = ColonyMessage.from_api({
            "id": "m1",
            "sender": {"username": "alice"},
            "body": "Hello!",
            "is_read": True,
        })
        assert msg.sender.username == "alice"
        assert msg.body == "Hello!"
        assert msg.is_read is True

    def test_from_api_legacy_from(self):
        msg = ColonyMessage.from_api({"id": "m2", "from": "legacy-user", "body": "Old format"})
        assert msg.sender.username == "legacy-user"


class TestColonyConversation:
    def test_from_api(self):
        conv = ColonyConversation.from_api({
            "id": "conv1",
            "other_user": {"username": "bob"},
            "messages": [
                {"id": "m1", "sender": {"username": "alice"}, "body": "Hi"},
                {"id": "m2", "sender": {"username": "bob"}, "body": "Hey!"},
            ],
        })
        assert conv.other_user.username == "bob"
        assert len(conv.messages) == 2

    def test_format(self):
        conv = ColonyConversation.from_api({
            "id": "conv2",
            "messages": [
                {"id": "m1", "sender": {"username": "alice"}, "body": "Hello"},
            ],
        })
        text = conv.format()
        assert "alice" in text
        assert "Hello" in text

    def test_format_empty(self):
        conv = ColonyConversation.from_api({"id": "conv3", "messages": []})
        assert conv.format() == "No messages in conversation."
