# --- duplicate-write guard (2026-07-19) -------------------------------------------
class TestCommentIdempotency:
    """A graph that emits the same comment twice must not create two comments.

    Observed in the langford dogfood agent roughly monthly since May 2026: the model
    calls colony_comment_on_post, does not treat the result as terminal, and calls
    again. Prompt-level guards did not stop it, which is expected — prompting is a
    request, not a constraint.
    """

    def _tool(self):
        from unittest.mock import MagicMock

        from langchain_colony.tools import ColonyCommentOnPost

        ColonyCommentOnPost._sent.clear()  # process-scoped cache; isolate the test
        client = MagicMock()
        client.create_comment.return_value = {"id": "c-1"}
        return ColonyCommentOnPost(client=client), client

    def test_identical_second_call_does_not_reach_the_api(self):
        tool, client = self._tool()
        first = tool._run(post_id="p1", body="hello")
        second = tool._run(post_id="p1", body="hello")
        assert client.create_comment.call_count == 1
        assert "c-1" in first
        assert "already posted" in second

    def test_a_different_body_still_posts(self):
        """The guard must key on content, not merely on the post."""
        tool, client = self._tool()
        tool._run(post_id="p1", body="hello")
        tool._run(post_id="p1", body="a genuinely different point")
        assert client.create_comment.call_count == 2

    def test_same_body_on_a_different_post_still_posts(self):
        tool, client = self._tool()
        tool._run(post_id="p1", body="hello")
        tool._run(post_id="p2", body="hello")
        assert client.create_comment.call_count == 2

    def test_threading_is_part_of_the_key(self):
        """Same text as a top-level comment and as a threaded reply are two acts."""
        tool, client = self._tool()
        tool._run(post_id="p1", body="hello")
        tool._run(post_id="p1", body="hello", parent_id="cmt-9")
        assert client.create_comment.call_count == 2

    def test_a_failed_first_call_is_not_cached(self):
        """Only successes are remembered — a transient error must stay retryable."""
        from unittest.mock import MagicMock

        from langchain_colony.tools import ColonyCommentOnPost

        ColonyCommentOnPost._sent.clear()
        client = MagicMock()
        client.create_comment.side_effect = [Exception("boom"), {"id": "c-2"}]
        tool = ColonyCommentOnPost(client=client)
        tool._run(post_id="p1", body="hello")  # error path returns a string
        second = tool._run(post_id="p1", body="hello")  # must genuinely retry
        assert client.create_comment.call_count == 2
        assert "c-2" in second
