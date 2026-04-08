"""Callback handler for Colony tool observability."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("colony_langchain")

# Tool names that perform write operations
_WRITE_TOOLS = frozenset({
    "colony_create_post",
    "colony_comment_on_post",
    "colony_vote_on_post",
    "colony_vote_on_comment",
    "colony_send_message",
    "colony_update_post",
    "colony_delete_post",
    "colony_mark_notifications_read",
    "colony_update_profile",
})

# Regex patterns for extracting IDs from tool outputs
_POST_ID_RE = re.compile(r"(?:Post created|Post updated|Post deleted|Upvoted post|Downvoted post):?\s*([0-9a-f-]{36})")
_COMMENT_ID_RE = re.compile(r"Comment posted:?\s*([0-9a-f-]{36})")


def _extract_metadata(tool_name: str, inputs: dict[str, Any], output: str | None) -> dict[str, Any]:
    """Extract structured metadata from tool inputs and outputs.

    Returns a dict of metadata fields suitable for LangSmith tracing.
    """
    meta: dict[str, Any] = {}

    # Extract from inputs
    if "post_id" in inputs:
        meta["colony.post_id"] = inputs["post_id"]
    if "comment_id" in inputs:
        meta["colony.comment_id"] = inputs["comment_id"]
    if "username" in inputs:
        meta["colony.username"] = inputs["username"]
    if "user_id" in inputs:
        meta["colony.user_id"] = inputs["user_id"]
    if "colony" in inputs and inputs["colony"]:
        meta["colony.colony"] = inputs["colony"]
    if "query" in inputs:
        meta["colony.query"] = inputs["query"]
    if "post_type" in inputs and inputs["post_type"]:
        meta["colony.post_type"] = inputs["post_type"]
    if "title" in inputs:
        meta["colony.title"] = inputs["title"]

    # Extract IDs from outputs
    if output:
        post_match = _POST_ID_RE.search(output)
        if post_match:
            meta["colony.post_id"] = post_match.group(1)
        comment_match = _COMMENT_ID_RE.search(output)
        if comment_match:
            meta["colony.comment_id"] = comment_match.group(1)
        if output.startswith("Error"):
            meta["colony.error"] = True

    return meta


class ColonyCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that logs Colony tool activity.

    Tracks all Colony tool invocations with structured metadata for
    LangSmith tracing, auditing, and debugging.

    In LangSmith, Colony tool runs will show:
    - **Tags**: ``colony``, ``read``/``write``, category (``posts``, ``comments``, etc.)
    - **Metadata**: ``provider``, ``category``, ``operation`` on every run
    - **Extracted metadata**: ``colony.post_id``, ``colony.username``, ``colony.query``,
      etc. extracted from inputs and outputs

    Usage::

        from colony_langchain import ColonyCallbackHandler

        handler = ColonyCallbackHandler()

        # Pass to agent
        agent.invoke({"messages": [...]}, config={"callbacks": [handler]})

        # After run, inspect activity
        print(handler.summary())
        print(handler.actions)  # list of dicts with structured metadata

    Args:
        log_level: Logging level for automatic log output. Set to ``None``
            to disable logging and only collect actions. Defaults to ``logging.INFO``.
    """

    def __init__(self, log_level: int | None = logging.INFO) -> None:
        self.log_level = log_level
        self.actions: list[dict[str, Any]] = []
        self._pending: dict[str, dict[str, Any]] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "")
        if not name.startswith("colony_"):
            return

        action: dict[str, Any] = {
            "tool": name,
            "inputs": inputs or {},
            "is_write": name in _WRITE_TOOLS,
            "metadata": _extract_metadata(name, inputs or {}, None),
        }
        self._pending[str(run_id)] = action

        if self.log_level is not None:
            label = "WRITE" if action["is_write"] else "READ"
            logger.log(self.log_level, "[Colony %s] %s called", label, name)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        action = self._pending.pop(str(run_id), None)
        if action is None:
            return

        action["output"] = output
        action["error"] = None
        # Enrich metadata with output-derived fields
        action["metadata"].update(
            _extract_metadata(action["tool"], action["inputs"], output)
        )
        self.actions.append(action)

        if self.log_level is not None:
            logger.log(self.log_level, "[Colony] %s -> %s", action["tool"], output[:120])

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        action = self._pending.pop(str(run_id), None)
        if action is None:
            return

        action["output"] = None
        action["error"] = str(error)
        action["metadata"]["colony.error"] = True
        self.actions.append(action)

        if self.log_level is not None:
            logger.log(logging.WARNING, "[Colony] %s FAILED: %s", action["tool"], error)

    def summary(self) -> str:
        """Return a human-readable summary of all Colony actions taken."""
        if not self.actions:
            return "No Colony actions recorded."

        reads = [a for a in self.actions if not a["is_write"]]
        writes = [a for a in self.actions if a["is_write"]]
        errors = [a for a in self.actions if a["error"]]

        lines = [f"Colony activity: {len(self.actions)} actions ({len(reads)} reads, {len(writes)} writes)"]

        if errors:
            lines.append(f"  Errors: {len(errors)}")

        for a in writes:
            status = "OK" if a["error"] is None else f"FAILED: {a['error']}"
            lines.append(f"  - {a['tool']}: {status}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all recorded actions."""
        self.actions.clear()
        self._pending.clear()
