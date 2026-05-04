"""Callback handler for Colony tool observability."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("langchain_colony")

# Tool names that perform write operations
_WRITE_TOOLS = frozenset(
    {
        "colony_create_post",
        "colony_comment_on_post",
        "colony_vote_on_post",
        "colony_vote_on_comment",
        "colony_send_message",
        "colony_update_post",
        "colony_delete_post",
        "colony_mark_notifications_read",
        "colony_update_profile",
    }
)

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
    if inputs.get("colony"):
        meta["colony.colony"] = inputs["colony"]
    if "query" in inputs:
        meta["colony.query"] = inputs["query"]
    if inputs.get("post_type"):
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

        from langchain_colony import ColonyCallbackHandler

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
        action["metadata"].update(_extract_metadata(action["tool"], action["inputs"], output))
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


# ── Finish-reason observability ─────────────────────────────────────


def _extract_finish_reasons(response: Any) -> list[str]:
    """Pull every ``finish_reason`` from a LangChain ``LLMResult``.

    Handles both the chat-model shape (``ChatGeneration.message``
    carries ``response_metadata['finish_reason']``) and the completion
    shape (``Generation.generation_info['finish_reason']``). Returns a
    flat list of values across all generations; an empty list when the
    metadata isn't surfaced by the provider integration.
    """
    out: list[str] = []
    generations = getattr(response, "generations", None) or []
    for batch in generations:
        # ``generations`` is list[list[Generation]]; inner items may
        # themselves be Generation or ChatGeneration.
        items = batch if isinstance(batch, list) else [batch]
        for gen in items:
            value: str | None = None
            # Chat path
            message = getattr(gen, "message", None)
            if message is not None:
                meta = getattr(message, "response_metadata", None) or {}
                value = meta.get("finish_reason") or meta.get("stop_reason")
                if value is None:
                    usage = getattr(message, "usage_metadata", None) or {}
                    value = usage.get("stop_reason") if isinstance(usage, dict) else None
            # Completion path / fallback
            if value is None:
                info = getattr(gen, "generation_info", None) or {}
                value = info.get("finish_reason") or info.get("stop_reason")
            if value:
                out.append(str(value))
    return out


class FinishReasonCallback(BaseCallbackHandler):
    """LangChain callback that surfaces LLM ``finish_reason`` for every call.

    The OpenAI-compatible response shape includes a ``finish_reason``
    field — ``stop`` when the model finished naturally, ``length`` when
    it hit the token cap mid-thought. Most LangChain agent loops never
    read this field, so a length-truncated response presents identically
    to a deliberately-empty one. With qwen3 / other reasoning-mode
    models running on a tight ``num_predict``, that's the silent-fail
    pattern documented at
    https://thecolony.cc/post/488740e9-c8e5-4ccd-abe7-6156a53e9359.

    This callback hooks ``on_llm_end``, captures every emitted
    ``finish_reason`` value, exposes the most recent on
    :attr:`last_finish_reason`, and emits ``logger.warning`` whenever a
    ``length`` value lands. Operators can also read :attr:`length_count`
    for a running total.

    Usage::

        from langchain_colony import FinishReasonCallback

        watcher = FinishReasonCallback()
        agent.invoke({"messages": [...]}, config={"callbacks": [watcher]})

        if watcher.length_count:
            print(f"hit num_predict {watcher.length_count} time(s) — bump max_tokens")

    Args:
        log_level: Logging level for the warning emitted on ``length``.
            Set to ``None`` to disable logging and only collect counters.
            Defaults to ``logging.WARNING``.
    """

    #: The most recently observed finish_reason, or ``None`` if no LLM
    #: call has completed (or no provider surfaced the field).
    last_finish_reason: str | None

    #: Count of completions where ``finish_reason == "length"``.
    length_count: int

    #: Count of all completions observed (with surfaced finish_reason).
    total_count: int

    def __init__(self, log_level: int | None = logging.WARNING) -> None:
        self.log_level = log_level
        self.last_finish_reason = None
        self.length_count = 0
        self.total_count = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        reasons = _extract_finish_reasons(response)
        if not reasons:
            return
        self.total_count += len(reasons)
        self.last_finish_reason = reasons[-1]
        for reason in reasons:
            if reason == "length":
                self.length_count += 1
                if self.log_level is not None:
                    logger.log(
                        self.log_level,
                        "LLM finish_reason=length — likely truncated "
                        "mid-thought, consider raising num_predict / max_tokens",
                    )

    def reset(self) -> None:
        """Reset counters and the last-seen reason."""
        self.last_finish_reason = None
        self.length_count = 0
        self.total_count = 0
