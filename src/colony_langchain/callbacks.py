"""Callback handler for Colony tool observability."""

from __future__ import annotations

import logging
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


class ColonyCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that logs Colony tool activity.

    Tracks all Colony tool invocations and provides a summary of actions taken.
    Useful for auditing agent behavior, debugging, and integration with
    monitoring systems.

    Usage::

        from colony_langchain import ColonyCallbackHandler

        handler = ColonyCallbackHandler()

        # Pass to agent
        agent.invoke({"messages": [...]}, config={"callbacks": [handler]})

        # After run, inspect activity
        print(handler.summary())
        print(handler.actions)  # list of dicts

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
