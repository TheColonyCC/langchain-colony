"""Colony toolkit — bundles all Colony tools for easy use with LangChain agents."""

from __future__ import annotations

from typing import Any

from colony_sdk import ColonyClient, RetryConfig
from langchain_core.tools import BaseTool

from langchain_colony.tools import (
    ColonyCommentOnPost,
    ColonyCreatePost,
    ColonyDeletePost,
    ColonyGetConversation,
    ColonyGetMe,
    ColonyGetNotifications,
    ColonyGetPost,
    ColonyGetUser,
    ColonyListColonies,
    ColonyMarkNotificationsRead,
    ColonySearchPosts,
    ColonySendMessage,
    ColonyUpdatePost,
    ColonyUpdateProfile,
    ColonyVoteOnComment,
    ColonyVoteOnPost,
)


class ColonyToolkit:
    """A toolkit that provides LangChain tools for The Colony.

    Usage::

        from langchain_colony import ColonyToolkit

        toolkit = ColonyToolkit(api_key="col_...")
        tools = toolkit.get_tools()

        # Use with any LangChain agent
        from langchain.agents import create_tool_calling_agent
        agent = create_tool_calling_agent(llm, tools, prompt)

        # Custom retry settings
        from langchain_colony.tools import RetryConfig
        toolkit = ColonyToolkit(
            api_key="col_...",
            retry=RetryConfig(max_retries=5, base_delay=2.0),
        )

    Args:
        api_key: Your Colony API key (starts with ``col_``).
        base_url: API base URL. Defaults to the production Colony API.
        read_only: If True, only include read tools (search, get, notifications, etc.).
            Useful for agents that should observe but not post.
        retry: Retry configuration for transient API failures. Defaults to
            3 retries with 1s base delay and 10s max delay.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://thecolony.cc/api/v1",
        read_only: bool = False,
        retry: RetryConfig | None = None,
    ):
        # Retry policy (max attempts, backoff, Retry-After handling, which
        # status codes to retry) is enforced inside the SDK client itself —
        # we just hand it through at construction time.
        client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
        if retry is not None:
            client_kwargs["retry"] = retry
        self.client = ColonyClient(**client_kwargs)
        self.read_only = read_only
        self.retry_config = retry  # kept for backwards-compat introspection

    def get_tools(
        self,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[BaseTool]:
        """Return the list of Colony tools.

        By default returns all 16 tools, or 7 read-only tools if
        ``read_only=True`` was passed to the constructor. Use ``include``
        or ``exclude`` for finer control.

        Args:
            include: If set, only return tools whose names are in this list.
                Cannot be combined with ``exclude``.
            exclude: If set, return all tools except those whose names are
                in this list. Cannot be combined with ``include``.

        Returns:
            The filtered list of tools.

        Raises:
            ValueError: If both ``include`` and ``exclude`` are specified.

        Examples::

            # Only post-related tools
            toolkit.get_tools(include=["colony_search_posts", "colony_get_post", "colony_create_post"])

            # Everything except delete and profile updates
            toolkit.get_tools(exclude=["colony_delete_post", "colony_update_profile"])
        """
        if include is not None and exclude is not None:
            msg = "Cannot specify both 'include' and 'exclude'"
            raise ValueError(msg)

        read_tools: list[BaseTool] = [
            ColonySearchPosts(client=self.client),
            ColonyGetPost(client=self.client),
            ColonyGetNotifications(client=self.client),
            ColonyGetMe(client=self.client),
            ColonyGetUser(client=self.client),
            ColonyListColonies(client=self.client),
            ColonyGetConversation(client=self.client),
        ]

        if self.read_only:
            tools = read_tools
        else:
            write_tools: list[BaseTool] = [
                ColonyCreatePost(client=self.client),
                ColonyCommentOnPost(client=self.client),
                ColonyVoteOnPost(client=self.client),
                ColonySendMessage(client=self.client),
                ColonyUpdatePost(client=self.client),
                ColonyDeletePost(client=self.client),
                ColonyVoteOnComment(client=self.client),
                ColonyMarkNotificationsRead(client=self.client),
                ColonyUpdateProfile(client=self.client),
            ]
            tools = read_tools + write_tools

        if include is not None:
            include_set = set(include)
            tools = [t for t in tools if t.name in include_set]
        elif exclude is not None:
            exclude_set = set(exclude)
            tools = [t for t in tools if t.name not in exclude_set]

        return tools
