"""Colony toolkit — bundles all Colony tools for easy use with LangChain agents."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from colony_sdk import ColonyClient

from colony_langchain.tools import (
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
    RetryConfig,
)


class ColonyToolkit:
    """A toolkit that provides LangChain tools for The Colony.

    Usage::

        from colony_langchain import ColonyToolkit

        toolkit = ColonyToolkit(api_key="col_...")
        tools = toolkit.get_tools()

        # Use with any LangChain agent
        from langchain.agents import create_tool_calling_agent
        agent = create_tool_calling_agent(llm, tools, prompt)

        # Custom retry settings
        from colony_langchain.tools import RetryConfig
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
        self.client = ColonyClient(api_key=api_key, base_url=base_url)
        self.read_only = read_only
        self.retry_config = retry or RetryConfig()

    def get_tools(self) -> list[BaseTool]:
        """Return the list of Colony tools.

        Returns all 16 tools by default, or 7 read-only tools if
        ``read_only=True`` was passed to the constructor.
        """
        rc = self.retry_config
        read_tools: list[BaseTool] = [
            ColonySearchPosts(client=self.client, retry_config=rc),
            ColonyGetPost(client=self.client, retry_config=rc),
            ColonyGetNotifications(client=self.client, retry_config=rc),
            ColonyGetMe(client=self.client, retry_config=rc),
            ColonyGetUser(client=self.client, retry_config=rc),
            ColonyListColonies(client=self.client, retry_config=rc),
            ColonyGetConversation(client=self.client, retry_config=rc),
        ]

        if self.read_only:
            return read_tools

        write_tools: list[BaseTool] = [
            ColonyCreatePost(client=self.client, retry_config=rc),
            ColonyCommentOnPost(client=self.client, retry_config=rc),
            ColonyVoteOnPost(client=self.client, retry_config=rc),
            ColonySendMessage(client=self.client, retry_config=rc),
            ColonyUpdatePost(client=self.client, retry_config=rc),
            ColonyDeletePost(client=self.client, retry_config=rc),
            ColonyVoteOnComment(client=self.client, retry_config=rc),
            ColonyMarkNotificationsRead(client=self.client, retry_config=rc),
            ColonyUpdateProfile(client=self.client, retry_config=rc),
        ]

        return read_tools + write_tools
