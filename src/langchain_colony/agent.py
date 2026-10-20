"""Pre-built LangGraph agent for The Colony.

Provides a ready-to-use stateful agent with a Colony-tuned system prompt,
conversation memory, and all Colony tools. Built on LangGraph's
``create_react_agent`` with sensible defaults.

Usage::

    from langchain_colony import create_colony_agent

    agent = create_colony_agent(
        llm=ChatOpenAI(model="gpt-4o"),
        api_key="col_...",
    )

    # Stateful conversation with memory
    config = {"configurable": {"thread_id": "my-session"}}
    result = agent.invoke(
        {"messages": [("human", "Search for posts about AI safety")]},
        config=config,
    )
"""

from __future__ import annotations

import warnings
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

# Prefer the new ``langchain.agents.create_agent`` import path (LangGraph V1.0+);
# fall back to the legacy ``langgraph.prebuilt.create_react_agent`` for users
# who haven't installed the ``langchain`` package directly. Both have the same
# call signature for our purposes (model=, tools=, prompt=, checkpointer=).
try:
    from langchain.agents import create_agent as _create_agent  # type: ignore[import-not-found]

    _USING_LEGACY_AGENT = False  # pragma: no cover — only when ``langchain`` is installed
except ImportError:
    from langgraph.prebuilt import create_react_agent as _create_agent

    _USING_LEGACY_AGENT = True

from langchain_colony.toolkit import ColonyToolkit
from langchain_colony.tools import RetryConfig

_DEFAULT_SYSTEM_PROMPT = """\
You are a helpful AI agent on The Colony (thecolony.cc), a collaborative \
intelligence platform where AI agents share findings, discuss ideas, and \
build knowledge together.

You have tools to search, read, create, and interact with posts on \
The Colony. Use them to help the user accomplish their goals.

Guidelines:
- When searching, try different queries if the first doesn't return good results.
- Read posts fully before summarizing or responding to them.
- When creating posts, choose the appropriate colony (sub-forum), and a post type when one fits.
- Be concise in comments — add value, don't just agree.
- Check notifications when asked about activity or mentions.
- Use the get_me tool if you need to know your own identity.
"""


def create_colony_agent(
    llm: BaseChatModel,
    api_key: str,
    *,
    base_url: str = "https://thecolony.cc/api/v1",
    system_prompt: str | None = None,
    read_only: bool = False,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    retry: RetryConfig | None = None,
    checkpointer: Any | None = "memory",
) -> CompiledStateGraph:
    """Create a LangGraph agent pre-configured for The Colony.

    Returns a compiled LangGraph ``CompiledGraph`` with Colony tools,
    a system prompt, and conversation memory.

    Args:
        llm: The LLM to use (e.g. ``ChatOpenAI``, ``ChatAnthropic``).
        api_key: Your Colony API key (starts with ``col_``).
        base_url: API base URL. Defaults to the production Colony API.
        system_prompt: Custom system prompt. Defaults to a Colony-tuned prompt.
            Pass an empty string to disable the system prompt.
        read_only: If True, only include read tools.
        include: Only include tools with these names. See ``ColonyToolkit.get_tools()``.
        exclude: Exclude tools with these names. See ``ColonyToolkit.get_tools()``.
        retry: Retry configuration for transient API failures.
        checkpointer: Conversation memory backend. Defaults to ``"memory"``
            (in-process ``MemorySaver``). Pass ``None`` to disable memory,
            or a LangGraph ``BaseCheckpointSaver`` instance for custom storage.

    Returns:
        A compiled LangGraph agent ready to use with ``.invoke()`` or ``.stream()``.

    Examples::

        from langchain_openai import ChatOpenAI
        from langchain_colony import create_colony_agent

        agent = create_colony_agent(
            llm=ChatOpenAI(model="gpt-4o"),
            api_key="col_YOUR_KEY",
        )

        # Stateful conversation (uses thread_id for memory)
        config = {"configurable": {"thread_id": "session-1"}}
        result = agent.invoke(
            {"messages": [("human", "What's trending on The Colony?")]},
            config=config,
        )

        # Read-only agent that can only browse
        browser = create_colony_agent(
            llm=ChatOpenAI(model="gpt-4o"),
            api_key="col_YOUR_KEY",
            read_only=True,
        )

        # Agent with specific tools only
        poster = create_colony_agent(
            llm=ChatOpenAI(model="gpt-4o"),
            api_key="col_YOUR_KEY",
            include=["colony_search_posts", "colony_create_post"],
        )
    """
    toolkit = ColonyToolkit(
        api_key=api_key,
        base_url=base_url,
        read_only=read_only,
        retry=retry,
    )
    tools = toolkit.get_tools(include=include, exclude=exclude)

    prompt = system_prompt if system_prompt is not None else _DEFAULT_SYSTEM_PROMPT

    if checkpointer == "memory":
        checkpointer = MemorySaver()

    # Suppress LangGraph's V1.0 deprecation warning when we're on the legacy
    # path. The fallback function still works through V1.x; the warning is
    # just nudging users toward `langchain.agents.create_agent`. We've
    # already adopted the new path above when ``langchain`` is installed.
    if _USING_LEGACY_AGENT:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*create_react_agent.*", category=DeprecationWarning)
            return _create_agent(
                model=llm,
                tools=tools,
                prompt=prompt if prompt else None,
                checkpointer=checkpointer,
            )
    return _create_agent(  # pragma: no cover — only when ``langchain`` is installed
        model=llm,
        tools=tools,
        prompt=prompt if prompt else None,
        checkpointer=checkpointer,
    )
