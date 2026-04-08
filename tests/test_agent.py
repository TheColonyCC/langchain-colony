"""Tests for the pre-built Colony LangGraph agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langgraph.graph.state import CompiledStateGraph

from colony_langchain.agent import _DEFAULT_SYSTEM_PROMPT, create_colony_agent


def _mock_llm():
    """Create a mock LLM that satisfies BaseChatModel interface."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


class TestCreateColonyAgent:
    def test_returns_compiled_graph(self):
        with patch("colony_langchain.toolkit.ColonyClient"):
            agent = create_colony_agent(llm=_mock_llm(), api_key="col_test")
        assert isinstance(agent, CompiledStateGraph)

    def test_default_has_memory(self):
        with patch("colony_langchain.toolkit.ColonyClient"):
            agent = create_colony_agent(llm=_mock_llm(), api_key="col_test")
        # CompiledGraph with checkpointer should have it set
        assert agent.checkpointer is not None

    def test_no_memory(self):
        with patch("colony_langchain.toolkit.ColonyClient"):
            agent = create_colony_agent(llm=_mock_llm(), api_key="col_test", checkpointer=None)
        assert agent.checkpointer is None

    def test_read_only(self):
        with patch("colony_langchain.toolkit.ColonyClient"):
            agent = create_colony_agent(llm=_mock_llm(), api_key="col_test", read_only=True)
        assert isinstance(agent, CompiledStateGraph)

    def test_include_filter(self):
        with patch("colony_langchain.toolkit.ColonyClient"):
            agent = create_colony_agent(
                llm=_mock_llm(),
                api_key="col_test",
                include=["colony_search_posts", "colony_get_post"],
            )
        assert isinstance(agent, CompiledStateGraph)

    def test_exclude_filter(self):
        with patch("colony_langchain.toolkit.ColonyClient"):
            agent = create_colony_agent(
                llm=_mock_llm(),
                api_key="col_test",
                exclude=["colony_delete_post"],
            )
        assert isinstance(agent, CompiledStateGraph)

    def test_custom_system_prompt(self):
        with patch("colony_langchain.toolkit.ColonyClient"):
            agent = create_colony_agent(
                llm=_mock_llm(),
                api_key="col_test",
                system_prompt="You are a research bot.",
            )
        assert isinstance(agent, CompiledStateGraph)

    def test_empty_system_prompt_disables(self):
        with patch("colony_langchain.toolkit.ColonyClient"):
            agent = create_colony_agent(
                llm=_mock_llm(),
                api_key="col_test",
                system_prompt="",
            )
        assert isinstance(agent, CompiledStateGraph)

    def test_default_system_prompt_exists(self):
        assert "Colony" in _DEFAULT_SYSTEM_PROMPT
        assert "thecolony.cc" in _DEFAULT_SYSTEM_PROMPT

    def test_custom_retry(self):
        from colony_langchain.tools import RetryConfig

        with patch("colony_langchain.toolkit.ColonyClient"):
            agent = create_colony_agent(
                llm=_mock_llm(),
                api_key="col_test",
                retry=RetryConfig(max_retries=1),
            )
        assert isinstance(agent, CompiledStateGraph)
