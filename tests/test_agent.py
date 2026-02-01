"""Tests for rag_system.agent module.

Verifies agent creation, query execution, empty query handling,
error handling, and tool call extraction using mocked LLM and tools.
"""

from unittest import mock

import pytest
from langchain_core.tools import BaseTool

from rag_system.agent import (
    AGENT_PROMPT,
    _get_tools,
    create_agent,
    run_agent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm():
    """Create a mock OllamaLLM instance."""
    return mock.MagicMock()


@pytest.fixture()
def mock_tools() -> list[BaseTool]:
    """Create a list of mock LangChain tools for agent creation."""
    tools = []
    for name, desc in [
        ("legal_framework_search", "法令を検索する"),
        ("precedent_search", "判例を検索する"),
        ("archive_stats", "書庫統計を取得する"),
    ]:
        tool = mock.MagicMock(spec=BaseTool)
        tool.name = name
        tool.description = desc
        tools.append(tool)
    return tools


@pytest.fixture()
def mock_agent_executor():
    """Create a mock AgentExecutor for run_agent tests."""
    executor = mock.MagicMock()
    return executor


# ---------------------------------------------------------------------------
# Tests: _get_tools
# ---------------------------------------------------------------------------


class TestGetTools:
    """Test the _get_tools helper function."""

    def test_returns_three_tools(self):
        tools = _get_tools()
        assert len(tools) == 3

    def test_returns_base_tool_instances(self):
        tools = _get_tools()
        for tool in tools:
            assert isinstance(tool, BaseTool)

    def test_contains_expected_tool_names(self):
        tools = _get_tools()
        names = {tool.name for tool in tools}
        assert "legal_framework_search" in names
        assert "precedent_search" in names
        assert "archive_stats" in names


# ---------------------------------------------------------------------------
# Tests: AGENT_PROMPT
# ---------------------------------------------------------------------------


class TestAgentPrompt:
    """Test the agent prompt template configuration."""

    def test_prompt_has_required_input_variables(self):
        assert "tools" in AGENT_PROMPT.input_variables
        assert "tool_names" in AGENT_PROMPT.input_variables
        assert "input" in AGENT_PROMPT.input_variables
        assert "agent_scratchpad" in AGENT_PROMPT.input_variables

    def test_prompt_template_contains_react_format(self):
        assert "Thought:" in AGENT_PROMPT.template
        assert "Action:" in AGENT_PROMPT.template
        assert "Action Input:" in AGENT_PROMPT.template
        assert "Observation:" in AGENT_PROMPT.template
        assert "Final Answer:" in AGENT_PROMPT.template


# ---------------------------------------------------------------------------
# Tests: create_agent
# ---------------------------------------------------------------------------


class TestCreateAgent:
    """Test the create_agent factory function."""

    def test_creates_agent_with_custom_tools(self, mock_tools):
        with mock.patch("rag_system.judge.create_llm") as mock_create_llm:
            mock_create_llm.return_value = mock.MagicMock()

            executor = create_agent(tools=mock_tools)

        assert executor is not None
        assert executor.tools == mock_tools

    def test_calls_create_llm_with_defaults(self, mock_tools):
        with mock.patch("rag_system.judge.create_llm") as mock_create_llm:
            mock_create_llm.return_value = mock.MagicMock()

            create_agent(tools=mock_tools)

        mock_create_llm.assert_called_once_with(
            model_name=None,
            temperature=None,
            num_ctx=None,
        )

    def test_passes_model_name_to_create_llm(self, mock_tools):
        with mock.patch("rag_system.judge.create_llm") as mock_create_llm:
            mock_create_llm.return_value = mock.MagicMock()

            create_agent(tools=mock_tools, model_name="test-model")

        mock_create_llm.assert_called_once_with(
            model_name="test-model",
            temperature=None,
            num_ctx=None,
        )

    def test_passes_temperature_to_create_llm(self, mock_tools):
        with mock.patch("rag_system.judge.create_llm") as mock_create_llm:
            mock_create_llm.return_value = mock.MagicMock()

            create_agent(tools=mock_tools, temperature=0.5)

        mock_create_llm.assert_called_once_with(
            model_name=None,
            temperature=0.5,
            num_ctx=None,
        )

    def test_passes_num_ctx_to_create_llm(self, mock_tools):
        with mock.patch("rag_system.judge.create_llm") as mock_create_llm:
            mock_create_llm.return_value = mock.MagicMock()

            create_agent(tools=mock_tools, num_ctx=8192)

        mock_create_llm.assert_called_once_with(
            model_name=None,
            temperature=None,
            num_ctx=8192,
        )

    def test_sets_max_iterations(self, mock_tools):
        with mock.patch("rag_system.judge.create_llm") as mock_create_llm:
            mock_create_llm.return_value = mock.MagicMock()

            executor = create_agent(tools=mock_tools, max_iterations=5)

        assert executor.max_iterations == 5

    def test_default_max_iterations_is_10(self, mock_tools):
        with mock.patch("rag_system.judge.create_llm") as mock_create_llm:
            mock_create_llm.return_value = mock.MagicMock()

            executor = create_agent(tools=mock_tools)

        assert executor.max_iterations == 10

    def test_sets_handle_parsing_errors(self, mock_tools):
        with mock.patch("rag_system.judge.create_llm") as mock_create_llm:
            mock_create_llm.return_value = mock.MagicMock()

            executor = create_agent(
                tools=mock_tools, handle_parsing_errors=False,
            )

        assert executor.handle_parsing_errors is False

    def test_default_handle_parsing_errors_is_true(self, mock_tools):
        with mock.patch("rag_system.judge.create_llm") as mock_create_llm:
            mock_create_llm.return_value = mock.MagicMock()

            executor = create_agent(tools=mock_tools)

        assert executor.handle_parsing_errors is True

    def test_returns_intermediate_steps(self, mock_tools):
        with mock.patch("rag_system.judge.create_llm") as mock_create_llm:
            mock_create_llm.return_value = mock.MagicMock()

            executor = create_agent(tools=mock_tools)

        assert executor.return_intermediate_steps is True

    def test_loads_default_tools_when_none(self):
        with mock.patch("rag_system.judge.create_llm") as mock_create_llm:
            mock_create_llm.return_value = mock.MagicMock()

            executor = create_agent(tools=None)

        tool_names = {tool.name for tool in executor.tools}
        assert "legal_framework_search" in tool_names
        assert "precedent_search" in tool_names
        assert "archive_stats" in tool_names

    def test_raises_connection_error_when_ollama_unavailable(self, mock_tools):
        with mock.patch(
            "rag_system.judge.create_llm",
            side_effect=ConnectionError("Ollamaサーバーに接続できません"),
        ):
            with pytest.raises(ConnectionError, match="Ollama"):
                create_agent(tools=mock_tools)


# ---------------------------------------------------------------------------
# Tests: run_agent
# ---------------------------------------------------------------------------


class TestRunAgent:
    """Test the run_agent query execution function."""

    def test_returns_dict_with_expected_keys(self, mock_agent_executor):
        mock_agent_executor.invoke.return_value = {
            "output": "窃盗罪の回答です。",
            "intermediate_steps": [],
        }

        result = run_agent(mock_agent_executor, "窃盗罪について")

        assert "query" in result
        assert "result" in result
        assert "tool_calls" in result

    def test_returns_query_in_result(self, mock_agent_executor):
        mock_agent_executor.invoke.return_value = {
            "output": "回答",
            "intermediate_steps": [],
        }

        result = run_agent(mock_agent_executor, "窃盗罪について")

        assert result["query"] == "窃盗罪について"

    def test_returns_output_as_result(self, mock_agent_executor):
        mock_agent_executor.invoke.return_value = {
            "output": "窃盗罪の量刑基準は...",
            "intermediate_steps": [],
        }

        result = run_agent(mock_agent_executor, "窃盗罪の量刑基準")

        assert result["result"] == "窃盗罪の量刑基準は..."

    def test_passes_input_to_agent(self, mock_agent_executor):
        mock_agent_executor.invoke.return_value = {
            "output": "回答",
            "intermediate_steps": [],
        }

        run_agent(mock_agent_executor, "窃盗罪について")

        mock_agent_executor.invoke.assert_called_once_with(
            {"input": "窃盗罪について"},
        )

    def test_extracts_tool_calls_from_intermediate_steps(
        self, mock_agent_executor,
    ):
        mock_action_1 = mock.MagicMock()
        mock_action_1.tool = "legal_framework_search"
        mock_action_1.tool_input = "窃盗罪"

        mock_action_2 = mock.MagicMock()
        mock_action_2.tool = "precedent_search"
        mock_action_2.tool_input = "窃盗罪の判例"

        mock_agent_executor.invoke.return_value = {
            "output": "最終回答",
            "intermediate_steps": [
                (mock_action_1, "法令の検索結果"),
                (mock_action_2, "判例の検索結果"),
            ],
        }

        result = run_agent(mock_agent_executor, "窃盗罪について")

        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["tool"] == "legal_framework_search"
        assert result["tool_calls"][0]["input"] == "窃盗罪"
        assert result["tool_calls"][0]["output"] == "法令の検索結果"
        assert result["tool_calls"][1]["tool"] == "precedent_search"

    def test_empty_intermediate_steps(self, mock_agent_executor):
        mock_agent_executor.invoke.return_value = {
            "output": "直接回答",
            "intermediate_steps": [],
        }

        result = run_agent(mock_agent_executor, "テスト質問")

        assert result["tool_calls"] == []

    def test_no_intermediate_steps_key(self, mock_agent_executor):
        mock_agent_executor.invoke.return_value = {
            "output": "回答",
        }

        result = run_agent(mock_agent_executor, "テスト質問")

        assert result["tool_calls"] == []


# ---------------------------------------------------------------------------
# Tests: run_agent — Empty Query Handling
# ---------------------------------------------------------------------------


class TestRunAgentEmptyQuery:
    """Test that run_agent handles empty queries correctly."""

    def test_empty_string_returns_error(self, mock_agent_executor):
        result = run_agent(mock_agent_executor, "")

        assert "エラー" in result["result"]
        assert result["query"] == ""
        assert result["tool_calls"] == []
        mock_agent_executor.invoke.assert_not_called()

    def test_whitespace_only_returns_error(self, mock_agent_executor):
        result = run_agent(mock_agent_executor, "   ")

        assert "エラー" in result["result"]
        assert result["tool_calls"] == []
        mock_agent_executor.invoke.assert_not_called()

    def test_none_query_returns_error(self, mock_agent_executor):
        result = run_agent(mock_agent_executor, None)

        assert "エラー" in result["result"]
        assert result["tool_calls"] == []
        mock_agent_executor.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: run_agent — Error Handling
# ---------------------------------------------------------------------------


class TestRunAgentErrorHandling:
    """Test that run_agent handles exceptions correctly."""

    def test_handles_connection_error(self, mock_agent_executor):
        mock_agent_executor.invoke.side_effect = ConnectionError(
            "Ollamaサーバーとの接続が切れました",
        )

        result = run_agent(mock_agent_executor, "窃盗罪について")

        assert "エラー" in result["result"]
        assert "Ollama" in result["result"]
        assert result["query"] == "窃盗罪について"
        assert result["tool_calls"] == []

    def test_handles_unexpected_exception(self, mock_agent_executor):
        mock_agent_executor.invoke.side_effect = RuntimeError(
            "Unexpected error",
        )

        result = run_agent(mock_agent_executor, "窃盗罪について")

        assert "エラー" in result["result"]
        assert result["query"] == "窃盗罪について"
        assert result["tool_calls"] == []

    def test_handles_keyboard_interrupt_propagation(self, mock_agent_executor):
        """KeyboardInterrupt is a BaseException, not Exception — verify it propagates."""
        mock_agent_executor.invoke.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            run_agent(mock_agent_executor, "窃盗罪について")
