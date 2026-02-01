"""Chat page for the RAG Accuracy Verification & Comparison Tool.

Provides a web-based chat interface that replicates the CLI functionality
in :mod:`rag_system.main`.  Users can submit legal questions and receive
structured judicial decisions with source document references.

The page uses ``st.session_state`` to maintain conversation history and
the current judicial chain across Streamlit reruns.

Usage:
    from ui.chat_page import render_chat_page

    render_chat_page()
"""

from __future__ import annotations

import logging
import time
from typing import Any

import streamlit as st

from rag_system.config import (
    LLM_MODEL_NAME,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    RETRIEVAL_K,
)
from ui.components import source_document_expander

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session State Keys
# ---------------------------------------------------------------------------

_KEY_CHAT_HISTORY = "chat_history"
_KEY_JUDICIAL_CHAIN = "judicial_chain"
_KEY_CHAIN_CONFIG = "chain_config"
_KEY_AGENT_MODE = "agent_mode"
_KEY_AGENT_EXECUTOR = "agent_executor"
_KEY_AGENT_CONFIG = "agent_config"


# ---------------------------------------------------------------------------
# Chain Management
# ---------------------------------------------------------------------------


def _get_current_config() -> dict[str, Any]:
    """Build the current chain configuration from session state.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary with model_name, temperature, num_ctx,
        and retrieval_k.
    """
    return {
        "model_name": st.session_state.get("selected_model", LLM_MODEL_NAME),
        "temperature": st.session_state.get("param_temperature", LLM_TEMPERATURE),
        "num_ctx": st.session_state.get("param_num_ctx", LLM_NUM_CTX),
        "retrieval_k": st.session_state.get("param_retrieval_k", RETRIEVAL_K),
    }


def _build_chain(config: dict[str, Any]) -> Any:
    """Build a judicial chain from the given configuration.

    Uses lazy imports to avoid loading heavy ML dependencies until needed.

    Parameters
    ----------
    config:
        Configuration dictionary (model_name, temperature, num_ctx,
        retrieval_k).

    Returns
    -------
    RetrievalQA or None
        The configured chain, or ``None`` if construction fails.
    """
    try:
        from rag_system.backend_adapter import BackendConfig, create_backend

        backend_config = BackendConfig(
            name="chat",
            model_name=config["model_name"],
            temperature=config["temperature"],
            num_ctx=config["num_ctx"],
            retrieval_k=config["retrieval_k"],
        )
        chain = create_backend(backend_config)
        logger.info(
            "チャット用チェーンを構築: model='%s', temperature=%.2f",
            config["model_name"],
            config["temperature"],
        )
        return chain
    except ConnectionError:
        logger.error("Ollamaサーバーに接続できません")
        return None
    except FileNotFoundError:
        logger.error("ChromaDBが初期化されていません")
        return None
    except Exception:
        logger.exception("チェーン構築中にエラーが発生しました")
        return None


def _get_or_rebuild_chain() -> Any:
    """Return the cached judicial chain, rebuilding if config changed.

    Returns
    -------
    RetrievalQA or None
        The judicial chain, or ``None`` if unavailable.
    """
    current_config = _get_current_config()
    cached_config = st.session_state.get(_KEY_CHAIN_CONFIG)

    if (
        st.session_state.get(_KEY_JUDICIAL_CHAIN) is not None
        and cached_config == current_config
    ):
        return st.session_state[_KEY_JUDICIAL_CHAIN]

    chain = _build_chain(current_config)
    st.session_state[_KEY_JUDICIAL_CHAIN] = chain
    st.session_state[_KEY_CHAIN_CONFIG] = current_config
    return chain


def _execute_query(chain: Any, query: str) -> dict[str, Any]:
    """Execute a judicial query and return the result dict.

    Wraps :func:`rag_system.judge.judge` with timing information.

    Parameters
    ----------
    chain:
        The judicial RetrievalQA chain.
    query:
        The legal question.

    Returns
    -------
    dict[str, Any]
        Result with keys ``query``, ``result``, ``source_documents``,
        and ``elapsed_time``.
    """
    from rag_system.judge import judge

    start = time.time()
    result = judge(chain, query)
    elapsed = time.time() - start

    result["elapsed_time"] = elapsed
    return result


# ---------------------------------------------------------------------------
# Agent Management
# ---------------------------------------------------------------------------


def _get_or_rebuild_agent() -> Any:
    """Return the cached agent executor, rebuilding if config changed.

    Returns
    -------
    AgentExecutor or None
        The agent executor, or ``None`` if unavailable.
    """
    current_config = _get_current_config()
    cached_config = st.session_state.get(_KEY_AGENT_CONFIG)

    if (
        st.session_state.get(_KEY_AGENT_EXECUTOR) is not None
        and cached_config == current_config
    ):
        return st.session_state[_KEY_AGENT_EXECUTOR]

    try:
        from rag_system.agent import create_agent

        agent = create_agent(
            model_name=current_config["model_name"],
            temperature=current_config["temperature"],
            num_ctx=current_config["num_ctx"],
        )
        st.session_state[_KEY_AGENT_EXECUTOR] = agent
        st.session_state[_KEY_AGENT_CONFIG] = current_config
        logger.info(
            "エージェントを構築: model='%s', temperature=%.2f",
            current_config["model_name"],
            current_config["temperature"],
        )
        return agent
    except ConnectionError:
        logger.error("Ollamaサーバーに接続できません")
        return None
    except Exception:
        logger.exception("エージェント構築中にエラーが発生しました")
        return None


def _execute_agent_query(agent: Any, query: str) -> dict[str, Any]:
    """Execute a query using the agent and return the result dict.

    Parameters
    ----------
    agent:
        The LangChain agent executor.
    query:
        The legal question.

    Returns
    -------
    dict[str, Any]
        Result with keys ``query``, ``result``, ``tool_calls``,
        and ``elapsed_time``.
    """
    from rag_system.agent import run_agent

    start = time.time()
    result = run_agent(agent, query)
    elapsed = time.time() - start

    result["elapsed_time"] = elapsed
    return result


# ---------------------------------------------------------------------------
# Chat History
# ---------------------------------------------------------------------------


def _init_chat_history() -> None:
    """Initialize chat history in session state if absent."""
    if _KEY_CHAT_HISTORY not in st.session_state:
        st.session_state[_KEY_CHAT_HISTORY] = []


def _append_message(role: str, content: str, **extra: Any) -> None:
    """Append a message to the chat history.

    Parameters
    ----------
    role:
        Message role (``"user"`` or ``"assistant"``).
    content:
        Message text content.
    **extra:
        Additional data (e.g. ``source_documents``, ``elapsed_time``).
    """
    st.session_state[_KEY_CHAT_HISTORY].append(
        {"role": role, "content": content, **extra}
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_message(msg: dict[str, Any]) -> None:
    """Render a single chat message.

    Parameters
    ----------
    msg:
        Message dict with ``role``, ``content``, and optional
        ``source_documents`` and ``elapsed_time``.
    """
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            elapsed = msg.get("elapsed_time")
            if elapsed is not None:
                st.caption(f"応答時間: {elapsed:.1f} 秒")

            source_docs = msg.get("source_documents", [])
            if source_docs:
                source_document_expander(source_docs)

            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                with st.expander(
                    f"使用ツール ({len(tool_calls)}件)", expanded=False
                ):
                    for tc in tool_calls:
                        st.markdown(f"**{tc['tool']}**")
                        st.code(tc.get("input", ""), language="text")
                        st.text(tc.get("output", "")[:500])


def _render_history() -> None:
    """Render the full chat history."""
    for msg in st.session_state.get(_KEY_CHAT_HISTORY, []):
        _render_message(msg)


# ---------------------------------------------------------------------------
# Page Entry Point
# ---------------------------------------------------------------------------


def render_chat_page() -> None:
    """Render the chat page with question input and judicial decision display.

    This is the main entry point called by the app router.  It handles:

    1. Displaying existing conversation history
    2. Accepting new questions via ``st.chat_input``
    3. Building/caching the judicial chain
    4. Executing queries with a spinner
    5. Displaying results with source document references
    """
    st.header("チャット")
    st.caption(
        "法的質問を入力すると、「謎の国家」の最高裁判所が司法判断を下します。"
    )

    # --- Agent mode toggle ---
    if _KEY_AGENT_MODE not in st.session_state:
        st.session_state[_KEY_AGENT_MODE] = False
    use_agent = st.toggle(
        "エージェントモード（書庫ツール使用）",
        key=_KEY_AGENT_MODE,
        help=(
            "有効にすると、エージェントが法令検索・判例検索・書庫統計の"
            "ツールを自動的に選択・実行して回答します。"
        ),
    )

    _init_chat_history()

    # --- Render existing history ---
    _render_history()

    # --- Question input ---
    query = st.chat_input(
        "法的質問を入力してください...",
        key="chat_input",
    )

    if not query:
        return

    # Display user message immediately
    _append_message("user", query)
    with st.chat_message("user"):
        st.markdown(query)

    # --- Build chain/agent and execute ---
    with st.chat_message("assistant"):
        if use_agent:
            _handle_agent_query(query)
        else:
            _handle_chain_query(query)

    # --- Clear history button ---
    if st.session_state.get(_KEY_CHAT_HISTORY):
        if st.button("会話履歴をクリア", key="clear_chat"):
            st.session_state[_KEY_CHAT_HISTORY] = []
            st.session_state.pop(_KEY_JUDICIAL_CHAIN, None)
            st.session_state.pop(_KEY_CHAIN_CONFIG, None)
            st.session_state.pop(_KEY_AGENT_EXECUTOR, None)
            st.session_state.pop(_KEY_AGENT_CONFIG, None)
            st.rerun()


# ---------------------------------------------------------------------------
# Query Handlers
# ---------------------------------------------------------------------------


def _handle_chain_query(query: str) -> None:
    """Handle a query using the standard RetrievalQA chain.

    Parameters
    ----------
    query:
        The legal question.
    """
    with st.spinner("司法判断を生成中..."):
        chain = _get_or_rebuild_chain()

        if chain is None:
            error_msg = (
                "エラー: 司法推論チェーンを構築できません。\n\n"
                "以下を確認してください:\n"
                "- Ollamaサーバーが起動していること (`ollama serve`)\n"
                "- ドキュメントがインジェスト済みであること "
                "(`python rag_system/ingest.py`)"
            )
            st.error(error_msg)
            _append_message("assistant", error_msg)
            return

        result = _execute_query(chain, query)

    # Display the judicial decision
    response_text = result.get("result", "")
    st.markdown(response_text)

    elapsed = result.get("elapsed_time")
    if elapsed is not None:
        st.caption(f"応答時間: {elapsed:.1f} 秒")

    source_docs = result.get("source_documents", [])
    if source_docs:
        source_document_expander(source_docs)

    # Save to history
    _append_message(
        "assistant",
        response_text,
        source_documents=source_docs,
        elapsed_time=elapsed,
    )


def _handle_agent_query(query: str) -> None:
    """Handle a query using the LangChain agent with tools.

    Parameters
    ----------
    query:
        The legal question.
    """
    with st.spinner("エージェントが書庫ツールを使用して回答を生成中..."):
        agent = _get_or_rebuild_agent()

        if agent is None:
            error_msg = (
                "エラー: エージェントを構築できません。\n\n"
                "以下を確認してください:\n"
                "- Ollamaサーバーが起動していること (`ollama serve`)\n"
                "- ドキュメントがインジェスト済みであること "
                "(`python rag_system/ingest.py`)"
            )
            st.error(error_msg)
            _append_message("assistant", error_msg)
            return

        result = _execute_agent_query(agent, query)

    # Display the agent response
    response_text = result.get("result", "")
    st.markdown(response_text)

    elapsed = result.get("elapsed_time")
    if elapsed is not None:
        st.caption(f"応答時間: {elapsed:.1f} 秒")

    tool_calls = result.get("tool_calls", [])
    if tool_calls:
        with st.expander(
            f"使用ツール ({len(tool_calls)}件)", expanded=False
        ):
            for tc in tool_calls:
                st.markdown(f"**{tc['tool']}**")
                st.code(tc.get("input", ""), language="text")
                st.text(tc.get("output", "")[:500])

    # Save to history
    _append_message(
        "assistant",
        response_text,
        tool_calls=tool_calls,
        elapsed_time=elapsed,
    )
