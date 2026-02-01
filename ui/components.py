"""Shared UI components for the RAG Accuracy Verification & Comparison Tool.

Provides reusable Streamlit widgets used across multiple pages:

- :func:`server_status_indicator` — Ollama server connectivity badge
- :func:`model_selector` — model selection dropdown
- :func:`parameter_sliders` — LLM / retrieval parameter controls
- :func:`metrics_card` — single-metric display card
- :func:`source_document_expander` — expandable source document viewer

Usage:
    from ui.components import server_status_indicator, model_selector

    server_status_indicator()
    selected = model_selector(models=["llama3.1:8b"])
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

import streamlit as st

from rag_system.config import (
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    RETRIEVAL_K,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server Status
# ---------------------------------------------------------------------------


def server_status_indicator(base_url: str | None = None) -> bool:
    """Display the Ollama server connectivity status.

    Shows a success or error badge in the current Streamlit context.

    Parameters
    ----------
    base_url:
        The Ollama server URL to check.  Defaults to
        :data:`~rag_system.config.OLLAMA_BASE_URL`.

    Returns
    -------
    bool
        ``True`` if the server is reachable, ``False`` otherwise.
    """
    if base_url is None:
        base_url = OLLAMA_BASE_URL

    try:
        req = urllib.request.Request(base_url, method="GET")
        with urllib.request.urlopen(req, timeout=5):
            pass
        st.success("Ollama: 接続済み", icon="\u2705")
        return True
    except (urllib.error.URLError, OSError):
        st.error("Ollama: 未接続", icon="\u274c")
        st.info(
            "Ollamaサーバーを起動してください:\n\n"
            "```\nollama serve\n```"
        )
        return False


# ---------------------------------------------------------------------------
# Model Selector
# ---------------------------------------------------------------------------


def model_selector(
    models: list[str],
    key: str = "component_model_selector",
    label: str = "使用モデル",
) -> str | None:
    """Render a model selection dropdown.

    Remembers the previous selection via ``st.session_state``.

    Parameters
    ----------
    models:
        Available model names.
    key:
        Streamlit widget key (must be unique per page).
    label:
        Display label for the dropdown.

    Returns
    -------
    str or None
        The selected model name, or ``None`` if no models are available.
    """
    if not models:
        st.warning("利用可能なモデルがありません")
        return None

    current = st.session_state.get("selected_model", models[0])
    if current not in models:
        current = models[0]

    selected = st.selectbox(
        label,
        options=models,
        index=models.index(current),
        key=key,
    )
    st.session_state["selected_model"] = selected
    return selected


# ---------------------------------------------------------------------------
# Parameter Sliders
# ---------------------------------------------------------------------------


def parameter_sliders(
    key_prefix: str = "param",
) -> dict[str, Any]:
    """Render LLM and retrieval parameter sliders.

    Returns a dictionary with the user-selected values for
    ``temperature``, ``num_ctx``, and ``retrieval_k``.

    Parameters
    ----------
    key_prefix:
        Prefix for Streamlit widget keys to avoid collisions.

    Returns
    -------
    dict[str, Any]
        ``{"temperature": float, "num_ctx": int, "retrieval_k": int}``
    """
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.get(f"{key_prefix}_temperature", LLM_TEMPERATURE),
        step=0.05,
        key=f"{key_prefix}_temperature_slider",
        help="低い値は安定した回答、高い値は多様な回答を生成します。",
    )
    st.session_state[f"{key_prefix}_temperature"] = temperature

    num_ctx = st.slider(
        "コンテキストウィンドウ",
        min_value=512,
        max_value=32768,
        value=st.session_state.get(f"{key_prefix}_num_ctx", LLM_NUM_CTX),
        step=512,
        key=f"{key_prefix}_num_ctx_slider",
        help="LLMが一度に処理できるトークン数。大きいほど多くの文脈を参照できます。",
    )
    st.session_state[f"{key_prefix}_num_ctx"] = num_ctx

    retrieval_k = st.slider(
        "検索件数 (K)",
        min_value=1,
        max_value=20,
        value=st.session_state.get(f"{key_prefix}_retrieval_k", RETRIEVAL_K),
        step=1,
        key=f"{key_prefix}_retrieval_k_slider",
        help="ベクトル検索で取得する関連ドキュメントの数。",
    )
    st.session_state[f"{key_prefix}_retrieval_k"] = retrieval_k

    return {
        "temperature": temperature,
        "num_ctx": num_ctx,
        "retrieval_k": retrieval_k,
    }


# ---------------------------------------------------------------------------
# Metrics Card
# ---------------------------------------------------------------------------


def metrics_card(
    title: str,
    value: str | int | float,
    subtitle: str | None = None,
    delta: str | None = None,
) -> None:
    """Display a single metric in a styled card.

    Uses :func:`st.metric` with an optional subtitle rendered below.

    Parameters
    ----------
    title:
        Metric label.
    value:
        The primary metric value.
    subtitle:
        Optional description displayed below the metric.
    delta:
        Optional delta indicator (e.g. ``"+5%"``).
    """
    st.metric(label=title, value=value, delta=delta)
    if subtitle:
        st.caption(subtitle)


# ---------------------------------------------------------------------------
# Source Document Expander
# ---------------------------------------------------------------------------


def source_document_expander(
    source_documents: list[Any],
    header: str = "参照資料",
) -> None:
    """Render source documents in expandable sections.

    Each document is displayed with its metadata (source path, document
    type, case ID) and a content snippet.

    Parameters
    ----------
    source_documents:
        List of LangChain :class:`Document` objects (or any object with
        ``.metadata`` and ``.page_content`` attributes).
    header:
        The section header text.
    """
    if not source_documents:
        return

    st.subheader(f"{header}（{len(source_documents)} 件）")

    for i, doc in enumerate(source_documents, 1):
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        source = metadata.get("source", "不明")
        doc_type = metadata.get("document_type", "不明")
        case_id = metadata.get("case_id", "")

        label = f"[{i}] {source}"
        if case_id:
            label += f" (判例: {case_id})"

        with st.expander(label, expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"種別: {doc_type}")
            with col2:
                if case_id:
                    st.caption(f"判例ID: {case_id}")

            content = doc.page_content if hasattr(doc, "page_content") else ""
            if content:
                st.markdown(content)
