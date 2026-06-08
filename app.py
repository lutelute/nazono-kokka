"""Streamlit entry point for the RAG Accuracy Verification & Comparison Tool.

Provides a multi-page web interface for the judicial RAG system with:
- Chat page: interactive legal Q&A (replicating CLI functionality)
- Settings page: AI backend configuration (model selection, parameters)
- Test case page: verification case management (CRUD)
- Comparison page: accuracy comparison dashboard with metrics

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

import streamlit as st

from rag_system.config import OLLAMA_BASE_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

PAGE_CHAT = "チャット"
PAGE_VISUALIZATION = "ベクトル可視化"
PAGE_SETTINGS = "設定"
PAGE_TESTCASES = "テストケース"
PAGE_COMPARISON = "精度比較"

PAGES = [PAGE_CHAT, PAGE_VISUALIZATION, PAGE_SETTINGS, PAGE_TESTCASES, PAGE_COMPARISON]


# ---------------------------------------------------------------------------
# Ollama Server Utilities
# ---------------------------------------------------------------------------


def check_ollama_status() -> bool:
    """Check whether the Ollama server is reachable.

    Returns
    -------
    bool
        ``True`` if the server responds, ``False`` otherwise.
    """
    try:
        req = urllib.request.Request(OLLAMA_BASE_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5):
            pass
        return True
    except (urllib.error.URLError, OSError):
        return False


def fetch_available_models() -> list[str]:
    """Fetch the list of available models from the Ollama API.

    Returns
    -------
    list[str]
        Model names available on the Ollama server.
        Returns an empty list if the server is unreachable.
    """
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags", method="GET"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            return sorted(models)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        logger.warning("Ollamaサーバーからモデル一覧を取得できません")
        return []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    """Render the sidebar with server status, model selection, and navigation.

    Returns
    -------
    str
        The selected page name.
    """
    with st.sidebar:
        st.title("謎の国家")
        st.caption("RAG 精度検証・比較ツール")
        st.divider()

        # --- Ollama server status ---
        ollama_online = check_ollama_status()
        if ollama_online:
            st.success("Ollama: 接続済み", icon="\u2705")
        else:
            st.error("Ollama: 未接続", icon="\u274c")
            st.info(
                "Ollamaサーバーを起動してください:\n\n"
                "```\nollama serve\n```"
            )

        # --- Model selection ---
        st.subheader("モデル選択")
        models = fetch_available_models() if ollama_online else []

        if models:
            # Use session_state default if previously selected
            current_model = st.session_state.get("selected_model", models[0])
            if current_model not in models:
                current_model = models[0]

            selected = st.selectbox(
                "使用モデル",
                options=models,
                index=models.index(current_model),
                key="model_selector",
            )
            st.session_state["selected_model"] = selected
        else:
            st.warning("利用可能なモデルがありません")
            if ollama_online:
                st.info(
                    "モデルをダウンロードしてください:\n\n"
                    "```\nollama pull llama3.1:8b\n```"
                )

        st.divider()

        # --- Page navigation ---
        st.subheader("ナビゲーション")
        selected_page = st.radio(
            "ページ選択",
            options=PAGES,
            index=0,
            key="page_selector",
            label_visibility="collapsed",
        )

    return selected_page


# ---------------------------------------------------------------------------
# Page Routing
# ---------------------------------------------------------------------------


def render_page(page: str) -> None:
    """Route to the appropriate page renderer.

    Parameters
    ----------
    page:
        The page name to render.
    """
    if page == PAGE_CHAT:
        _render_chat_placeholder()
    elif page == PAGE_VISUALIZATION:
        _render_visualization_placeholder()
    elif page == PAGE_SETTINGS:
        _render_settings_placeholder()
    elif page == PAGE_TESTCASES:
        _render_testcases_placeholder()
    elif page == PAGE_COMPARISON:
        _render_comparison_placeholder()


def _render_chat_placeholder() -> None:
    """Render the chat page (delegates to ui.chat_page)."""
    from ui.chat_page import render_chat_page

    render_chat_page()


def _render_visualization_placeholder() -> None:
    """Render the vector visualization page."""
    from ui.visualization_page import render_visualization_page

    render_visualization_page()


def _render_settings_placeholder() -> None:
    """Render the settings page (delegates to ui.settings_page)."""
    from ui.settings_page import render_settings_page

    render_settings_page()


def _render_testcases_placeholder() -> None:
    """Render the test case page (delegates to ui.testcase_page)."""
    from ui.testcase_page import render_testcase_page

    render_testcase_page()


def _render_comparison_placeholder() -> None:
    """Render the comparison page (delegates to ui.comparison_page)."""
    from ui.comparison_page import render_comparison_page

    render_comparison_page()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title="謎の国家 — RAG精度検証ツール",
        page_icon="\u2696\ufe0f",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    selected_page = render_sidebar()
    render_page(selected_page)


if __name__ == "__main__":
    main()
