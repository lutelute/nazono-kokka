"""Settings page for the RAG Accuracy Verification & Comparison Tool.

Provides an AI backend configuration interface where users can:

- Select Ollama models from the list of available models
- Adjust LLM parameters (temperature, context window size)
- Adjust retrieval parameters (K value)
- Filter documents by type (legal framework / precedents)
- Save and load named configuration presets

Usage:
    from ui.settings_page import render_settings_page

    render_settings_page()
"""

from __future__ import annotations

import json
import logging
from typing import Any

import streamlit as st

from rag_system.backend_adapter import BackendConfig
from rag_system.config import (
    LLM_MODEL_NAME,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    PROJECT_ROOT,
    RETRIEVAL_K,
)
from ui.components import (
    model_selector,
    parameter_sliders,
    server_status_indicator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KEY_SAVED_CONFIGS = "saved_configs"
_CONFIGS_PATH = PROJECT_ROOT / "test_cases" / "saved_configs.json"

_DOCUMENT_TYPE_OPTIONS: dict[str, str | None] = {
    "すべて": None,
    "法令のみ": "legal_framework",
    "判例のみ": "precedent",
}

_CASE_TYPE_OPTIONS: dict[str, str | None] = {
    "すべて": None,
    "刑事": "criminal",
    "民事": "civil",
    "憲法": "constitutional",
}

_VERDICT_OPTIONS: dict[str, str | None] = {
    "すべて": None,
    "有罪": "有罪",
    "無罪": "無罪",
}


# ---------------------------------------------------------------------------
# Saved Configs Persistence
# ---------------------------------------------------------------------------


def _load_saved_configs() -> dict[str, dict[str, Any]]:
    """Load named configurations from disk.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of config name to serialised :class:`BackendConfig` fields.
    """
    if _KEY_SAVED_CONFIGS in st.session_state:
        return st.session_state[_KEY_SAVED_CONFIGS]

    if _CONFIGS_PATH.exists():
        try:
            with open(_CONFIGS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            configs = data.get("configs", {})
            st.session_state[_KEY_SAVED_CONFIGS] = configs
            logger.info(
                "%d 件の保存済み設定を読み込みました", len(configs)
            )
            return configs
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("設定ファイルの読み込みに失敗しました: %s", exc)

    st.session_state[_KEY_SAVED_CONFIGS] = {}
    return st.session_state[_KEY_SAVED_CONFIGS]


def _save_configs_to_disk(configs: dict[str, dict[str, Any]]) -> None:
    """Persist named configurations to a JSON file.

    Parameters
    ----------
    configs:
        Mapping of config name to serialised :class:`BackendConfig` fields.
    """
    _CONFIGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_CONFIGS_PATH, "w", encoding="utf-8") as f:
            json.dump({"configs": configs}, f, ensure_ascii=False, indent=2)
        logger.info("%d 件の設定を保存しました", len(configs))
    except OSError as exc:
        logger.error("設定ファイルの保存に失敗しました: %s", exc)


# ---------------------------------------------------------------------------
# Ollama Model Listing
# ---------------------------------------------------------------------------


def _fetch_available_models(base_url: str | None = None) -> list[str]:
    """Fetch the list of available models from the Ollama server.

    Parameters
    ----------
    base_url:
        Ollama server URL.  Defaults to
        :data:`~rag_system.config.OLLAMA_BASE_URL`.

    Returns
    -------
    list[str]
        Sorted list of model names available on the server.
    """
    import urllib.error
    import urllib.request

    if base_url is None:
        base_url = OLLAMA_BASE_URL

    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = sorted(m["name"] for m in data.get("models", []))
        return models
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        logger.error("モデル一覧の取得に失敗しました: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Document Filter Section
# ---------------------------------------------------------------------------


def _render_document_filters(key_prefix: str = "settings") -> dict[str, str | None]:
    """Render document type filter controls.

    Parameters
    ----------
    key_prefix:
        Prefix for Streamlit widget keys.

    Returns
    -------
    dict[str, str | None]
        Dictionary with ``document_type``, ``case_type``, and ``verdict``.
    """
    st.subheader("ドキュメントフィルタ")

    doc_type_label = st.selectbox(
        "ドキュメント種別",
        options=list(_DOCUMENT_TYPE_OPTIONS.keys()),
        key=f"{key_prefix}_doc_type",
        help="検索対象のドキュメント種別を絞り込みます。",
    )
    document_type = _DOCUMENT_TYPE_OPTIONS[doc_type_label]

    case_type_label = st.selectbox(
        "判例カテゴリ",
        options=list(_CASE_TYPE_OPTIONS.keys()),
        key=f"{key_prefix}_case_type",
        help="判例のカテゴリで絞り込みます。",
    )
    case_type = _CASE_TYPE_OPTIONS[case_type_label]

    verdict_label = st.selectbox(
        "判決種別",
        options=list(_VERDICT_OPTIONS.keys()),
        key=f"{key_prefix}_verdict",
        help="判決の種別で絞り込みます。",
    )
    verdict = _VERDICT_OPTIONS[verdict_label]

    return {
        "document_type": document_type,
        "case_type": case_type,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Config Save / Load Section
# ---------------------------------------------------------------------------


def _render_config_management(current_config: dict[str, Any]) -> None:
    """Render the named configuration save/load controls.

    Parameters
    ----------
    current_config:
        The current settings dictionary to be saved.
    """
    st.divider()
    st.subheader("設定プリセット")

    saved = _load_saved_configs()

    # --- Save ---
    col_name, col_save = st.columns([3, 1])
    with col_name:
        config_name = st.text_input(
            "設定名",
            key="settings_config_name",
            placeholder="例: 高精度設定",
        )
    with col_save:
        st.write("")  # spacing
        st.write("")
        save_clicked = st.button("保存", key="settings_save_btn")

    if save_clicked:
        if not config_name:
            st.warning("設定名を入力してください。")
        else:
            saved[config_name] = current_config
            st.session_state[_KEY_SAVED_CONFIGS] = saved
            _save_configs_to_disk(saved)
            st.success(f"設定 '{config_name}' を保存しました。")

    # --- Load ---
    if saved:
        st.write("**保存済み設定:**")
        selected_name = st.selectbox(
            "読み込む設定を選択",
            options=list(saved.keys()),
            key="settings_load_select",
        )

        col_load, col_delete = st.columns(2)
        with col_load:
            if st.button("読み込み", key="settings_load_btn"):
                cfg = saved[selected_name]
                _apply_config_to_session(cfg)
                st.success(f"設定 '{selected_name}' を読み込みました。")
                st.rerun()

        with col_delete:
            if st.button("削除", key="settings_delete_btn"):
                del saved[selected_name]
                st.session_state[_KEY_SAVED_CONFIGS] = saved
                _save_configs_to_disk(saved)
                st.success(f"設定 '{selected_name}' を削除しました。")
                st.rerun()
    else:
        st.info("保存済みの設定はありません。")


def _apply_config_to_session(config: dict[str, Any]) -> None:
    """Apply a saved configuration to session state.

    Parameters
    ----------
    config:
        Serialised configuration dictionary.
    """
    if "model_name" in config:
        st.session_state["selected_model"] = config["model_name"]
    if "temperature" in config:
        st.session_state["param_temperature"] = config["temperature"]
    if "num_ctx" in config:
        st.session_state["param_num_ctx"] = config["num_ctx"]
    if "retrieval_k" in config:
        st.session_state["param_retrieval_k"] = config["retrieval_k"]


# ---------------------------------------------------------------------------
# Page Entry Point
# ---------------------------------------------------------------------------


def render_settings_page() -> None:
    """Render the AI backend configuration page.

    This is the main entry point called by the app router.  It provides:

    1. Ollama server status check
    2. Model selection from available models
    3. LLM parameter sliders (temperature, context window, retrieval K)
    4. Document type filters
    5. Named configuration save/load
    """
    st.header("AI バックエンド設定")
    st.caption("モデル選択、パラメータ調整、ドキュメントフィルタの設定を行います。")

    # --- Server status ---
    is_connected = server_status_indicator()

    st.divider()

    # --- Model selection ---
    st.subheader("モデル選択")
    if is_connected:
        models = _fetch_available_models()
        if not models:
            st.warning(
                "利用可能なモデルが見つかりません。\n\n"
                "以下のコマンドでモデルをダウンロードしてください:\n\n"
                f"```\nollama pull {LLM_MODEL_NAME}\n```"
            )
            models = [LLM_MODEL_NAME]
    else:
        models = [LLM_MODEL_NAME]

    selected_model = model_selector(
        models=models,
        key="settings_model_selector",
    )

    st.divider()

    # --- Parameter sliders ---
    st.subheader("LLM / 検索パラメータ")
    params = parameter_sliders(key_prefix="param")

    st.divider()

    # --- Document filters ---
    filters = _render_document_filters()

    # --- Build current config dict ---
    current_config: dict[str, Any] = {
        "model_name": selected_model or LLM_MODEL_NAME,
        "temperature": params["temperature"],
        "num_ctx": params["num_ctx"],
        "retrieval_k": params["retrieval_k"],
        "document_type": filters["document_type"],
        "case_type": filters["case_type"],
        "verdict": filters["verdict"],
    }

    # --- Config management ---
    _render_config_management(current_config)

    # --- Current config summary ---
    st.divider()
    st.subheader("現在の設定")
    st.json(current_config)
