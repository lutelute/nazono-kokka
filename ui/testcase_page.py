"""Test case management page for the RAG Accuracy Verification & Comparison Tool.

Provides a web interface for managing verification test cases:

- CRUD form for creating, viewing, editing, and deleting test cases
- Import/export functionality (JSON files)
- Filterable case list display with category and difficulty badges

Usage:
    from ui.testcase_page import render_testcase_page

    render_testcase_page()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import streamlit as st

from rag_system.test_cases import TestCase, TestCaseManager, _generate_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session State Keys
# ---------------------------------------------------------------------------

_KEY_MANAGER = "testcase_manager"
_KEY_EDITING_ID = "testcase_editing_id"

_CATEGORY_OPTIONS = ["criminal", "civil", "constitutional"]
_DIFFICULTY_OPTIONS = ["basic", "intermediate", "advanced"]

_CATEGORY_LABELS: dict[str, str] = {
    "criminal": "刑事",
    "civil": "民事",
    "constitutional": "憲法",
}

_DIFFICULTY_LABELS: dict[str, str] = {
    "basic": "基礎",
    "intermediate": "中級",
    "advanced": "上級",
}


# ---------------------------------------------------------------------------
# Manager Access
# ---------------------------------------------------------------------------


def _get_manager() -> TestCaseManager:
    """Return the shared :class:`TestCaseManager`, initializing if needed.

    On first call, loads the default test cases from disk.

    Returns
    -------
    TestCaseManager
        The shared test case manager stored in session state.
    """
    if _KEY_MANAGER not in st.session_state:
        manager = TestCaseManager()
        manager.load_default_cases()
        st.session_state[_KEY_MANAGER] = manager
        logger.info("テストケースマネージャを初期化しました")
    return st.session_state[_KEY_MANAGER]


# ---------------------------------------------------------------------------
# CRUD Form
# ---------------------------------------------------------------------------


def _render_case_form(
    existing: TestCase | None = None,
) -> TestCase | None:
    """Render the test case creation/edit form.

    Parameters
    ----------
    existing:
        If provided, populates the form with this test case's data for
        editing.  ``None`` creates a blank form for new cases.

    Returns
    -------
    TestCase or None
        The submitted test case, or ``None`` if the form was not submitted.
    """
    is_edit = existing is not None
    form_key = "testcase_edit_form" if is_edit else "testcase_add_form"
    submit_label = "更新" if is_edit else "追加"

    with st.form(form_key, clear_on_submit=not is_edit):
        st.subheader("編集" if is_edit else "新規テストケース")

        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox(
                "カテゴリ",
                options=_CATEGORY_OPTIONS,
                index=(
                    _CATEGORY_OPTIONS.index(existing.category)
                    if is_edit and existing.category in _CATEGORY_OPTIONS
                    else 0
                ),
                format_func=lambda x: _CATEGORY_LABELS.get(x, x),
                key=f"{form_key}_category",
            )
        with col2:
            difficulty = st.selectbox(
                "難易度",
                options=_DIFFICULTY_OPTIONS,
                index=(
                    _DIFFICULTY_OPTIONS.index(existing.difficulty)
                    if is_edit and existing.difficulty in _DIFFICULTY_OPTIONS
                    else 0
                ),
                format_func=lambda x: _DIFFICULTY_LABELS.get(x, x),
                key=f"{form_key}_difficulty",
            )

        query = st.text_area(
            "質問文",
            value=existing.query if is_edit else "",
            key=f"{form_key}_query",
            height=100,
        )

        description = st.text_input(
            "説明",
            value=existing.description if is_edit else "",
            key=f"{form_key}_description",
        )

        keywords_str = st.text_input(
            "期待キーワード（カンマ区切り）",
            value=", ".join(existing.expected_keywords) if is_edit else "",
            key=f"{form_key}_keywords",
            help="例: 刑法, 窃盗, 量刑",
        )

        statutes_str = st.text_input(
            "期待条文（カンマ区切り）",
            value=", ".join(existing.expected_statutes) if is_edit else "",
            key=f"{form_key}_statutes",
            help="例: 刑法第235条, 刑法第236条",
        )

        case_ids_str = st.text_input(
            "期待判例ID（カンマ区切り）",
            value=", ".join(existing.expected_case_ids) if is_edit else "",
            key=f"{form_key}_case_ids",
            help="例: CRIMINAL-2019-0012",
        )

        submitted = st.form_submit_button(submit_label)

    if not submitted:
        return None

    if not query.strip():
        st.warning("質問文を入力してください。")
        return None

    case_id = existing.id if is_edit else _generate_id()
    expected_keywords = [
        k.strip() for k in keywords_str.split(",") if k.strip()
    ]
    expected_statutes = [
        s.strip() for s in statutes_str.split(",") if s.strip()
    ]
    expected_case_ids = [
        c.strip() for c in case_ids_str.split(",") if c.strip()
    ]

    return TestCase(
        id=case_id,
        category=category,
        query=query.strip(),
        expected_keywords=expected_keywords,
        expected_statutes=expected_statutes,
        expected_case_ids=expected_case_ids,
        difficulty=difficulty,
        description=description.strip(),
    )


# ---------------------------------------------------------------------------
# Import / Export
# ---------------------------------------------------------------------------


def _render_import_export(manager: TestCaseManager) -> None:
    """Render the import/export controls.

    Parameters
    ----------
    manager:
        The test case manager to import into / export from.
    """
    st.subheader("インポート / エクスポート")

    # One-click loader for the data-derived suite generated by
    # scripts/generate_test_cases.py (127 cases across all case types).
    generated_path = Path(__file__).resolve().parent.parent / "test_cases" / "generated_cases.json"
    if generated_path.exists():
        with st.container(border=True):
            st.markdown(
                "**📦 判例由来の拡張テストスイート** ― "
                "全 case_type から自動生成した大規模テストケース"
                "（精度比較を統計的に意味のある規模にします）。"
            )
            if st.button("拡張スイートを読み込む", key="testcase_load_generated"):
                try:
                    data = json.loads(generated_path.read_text(encoding="utf-8"))
                    existing_ids = {tc.id for tc in manager.list_cases()}
                    added = 0
                    for item in data.get("test_cases", []):
                        tc = TestCase.from_dict(item)
                        if tc.id in existing_ids:
                            continue
                        manager.add(tc)
                        added += 1
                    manager.save_to_file()
                    st.success(f"拡張スイートから {added} 件を追加しました。")
                    st.rerun()
                except (json.JSONDecodeError, OSError, KeyError) as exc:
                    logger.error("拡張スイートの読み込みに失敗: %s", exc)
                    st.error("拡張スイートの読み込みに失敗しました。")

    col_import, col_export = st.columns(2)

    with col_import:
        uploaded = st.file_uploader(
            "JSONファイルをインポート",
            type=["json"],
            key="testcase_import_file",
        )
        if uploaded is not None:
            if st.button("インポート実行", key="testcase_import_btn"):
                try:
                    data = json.loads(uploaded.read().decode("utf-8"))
                    raw_cases = data.get("test_cases", [])
                    count = 0
                    for item in raw_cases:
                        tc = TestCase.from_dict(item)
                        manager.add(tc)
                        count += 1
                    manager.save_to_file()
                    st.success(f"{count} 件のテストケースをインポートしました。")
                    st.rerun()
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.error("インポートに失敗しました: %s", exc)
                    st.error("JSONファイルの形式が正しくありません。")

    with col_export:
        cases = manager.list_cases()
        if cases:
            payload = {
                "test_cases": [tc.to_dict() for tc in cases],
            }
            json_str = json.dumps(payload, ensure_ascii=False, indent=2)
            st.download_button(
                "エクスポート (JSON)",
                data=json_str,
                file_name="test_cases_export.json",
                mime="application/json",
                key="testcase_export_btn",
            )
        else:
            st.info("エクスポートするテストケースがありません。")


# ---------------------------------------------------------------------------
# Case List Display
# ---------------------------------------------------------------------------


def _render_case_list(manager: TestCaseManager) -> None:
    """Render the test case list with filter and action buttons.

    Parameters
    ----------
    manager:
        The test case manager providing the case data.
    """
    cases = manager.list_cases()

    st.subheader(f"テストケース一覧（{len(cases)} 件）")

    if not cases:
        st.info(
            "テストケースが登録されていません。\n\n"
            "上のフォームから新規作成するか、JSONファイルをインポートしてください。"
        )
        return

    # --- Filters ---
    col_cat, col_diff = st.columns(2)
    with col_cat:
        filter_category = st.selectbox(
            "カテゴリで絞り込み",
            options=["すべて"] + _CATEGORY_OPTIONS,
            format_func=lambda x: (
                _CATEGORY_LABELS.get(x, x) if x != "すべて" else x
            ),
            key="testcase_filter_category",
        )
    with col_diff:
        filter_difficulty = st.selectbox(
            "難易度で絞り込み",
            options=["すべて"] + _DIFFICULTY_OPTIONS,
            format_func=lambda x: (
                _DIFFICULTY_LABELS.get(x, x) if x != "すべて" else x
            ),
            key="testcase_filter_difficulty",
        )

    filtered = cases
    if filter_category != "すべて":
        filtered = [c for c in filtered if c.category == filter_category]
    if filter_difficulty != "すべて":
        filtered = [c for c in filtered if c.difficulty == filter_difficulty]

    if not filtered:
        st.info("条件に一致するテストケースがありません。")
        return

    # --- Case cards ---
    for tc in filtered:
        cat_label = _CATEGORY_LABELS.get(tc.category, tc.category)
        diff_label = _DIFFICULTY_LABELS.get(tc.difficulty, tc.difficulty)

        with st.expander(
            f"[{tc.id}] {cat_label} / {diff_label} — {tc.query[:60]}...",
            expanded=False,
        ):
            st.write(f"**質問:** {tc.query}")
            if tc.description:
                st.write(f"**説明:** {tc.description}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**期待キーワード:**")
                st.write(
                    ", ".join(tc.expected_keywords)
                    if tc.expected_keywords
                    else "（未設定）"
                )
            with col2:
                st.write("**期待条文:**")
                st.write(
                    ", ".join(tc.expected_statutes)
                    if tc.expected_statutes
                    else "（未設定）"
                )
            with col3:
                st.write("**期待判例ID:**")
                st.write(
                    ", ".join(tc.expected_case_ids)
                    if tc.expected_case_ids
                    else "（未設定）"
                )

            col_edit, col_del = st.columns(2)
            with col_edit:
                if st.button("編集", key=f"edit_{tc.id}"):
                    st.session_state[_KEY_EDITING_ID] = tc.id
                    st.rerun()
            with col_del:
                if st.button("削除", key=f"delete_{tc.id}"):
                    manager.delete(tc.id)
                    manager.save_to_file()
                    st.success(f"テストケース '{tc.id}' を削除しました。")
                    st.rerun()


# ---------------------------------------------------------------------------
# Page Entry Point
# ---------------------------------------------------------------------------


def render_testcase_page() -> None:
    """Render the test case management page.

    This is the main entry point called by the app router.  It provides:

    1. CRUD form for creating and editing test cases
    2. Import/export buttons for JSON files
    3. Filterable case list with category and difficulty badges
    """
    st.header("テストケース管理")
    st.caption("検証用テストケースの作成・編集・削除、インポート・エクスポートを行います。")

    manager = _get_manager()

    # --- Edit mode ---
    editing_id = st.session_state.get(_KEY_EDITING_ID)
    if editing_id is not None:
        existing = manager.get(editing_id)
        if existing is not None:
            if st.button("← 編集をキャンセル", key="testcase_cancel_edit"):
                st.session_state.pop(_KEY_EDITING_ID, None)
                st.rerun()

            updated = _render_case_form(existing=existing)
            if updated is not None:
                manager.update(
                    editing_id,
                    category=updated.category,
                    query=updated.query,
                    expected_keywords=updated.expected_keywords,
                    expected_statutes=updated.expected_statutes,
                    expected_case_ids=updated.expected_case_ids,
                    difficulty=updated.difficulty,
                    description=updated.description,
                )
                manager.save_to_file()
                st.session_state.pop(_KEY_EDITING_ID, None)
                st.success(f"テストケース '{editing_id}' を更新しました。")
                st.rerun()
            return
        else:
            st.session_state.pop(_KEY_EDITING_ID, None)

    # --- Add form ---
    new_case = _render_case_form()
    if new_case is not None:
        manager.add(new_case)
        manager.save_to_file()
        st.success(f"テストケース '{new_case.id}' を追加しました。")
        st.rerun()

    st.divider()

    # --- Import / Export ---
    _render_import_export(manager)

    st.divider()

    # --- Case list ---
    _render_case_list(manager)
