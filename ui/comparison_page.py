"""Comparison dashboard page for the RAG Accuracy Verification & Comparison Tool.

Provides the core comparison dashboard where users can:

- Select multiple saved configurations for side-by-side comparison
- Choose test case sets to run against
- Execute comparisons with real-time progress tracking
- View results in a side-by-side table
- Visualise metrics with Plotly bar and radar charts

Usage:
    from ui.comparison_page import render_comparison_page

    render_comparison_page()
"""

from __future__ import annotations

import json
import logging
from typing import Any, List

import streamlit as st

from rag_system.backend_adapter import BackendConfig
from rag_system.comparison_engine import ComparisonEngine
from rag_system.config import LLM_MODEL_NAME, PROJECT_ROOT
from rag_system.metrics import EvaluationResult
from rag_system.test_cases import TestCase, TestCaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants / Session State Keys
# ---------------------------------------------------------------------------

_KEY_SAVED_CONFIGS = "saved_configs"
_KEY_COMPARISON_RESULTS = "comparison_results"
_KEY_TESTCASE_MANAGER = "testcase_manager"

_CONFIGS_PATH = PROJECT_ROOT / "test_cases" / "saved_configs.json"

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
# Helpers
# ---------------------------------------------------------------------------


def _load_saved_configs() -> dict[str, dict[str, Any]]:
    """Load named configurations from disk or session state.

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
            return configs
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("設定ファイルの読み込みに失敗しました: %s", exc)

    st.session_state[_KEY_SAVED_CONFIGS] = {}
    return st.session_state[_KEY_SAVED_CONFIGS]


def _get_testcase_manager() -> TestCaseManager:
    """Return the shared :class:`TestCaseManager`, initializing if needed.

    Returns
    -------
    TestCaseManager
        The shared test case manager stored in session state.
    """
    if _KEY_TESTCASE_MANAGER not in st.session_state:
        manager = TestCaseManager()
        manager.load_default_cases()
        st.session_state[_KEY_TESTCASE_MANAGER] = manager
    return st.session_state[_KEY_TESTCASE_MANAGER]


def _build_backend_config(name: str, cfg: dict[str, Any]) -> BackendConfig:
    """Build a :class:`BackendConfig` from a saved config dict.

    Parameters
    ----------
    name:
        Config name (used as identifier in comparison results).
    cfg:
        Serialised configuration dictionary.

    Returns
    -------
    BackendConfig
        The constructed backend configuration.
    """
    return BackendConfig(
        name=name,
        model_name=cfg.get("model_name", LLM_MODEL_NAME),
        temperature=cfg.get("temperature", 0.1),
        num_ctx=cfg.get("num_ctx", 4096),
        retrieval_k=cfg.get("retrieval_k", 4),
        document_type=cfg.get("document_type"),
        case_type=cfg.get("case_type"),
        verdict=cfg.get("verdict"),
    )


# ---------------------------------------------------------------------------
# Config Selection Section
# ---------------------------------------------------------------------------


def _render_config_selection(
    saved_configs: dict[str, dict[str, Any]],
) -> list[str]:
    """Render multi-select for saved configurations.

    Parameters
    ----------
    saved_configs:
        Available named configurations.

    Returns
    -------
    list[str]
        Selected configuration names.
    """
    st.subheader("比較対象の設定")

    if not saved_configs:
        st.warning(
            "保存済みの設定がありません。\n\n"
            "「設定」ページで設定を作成・保存してください。"
        )
        return []

    config_names = list(saved_configs.keys())
    selected = st.multiselect(
        "比較する設定を選択（複数可）",
        options=config_names,
        default=[],
        key="comparison_config_select",
        help="2つ以上の設定を選択して精度を比較できます。",
    )

    if selected:
        with st.expander("選択した設定の詳細", expanded=False):
            for name in selected:
                cfg = saved_configs[name]
                st.write(f"**{name}:**")
                st.json(cfg)
    return selected


# ---------------------------------------------------------------------------
# Test Case Selection Section
# ---------------------------------------------------------------------------


def _render_testcase_selection(
    manager: TestCaseManager,
) -> list[TestCase]:
    """Render test case set selection with category/difficulty filters.

    Parameters
    ----------
    manager:
        The test case manager providing case data.

    Returns
    -------
    list[TestCase]
        The selected test cases.
    """
    st.subheader("テストケースの選択")

    all_cases = manager.list_cases()
    if not all_cases:
        st.warning(
            "テストケースが登録されていません。\n\n"
            "「テストケース」ページでテストケースを作成してください。"
        )
        return []

    # --- Filters ---
    col_cat, col_diff = st.columns(2)
    categories = sorted({tc.category for tc in all_cases})
    difficulties = sorted({tc.difficulty for tc in all_cases})

    with col_cat:
        filter_categories = st.multiselect(
            "カテゴリで絞り込み",
            options=categories,
            default=categories,
            format_func=lambda x: _CATEGORY_LABELS.get(x, x),
            key="comparison_filter_category",
        )
    with col_diff:
        filter_difficulties = st.multiselect(
            "難易度で絞り込み",
            options=difficulties,
            default=difficulties,
            format_func=lambda x: _DIFFICULTY_LABELS.get(x, x),
            key="comparison_filter_difficulty",
        )

    filtered = [
        tc
        for tc in all_cases
        if tc.category in filter_categories and tc.difficulty in filter_difficulties
    ]

    st.info(f"対象テストケース: {len(filtered)} / {len(all_cases)} 件")

    if filtered:
        with st.expander("対象テストケース一覧", expanded=False):
            for tc in filtered:
                cat_label = _CATEGORY_LABELS.get(tc.category, tc.category)
                diff_label = _DIFFICULTY_LABELS.get(tc.difficulty, tc.difficulty)
                st.write(f"- **[{tc.id}]** {cat_label}/{diff_label} — {tc.query[:60]}")

    return filtered


# ---------------------------------------------------------------------------
# Comparison Execution
# ---------------------------------------------------------------------------


def _render_comparison_execution(
    selected_config_names: list[str],
    saved_configs: dict[str, dict[str, Any]],
    test_cases: list[TestCase],
) -> None:
    """Render the comparison execution button and progress bar.

    Parameters
    ----------
    selected_config_names:
        Names of the selected configurations.
    saved_configs:
        Full saved configurations dictionary.
    test_cases:
        Test cases to run.
    """
    st.divider()
    st.subheader("比較実行")

    can_run = len(selected_config_names) >= 1 and len(test_cases) >= 1
    total_steps = len(selected_config_names) * len(test_cases)

    if can_run:
        st.write(
            f"**{len(selected_config_names)}** 設定 × **{len(test_cases)}** "
            f"テストケース = **{total_steps}** ステップを実行します。"
        )
    else:
        if len(selected_config_names) < 1:
            st.info("比較対象の設定を1つ以上選択してください。")
        if len(test_cases) < 1:
            st.info("テストケースを1つ以上選択してください。")

    run_clicked = st.button(
        "比較を実行",
        disabled=not can_run,
        key="comparison_run_btn",
        type="primary",
    )

    if run_clicked:
        configs = [
            _build_backend_config(name, saved_configs[name])
            for name in selected_config_names
        ]

        engine = ComparisonEngine()
        progress_bar = st.progress(0)
        status_text = st.empty()

        def _progress_callback(current: int, total: int, message: str) -> None:
            progress_bar.progress(current / total if total > 0 else 0)
            status_text.text(f"[{current}/{total}] {message}")

        with st.spinner("比較実行中..."):
            results = engine.run_comparison(
                configs=configs,
                test_cases=test_cases,
                progress_callback=_progress_callback,
                save_results=True,
            )

        progress_bar.progress(1.0)
        status_text.text("比較実行が完了しました。")

        if results:
            st.session_state[_KEY_COMPARISON_RESULTS] = results
            st.success(f"比較完了: {len(results)} 件の結果を取得しました。")
        else:
            st.warning("比較結果が空です。設定やテストケースを確認してください。")


# ---------------------------------------------------------------------------
# Results Table (Side-by-Side)
# ---------------------------------------------------------------------------


def _render_results_table(results: list[EvaluationResult]) -> None:
    """Render side-by-side comparison results table.

    Parameters
    ----------
    results:
        Evaluation results from the comparison engine.
    """
    try:
        import pandas as pd
    except ImportError:
        st.error("pandas がインストールされていません。")
        return

    st.subheader("比較結果テーブル")

    rows = []
    for r in results:
        rows.append(
            {
                "設定名": r.config_name,
                "テストケースID": r.test_case_id,
                "総合スコア": round(r.overall_score, 3),
                "キーワードスコア": round(r.keyword_score, 3),
                "条文スコア": round(r.statute_score, 3),
                "判例IDスコア": round(r.case_id_score, 3),
                "応答時間(秒)": round(r.elapsed_time, 2),
                "ソース数": r.source_count,
            }
        )

    df = pd.DataFrame(rows)

    # --- Summary by config ---
    st.write("**設定ごとの平均スコア:**")
    summary = (
        df.groupby("設定名")
        .agg(
            {
                "総合スコア": "mean",
                "キーワードスコア": "mean",
                "条文スコア": "mean",
                "判例IDスコア": "mean",
                "応答時間(秒)": "mean",
                "ソース数": "mean",
            }
        )
        .round(3)
    )
    st.dataframe(summary, use_container_width=True)

    # --- Full detail table ---
    with st.expander("全結果の詳細テーブル", expanded=False):
        st.dataframe(df, use_container_width=True)

    # --- Response details ---
    with st.expander("応答テキストの詳細", expanded=False):
        for r in results:
            st.write(f"**{r.config_name} / {r.test_case_id}:**")
            st.text(r.response_text[:500] if r.response_text else "（応答なし）")
            st.divider()


# ---------------------------------------------------------------------------
# Metrics Charts (Plotly)
# ---------------------------------------------------------------------------


def _render_metrics_charts(results: list[EvaluationResult]) -> None:
    """Render Plotly charts for comparison metrics.

    Includes bar charts for scores and response times, and a radar chart
    for multi-dimensional score comparison.

    Parameters
    ----------
    results:
        Evaluation results from the comparison engine.
    """
    try:
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        st.warning("plotly または pandas がインストールされていません。チャートを表示できません。")
        return

    st.subheader("メトリクスチャート")

    # --- Aggregate by config ---
    config_names = sorted({r.config_name for r in results})
    agg_data: dict[str, dict[str, float]] = {}

    for config_name in config_names:
        config_results = [r for r in results if r.config_name == config_name]
        n = len(config_results)
        if n == 0:
            continue
        agg_data[config_name] = {
            "総合スコア": sum(r.overall_score for r in config_results) / n,
            "キーワードスコア": sum(r.keyword_score for r in config_results) / n,
            "条文スコア": sum(r.statute_score for r in config_results) / n,
            "判例IDスコア": sum(r.case_id_score for r in config_results) / n,
            "平均応答時間": sum(r.elapsed_time for r in config_results) / n,
        }

    if not agg_data:
        st.info("チャートを表示するデータがありません。")
        return

    # --- Bar Chart: Score Comparison ---
    st.write("**スコア比較（棒グラフ）**")
    bar_rows = []
    for config_name, metrics in agg_data.items():
        for metric_name in ["総合スコア", "キーワードスコア", "条文スコア", "判例IDスコア"]:
            bar_rows.append(
                {
                    "設定名": config_name,
                    "メトリクス": metric_name,
                    "スコア": round(metrics[metric_name], 3),
                }
            )

    bar_df = pd.DataFrame(bar_rows)
    fig_bar = px.bar(
        bar_df,
        x="メトリクス",
        y="スコア",
        color="設定名",
        barmode="group",
        title="設定ごとのスコア比較",
        range_y=[0, 1],
    )
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Bar Chart: Response Time ---
    st.write("**応答時間比較（棒グラフ）**")
    time_rows = []
    for config_name, metrics in agg_data.items():
        time_rows.append(
            {
                "設定名": config_name,
                "平均応答時間(秒)": round(metrics["平均応答時間"], 2),
            }
        )

    time_df = pd.DataFrame(time_rows)
    fig_time = px.bar(
        time_df,
        x="設定名",
        y="平均応答時間(秒)",
        color="設定名",
        title="設定ごとの平均応答時間",
    )
    fig_time.update_layout(height=350)
    st.plotly_chart(fig_time, use_container_width=True)

    # --- Radar Chart: Multi-dimensional comparison ---
    if len(config_names) >= 2:
        st.write("**レーダーチャート（多次元比較）**")
        radar_categories = ["総合スコア", "キーワードスコア", "条文スコア", "判例IDスコア"]

        fig_radar = go.Figure()
        for config_name in config_names:
            if config_name not in agg_data:
                continue
            metrics = agg_data[config_name]
            values = [metrics[cat] for cat in radar_categories]
            # Close the radar shape
            values.append(values[0])
            cats = radar_categories + [radar_categories[0]]

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=cats,
                    fill="toself",
                    name=config_name,
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="設定間の多次元スコア比較",
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ---------------------------------------------------------------------------
# Past Results Viewer
# ---------------------------------------------------------------------------


def _render_past_results() -> None:
    """Render a viewer for previously saved comparison results."""
    engine = ComparisonEngine()
    result_files = engine.list_result_files()

    if not result_files:
        st.info("過去の比較結果はありません。")
        return

    st.subheader("過去の比較結果")

    file_labels = [f.stem for f in result_files]
    selected_label = st.selectbox(
        "結果ファイルを選択",
        options=file_labels,
        key="comparison_past_results_select",
    )

    if selected_label is None:
        return

    selected_idx = file_labels.index(selected_label)
    selected_file = result_files[selected_idx]

    raw_results = engine.load_results(selected_file)
    if not raw_results:
        st.warning("結果データが空です。")
        return

    # Convert raw dicts to EvaluationResult objects
    eval_results: list[EvaluationResult] = []
    for r in raw_results:
        eval_results.append(
            EvaluationResult(
                config_name=r.get("config_name", ""),
                test_case_id=r.get("test_case_id", ""),
                response_text=r.get("response_text", ""),
                source_documents=[],
                elapsed_time=r.get("elapsed_time", 0.0),
                keyword_hits=r.get("keyword_hits", 0),
                keyword_total=r.get("keyword_total", 0),
                statute_hits=r.get("statute_hits", 0),
                statute_total=r.get("statute_total", 0),
                case_id_hits=r.get("case_id_hits", 0),
                case_id_total=r.get("case_id_total", 0),
                source_count=r.get("source_count", 0),
            )
        )

    _render_results_table(eval_results)
    _render_metrics_charts(eval_results)


# ---------------------------------------------------------------------------
# Page Entry Point
# ---------------------------------------------------------------------------


def render_comparison_page() -> None:
    """Render the comparison dashboard page.

    This is the main entry point called by the app router.  It provides:

    1. Multi-select for saved configurations
    2. Test case set selection with category/difficulty filters
    3. Comparison execution with progress bar
    4. Side-by-side results table
    5. Plotly bar and radar charts for metrics visualisation
    6. Past results viewer
    """
    st.header("精度比較ダッシュボード")
    st.caption(
        "複数のAI設定を同一テストケースで比較し、精度・応答速度を可視化します。"
    )

    # --- Load data ---
    saved_configs = _load_saved_configs()
    manager = _get_testcase_manager()

    # --- Tab layout ---
    tab_new, tab_history = st.tabs(["新規比較", "過去の結果"])

    with tab_new:
        # --- Config selection ---
        selected_names = _render_config_selection(saved_configs)

        st.divider()

        # --- Test case selection ---
        selected_cases = _render_testcase_selection(manager)

        # --- Run comparison ---
        _render_comparison_execution(selected_names, saved_configs, selected_cases)

        # --- Display results ---
        results = st.session_state.get(_KEY_COMPARISON_RESULTS)
        if results:
            st.divider()
            _render_results_table(results)
            _render_metrics_charts(results)

    with tab_history:
        _render_past_results()
