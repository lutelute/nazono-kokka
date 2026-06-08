"""Integration smoke tests for the vector visualization Streamlit page.

These tests use Streamlit's AppTest harness to actually run the page,
shake out import errors, and verify that all 8 sections render under
varied UI states (PCA vs t-SNE, 2D vs 3D, different cluster counts).

They skip if ChromaDB is not initialized, since the page is meaningless
without the corpus.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest


warnings.filterwarnings("ignore")


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_APP_PATH = _PROJECT_ROOT / "app.py"
_CHROMA_DIR = _PROJECT_ROOT / "chroma_db"


pytestmark = pytest.mark.skipif(
    not _CHROMA_DIR.exists(),
    reason="ChromaDB が初期化されていません",
)


def _run_page(**overrides) -> AppTest:
    """Launch the app on the visualization page with optional session state."""
    app = AppTest.from_file(str(_APP_PATH), default_timeout=300)
    app.session_state["page_selector"] = "ベクトル可視化"
    for key, value in overrides.items():
        app.session_state[key] = value
    app.run()
    return app


def _page_errors(app: AppTest) -> list[str]:
    """Errors raised by the visualization page itself.

    Filters out the sidebar's "Ollama: 未接続" badge, which is environmental
    (depends on whether the dev has the Ollama daemon running), not a real
    failure of the page logic.
    """
    return [
        str(e.value) for e in app.error
        if "Ollama" not in str(e.value)
    ]


class TestVisualizationPage:
    def test_default_view_renders(self) -> None:
        app = _run_page()
        assert len(app.exception) == 0, [str(e.value)[:200] for e in app.exception]
        assert _page_errors(app) == []
        # All 10 visualization sections (0..9) + sidebar
        section_titles = [s.value for s in app.subheader]
        for n in range(0, 10):
            assert any(s.startswith(f"{n}.") for s in section_titles), (
                f"section {n} missing from {section_titles}"
            )

    def test_3d_archive_map(self) -> None:
        app = _run_page(viz_archive_dims="3D")
        assert len(app.exception) == 0
        assert _page_errors(app) == []

    def test_tsne_archive_map(self) -> None:
        app = _run_page(viz_archive_method="tsne")
        assert len(app.exception) == 0
        assert _page_errors(app) == []

    def test_top_level_metrics_present(self) -> None:
        app = _run_page()
        labels = {m.label for m in app.metric}
        assert "総チャンク数" in labels
        assert "ベクトル次元" in labels

    def test_query_neighborhood_metrics_present(self) -> None:
        app = _run_page()
        labels = [m.label for m in app.metric]
        # Min/max/mean/median similarity from the histogram section
        assert "最大類似度" in labels
        assert "平均類似度" in labels

    def test_story_lab_cross_probe_runs(self) -> None:
        app = _run_page(viz_story_cross_probe="桃太郎")
        assert len(app.exception) == 0
        # Either success or warning result from the boundary check
        all_messages = (
            [s.value for s in app.success]
            + [w.value for w in app.warning]
        )
        assert any("類似度" in m for m in all_messages)

    def test_changing_kmeans_clusters(self) -> None:
        app = _run_page(viz_cluster_k=12)
        assert len(app.exception) == 0
        assert _page_errors(app) == []

    def test_rag_flow_section_runs_without_ollama(self) -> None:
        """The RAG flow section must show steps 1-3 even if Ollama is offline."""
        app = _run_page(viz_rag_flow_run_llm=False)
        assert len(app.exception) == 0
        # Section 0 should be present
        assert any(
            s.value.startswith("0.")
            for s in app.subheader
        )

    def test_pipeline_lab_present(self) -> None:
        """Section 9 (pipeline comparison) renders and is idle until run."""
        app = _run_page()
        assert len(app.exception) == 0
        assert any(s.value.startswith("9.") for s in app.subheader)

    def test_pipeline_lab_runs_on_button(self) -> None:
        """Pressing the run button executes dense→hybrid→rerank without error."""
        app = _run_page(viz_pipeline_query="窃盗罪の量刑")
        # Find and click the pipeline run button.
        run_btns = [b for b in app.button if b.key == "viz_pipe_run"]
        assert run_btns, "pipeline run button not found"
        run_btns[0].set_value(True).run()
        assert len(app.exception) == 0, [str(e.value)[:200] for e in app.exception]
        # The three-pipeline comparison header should appear.
        markdowns = [m.value for m in app.markdown]
        assert any("最終 top-k の比較" in m for m in markdowns)
