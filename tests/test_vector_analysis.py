"""Tests for rag_system.vector_analysis module.

Covers the pure-numpy primitives (cosine similarity, top-K search,
dimensionality reduction, group labels) with no dependency on heavy
ML libraries, plus a couple of light integration smokes against the
persisted ChromaDB collection when present.
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from rag_system.vector_analysis import (
    _embedding_fingerprint,
    cosine_similarity,
    cosine_similarity_matrix,
    group_label,
    kmeans_clusters,
    project_cached,
    project_to_2d,
    project_to_3d,
    short_preview,
    top_k_similar,
)


# ---------------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_vectors(self) -> None:
        u = np.array([1.0, 0.0], dtype=np.float32)
        v = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(u, v) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_returns_zero(self) -> None:
        u = np.zeros(3, dtype=np.float32)
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(u, v) == 0.0
        assert cosine_similarity(v, u) == 0.0


# ---------------------------------------------------------------------------
# Cosine Similarity Matrix
# ---------------------------------------------------------------------------


class TestCosineSimilarityMatrix:
    def test_self_similarity_diagonal_is_one(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(size=(5, 8)).astype(np.float32)
        sim = cosine_similarity_matrix(a)
        assert sim.shape == (5, 5)
        np.testing.assert_allclose(np.diag(sim), np.ones(5), atol=1e-5)

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(size=(4, 6)).astype(np.float32)
        sim = cosine_similarity_matrix(a)
        np.testing.assert_allclose(sim, sim.T, atol=1e-5)

    def test_a_vs_b_shape(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.normal(size=(3, 4)).astype(np.float32)
        b = rng.normal(size=(7, 4)).astype(np.float32)
        sim = cosine_similarity_matrix(a, b)
        assert sim.shape == (3, 7)

    def test_empty_input(self) -> None:
        a = np.empty((0, 4), dtype=np.float32)
        b = np.empty((0, 4), dtype=np.float32)
        assert cosine_similarity_matrix(a).shape == (0, 0)
        assert cosine_similarity_matrix(a, b).shape == (0, 0)

    def test_values_in_unit_range(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.normal(size=(10, 16)).astype(np.float32)
        sim = cosine_similarity_matrix(a)
        assert sim.min() >= -1.0 - 1e-5
        assert sim.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Top-K Similar
# ---------------------------------------------------------------------------


class TestTopKSimilar:
    def test_single_query_returns_sorted(self) -> None:
        corpus = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        query = np.array([1.0, 0.0], dtype=np.float32)
        idx, scores = top_k_similar(query, corpus, k=3)
        assert list(idx) == [0, 1, 2]
        assert scores[0] > scores[1] > scores[2]

    def test_k_larger_than_corpus_clamps(self) -> None:
        corpus = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        query = np.array([1.0, 0.0], dtype=np.float32)
        idx, scores = top_k_similar(query, corpus, k=10)
        assert idx.shape == (2,)
        assert scores.shape == (2,)

    def test_batch_query(self) -> None:
        corpus = np.eye(5, dtype=np.float32)
        queries = np.eye(3, 5, dtype=np.float32)
        idx, scores = top_k_similar(queries, corpus, k=2)
        assert idx.shape == (3, 2)
        assert scores.shape == (3, 2)
        # Best match for query i is corpus row i
        for i in range(3):
            assert idx[i, 0] == i
            assert scores[i, 0] == pytest.approx(1.0, abs=1e-6)

    def test_empty_corpus(self) -> None:
        corpus = np.empty((0, 4), dtype=np.float32)
        query = np.ones(4, dtype=np.float32)
        idx, scores = top_k_similar(query, corpus, k=5)
        assert idx.shape == (0,)
        assert scores.shape == (0,)


# ---------------------------------------------------------------------------
# Dimensionality Reduction
# ---------------------------------------------------------------------------


class TestProjection:
    def test_pca_2d_shape(self) -> None:
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(50, 16)).astype(np.float32)
        coords = project_to_2d(emb, method="pca")
        assert coords.shape == (50, 2)
        assert coords.dtype == np.float32

    def test_pca_3d_shape(self) -> None:
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(50, 16)).astype(np.float32)
        coords = project_to_3d(emb, method="pca")
        assert coords.shape == (50, 3)

    def test_tsne_2d_shape(self) -> None:
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(40, 12)).astype(np.float32)
        coords = project_to_2d(emb, method="tsne")
        assert coords.shape == (40, 2)

    def test_invalid_method_raises(self) -> None:
        emb = np.ones((10, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            project_to_2d(emb, method="not-a-method")

    def test_empty_input_returns_empty(self) -> None:
        emb = np.empty((0, 8), dtype=np.float32)
        coords = project_to_2d(emb, method="pca")
        assert coords.shape == (0, 2)


# ---------------------------------------------------------------------------
# K-means Clusters
# ---------------------------------------------------------------------------


class TestKmeansClusters:
    def test_basic_shape(self) -> None:
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(40, 8)).astype(np.float32)
        labels, centroids = kmeans_clusters(emb, n_clusters=4)
        assert labels.shape == (40,)
        assert labels.dtype == np.int64
        assert centroids.shape == (4, 8)
        assert set(labels.tolist()).issubset({0, 1, 2, 3})

    def test_well_separated_clusters_recovered(self) -> None:
        # Three well-separated blobs
        rng = np.random.default_rng(42)
        a = rng.normal(loc=0.0, size=(20, 4)).astype(np.float32)
        b = rng.normal(loc=10.0, size=(20, 4)).astype(np.float32)
        c = rng.normal(loc=-10.0, size=(20, 4)).astype(np.float32)
        emb = np.vstack([a, b, c])
        labels, _ = kmeans_clusters(emb, n_clusters=3)
        # Within each blob, almost all labels should agree
        for start in (0, 20, 40):
            blob = labels[start: start + 20]
            mode_count = max(int((blob == k).sum()) for k in range(3))
            assert mode_count >= 18

    def test_clamps_n_clusters_to_n_points(self) -> None:
        emb = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
        labels, centroids = kmeans_clusters(emb, n_clusters=100)
        assert labels.shape == (3,)
        assert centroids.shape[0] <= 3

    def test_empty_input(self) -> None:
        emb = np.empty((0, 4), dtype=np.float32)
        labels, centroids = kmeans_clusters(emb, n_clusters=5)
        assert labels.shape == (0,)
        assert centroids.shape[0] == 0


# ---------------------------------------------------------------------------
# Group Label
# ---------------------------------------------------------------------------


class TestGroupLabel:
    def test_legal_framework_constitution(self) -> None:
        meta = {"document_type": "legal_framework", "filename": "constitution"}
        assert group_label(meta) == "憲法"

    def test_legal_framework_criminal_code(self) -> None:
        meta = {"document_type": "legal_framework", "filename": "criminal_code"}
        assert group_label(meta) == "刑法"

    def test_legal_framework_unknown_filename(self) -> None:
        meta = {"document_type": "legal_framework", "filename": "mystery"}
        assert group_label(meta) == "法令"

    def test_precedent_criminal(self) -> None:
        meta = {"document_type": "precedent", "case_type": "criminal"}
        assert group_label(meta) == "判例 (刑事)"

    def test_precedent_civil(self) -> None:
        meta = {"document_type": "precedent", "case_type": "civil"}
        assert group_label(meta) == "判例 (民事)"

    def test_precedent_constitutional(self) -> None:
        meta = {"document_type": "precedent", "case_type": "constitutional"}
        assert group_label(meta) == "判例 (憲法)"

    def test_unknown_type_falls_back(self) -> None:
        assert group_label({}) == "その他"


# ---------------------------------------------------------------------------
# Short Preview
# ---------------------------------------------------------------------------


class TestShortPreview:
    def test_short_text_passes_through(self) -> None:
        assert short_preview("短い", max_chars=10) == "短い"

    def test_newlines_collapsed(self) -> None:
        assert short_preview("ab\ncd", max_chars=10) == "ab cd"

    def test_truncation_with_ellipsis(self) -> None:
        long = "あ" * 200
        result = short_preview(long, max_chars=10)
        assert len(result) == 10
        assert result.endswith("…")

    def test_empty_input(self) -> None:
        assert short_preview("") == ""


# ---------------------------------------------------------------------------
# Disk-Cached Projection
# ---------------------------------------------------------------------------


class TestProjectCached:
    def test_fingerprint_stable_for_same_input(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(size=(30, 16)).astype(np.float32)
        assert _embedding_fingerprint(a) == _embedding_fingerprint(a)

    def test_fingerprint_differs_for_different_input(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(size=(30, 16)).astype(np.float32)
        b = rng.normal(size=(30, 16)).astype(np.float32)
        assert _embedding_fingerprint(a) != _embedding_fingerprint(b)

    def test_round_trip_returns_consistent_shape(self, tmp_path) -> None:
        # Monkeypatch the cache dir to keep the test hermetic
        from rag_system import vector_analysis as va

        orig = va._cache_dir
        va._cache_dir = lambda: tmp_path  # type: ignore[assignment]
        try:
            rng = np.random.default_rng(0)
            emb = rng.normal(size=(40, 12)).astype(np.float32)
            coords1 = project_cached(emb, n_components=2, method="pca")
            assert coords1.shape == (40, 2)
            # Second call should hit the cache and return the same values
            coords2 = project_cached(emb, n_components=2, method="pca")
            np.testing.assert_allclose(coords1, coords2)
        finally:
            va._cache_dir = orig  # type: ignore[assignment]

    def test_empty_returns_empty(self, tmp_path) -> None:
        from rag_system import vector_analysis as va

        orig = va._cache_dir
        va._cache_dir = lambda: tmp_path  # type: ignore[assignment]
        try:
            coords = project_cached(
                np.empty((0, 8), dtype=np.float32),
                n_components=2,
                method="pca",
            )
            assert coords.shape == (0, 2)
        finally:
            va._cache_dir = orig  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CLI Smoke Tests
# ---------------------------------------------------------------------------


class TestCli:
    def test_cli_stats_calls_stats(self) -> None:
        from rag_system import vector_analysis as va

        with mock.patch.object(
            va, "get_collection_stats",
            return_value={
                "total": 100,
                "dimension": 768,
                "by_document_type": {"legal_framework": 30, "precedent": 70},
                "by_case_type": {"criminal": 40, "civil": 30},
            },
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                va._cli_stats()
            out = buf.getvalue()
            assert "総チャンク数" in out
            assert "100" in out
            assert "768" in out

    def test_cli_compare_outputs_similarity(self) -> None:
        from rag_system import vector_analysis as va

        # Fake embed_texts to avoid loading the heavy model
        with mock.patch.object(
            va, "embed_texts",
            return_value=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                va._cli_compare("foo", "bar")
            out = buf.getvalue()
            assert "コサイン類似度" in out
            assert "ほぼ無関係" in out

    def test_cli_compare_high_similarity_label(self) -> None:
        from rag_system import vector_analysis as va

        with mock.patch.object(
            va, "embed_texts",
            return_value=np.array(
                [[1.0, 0.0], [1.0, 0.01]], dtype=np.float32
            ),
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                va._cli_compare("a", "b")
            out = buf.getvalue()
            assert "非常に類似" in out


# ---------------------------------------------------------------------------
# Integration: real ChromaDB (skipped if collection missing)
# ---------------------------------------------------------------------------


_CHROMA_DIR = (
    Path(__file__).resolve().parent.parent / "chroma_db"
)


@pytest.mark.skipif(
    not _CHROMA_DIR.exists(),
    reason="ChromaDB が初期化されていません (rag_system.ingest を先に実行してください)",
)
class TestIntegrationChromaDB:
    def test_fetch_limited_returns_embeddings(self) -> None:
        from rag_system.vector_analysis import fetch_all_embeddings

        data = fetch_all_embeddings(limit=5)
        assert data["embeddings"].shape[0] <= 5
        if data["embeddings"].shape[0] > 0:
            assert data["embeddings"].shape[1] == 768

    def test_get_collection_stats(self) -> None:
        from rag_system.vector_analysis import get_collection_stats

        stats = get_collection_stats()
        assert "total" in stats
        assert stats["dimension"] == 768
        assert isinstance(stats["by_document_type"], dict)

    def test_filter_by_document_type(self) -> None:
        from rag_system.vector_analysis import fetch_all_embeddings

        legal = fetch_all_embeddings(limit=3, document_type="legal_framework")
        for m in legal["metadatas"]:
            assert m.get("document_type") == "legal_framework"
