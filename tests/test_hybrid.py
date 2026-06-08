"""Tests for rag_system.hybrid (BM25 tokenisation + RRF fusion).

These exercise the pure logic — tokenisation, BM25 ranking over a tiny
in-memory corpus, and Reciprocal Rank Fusion — with no ChromaDB or model
dependency.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag_system.hybrid import (
    BM25Index,
    FusedResult,
    reciprocal_rank_fusion,
    tokenize,
)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize(None) == []  # type: ignore[arg-type]


def test_tokenize_ascii_lowercased():
    tokens = tokenize("CRIM-2020 第235")
    assert "crim" in tokens
    assert "2020" in tokens
    assert "235" in tokens


def test_tokenize_cjk_bigrams():
    tokens = tokenize("窃盗罪")
    # 窃盗, 盗罪
    assert "窃盗" in tokens
    assert "盗罪" in tokens


def test_tokenize_single_cjk_char_emitted():
    assert tokenize("罪") == ["罪"]


def test_tokenize_mixed():
    tokens = tokenize("刑法第235条 theft")
    assert "theft" in tokens
    assert "235" in tokens
    assert any(t == "刑法" for t in tokens)


# ---------------------------------------------------------------------------
# BM25 over a tiny corpus
# ---------------------------------------------------------------------------


def _build_index(texts: list[str]) -> BM25Index:
    from rank_bm25 import BM25Okapi

    docs = [Document(page_content=t, metadata={"filename": f"d{i}"})
            for i, t in enumerate(texts)]
    corpus = [tokenize(t) for t in texts]
    return BM25Index(documents=docs, _bm25=BM25Okapi(corpus))


def test_bm25_exact_keyword_wins():
    index = _build_index([
        "被告人は他人の財物を窃取した窃盗の事案である",
        "桃太郎は鬼ヶ島へ鬼を退治しに行った",
        "契約の解除と損害賠償について述べる",
    ])
    results = index.search("窃盗", k=3)
    assert results, "BM25 should return results"
    # The doc literally containing 窃盗 must rank first.
    assert results[0][0].metadata["filename"] == "d0"
    assert results[0][1] > 0


def test_bm25_empty_query():
    index = _build_index(["あ", "い"])
    assert index.search("", k=3) == []


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _doc(name: str) -> Document:
    return Document(page_content=f"content of {name}", metadata={"filename": name})


def test_rrf_combines_rankings():
    a, b, c = _doc("A"), _doc("B"), _doc("C")
    dense = [(a, 0.9), (b, 0.8), (c, 0.7)]
    sparse = [(c, 5.0), (a, 4.0), (b, 1.0)]

    fused = reciprocal_rank_fusion(dense, sparse, k=3, dense_weight=0.5)

    assert len(fused) == 3
    assert all(isinstance(f, FusedResult) for f in fused)
    # A is rank1 in dense and rank2 in sparse -> should top the fused list.
    assert fused[0].document.metadata["filename"] == "A"
    # Scores are sorted descending.
    scores = [f.rrf_score for f in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_provenance_recorded():
    a = _doc("A")
    fused = reciprocal_rank_fusion([(a, 0.9)], [(a, 2.0)], k=1)
    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank == 1
    assert fused[0].dense_score == pytest.approx(0.9)
    assert fused[0].sparse_score == pytest.approx(2.0)


def test_rrf_dense_weight_extremes():
    a, b = _doc("A"), _doc("B")
    dense = [(a, 0.9), (b, 0.1)]
    sparse = [(b, 9.0), (a, 1.0)]

    dense_only = reciprocal_rank_fusion(dense, sparse, k=2, dense_weight=1.0)
    sparse_only = reciprocal_rank_fusion(dense, sparse, k=2, dense_weight=0.0)

    assert dense_only[0].document.metadata["filename"] == "A"
    assert sparse_only[0].document.metadata["filename"] == "B"


def test_rrf_truncates_to_k():
    docs = [_doc(str(i)) for i in range(10)]
    dense = [(d, 1.0 - i * 0.1) for i, d in enumerate(docs)]
    fused = reciprocal_rank_fusion(dense, [], k=3)
    assert len(fused) == 3
