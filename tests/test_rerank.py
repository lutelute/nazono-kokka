"""Tests for rag_system.rerank.

The cross-encoder model is heavy and may be unavailable offline, so these
tests focus on the graceful-fallback path and the rank-bookkeeping logic by
mocking the model. A single opt-in integration test exercises the real model
when ``RUN_RERANK_MODEL=1``.
"""

from __future__ import annotations

import math
import os
from unittest import mock

import pytest
from langchain_core.documents import Document

import rag_system.rerank as rr
from rag_system.rerank import RerankedDocument, rerank


def _docs(n: int) -> list[Document]:
    return [Document(page_content=f"doc {i}", metadata={"filename": f"d{i}"})
            for i in range(n)]


def test_rerank_empty_input():
    assert rerank("q", []) == []


def test_rerank_fallback_preserves_order(monkeypatch):
    # Force the reranker to be unavailable.
    monkeypatch.setattr(rr, "get_reranker", lambda: None)
    docs = _docs(3)
    out = rerank("q", docs, top_n=2)
    assert len(out) == 2
    assert [r.document.metadata["filename"] for r in out] == ["d0", "d1"]
    # NaN score signals "reranking did not run".
    assert all(math.isnan(r.score) for r in out)
    assert out[0].original_rank == 1 and out[0].new_rank == 1


def test_rerank_reorders_by_score(monkeypatch):
    docs = _docs(3)

    class FakeModel:
        def predict(self, pairs):
            # Make the last doc the most relevant.
            return [0.1, 0.2, 0.9]

    monkeypatch.setattr(rr, "get_reranker", lambda: FakeModel())
    out = rerank("q", docs)
    assert [r.document.metadata["filename"] for r in out] == ["d2", "d1", "d0"]
    # d2 was originally rank 3, now rank 1 -> delta +2.
    assert out[0].original_rank == 3
    assert out[0].new_rank == 1
    assert out[0].rank_delta == 2


def test_rerank_top_n_truncation(monkeypatch):
    docs = _docs(5)

    class FakeModel:
        def predict(self, pairs):
            return [5, 4, 3, 2, 1]

    monkeypatch.setattr(rr, "get_reranker", lambda: FakeModel())
    out = rerank("q", docs, top_n=2)
    assert len(out) == 2
    assert out[0].document.metadata["filename"] == "d0"


def test_rank_delta_property():
    doc = Document(page_content="x", metadata={})
    rd = RerankedDocument(document=doc, score=1.0, original_rank=7, new_rank=2)
    assert rd.rank_delta == 5


@pytest.mark.skipif(
    os.environ.get("RUN_RERANK_MODEL") != "1",
    reason="set RUN_RERANK_MODEL=1 to exercise the real cross-encoder",
)
def test_rerank_real_model_smoke():
    # Reset cached state so the real model loads.
    rr._RERANKER = None
    rr._LOAD_FAILED = False
    docs = [
        Document(page_content="被告人は他人の財物を窃取した窃盗の事案", metadata={"filename": "law"}),
        Document(page_content="桃太郎は鬼ヶ島へ鬼退治に行った", metadata={"filename": "tale"}),
    ]
    out = rerank("窃盗罪の量刑", docs)
    assert out[0].document.metadata["filename"] == "law"
