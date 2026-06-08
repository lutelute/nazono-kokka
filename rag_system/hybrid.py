"""Hybrid retrieval: sparse BM25 keyword search fused with dense vectors.

Dense (embedding) search captures *semantic* similarity but can miss exact
keywords — statute numbers like "刑法第235条" or rare proper nouns. Sparse
BM25 search captures *lexical* overlap but misses paraphrases. Fusing the two
with Reciprocal Rank Fusion (RRF) combines their strengths and is one of the
single most reliable accuracy upgrades for a RAG retriever.

This module is dependency-light: BM25 comes from ``rank_bm25`` and Japanese
tokenisation uses a character-bigram scheme (no MeCab/fugashi required), which
is a well-established cheap tokeniser for CJK BM25.

Usage::

    from rag_system.hybrid import get_bm25_index, hybrid_search
    index = get_bm25_index(vectorstore)
    results = hybrid_search("窃盗罪の量刑", vectorstore, index, k=5)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from rag_system.config import (
    HYBRID_DENSE_WEIGHT,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_K,
    RRF_K,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

# CJK ranges (Hiragana, Katakana, CJK ideographs) plus ASCII word characters.
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿々〆ヵヶ]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenise mixed Japanese/ASCII text for BM25.

    ASCII runs (statute numbers, latin words) are kept whole and lower-cased.
    CJK runs are turned into overlapping character bigrams, which approximate
    word boundaries well enough for keyword matching without a morphological
    analyser. Single leftover CJK characters are also emitted so one-character
    queries still match.
    """
    if not text:
        return []

    tokens: list[str] = []

    # ASCII words (e.g. "235", "CRIM", "2020")
    for m in _ASCII_WORD_RE.findall(text):
        tokens.append(m.lower())

    # CJK character bigrams
    cjk_chars = _CJK_RE.findall(text)
    if len(cjk_chars) == 1:
        tokens.append(cjk_chars[0])
    else:
        for i in range(len(cjk_chars) - 1):
            tokens.append(cjk_chars[i] + cjk_chars[i + 1])

    return tokens


# ---------------------------------------------------------------------------
# BM25 index over the corpus
# ---------------------------------------------------------------------------


@dataclass
class BM25Index:
    """A BM25 index over the full ChromaDB corpus.

    Holds the parallel arrays of document objects and their pre-tokenised
    forms alongside the fitted ``rank_bm25`` model.
    """

    documents: list[Document]
    _bm25: Any
    _id_to_pos: dict[str, int] = field(default_factory=dict)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Return the top-``k`` documents by BM25 score for ``query``."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # argsort descending without numpy dependency at call site
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [(self.documents[i], float(scores[i])) for i in ranked]


# Module-level cache so we only tokenise the corpus once per process.
_INDEX_CACHE: dict[int, BM25Index] = {}


def build_bm25_index(vectorstore: Any) -> BM25Index:
    """Build a :class:`BM25Index` from every chunk stored in ``vectorstore``.

    Pulls all documents out of the Chroma collection (content + metadata),
    tokenises each one, and fits a BM25Okapi model over the corpus.
    """
    from rank_bm25 import BM25Okapi

    raw = vectorstore.get(include=["documents", "metadatas"])
    contents: list[str] = raw.get("documents") or []
    metadatas: list[dict] = raw.get("metadatas") or [{}] * len(contents)
    ids: list[str] = raw.get("ids") or [""] * len(contents)

    documents = [
        Document(page_content=c, metadata=(metadatas[i] or {}))
        for i, c in enumerate(contents)
    ]
    corpus_tokens = [tokenize(c) for c in contents]
    bm25 = BM25Okapi(corpus_tokens)

    id_to_pos = {doc_id: i for i, doc_id in enumerate(ids)}
    logger.info("BM25 インデックスを構築: %d 文書", len(documents))
    return BM25Index(documents=documents, _bm25=bm25, _id_to_pos=id_to_pos)


def get_bm25_index(vectorstore: Any) -> BM25Index:
    """Return a cached BM25 index for ``vectorstore``, building it on first use."""
    key = id(vectorstore)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = build_bm25_index(vectorstore)
    return _INDEX_CACHE[key]


def clear_bm25_cache() -> None:
    """Drop all cached BM25 indexes (call after re-ingesting the corpus)."""
    _INDEX_CACHE.clear()


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _doc_key(doc: Document) -> str:
    """A stable identity for a chunk: case id / filename + a content hash."""
    meta = doc.metadata or {}
    base = meta.get("case_id") or meta.get("filename") or meta.get("source") or ""
    # Content prefix disambiguates multiple chunks of the same source file.
    return f"{base}::{doc.page_content[:64]}"


@dataclass
class FusedResult:
    """A single fused hit with provenance for transparent display."""

    document: Document
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None


def reciprocal_rank_fusion(
    dense: list[tuple[Document, float]],
    sparse: list[tuple[Document, float]],
    *,
    k: int = RETRIEVAL_K,
    rrf_k: int = RRF_K,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
) -> list[FusedResult]:
    """Fuse dense and sparse rankings via weighted Reciprocal Rank Fusion.

    RRF score for a document is ``sum(weight / (rrf_k + rank))`` across the
    rankings it appears in (rank is 1-based). It depends only on *rank*, not
    on the raw scores, which makes it robust to the different score scales of
    cosine similarity and BM25.

    Parameters
    ----------
    dense, sparse:
        Ranked ``(document, score)`` lists, best first.
    k:
        Number of fused results to return.
    rrf_k:
        RRF damping constant (larger flattens rank influence).
    dense_weight:
        Weight on the dense ranking; the sparse ranking gets
        ``1 - dense_weight``.

    Returns
    -------
    list[FusedResult]
        Fused hits sorted by descending RRF score, length ``<= k``.
    """
    sparse_weight = 1.0 - dense_weight
    table: dict[str, FusedResult] = {}

    for rank, (doc, score) in enumerate(dense, start=1):
        key = _doc_key(doc)
        entry = table.setdefault(key, FusedResult(document=doc, rrf_score=0.0))
        entry.rrf_score += dense_weight / (rrf_k + rank)
        entry.dense_rank = rank
        entry.dense_score = score

    for rank, (doc, score) in enumerate(sparse, start=1):
        key = _doc_key(doc)
        entry = table.setdefault(key, FusedResult(document=doc, rrf_score=0.0))
        entry.rrf_score += sparse_weight / (rrf_k + rank)
        entry.sparse_rank = rank
        entry.sparse_score = score

    fused = sorted(table.values(), key=lambda e: e.rrf_score, reverse=True)
    return fused[:k]


# ---------------------------------------------------------------------------
# High-level hybrid search
# ---------------------------------------------------------------------------


def hybrid_search(
    query: str,
    vectorstore: Any,
    bm25_index: BM25Index | None = None,
    *,
    k: int = RETRIEVAL_K,
    fetch_k: int = RETRIEVAL_FETCH_K,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    where: dict[str, Any] | None = None,
) -> list[FusedResult]:
    """Run dense + BM25 search and fuse the candidate pools with RRF.

    Both retrievers fetch ``fetch_k`` candidates; the fused list is truncated
    to ``k``.

    Parameters
    ----------
    query:
        Natural-language query.
    vectorstore:
        A Chroma vector store.
    bm25_index:
        A prebuilt BM25 index; built (and cached) from ``vectorstore`` if
        ``None``.
    k:
        Final number of fused results.
    fetch_k:
        Candidate pool size for each retriever.
    dense_weight:
        Weight on the dense ranking during fusion.
    where:
        Optional Chroma metadata filter applied to the dense search.

    Returns
    -------
    list[FusedResult]
    """
    if not query or not query.strip():
        return []

    if bm25_index is None:
        bm25_index = get_bm25_index(vectorstore)

    dense_kwargs: dict[str, Any] = {}
    if where is not None:
        dense_kwargs["filter"] = where
    # Distance (lower = closer); negate so higher = better, consistent with
    # BM25 scores and RRF's best-first expectation.
    raw = vectorstore.similarity_search_with_score(query, k=fetch_k, **dense_kwargs)
    dense = [(doc, -float(dist)) for doc, dist in raw]

    sparse = bm25_index.search(query, fetch_k)

    return reciprocal_rank_fusion(
        dense, sparse, k=k, dense_weight=dense_weight
    )
