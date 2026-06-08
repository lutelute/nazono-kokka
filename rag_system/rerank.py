"""Cross-encoder reranking for retrieved candidates.

A bi-encoder (the embedding model) scores query and document *independently*
then compares vectors — fast, but it never sees the two together. A
cross-encoder feeds ``[query, document]`` through the transformer jointly and
outputs a single relevance score, so it can judge fine-grained relevance the
bi-encoder misses. The standard RAG pattern is therefore:

    dense/hybrid retrieve a wide pool (fetch_k)  ->  cross-encoder reranks  ->  keep top-k

This module loads the reranker lazily and degrades gracefully: if the model
cannot be loaded (offline, missing weights), reranking is skipped and the
input order is preserved, so the pipeline never hard-fails on its account.

Usage::

    from rag_system.rerank import rerank
    reranked = rerank("窃盗罪の量刑", documents, top_n=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from rag_system.config import RERANKER_MODEL_NAME

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_RERANKER = None
_LOAD_FAILED = False


def get_reranker():
    """Return a cached CrossEncoder instance, or ``None`` if unavailable.

    The first call downloads/loads the model (which can take a few seconds).
    A failure is remembered so we don't retry on every query.
    """
    global _RERANKER, _LOAD_FAILED

    if _RERANKER is not None:
        return _RERANKER
    if _LOAD_FAILED:
        return None

    try:
        # Quiet the noisy HF/transformers download + weight-loading bars so
        # they don't flood Streamlit / CLI logs.
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            import transformers

            transformers.logging.set_verbosity_error()
        except Exception:
            pass

        from sentence_transformers import CrossEncoder

        logger.info("リランカーモデルを読み込み: %s", RERANKER_MODEL_NAME)
        _RERANKER = CrossEncoder(RERANKER_MODEL_NAME)
        return _RERANKER
    except Exception:
        logger.warning(
            "リランカーモデルを読み込めませんでした (%s)。リランクをスキップします。",
            RERANKER_MODEL_NAME,
            exc_info=True,
        )
        _LOAD_FAILED = True
        return None


def reranker_available() -> bool:
    """Whether the cross-encoder can be loaded (without forcing a download)."""
    return get_reranker() is not None


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


@dataclass
class RerankedDocument:
    """A document with its cross-encoder score and rank movement."""

    document: Document
    score: float
    original_rank: int  # 1-based position before reranking
    new_rank: int  # 1-based position after reranking

    @property
    def rank_delta(self) -> int:
        """Positions gained (positive) or lost (negative) by reranking."""
        return self.original_rank - self.new_rank


def rerank(
    query: str,
    documents: list[Document],
    *,
    top_n: int | None = None,
) -> list[RerankedDocument]:
    """Rerank ``documents`` against ``query`` with the cross-encoder.

    Parameters
    ----------
    query:
        The natural-language query.
    documents:
        Candidate documents, in their pre-rerank order.
    top_n:
        Keep only this many top results. ``None`` keeps all.

    Returns
    -------
    list[RerankedDocument]
        Documents sorted by descending cross-encoder score. If the reranker
        is unavailable, the original order is preserved and scores are set to
        ``nan`` so callers can tell reranking did not run.
    """
    if not documents:
        return []

    model = get_reranker()

    if model is None:
        # Graceful fallback: identity ordering.
        out = [
            RerankedDocument(
                document=doc,
                score=float("nan"),
                original_rank=i + 1,
                new_rank=i + 1,
            )
            for i, doc in enumerate(documents)
        ]
        return out[:top_n] if top_n else out

    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    order = sorted(
        range(len(documents)), key=lambda i: scores[i], reverse=True
    )

    reranked = [
        RerankedDocument(
            document=documents[orig_idx],
            score=float(scores[orig_idx]),
            original_rank=orig_idx + 1,
            new_rank=new_idx + 1,
        )
        for new_idx, orig_idx in enumerate(order)
    ]

    return reranked[:top_n] if top_n else reranked
