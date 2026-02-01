"""Retrieval logic for the RAG judicial system.

Provides vector search with optional metadata filtering against the
ChromaDB collection.  Supports filtering by document type (legal
framework vs. precedent), case type (criminal, civil, constitutional),
verdict, and other metadata fields stored during ingestion.

Usage:
    from rag_system.retriever import create_retriever, retrieve

    retriever = create_retriever()
    results = retrieve(retriever, "窃盗罪の判例")
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from rag_system.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDINGS_MODEL_NAME,
    RETRIEVAL_K,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding & Vector Store Initialization
# ---------------------------------------------------------------------------


def create_embeddings() -> HuggingFaceEmbeddings:
    """Create the HuggingFace embedding model for multilingual support."""
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)


def load_vectorstore(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    """Load the existing ChromaDB vector store.

    Parameters
    ----------
    embeddings:
        The embedding model to use.  If ``None``, a new instance is created.

    Returns
    -------
    Chroma
        The vector store connected to the persisted ChromaDB collection.

    Raises
    ------
    FileNotFoundError
        If the ChromaDB directory does not exist.
    """
    import os

    if not os.path.isdir(CHROMA_DB_PATH):
        raise FileNotFoundError(
            f"ChromaDB ディレクトリが見つかりません: {CHROMA_DB_PATH}\n"
            "先に 'python rag_system/ingest.py' を実行してドキュメントを取り込んでください。"
        )

    if embeddings is None:
        embeddings = create_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    return vectorstore


# ---------------------------------------------------------------------------
# Metadata Filter Builders
# ---------------------------------------------------------------------------


def _build_where_filter(
    document_type: str | None = None,
    case_type: str | None = None,
    verdict: str | None = None,
) -> dict[str, Any] | None:
    """Build a ChromaDB ``where`` filter from optional metadata criteria.

    Parameters
    ----------
    document_type:
        Filter by document type (``"legal_framework"`` or ``"precedent"``).
    case_type:
        Filter by case category (``"criminal"``, ``"civil"``,
        ``"constitutional"``).  Only applicable to precedent documents.
    verdict:
        Filter by verdict string (e.g. ``"有罪"``, ``"無罪"``).

    Returns
    -------
    dict or None
        A ChromaDB-compatible ``where`` clause, or ``None`` if no filters
        are specified.
    """
    conditions: list[dict[str, Any]] = []

    if document_type is not None:
        conditions.append({"document_type": {"$eq": document_type}})
    if case_type is not None:
        conditions.append({"case_type": {"$eq": case_type}})
    if verdict is not None:
        conditions.append({"verdict": {"$eq": verdict}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# Retriever Creation
# ---------------------------------------------------------------------------


def create_retriever(
    vectorstore: Chroma | None = None,
    k: int | None = None,
    document_type: str | None = None,
    case_type: str | None = None,
    verdict: str | None = None,
):
    """Create a LangChain retriever with optional metadata filtering.

    Parameters
    ----------
    vectorstore:
        An existing Chroma vector store.  If ``None``, one is loaded from
        the persisted ChromaDB directory.
    k:
        Number of documents to retrieve.  Defaults to ``RETRIEVAL_K`` from
        config.
    document_type:
        Restrict results to a specific document type.
    case_type:
        Restrict results to a specific case category.
    verdict:
        Restrict results to a specific verdict.

    Returns
    -------
    langchain_core.retrievers.BaseRetriever
        A configured retriever instance.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    if k is None:
        k = RETRIEVAL_K

    search_kwargs: dict[str, Any] = {"k": k}

    where_filter = _build_where_filter(
        document_type=document_type,
        case_type=case_type,
        verdict=verdict,
    )
    if where_filter is not None:
        search_kwargs["filter"] = where_filter

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    logger.info(
        "リトリーバーを作成 (k=%d, フィルタ=%s)",
        k,
        where_filter if where_filter else "なし",
    )
    return retriever


# ---------------------------------------------------------------------------
# Convenience Retrieval Functions
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    *,
    vectorstore: Chroma | None = None,
    k: int | None = None,
    document_type: str | None = None,
    case_type: str | None = None,
    verdict: str | None = None,
) -> list[Document]:
    """Run a vector similarity search and return matching documents.

    This is a convenience wrapper that creates a retriever and invokes it
    in a single call.

    Parameters
    ----------
    query:
        The natural-language query string.
    vectorstore:
        An existing Chroma vector store (optional).
    k:
        Number of documents to retrieve.
    document_type:
        Restrict results to a specific document type.
    case_type:
        Restrict results to a specific case category.
    verdict:
        Restrict results to a specific verdict.

    Returns
    -------
    list[Document]
        The retrieved documents sorted by relevance.
    """
    if not query or not query.strip():
        logger.warning("空のクエリが指定されました")
        return []

    retriever = create_retriever(
        vectorstore=vectorstore,
        k=k,
        document_type=document_type,
        case_type=case_type,
        verdict=verdict,
    )

    try:
        results = retriever.invoke(query)
        logger.info("クエリ '%s' に対して %d 件の結果を取得", query[:50], len(results))
        return results
    except Exception:
        logger.exception("検索中にエラーが発生しました: %s", query[:50])
        return []


def retrieve_legal_framework(
    query: str,
    *,
    vectorstore: Chroma | None = None,
    k: int | None = None,
) -> list[Document]:
    """Retrieve only legal framework documents matching the query.

    Parameters
    ----------
    query:
        The natural-language query string.
    vectorstore:
        An existing Chroma vector store (optional).
    k:
        Number of documents to retrieve.

    Returns
    -------
    list[Document]
        Matching legal framework document chunks.
    """
    return retrieve(
        query,
        vectorstore=vectorstore,
        k=k,
        document_type="legal_framework",
    )


def retrieve_precedents(
    query: str,
    *,
    vectorstore: Chroma | None = None,
    k: int | None = None,
    case_type: str | None = None,
    verdict: str | None = None,
) -> list[Document]:
    """Retrieve only precedent documents matching the query.

    Parameters
    ----------
    query:
        The natural-language query string.
    vectorstore:
        An existing Chroma vector store (optional).
    k:
        Number of documents to retrieve.
    case_type:
        Restrict results to a specific case category.
    verdict:
        Restrict results to a specific verdict.

    Returns
    -------
    list[Document]
        Matching precedent document chunks.
    """
    return retrieve(
        query,
        vectorstore=vectorstore,
        k=k,
        document_type="precedent",
        case_type=case_type,
        verdict=verdict,
    )


def retrieve_with_scores(
    query: str,
    *,
    vectorstore: Chroma | None = None,
    k: int | None = None,
    document_type: str | None = None,
    case_type: str | None = None,
    verdict: str | None = None,
) -> list[tuple[Document, float]]:
    """Retrieve documents with their similarity scores.

    Useful for debugging and transparency — the score indicates how
    closely each result matches the query embedding.

    Parameters
    ----------
    query:
        The natural-language query string.
    vectorstore:
        An existing Chroma vector store (optional).
    k:
        Number of documents to retrieve.
    document_type:
        Restrict results to a specific document type.
    case_type:
        Restrict results to a specific case category.
    verdict:
        Restrict results to a specific verdict.

    Returns
    -------
    list[tuple[Document, float]]
        Pairs of (document, similarity_score) sorted by relevance.
    """
    if not query or not query.strip():
        logger.warning("空のクエリが指定されました")
        return []

    if vectorstore is None:
        vectorstore = load_vectorstore()

    if k is None:
        k = RETRIEVAL_K

    where_filter = _build_where_filter(
        document_type=document_type,
        case_type=case_type,
        verdict=verdict,
    )

    kwargs: dict[str, Any] = {}
    if where_filter is not None:
        kwargs["filter"] = where_filter

    try:
        results = vectorstore.similarity_search_with_score(
            query, k=k, **kwargs
        )
        logger.info(
            "クエリ '%s' に対して %d 件のスコア付き結果を取得",
            query[:50],
            len(results),
        )
        return results
    except Exception:
        logger.exception("スコア付き検索中にエラーが発生しました: %s", query[:50])
        return []
