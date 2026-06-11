"""LangChain Retriever interface for the staged retrieval pipeline.

:func:`rag_system.retriever.retrieve_advanced` implements the full
dense → BM25-hybrid → cross-encoder-rerank pipeline, but it returns a
:class:`StagedRetrieval` object and therefore cannot be plugged into
``RetrievalQA`` directly. This module wraps it in a
:class:`~langchain_core.retrievers.BaseRetriever` so the judicial chain
can use hybrid + rerank retrieval transparently.

It also adds a *legal-document quota*: the cross-encoder tends to rank
precedents above statute texts (measured legal-doc presence drops from
0.63 to 0.35 with rerank on the 139-case set), yet the judicial prompt
asks for an 【適用法令】 section. The quota guarantees at least
``min_legal_docs`` legal-framework documents in the final context by
swapping in the best dense legal hits when they are missing.

Usage::

    from rag_system.advanced_retriever import AdvancedRetriever
    retriever = AdvancedRetriever(vectorstore=load_vectorstore())
    chain = create_judicial_chain(llm=llm, retriever=retriever)
"""

from __future__ import annotations

import logging
from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_system.config import HYBRID_DENSE_WEIGHT, RETRIEVAL_K

logger = logging.getLogger(__name__)


def is_legal_framework(doc: Document) -> bool:
    """Return True if the document comes from the legal framework corpus."""
    meta = doc.metadata or {}
    if meta.get("document_type") == "legal_framework":
        return True
    return "legal_framework" in str(meta.get("source", ""))


class AdvancedRetriever(BaseRetriever):
    """Staged retrieval (dense → hybrid → rerank) with a legal-doc quota.

    Attributes
    ----------
    vectorstore:
        Loaded Chroma vectorstore.
    k:
        Number of documents to return.
    use_hybrid / use_rerank:
        Stage toggles, passed through to ``retrieve_advanced``.
    dense_weight:
        Dense vs. sparse weight for RRF fusion.
    min_legal_docs:
        Minimum number of legal-framework documents to guarantee in the
        final list (0 disables the quota).
    """

    vectorstore: Any
    k: int = RETRIEVAL_K
    use_hybrid: bool = True
    use_rerank: bool = True
    dense_weight: float = HYBRID_DENSE_WEIGHT
    min_legal_docs: int = 1

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        from rag_system.retriever import retrieve_advanced

        out = retrieve_advanced(
            query,
            vectorstore=self.vectorstore,
            k=self.k,
            use_hybrid=self.use_hybrid,
            use_rerank=self.use_rerank,
            dense_weight=self.dense_weight,
        )
        docs = list(out.final)
        if self.min_legal_docs > 0:
            docs = self._apply_legal_quota(docs, out)
        return docs

    # ------------------------------------------------------------------
    # Legal-document quota
    # ------------------------------------------------------------------

    def _apply_legal_quota(
        self, docs: List[Document], staged: Any
    ) -> List[Document]:
        """Ensure at least ``min_legal_docs`` legal-framework documents.

        Missing legal documents are pulled from the dense candidate pool
        (ordered by dense score) and swapped in from the tail of the final
        list, so the strongest precedents at the head are preserved.
        """
        n_legal = sum(1 for d in docs if is_legal_framework(d))
        if n_legal >= self.min_legal_docs:
            return docs

        have = {id(d) for d in docs}
        have_content = {d.page_content for d in docs}
        # dense は (Document, score) のリスト、スコア降順
        candidates = [
            doc
            for doc, _score in getattr(staged, "dense", [])
            if is_legal_framework(doc)
            and id(doc) not in have
            and doc.page_content not in have_content
        ]

        needed = self.min_legal_docs - n_legal
        for legal_doc in candidates[:needed]:
            # 末尾側の非法令文書と入れ替える
            swap_idx = None
            for i in range(len(docs) - 1, -1, -1):
                if not is_legal_framework(docs[i]):
                    swap_idx = i
                    break
            if swap_idx is None:
                break
            logger.info(
                "法令クォータ適用: %s を %s と入替",
                legal_doc.metadata.get("source", "?"),
                docs[swap_idx].metadata.get("source", "?"),
            )
            docs[swap_idx] = legal_doc

        return docs
