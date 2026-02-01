"""AI backend adapter for the RAG judicial system.

Provides a configurable adapter pattern that wraps existing factory
functions (:func:`~rag_system.judge.create_llm`,
:func:`~rag_system.retriever.create_retriever`,
:func:`~rag_system.judge.create_judicial_chain`) into a single
configuration-driven interface.  This enables switching between
multiple AI backends (different Ollama models, RAG parameters) at
runtime for accuracy comparison.

Usage:
    from rag_system.backend_adapter import BackendConfig, create_backend

    config = BackendConfig(name="llama3-default", model_name="llama3.1:8b")
    chain = create_backend(config)
    result = chain.invoke({"query": "窃盗罪の量刑基準を示せ"})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_classic.chains import RetrievalQA

from rag_system.config import (
    LLM_MODEL_NAME,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    RETRIEVAL_K,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend Configuration
# ---------------------------------------------------------------------------


@dataclass
class BackendConfig:
    """AIバックエンド設定を表すデータクラス。

    比較実行時の識別子として ``name`` を使用し、LLM パラメータおよび
    検索フィルタをまとめて管理する。

    Attributes
    ----------
    name:
        設定名（比較時の識別子）。
    model_name:
        Ollama モデル名。
    temperature:
        サンプリング温度。
    num_ctx:
        コンテキストウィンドウサイズ。
    retrieval_k:
        検索結果の取得件数。
    document_type:
        ドキュメント種別フィルタ (``"legal_framework"`` or ``"precedent"``)。
    case_type:
        判例カテゴリフィルタ (``"criminal"``, ``"civil"``,
        ``"constitutional"``)。
    verdict:
        判決フィルタ (e.g. ``"有罪"``, ``"無罪"``)。
    """

    name: str
    model_name: str
    temperature: float = LLM_TEMPERATURE
    num_ctx: int = LLM_NUM_CTX
    retrieval_k: int = RETRIEVAL_K
    document_type: str | None = None
    case_type: str | None = None
    verdict: str | None = None


# ---------------------------------------------------------------------------
# Backend Factory
# ---------------------------------------------------------------------------


def create_backend(config: BackendConfig) -> RetrievalQA:
    """設定から LLM + Retriever + Chain を構築するファクトリ関数。

    既存の :func:`~rag_system.judge.create_llm`,
    :func:`~rag_system.retriever.create_retriever`,
    :func:`~rag_system.judge.create_judicial_chain` をラップし、
    :class:`BackendConfig` から一括で司法推論チェーンを構築する。

    Parameters
    ----------
    config:
        AIバックエンド設定。

    Returns
    -------
    RetrievalQA
        設定に基づいて構築された司法推論チェーン。

    Raises
    ------
    ConnectionError
        Ollama サーバーに接続できない場合。
    FileNotFoundError
        ChromaDB ディレクトリが見つからない場合。
    """
    from rag_system.judge import create_judicial_chain, create_llm
    from rag_system.retriever import create_retriever

    logger.info(
        "バックエンドを構築: name='%s', model='%s', temperature=%.2f, "
        "num_ctx=%d, k=%d",
        config.name,
        config.model_name,
        config.temperature,
        config.num_ctx,
        config.retrieval_k,
    )

    llm = create_llm(
        model_name=config.model_name,
        temperature=config.temperature,
        num_ctx=config.num_ctx,
    )

    retriever = create_retriever(
        k=config.retrieval_k,
        document_type=config.document_type,
        case_type=config.case_type,
        verdict=config.verdict,
    )

    chain = create_judicial_chain(llm=llm, retriever=retriever)

    logger.info("バックエンド '%s' の構築が完了しました", config.name)
    return chain
