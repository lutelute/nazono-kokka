"""LangChain Tool definitions for the RAG judicial system archive.

Provides structured tool access to the legal document archive (書庫),
enabling LLM agents to search legal frameworks, precedents, and
retrieve archive statistics programmatically.

Usage:
    from rag_system.tools import legal_framework_search

    result = legal_framework_search.invoke({"query": "窃盗罪の構成要件"})
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from rag_system.config import RETRIEVAL_K
from rag_system.retriever import retrieve_legal_framework, retrieve_precedents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Input Schemas
# ---------------------------------------------------------------------------


class LegalSearchInput(BaseModel):
    """法令検索ツールの入力スキーマ。"""

    query: str = Field(description="法令データベースを検索するためのクエリ文字列")
    k: int = Field(
        default=5,
        description="取得する検索結果の件数",
        ge=1,
        le=20,
    )


class PrecedentSearchInput(BaseModel):
    """判例検索ツールの入力スキーマ。"""

    query: str = Field(description="判例データベースを検索するためのクエリ文字列")
    k: int = Field(
        default=5,
        description="取得する検索結果の件数",
        ge=1,
        le=20,
    )
    case_type: str | None = Field(
        default=None,
        description="事件類型でフィルタリング（'criminal', 'civil', 'constitutional'）",
    )
    verdict: str | None = Field(
        default=None,
        description="判決結果でフィルタリング（例: '有罪', '無罪'）",
    )


# ---------------------------------------------------------------------------
# Document Formatting Helpers
# ---------------------------------------------------------------------------


def _format_legal_documents(docs: list[Any]) -> str:
    """法令文書のリストをフォーマット済み日本語文字列に変換する。

    Parameters
    ----------
    docs:
        検索結果のDocumentオブジェクトのリスト。

    Returns
    -------
    str
        フォーマット済みの法令文書テキスト。
    """
    if not docs:
        return "該当する法令文書が見つかりませんでした。"

    formatted_parts: list[str] = []
    formatted_parts.append(f"【法令検索結果】（{len(docs)}件）\n")

    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        source = metadata.get("source", "不明")
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)

        formatted_parts.append(f"--- 結果 {i} ---")
        formatted_parts.append(f"出典: {source}")
        formatted_parts.append(f"内容:\n{content}")
        formatted_parts.append("")

    return "\n".join(formatted_parts)


def _format_precedent_documents(docs: list[Any]) -> str:
    """判例文書のリストをフォーマット済み日本語文字列に変換する。

    Parameters
    ----------
    docs:
        検索結果のDocumentオブジェクトのリスト。

    Returns
    -------
    str
        フォーマット済みの判例文書テキスト。
    """
    if not docs:
        return "該当する判例が見つかりませんでした。"

    formatted_parts: list[str] = []
    formatted_parts.append(f"【判例検索結果】（{len(docs)}件）\n")

    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        source = metadata.get("source", "不明")
        case_type = metadata.get("case_type", "不明")
        verdict = metadata.get("verdict", "不明")
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)

        formatted_parts.append(f"--- 結果 {i} ---")
        formatted_parts.append(f"出典: {source}")
        formatted_parts.append(f"事件類型: {case_type}")
        formatted_parts.append(f"判決: {verdict}")
        formatted_parts.append(f"内容:\n{content}")
        formatted_parts.append("")

    return "\n".join(formatted_parts)


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------


@tool("legal_framework_search", args_schema=LegalSearchInput)
def legal_framework_search(query: str, k: int = 5) -> str:
    """法令データベース（憲法、刑法、民法、行政法、文化規制、倫理指針）を検索し、関連する法律条文を返します。

    法令の条文番号や具体的な法的要件を調べる際に使用してください。
    """
    if not query or not query.strip():
        return "エラー: 検索クエリが空です。検索キーワードを指定してください。"

    logger.info("法令検索ツール実行: query='%s', k=%d", query[:50], k)

    try:
        docs = retrieve_legal_framework(query, k=k)
        result = _format_legal_documents(docs)
        logger.info("法令検索完了: %d 件の結果", len(docs))
        return result
    except FileNotFoundError as e:
        logger.error("ChromaDBが初期化されていません: %s", e)
        return (
            "エラー: 法令データベースが初期化されていません。\n"
            "'python rag_system/ingest.py' を実行してデータを取り込んでください。"
        )
    except Exception as e:
        logger.exception("法令検索中に予期しないエラーが発生しました")
        return f"エラー: 法令検索中にエラーが発生しました: {e}"


@tool("precedent_search", args_schema=PrecedentSearchInput)
def precedent_search(
    query: str,
    k: int = 5,
    case_type: str | None = None,
    verdict: str | None = None,
) -> str:
    """判例データベースを検索し、関連する過去の裁判事例を返します。

    事件類型（criminal, civil, constitutional）や判決結果（有罪、無罪）でフィルタリングできます。
    過去の判例や裁判の先例を調べる際に使用してください。
    """
    if not query or not query.strip():
        return "エラー: 検索クエリが空です。検索キーワードを指定してください。"

    logger.info(
        "判例検索ツール実行: query='%s', k=%d, case_type=%s, verdict=%s",
        query[:50],
        k,
        case_type,
        verdict,
    )

    try:
        docs = retrieve_precedents(query, k=k, case_type=case_type, verdict=verdict)
        result = _format_precedent_documents(docs)
        logger.info("判例検索完了: %d 件の結果", len(docs))
        return result
    except FileNotFoundError as e:
        logger.error("ChromaDBが初期化されていません: %s", e)
        return (
            "エラー: 判例データベースが初期化されていません。\n"
            "'python rag_system/ingest.py' を実行してデータを取り込んでください。"
        )
    except Exception as e:
        logger.exception("判例検索中に予期しないエラーが発生しました")
        return f"エラー: 判例検索中にエラーが発生しました: {e}"
