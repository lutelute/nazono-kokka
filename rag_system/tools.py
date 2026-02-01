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

from rag_system.config import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH, RETRIEVAL_K
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


@tool("archive_stats")
def archive_stats() -> str:
    """書庫（法令・判例データベース）の統計情報を返します。

    格納されているドキュメント数、文書タイプ別の件数、事件類型別の件数など、
    書庫の概要を把握する際に使用してください。引数は不要です。
    """
    logger.info("書庫統計ツール実行")

    try:
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

        total_count = collection.count()

        # メタデータからドキュメントタイプ別・事件類型別の統計を集計
        all_data = collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])

        type_counts: dict[str, int] = {}
        case_type_counts: dict[str, int] = {}
        verdict_counts: dict[str, int] = {}
        sources: set[str] = set()

        for meta in metadatas:
            if not meta:
                continue
            doc_type = meta.get("document_type", "不明")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

            case_type = meta.get("case_type")
            if case_type:
                case_type_counts[case_type] = case_type_counts.get(case_type, 0) + 1

            verdict = meta.get("verdict")
            if verdict:
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

            source = meta.get("source")
            if source:
                sources.add(source)

        # フォーマット済み統計文字列を構築
        formatted_parts: list[str] = []
        formatted_parts.append("【書庫統計情報】\n")
        formatted_parts.append(f"総ドキュメント数: {total_count} 件")
        formatted_parts.append(f"ソースファイル数: {len(sources)} 件")

        formatted_parts.append("\n--- 文書タイプ別 ---")
        for doc_type, count in sorted(type_counts.items()):
            formatted_parts.append(f"  {doc_type}: {count} 件")

        if case_type_counts:
            formatted_parts.append("\n--- 事件類型別 ---")
            for case_type, count in sorted(case_type_counts.items()):
                formatted_parts.append(f"  {case_type}: {count} 件")

        if verdict_counts:
            formatted_parts.append("\n--- 判決結果別 ---")
            for verdict, count in sorted(verdict_counts.items()):
                formatted_parts.append(f"  {verdict}: {count} 件")

        result = "\n".join(formatted_parts)
        logger.info("書庫統計完了: 総ドキュメント数=%d", total_count)
        return result

    except FileNotFoundError as e:
        logger.error("ChromaDBが初期化されていません: %s", e)
        return (
            "エラー: 書庫データベースが初期化されていません。\n"
            "'python rag_system/ingest.py' を実行してデータを取り込んでください。"
        )
    except Exception as e:
        logger.exception("書庫統計取得中に予期しないエラーが発生しました")
        return f"エラー: 書庫統計の取得中にエラーが発生しました: {e}"
