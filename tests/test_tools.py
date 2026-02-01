"""Tests for rag_system.tools module.

Verifies the three LangChain tool definitions (legal_framework_search,
precedent_search, archive_stats) including normal operation, empty query
handling, error handling, and output formatting.
"""

from unittest import mock

import pytest
from langchain_core.documents import Document

from rag_system.tools import (
    _format_legal_documents,
    _format_precedent_documents,
    archive_stats,
    legal_framework_search,
    precedent_search,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def legal_documents() -> list[Document]:
    """Return sample legal framework documents."""
    return [
        Document(
            page_content="刑法第235条：他人の財物を窃取した者は、窃盗の罪とし、十年以下の懲役に処する。",
            metadata={
                "document_type": "legal_framework",
                "source": "criminal_code.txt",
            },
        ),
        Document(
            page_content="刑法第236条：暴行又は脅迫を用いて他人の財物を強取した者は、強盗の罪とし、五年以上の有期懲役に処する。",
            metadata={
                "document_type": "legal_framework",
                "source": "criminal_code.txt",
            },
        ),
    ]


@pytest.fixture()
def precedent_documents() -> list[Document]:
    """Return sample precedent documents."""
    return [
        Document(
            page_content="窃盗罪に関する判例。被告人はコンビニにおいて商品を窃取した。",
            metadata={
                "document_type": "precedent",
                "case_type": "criminal",
                "verdict": "有罪",
                "source": "case_001.txt",
            },
        ),
        Document(
            page_content="民事損害賠償請求事件。原告の主張が認められた。",
            metadata={
                "document_type": "precedent",
                "case_type": "civil",
                "verdict": "原告勝訴",
                "source": "case_002.txt",
            },
        ),
    ]


@pytest.fixture()
def mock_collection_data() -> dict:
    """Return mock ChromaDB collection metadata for archive_stats."""
    return {
        "metadatas": [
            {"document_type": "legal_framework", "source": "criminal_code.txt"},
            {"document_type": "legal_framework", "source": "civil_code.txt"},
            {
                "document_type": "precedent",
                "case_type": "criminal",
                "verdict": "有罪",
                "source": "case_001.txt",
            },
            {
                "document_type": "precedent",
                "case_type": "civil",
                "verdict": "原告勝訴",
                "source": "case_002.txt",
            },
            {
                "document_type": "precedent",
                "case_type": "criminal",
                "verdict": "無罪",
                "source": "case_003.txt",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests: legal_framework_search
# ---------------------------------------------------------------------------


class TestLegalFrameworkSearchTool:
    """Test the legal_framework_search tool."""

    def test_returns_formatted_results(self, legal_documents):
        with mock.patch(
            "rag_system.tools.retrieve_legal_framework",
            return_value=legal_documents,
        ):
            result = legal_framework_search.invoke({"query": "窃盗罪の構成要件"})

        assert "【法令検索結果】" in result
        assert "2件" in result
        assert "刑法第235条" in result
        assert "criminal_code.txt" in result

    def test_passes_k_parameter(self, legal_documents):
        with mock.patch(
            "rag_system.tools.retrieve_legal_framework",
            return_value=legal_documents,
        ) as mock_retrieve:
            legal_framework_search.invoke({"query": "窃盗罪", "k": 3})

        mock_retrieve.assert_called_once_with("窃盗罪", k=3)

    def test_default_k_is_5(self, legal_documents):
        with mock.patch(
            "rag_system.tools.retrieve_legal_framework",
            return_value=legal_documents,
        ) as mock_retrieve:
            legal_framework_search.invoke({"query": "窃盗罪"})

        mock_retrieve.assert_called_once_with("窃盗罪", k=5)

    def test_empty_query_returns_error(self):
        result = legal_framework_search.invoke({"query": ""})
        assert "エラー" in result
        assert "空" in result

    def test_whitespace_query_returns_error(self):
        result = legal_framework_search.invoke({"query": "   "})
        assert "エラー" in result
        assert "空" in result

    def test_no_results_returns_message(self):
        with mock.patch(
            "rag_system.tools.retrieve_legal_framework",
            return_value=[],
        ):
            result = legal_framework_search.invoke({"query": "存在しない法令"})

        assert "見つかりませんでした" in result

    def test_handles_file_not_found_error(self):
        with mock.patch(
            "rag_system.tools.retrieve_legal_framework",
            side_effect=FileNotFoundError("ChromaDB not found"),
        ):
            result = legal_framework_search.invoke({"query": "窃盗罪"})

        assert "エラー" in result
        assert "初期化されていません" in result

    def test_handles_unexpected_exception(self):
        with mock.patch(
            "rag_system.tools.retrieve_legal_framework",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = legal_framework_search.invoke({"query": "窃盗罪"})

        assert "エラー" in result
        assert "Unexpected error" in result


# ---------------------------------------------------------------------------
# Tests: precedent_search
# ---------------------------------------------------------------------------


class TestPrecedentSearchTool:
    """Test the precedent_search tool."""

    def test_returns_formatted_results(self, precedent_documents):
        with mock.patch(
            "rag_system.tools.retrieve_precedents",
            return_value=precedent_documents,
        ):
            result = precedent_search.invoke({"query": "窃盗罪の判例"})

        assert "【判例検索結果】" in result
        assert "2件" in result
        assert "窃盗罪に関する判例" in result
        assert "事件類型: criminal" in result
        assert "判決: 有罪" in result

    def test_passes_k_parameter(self, precedent_documents):
        with mock.patch(
            "rag_system.tools.retrieve_precedents",
            return_value=precedent_documents,
        ) as mock_retrieve:
            precedent_search.invoke({"query": "窃盗罪", "k": 10})

        mock_retrieve.assert_called_once_with(
            "窃盗罪", k=10, case_type=None, verdict=None,
        )

    def test_passes_case_type_filter(self, precedent_documents):
        with mock.patch(
            "rag_system.tools.retrieve_precedents",
            return_value=precedent_documents,
        ) as mock_retrieve:
            precedent_search.invoke({
                "query": "窃盗罪",
                "case_type": "criminal",
            })

        mock_retrieve.assert_called_once_with(
            "窃盗罪", k=5, case_type="criminal", verdict=None,
        )

    def test_passes_verdict_filter(self, precedent_documents):
        with mock.patch(
            "rag_system.tools.retrieve_precedents",
            return_value=precedent_documents,
        ) as mock_retrieve:
            precedent_search.invoke({
                "query": "窃盗罪",
                "verdict": "有罪",
            })

        mock_retrieve.assert_called_once_with(
            "窃盗罪", k=5, case_type=None, verdict="有罪",
        )

    def test_passes_all_filters(self, precedent_documents):
        with mock.patch(
            "rag_system.tools.retrieve_precedents",
            return_value=precedent_documents,
        ) as mock_retrieve:
            precedent_search.invoke({
                "query": "窃盗罪",
                "k": 3,
                "case_type": "criminal",
                "verdict": "有罪",
            })

        mock_retrieve.assert_called_once_with(
            "窃盗罪", k=3, case_type="criminal", verdict="有罪",
        )

    def test_empty_query_returns_error(self):
        result = precedent_search.invoke({"query": ""})
        assert "エラー" in result
        assert "空" in result

    def test_whitespace_query_returns_error(self):
        result = precedent_search.invoke({"query": "   "})
        assert "エラー" in result
        assert "空" in result

    def test_no_results_returns_message(self):
        with mock.patch(
            "rag_system.tools.retrieve_precedents",
            return_value=[],
        ):
            result = precedent_search.invoke({"query": "存在しない判例"})

        assert "見つかりませんでした" in result

    def test_handles_file_not_found_error(self):
        with mock.patch(
            "rag_system.tools.retrieve_precedents",
            side_effect=FileNotFoundError("ChromaDB not found"),
        ):
            result = precedent_search.invoke({"query": "窃盗罪"})

        assert "エラー" in result
        assert "初期化されていません" in result

    def test_handles_unexpected_exception(self):
        with mock.patch(
            "rag_system.tools.retrieve_precedents",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = precedent_search.invoke({"query": "窃盗罪"})

        assert "エラー" in result
        assert "Unexpected error" in result


# ---------------------------------------------------------------------------
# Tests: archive_stats
# ---------------------------------------------------------------------------


class TestArchiveStatsTool:
    """Test the archive_stats tool."""

    @pytest.fixture()
    def _mock_chromadb(self, mock_collection_data):
        """Set up mocked ChromaDB client and collection."""
        mock_collection = mock.MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.get.return_value = mock_collection_data

        mock_client = mock.MagicMock()
        mock_client.get_collection.return_value = mock_collection

        with mock.patch("chromadb.PersistentClient", return_value=mock_client):
            yield mock_collection

    def test_returns_formatted_stats(self, _mock_chromadb):
        result = archive_stats.invoke({})

        assert "【書庫統計情報】" in result
        assert "総ドキュメント数: 5 件" in result

    def test_shows_document_type_breakdown(self, _mock_chromadb):
        result = archive_stats.invoke({})

        assert "文書タイプ別" in result
        assert "legal_framework: 2 件" in result
        assert "precedent: 3 件" in result

    def test_shows_case_type_breakdown(self, _mock_chromadb):
        result = archive_stats.invoke({})

        assert "事件類型別" in result
        assert "criminal: 2 件" in result
        assert "civil: 1 件" in result

    def test_shows_verdict_breakdown(self, _mock_chromadb):
        result = archive_stats.invoke({})

        assert "判決結果別" in result
        assert "有罪: 1 件" in result
        assert "無罪: 1 件" in result
        assert "原告勝訴: 1 件" in result

    def test_shows_source_file_count(self, _mock_chromadb):
        result = archive_stats.invoke({})

        assert "ソースファイル数:" in result

    def test_handles_file_not_found_error(self):
        with mock.patch(
            "chromadb.PersistentClient",
            side_effect=FileNotFoundError("ChromaDB not found"),
        ):
            result = archive_stats.invoke({})

        assert "エラー" in result
        assert "初期化されていません" in result

    def test_handles_unexpected_exception(self):
        with mock.patch(
            "chromadb.PersistentClient",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = archive_stats.invoke({})

        assert "エラー" in result
        assert "Unexpected error" in result


# ---------------------------------------------------------------------------
# Tests: Empty Query Handling
# ---------------------------------------------------------------------------


class TestToolEmptyQuery:
    """Test that all search tools handle empty queries correctly."""

    def test_legal_search_empty_string(self):
        result = legal_framework_search.invoke({"query": ""})
        assert "エラー" in result

    def test_legal_search_whitespace_only(self):
        result = legal_framework_search.invoke({"query": "  \t  "})
        assert "エラー" in result

    def test_precedent_search_empty_string(self):
        result = precedent_search.invoke({"query": ""})
        assert "エラー" in result

    def test_precedent_search_whitespace_only(self):
        result = precedent_search.invoke({"query": "  \t  "})
        assert "エラー" in result


# ---------------------------------------------------------------------------
# Tests: Output Formatting
# ---------------------------------------------------------------------------


class TestToolFormatting:
    """Test that tool outputs are properly formatted Japanese strings."""

    def test_format_legal_documents_empty(self):
        result = _format_legal_documents([])
        assert result == "該当する法令文書が見つかりませんでした。"

    def test_format_legal_documents_with_data(self, legal_documents):
        result = _format_legal_documents(legal_documents)

        assert "【法令検索結果】（2件）" in result
        assert "--- 結果 1 ---" in result
        assert "--- 結果 2 ---" in result
        assert "出典: criminal_code.txt" in result
        assert "内容:" in result
        assert "刑法第235条" in result

    def test_format_legal_documents_missing_metadata(self):
        doc = Document(page_content="テスト内容", metadata={})
        result = _format_legal_documents([doc])

        assert "出典: 不明" in result
        assert "テスト内容" in result

    def test_format_precedent_documents_empty(self):
        result = _format_precedent_documents([])
        assert result == "該当する判例が見つかりませんでした。"

    def test_format_precedent_documents_with_data(self, precedent_documents):
        result = _format_precedent_documents(precedent_documents)

        assert "【判例検索結果】（2件）" in result
        assert "--- 結果 1 ---" in result
        assert "--- 結果 2 ---" in result
        assert "出典: case_001.txt" in result
        assert "事件類型: criminal" in result
        assert "判決: 有罪" in result
        assert "内容:" in result

    def test_format_precedent_documents_missing_metadata(self):
        doc = Document(page_content="テスト判例", metadata={})
        result = _format_precedent_documents([doc])

        assert "出典: 不明" in result
        assert "事件類型: 不明" in result
        assert "判決: 不明" in result
        assert "テスト判例" in result

    def test_legal_tool_output_is_string(self, legal_documents):
        with mock.patch(
            "rag_system.tools.retrieve_legal_framework",
            return_value=legal_documents,
        ):
            result = legal_framework_search.invoke({"query": "窃盗罪"})

        assert isinstance(result, str)

    def test_precedent_tool_output_is_string(self, precedent_documents):
        with mock.patch(
            "rag_system.tools.retrieve_precedents",
            return_value=precedent_documents,
        ):
            result = precedent_search.invoke({"query": "窃盗罪"})

        assert isinstance(result, str)

    def test_archive_stats_output_is_string(self):
        mock_collection = mock.MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"metadatas": []}

        mock_client = mock.MagicMock()
        mock_client.get_collection.return_value = mock_collection

        with mock.patch(
            "chromadb.PersistentClient",
            return_value=mock_client,
        ):
            result = archive_stats.invoke({})

        assert isinstance(result, str)
