"""Tests for rag_system.retriever module.

Verifies retriever creation, metadata filter building, and convenience
retrieval functions using mocked ChromaDB vector stores.
"""

from unittest import mock

import pytest
from langchain_core.documents import Document

from rag_system.retriever import (
    _build_where_filter,
    create_retriever,
    retrieve,
    retrieve_legal_framework,
    retrieve_precedents,
    retrieve_with_scores,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_documents() -> list[Document]:
    """Return a list of sample LangChain Documents for retriever results."""
    return [
        Document(
            page_content="窃盗罪に関する判例。被告人はコンビニにおいて商品を窃取した。",
            metadata={
                "document_type": "precedent",
                "case_type": "criminal",
                "verdict": "有罪",
                "case_id": "令和5年(あ)第123号",
            },
        ),
        Document(
            page_content="刑法第235条：他人の財物を窃取した者は、窃盗の罪とし、十年以下の懲役に処する。",
            metadata={
                "document_type": "legal_framework",
                "filename": "criminal_code",
            },
        ),
        Document(
            page_content="民事損害賠償請求事件。原告の主張が認められた。",
            metadata={
                "document_type": "precedent",
                "case_type": "civil",
                "verdict": "原告勝訴",
                "case_id": "令和5年(ワ)第456号",
            },
        ),
    ]


@pytest.fixture()
def mock_vectorstore(sample_documents: list[Document]):
    """Create a mock Chroma vector store that returns sample documents."""
    mock_vs = mock.MagicMock()
    mock_retriever = mock.MagicMock()
    mock_retriever.invoke.return_value = sample_documents
    mock_vs.as_retriever.return_value = mock_retriever
    mock_vs.similarity_search_with_score.return_value = [
        (doc, 0.9 - i * 0.1) for i, doc in enumerate(sample_documents)
    ]
    return mock_vs


# ---------------------------------------------------------------------------
# Tests: _build_where_filter
# ---------------------------------------------------------------------------


class TestBuildWhereFilter:
    """Test the _build_where_filter helper function."""

    def test_no_filters_returns_none(self):
        result = _build_where_filter()
        assert result is None

    def test_document_type_only(self):
        result = _build_where_filter(document_type="legal_framework")
        assert result == {"document_type": {"$eq": "legal_framework"}}

    def test_case_type_only(self):
        result = _build_where_filter(case_type="criminal")
        assert result == {"case_type": {"$eq": "criminal"}}

    def test_verdict_only(self):
        result = _build_where_filter(verdict="有罪")
        assert result == {"verdict": {"$eq": "有罪"}}

    def test_two_filters_combined_with_and(self):
        result = _build_where_filter(document_type="precedent", case_type="civil")
        assert result == {
            "$and": [
                {"document_type": {"$eq": "precedent"}},
                {"case_type": {"$eq": "civil"}},
            ]
        }

    def test_all_three_filters_combined_with_and(self):
        result = _build_where_filter(
            document_type="precedent",
            case_type="criminal",
            verdict="無罪",
        )
        assert result == {
            "$and": [
                {"document_type": {"$eq": "precedent"}},
                {"case_type": {"$eq": "criminal"}},
                {"verdict": {"$eq": "無罪"}},
            ]
        }

    def test_none_values_ignored(self):
        result = _build_where_filter(
            document_type=None,
            case_type="criminal",
            verdict=None,
        )
        assert result == {"case_type": {"$eq": "criminal"}}


# ---------------------------------------------------------------------------
# Tests: create_retriever
# ---------------------------------------------------------------------------


class TestCreateRetriever:
    """Test the create_retriever function."""

    def test_creates_retriever_with_default_k(self, mock_vectorstore):
        retriever = create_retriever(vectorstore=mock_vectorstore)

        mock_vectorstore.as_retriever.assert_called_once()
        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["k"] == 5  # RETRIEVAL_K default

    def test_creates_retriever_with_custom_k(self, mock_vectorstore):
        create_retriever(vectorstore=mock_vectorstore, k=10)

        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["k"] == 10

    def test_creates_retriever_without_filter(self, mock_vectorstore):
        create_retriever(vectorstore=mock_vectorstore)

        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        assert "filter" not in call_kwargs["search_kwargs"]

    def test_creates_retriever_with_document_type_filter(self, mock_vectorstore):
        create_retriever(
            vectorstore=mock_vectorstore,
            document_type="legal_framework",
        )

        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["filter"] == {
            "document_type": {"$eq": "legal_framework"}
        }

    def test_creates_retriever_with_multiple_filters(self, mock_vectorstore):
        create_retriever(
            vectorstore=mock_vectorstore,
            document_type="precedent",
            case_type="criminal",
            verdict="有罪",
        )

        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        expected_filter = {
            "$and": [
                {"document_type": {"$eq": "precedent"}},
                {"case_type": {"$eq": "criminal"}},
                {"verdict": {"$eq": "有罪"}},
            ]
        }
        assert call_kwargs["search_kwargs"]["filter"] == expected_filter

    def test_loads_vectorstore_when_none(self):
        with mock.patch("rag_system.retriever.load_vectorstore") as mock_load:
            mock_vs = mock.MagicMock()
            mock_load.return_value = mock_vs

            create_retriever(vectorstore=None)

            mock_load.assert_called_once()
            mock_vs.as_retriever.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: retrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    """Test the retrieve convenience function."""

    def test_returns_documents(self, mock_vectorstore, sample_documents):
        results = retrieve("窃盗罪の判例", vectorstore=mock_vectorstore)
        assert results == sample_documents

    def test_empty_query_returns_empty_list(self, mock_vectorstore):
        results = retrieve("", vectorstore=mock_vectorstore)
        assert results == []

    def test_whitespace_query_returns_empty_list(self, mock_vectorstore):
        results = retrieve("   ", vectorstore=mock_vectorstore)
        assert results == []

    def test_passes_k_parameter(self, mock_vectorstore):
        retrieve("窃盗罪", vectorstore=mock_vectorstore, k=3)

        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["k"] == 3

    def test_passes_filter_parameters(self, mock_vectorstore):
        retrieve(
            "窃盗罪",
            vectorstore=mock_vectorstore,
            document_type="precedent",
            case_type="criminal",
        )

        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        assert "filter" in call_kwargs["search_kwargs"]

    def test_handles_retriever_exception(self, mock_vectorstore):
        mock_retriever = mock.MagicMock()
        mock_retriever.invoke.side_effect = RuntimeError("ChromaDB error")
        mock_vectorstore.as_retriever.return_value = mock_retriever

        results = retrieve("窃盗罪", vectorstore=mock_vectorstore)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: retrieve_legal_framework
# ---------------------------------------------------------------------------


class TestRetrieveLegalFramework:
    """Test the retrieve_legal_framework convenience function."""

    def test_filters_by_legal_framework(self, mock_vectorstore):
        retrieve_legal_framework("憲法", vectorstore=mock_vectorstore)

        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["filter"] == {
            "document_type": {"$eq": "legal_framework"}
        }

    def test_passes_k_parameter(self, mock_vectorstore):
        retrieve_legal_framework("刑法", vectorstore=mock_vectorstore, k=3)

        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["k"] == 3

    def test_returns_documents(self, mock_vectorstore, sample_documents):
        results = retrieve_legal_framework("刑法", vectorstore=mock_vectorstore)
        assert results == sample_documents


# ---------------------------------------------------------------------------
# Tests: retrieve_precedents
# ---------------------------------------------------------------------------


class TestRetrievePrecedents:
    """Test the retrieve_precedents convenience function."""

    def test_filters_by_precedent(self, mock_vectorstore):
        retrieve_precedents("窃盗", vectorstore=mock_vectorstore)

        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        assert "filter" in call_kwargs["search_kwargs"]
        where_filter = call_kwargs["search_kwargs"]["filter"]
        # Should at least contain document_type filter
        assert {"document_type": {"$eq": "precedent"}} == where_filter or (
            "$and" in where_filter
            and {"document_type": {"$eq": "precedent"}} in where_filter["$and"]
        )

    def test_filters_by_case_type(self, mock_vectorstore):
        retrieve_precedents(
            "窃盗",
            vectorstore=mock_vectorstore,
            case_type="criminal",
        )

        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        where_filter = call_kwargs["search_kwargs"]["filter"]
        assert "$and" in where_filter
        assert {"document_type": {"$eq": "precedent"}} in where_filter["$and"]
        assert {"case_type": {"$eq": "criminal"}} in where_filter["$and"]

    def test_filters_by_verdict(self, mock_vectorstore):
        retrieve_precedents(
            "窃盗",
            vectorstore=mock_vectorstore,
            verdict="有罪",
        )

        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        where_filter = call_kwargs["search_kwargs"]["filter"]
        assert "$and" in where_filter
        assert {"verdict": {"$eq": "有罪"}} in where_filter["$and"]

    def test_returns_documents(self, mock_vectorstore, sample_documents):
        results = retrieve_precedents("窃盗", vectorstore=mock_vectorstore)
        assert results == sample_documents


# ---------------------------------------------------------------------------
# Tests: retrieve_with_scores
# ---------------------------------------------------------------------------


class TestRetrieveWithScores:
    """Test the retrieve_with_scores function."""

    def test_returns_document_score_pairs(self, mock_vectorstore, sample_documents):
        results = retrieve_with_scores("窃盗罪", vectorstore=mock_vectorstore)

        assert len(results) == len(sample_documents)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_scores_are_ordered(self, mock_vectorstore):
        results = retrieve_with_scores("窃盗罪", vectorstore=mock_vectorstore)

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty_list(self, mock_vectorstore):
        results = retrieve_with_scores("", vectorstore=mock_vectorstore)
        assert results == []

    def test_whitespace_query_returns_empty_list(self, mock_vectorstore):
        results = retrieve_with_scores("   ", vectorstore=mock_vectorstore)
        assert results == []

    def test_passes_k_parameter(self, mock_vectorstore):
        retrieve_with_scores("窃盗罪", vectorstore=mock_vectorstore, k=2)

        mock_vectorstore.similarity_search_with_score.assert_called_once()
        call_args = mock_vectorstore.similarity_search_with_score.call_args
        assert call_args[1].get("k", call_args[0][1] if len(call_args[0]) > 1 else None) == 2

    def test_passes_filter_parameters(self, mock_vectorstore):
        retrieve_with_scores(
            "窃盗罪",
            vectorstore=mock_vectorstore,
            document_type="precedent",
            case_type="criminal",
        )

        call_kwargs = mock_vectorstore.similarity_search_with_score.call_args[1]
        expected_filter = {
            "$and": [
                {"document_type": {"$eq": "precedent"}},
                {"case_type": {"$eq": "criminal"}},
            ]
        }
        assert call_kwargs["filter"] == expected_filter

    def test_no_filter_when_none_specified(self, mock_vectorstore):
        retrieve_with_scores("窃盗罪", vectorstore=mock_vectorstore)

        call_kwargs = mock_vectorstore.similarity_search_with_score.call_args[1]
        assert "filter" not in call_kwargs

    def test_handles_exception(self, mock_vectorstore):
        mock_vectorstore.similarity_search_with_score.side_effect = RuntimeError(
            "ChromaDB error"
        )

        results = retrieve_with_scores("窃盗罪", vectorstore=mock_vectorstore)
        assert results == []

    def test_loads_vectorstore_when_none(self):
        with mock.patch("rag_system.retriever.load_vectorstore") as mock_load:
            mock_vs = mock.MagicMock()
            mock_vs.similarity_search_with_score.return_value = []
            mock_load.return_value = mock_vs

            retrieve_with_scores("窃盗罪")

            mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: load_vectorstore
# ---------------------------------------------------------------------------


class TestLoadVectorstore:
    """Test the load_vectorstore function."""

    def test_raises_file_not_found_for_missing_dir(self, tmp_path):
        nonexistent = str(tmp_path / "nonexistent_chroma")

        with mock.patch("rag_system.retriever.CHROMA_DB_PATH", nonexistent):
            from rag_system.retriever import load_vectorstore

            with pytest.raises(FileNotFoundError, match="ChromaDB"):
                load_vectorstore()
