"""Tests for rag_system.ingest module.

Verifies document loading, chunking, text conversion, and the
split_documents pipeline using temporary directories and mock data.
"""

import json
from pathlib import Path
from unittest import mock

import pytest
from langchain_core.documents import Document

from rag_system.ingest import (
    _precedent_to_text,
    create_text_splitter,
    load_legal_framework_documents,
    load_precedent_documents,
    split_documents,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def legal_framework_dir(tmp_path: Path) -> Path:
    """Create a temporary legal framework directory with sample markdown files."""
    framework_dir = tmp_path / "legal_framework"
    framework_dir.mkdir()

    (framework_dir / "constitution.md").write_text(
        "# 憲法\n\n## 第1章 総則\n\n国民は法の下に平等である。\n\n## 第2章 権利\n\n基本的人権を保障する。\n",
        encoding="utf-8",
    )
    (framework_dir / "criminal_code.md").write_text(
        "# 刑法\n\n## 第1編 総則\n\n犯罪と刑罰に関する基本法。\n",
        encoding="utf-8",
    )
    return framework_dir


@pytest.fixture()
def empty_legal_framework_dir(tmp_path: Path) -> Path:
    """Create a temporary legal framework directory with no markdown files."""
    framework_dir = tmp_path / "legal_framework"
    framework_dir.mkdir()
    return framework_dir


@pytest.fixture()
def precedent_data() -> dict:
    """Return a sample precedent JSON object."""
    return {
        "title": "テスト事件",
        "case_id": "令和5年(あ)第123号",
        "date": "2023-06-15",
        "case_type": "criminal",
        "charges": ["窃盗罪", "詐欺罪"],
        "verdict": "有罪",
        "sentence": "懲役3年",
        "legal_principles": ["罪刑法定主義", "比例原則"],
        "referenced_statutes": ["刑法第235条", "刑法第246条"],
        "summary": "被告人はコンビニにおいて商品を窃取した。",
        "reasoning": "証拠により犯行が認定された。",
    }


@pytest.fixture()
def precedents_dir(tmp_path: Path, precedent_data: dict) -> Path:
    """Create a temporary precedents directory with sample JSON files."""
    precedents = tmp_path / "precedents"
    criminal_dir = precedents / "criminal"
    civil_dir = precedents / "civil"
    constitutional_dir = precedents / "constitutional"
    criminal_dir.mkdir(parents=True)
    civil_dir.mkdir(parents=True)
    constitutional_dir.mkdir(parents=True)

    (criminal_dir / "case_001.json").write_text(
        json.dumps(precedent_data, ensure_ascii=False),
        encoding="utf-8",
    )

    civil_data = {
        "title": "民事テスト事件",
        "case_id": "令和5年(ワ)第456号",
        "case_type": "civil",
        "verdict": "原告勝訴",
        "summary": "損害賠償請求事件。",
    }
    (civil_dir / "case_002.json").write_text(
        json.dumps(civil_data, ensure_ascii=False),
        encoding="utf-8",
    )

    return precedents


# ---------------------------------------------------------------------------
# Tests: _precedent_to_text
# ---------------------------------------------------------------------------


class TestPrecedentToText:
    """Test the _precedent_to_text helper function."""

    def test_full_precedent(self, precedent_data: dict):
        text = _precedent_to_text(precedent_data)

        assert "事件名: テスト事件" in text
        assert "事件番号: 令和5年(あ)第123号" in text
        assert "判決日: 2023-06-15" in text
        assert "事件種別: criminal" in text
        assert "罪状: 窃盗罪、詐欺罪" in text
        assert "判決: 有罪" in text
        assert "量刑: 懲役3年" in text
        assert "法的原則: 罪刑法定主義、比例原則" in text
        assert "参照法令: 刑法第235条、刑法第246条" in text
        assert "概要:\n被告人はコンビニにおいて商品を窃取した。" in text
        assert "判決理由:\n証拠により犯行が認定された。" in text

    def test_empty_data(self):
        text = _precedent_to_text({})
        assert text == ""

    def test_partial_data(self):
        data = {"title": "部分的事件", "verdict": "無罪"}
        text = _precedent_to_text(data)

        assert "事件名: 部分的事件" in text
        assert "判決: 無罪" in text
        assert "事件番号" not in text
        assert "罪状" not in text

    def test_charges_non_list_ignored(self):
        data = {"charges": "窃盗罪"}
        text = _precedent_to_text(data)
        assert "罪状" not in text

    def test_legal_principles_non_list_ignored(self):
        data = {"legal_principles": "罪刑法定主義"}
        text = _precedent_to_text(data)
        assert "法的原則" not in text

    def test_referenced_statutes_non_list_ignored(self):
        data = {"referenced_statutes": "刑法第235条"}
        text = _precedent_to_text(data)
        assert "参照法令" not in text


# ---------------------------------------------------------------------------
# Tests: load_legal_framework_documents
# ---------------------------------------------------------------------------


class TestLoadLegalFrameworkDocuments:
    """Test loading markdown legal framework documents."""

    def test_loads_markdown_files(self, legal_framework_dir: Path):
        with mock.patch("rag_system.ingest.LEGAL_FRAMEWORK_DIR", legal_framework_dir):
            docs = load_legal_framework_documents()

        assert len(docs) == 2

    def test_document_metadata(self, legal_framework_dir: Path):
        with mock.patch("rag_system.ingest.LEGAL_FRAMEWORK_DIR", legal_framework_dir):
            docs = load_legal_framework_documents()

        for doc in docs:
            assert doc.metadata["document_type"] == "legal_framework"
            assert "filename" in doc.metadata
            assert "source" in doc.metadata

    def test_document_content_not_empty(self, legal_framework_dir: Path):
        with mock.patch("rag_system.ingest.LEGAL_FRAMEWORK_DIR", legal_framework_dir):
            docs = load_legal_framework_documents()

        for doc in docs:
            assert len(doc.page_content.strip()) > 0

    def test_empty_directory_returns_empty_list(self, empty_legal_framework_dir: Path):
        with mock.patch("rag_system.ingest.LEGAL_FRAMEWORK_DIR", empty_legal_framework_dir):
            docs = load_legal_framework_documents()

        assert docs == []

    def test_skips_empty_files(self, legal_framework_dir: Path):
        (legal_framework_dir / "empty.md").write_text("", encoding="utf-8")

        with mock.patch("rag_system.ingest.LEGAL_FRAMEWORK_DIR", legal_framework_dir):
            docs = load_legal_framework_documents()

        filenames = [d.metadata["filename"] for d in docs]
        assert "empty" not in filenames

    def test_skips_whitespace_only_files(self, legal_framework_dir: Path):
        (legal_framework_dir / "whitespace.md").write_text("   \n\n  ", encoding="utf-8")

        with mock.patch("rag_system.ingest.LEGAL_FRAMEWORK_DIR", legal_framework_dir):
            docs = load_legal_framework_documents()

        filenames = [d.metadata["filename"] for d in docs]
        assert "whitespace" not in filenames

    def test_sorted_loading_order(self, legal_framework_dir: Path):
        with mock.patch("rag_system.ingest.LEGAL_FRAMEWORK_DIR", legal_framework_dir):
            docs = load_legal_framework_documents()

        filenames = [d.metadata["filename"] for d in docs]
        assert filenames == sorted(filenames)


# ---------------------------------------------------------------------------
# Tests: load_precedent_documents
# ---------------------------------------------------------------------------


class TestLoadPrecedentDocuments:
    """Test loading JSON precedent documents."""

    def test_loads_precedent_files(self, precedents_dir: Path):
        with (
            mock.patch("rag_system.ingest.PRECEDENTS_CRIMINAL_DIR", precedents_dir / "criminal"),
            mock.patch("rag_system.ingest.PRECEDENTS_CIVIL_DIR", precedents_dir / "civil"),
            mock.patch("rag_system.ingest.PRECEDENTS_CONSTITUTIONAL_DIR", precedents_dir / "constitutional"),
        ):
            docs = load_precedent_documents()

        assert len(docs) == 2

    def test_precedent_metadata(self, precedents_dir: Path):
        with (
            mock.patch("rag_system.ingest.PRECEDENTS_CRIMINAL_DIR", precedents_dir / "criminal"),
            mock.patch("rag_system.ingest.PRECEDENTS_CIVIL_DIR", precedents_dir / "civil"),
            mock.patch("rag_system.ingest.PRECEDENTS_CONSTITUTIONAL_DIR", precedents_dir / "constitutional"),
        ):
            docs = load_precedent_documents()

        for doc in docs:
            assert doc.metadata["document_type"] == "precedent"
            assert doc.metadata["case_type"] in ("criminal", "civil", "constitutional")
            assert "case_id" in doc.metadata
            assert "verdict" in doc.metadata
            assert "filename" in doc.metadata
            assert "source" in doc.metadata

    def test_criminal_precedent_content(self, precedents_dir: Path):
        with (
            mock.patch("rag_system.ingest.PRECEDENTS_CRIMINAL_DIR", precedents_dir / "criminal"),
            mock.patch("rag_system.ingest.PRECEDENTS_CIVIL_DIR", precedents_dir / "civil"),
            mock.patch("rag_system.ingest.PRECEDENTS_CONSTITUTIONAL_DIR", precedents_dir / "constitutional"),
        ):
            docs = load_precedent_documents()

        criminal_docs = [d for d in docs if d.metadata["case_type"] == "criminal"]
        assert len(criminal_docs) == 1
        assert "テスト事件" in criminal_docs[0].page_content

    def test_missing_directory_skipped(self, tmp_path: Path):
        nonexistent = tmp_path / "nonexistent"
        with (
            mock.patch("rag_system.ingest.PRECEDENTS_CRIMINAL_DIR", nonexistent),
            mock.patch("rag_system.ingest.PRECEDENTS_CIVIL_DIR", nonexistent),
            mock.patch("rag_system.ingest.PRECEDENTS_CONSTITUTIONAL_DIR", nonexistent),
        ):
            docs = load_precedent_documents()

        assert docs == []

    def test_invalid_json_skipped(self, precedents_dir: Path):
        (precedents_dir / "criminal" / "bad.json").write_text(
            "{invalid json",
            encoding="utf-8",
        )

        with (
            mock.patch("rag_system.ingest.PRECEDENTS_CRIMINAL_DIR", precedents_dir / "criminal"),
            mock.patch("rag_system.ingest.PRECEDENTS_CIVIL_DIR", precedents_dir / "civil"),
            mock.patch("rag_system.ingest.PRECEDENTS_CONSTITUTIONAL_DIR", precedents_dir / "constitutional"),
        ):
            docs = load_precedent_documents()

        # Original 2 files should still load; bad.json is skipped
        assert len(docs) == 2


# ---------------------------------------------------------------------------
# Tests: create_text_splitter
# ---------------------------------------------------------------------------


class TestCreateTextSplitter:
    """Test the text splitter factory function."""

    def test_returns_recursive_splitter(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = create_text_splitter()
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_splitter_chunk_size(self):
        from rag_system.config import CHUNK_SIZE

        splitter = create_text_splitter()
        assert splitter._chunk_size == CHUNK_SIZE

    def test_splitter_chunk_overlap(self):
        from rag_system.config import CHUNK_OVERLAP

        splitter = create_text_splitter()
        assert splitter._chunk_overlap == CHUNK_OVERLAP


# ---------------------------------------------------------------------------
# Tests: split_documents
# ---------------------------------------------------------------------------


class TestSplitDocuments:
    """Test the split_documents function."""

    def test_empty_list(self):
        chunks = split_documents([])
        assert chunks == []

    def test_short_legal_doc_stays_single_chunk(self):
        doc = Document(
            page_content="短い法的文書。",
            metadata={"document_type": "legal_framework"},
        )
        chunks = split_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].metadata["document_type"] == "legal_framework"

    def test_long_legal_doc_is_chunked(self):
        long_text = "法律条文。\n\n" * 500  # Well over CHUNK_SIZE
        doc = Document(
            page_content=long_text,
            metadata={"document_type": "legal_framework"},
        )
        chunks = split_documents([doc])
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["document_type"] == "legal_framework"

    def test_precedent_doc_processed(self):
        doc = Document(
            page_content="判例の短い内容。",
            metadata={"document_type": "precedent", "case_type": "criminal"},
        )
        chunks = split_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].metadata["document_type"] == "precedent"
        assert chunks[0].metadata["case_type"] == "criminal"

    def test_mixed_documents(self):
        legal_doc = Document(
            page_content="法的文書の内容。",
            metadata={"document_type": "legal_framework"},
        )
        precedent_doc = Document(
            page_content="判例の内容。",
            metadata={"document_type": "precedent", "case_type": "civil"},
        )
        chunks = split_documents([legal_doc, precedent_doc])
        assert len(chunks) >= 2

        doc_types = {c.metadata["document_type"] for c in chunks}
        assert "legal_framework" in doc_types
        assert "precedent" in doc_types

    def test_metadata_preserved_after_splitting(self):
        long_text = "法律条文の内容です。\n\n" * 500
        doc = Document(
            page_content=long_text,
            metadata={
                "document_type": "legal_framework",
                "source": "test/constitution.md",
                "filename": "constitution",
            },
        )
        chunks = split_documents([doc])
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["source"] == "test/constitution.md"
            assert chunk.metadata["filename"] == "constitution"
