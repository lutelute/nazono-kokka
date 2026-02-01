"""Document ingestion pipeline for the RAG judicial system.

Loads legal framework documents (markdown) and case precedents (JSON)
from disk, splits them into chunks, embeds them, and stores them in
ChromaDB for retrieval.

Usage:
    python rag_system/ingest.py [--reset]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from rag_system.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    EMBEDDINGS_MODEL_NAME,
    LEGAL_FRAMEWORK_DIR,
    PRECEDENTS_CIVIL_DIR,
    PRECEDENTS_CONSTITUTIONAL_DIR,
    PRECEDENTS_CRIMINAL_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document Loaders
# ---------------------------------------------------------------------------


def load_legal_framework_documents() -> list[Document]:
    """Load all markdown files from the legal framework directory.

    Each markdown file is loaded as a single ``Document`` with metadata
    indicating the source file and document type ``legal_framework``.
    """
    documents: list[Document] = []
    md_files = sorted(LEGAL_FRAMEWORK_DIR.glob("*.md"))

    if not md_files:
        logger.warning("法的枠組みディレクトリにMarkdownファイルが見つかりません: %s", LEGAL_FRAMEWORK_DIR)
        return documents

    for md_path in md_files:
        try:
            text = md_path.read_text(encoding="utf-8")
            if not text.strip():
                logger.warning("空のファイルをスキップ: %s", md_path.name)
                continue
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(md_path.relative_to(LEGAL_FRAMEWORK_DIR.parent)),
                    "document_type": "legal_framework",
                    "filename": md_path.stem,
                },
            )
            documents.append(doc)
            logger.info("法的文書を読み込み: %s (%d 文字)", md_path.name, len(text))
        except Exception:
            logger.exception("法的文書の読み込みに失敗: %s", md_path.name)

    return documents


def load_precedent_documents() -> list[Document]:
    """Load all JSON precedent files from the precedents directory.

    Each JSON file is converted into a ``Document`` whose page content is
    a human-readable text representation of the case.  Structured metadata
    fields (``case_id``, ``case_type``, ``verdict``, etc.) are preserved
    in the document metadata for filtered retrieval.
    """
    documents: list[Document] = []
    category_dirs = [
        ("criminal", PRECEDENTS_CRIMINAL_DIR),
        ("civil", PRECEDENTS_CIVIL_DIR),
        ("constitutional", PRECEDENTS_CONSTITUTIONAL_DIR),
    ]

    for category, dir_path in category_dirs:
        if not dir_path.exists():
            logger.warning("判例ディレクトリが存在しません: %s", dir_path)
            continue

        json_files = sorted(dir_path.glob("*.json"))
        if not json_files:
            logger.warning("%s の判例ファイルが見つかりません", category)
            continue

        loaded = 0
        for json_path in json_files:
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                page_content = _precedent_to_text(data)
                metadata = {
                    "source": str(json_path.relative_to(dir_path.parent.parent)),
                    "document_type": "precedent",
                    "case_type": category,
                    "case_id": data.get("case_id", ""),
                    "verdict": data.get("verdict", ""),
                    "filename": json_path.stem,
                }
                documents.append(
                    Document(page_content=page_content, metadata=metadata)
                )
                loaded += 1
            except json.JSONDecodeError:
                logger.exception("JSONパースエラー: %s", json_path.name)
            except Exception:
                logger.exception("判例の読み込みに失敗: %s", json_path.name)

        logger.info("%s 判例を %d 件読み込み", category, loaded)

    return documents


def _precedent_to_text(data: dict) -> str:
    """Convert a precedent JSON object to a human-readable text string."""
    parts: list[str] = []

    if title := data.get("title"):
        parts.append(f"事件名: {title}")
    if case_id := data.get("case_id"):
        parts.append(f"事件番号: {case_id}")
    if date := data.get("date"):
        parts.append(f"判決日: {date}")
    if case_type := data.get("case_type"):
        parts.append(f"事件種別: {case_type}")

    if charges := data.get("charges"):
        if isinstance(charges, list):
            parts.append("罪状: " + "、".join(charges))

    if verdict := data.get("verdict"):
        parts.append(f"判決: {verdict}")
    if sentence := data.get("sentence"):
        parts.append(f"量刑: {sentence}")

    if principles := data.get("legal_principles"):
        if isinstance(principles, list):
            parts.append("法的原則: " + "、".join(principles))

    if statutes := data.get("referenced_statutes"):
        if isinstance(statutes, list):
            parts.append("参照法令: " + "、".join(statutes))

    if summary := data.get("summary"):
        parts.append(f"\n概要:\n{summary}")
    if reasoning := data.get("reasoning"):
        parts.append(f"\n判決理由:\n{reasoning}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create a text splitter configured for legal markdown documents."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks using the configured text splitter.

    Legal framework documents are split into chunks while precedent
    documents are kept as-is (they are typically short enough to fit
    within a single chunk).
    """
    splitter = create_text_splitter()

    legal_docs = [d for d in documents if d.metadata.get("document_type") == "legal_framework"]
    precedent_docs = [d for d in documents if d.metadata.get("document_type") == "precedent"]

    chunks: list[Document] = []

    if legal_docs:
        legal_chunks = splitter.split_documents(legal_docs)
        logger.info("法的文書を %d チャンクに分割 (元文書: %d 件)", len(legal_chunks), len(legal_docs))
        chunks.extend(legal_chunks)

    if precedent_docs:
        precedent_chunks = splitter.split_documents(precedent_docs)
        logger.info("判例を %d チャンクに分割 (元文書: %d 件)", len(precedent_chunks), len(precedent_docs))
        chunks.extend(precedent_chunks)

    return chunks


# ---------------------------------------------------------------------------
# Embedding & Storage
# ---------------------------------------------------------------------------


def create_embeddings() -> HuggingFaceEmbeddings:
    """Create the HuggingFace embedding model for multilingual support."""
    logger.info("埋め込みモデルを初期化: %s", EMBEDDINGS_MODEL_NAME)
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)


def store_documents(chunks: list[Document], embeddings: HuggingFaceEmbeddings, *, reset: bool = False) -> Chroma:
    """Embed document chunks and store them in ChromaDB.

    Parameters
    ----------
    chunks:
        The document chunks to embed and store.
    embeddings:
        The embedding model to use.
    reset:
        If ``True``, delete the existing collection before ingesting.
    """
    if reset:
        import chromadb

        logger.info("既存のコレクションをリセット")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        existing = [c.name for c in client.list_collections()]
        if CHROMA_COLLECTION_NAME in existing:
            client.delete_collection(CHROMA_COLLECTION_NAME)
            logger.info("コレクション '%s' を削除しました", CHROMA_COLLECTION_NAME)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    logger.info(
        "ChromaDB にドキュメントを保存完了: %d チャンク (コレクション: %s)",
        len(chunks),
        CHROMA_COLLECTION_NAME,
    )
    return vectorstore


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def run_ingestion(*, reset: bool = False) -> None:
    """Execute the full ingestion pipeline.

    1. Load legal framework markdown files.
    2. Load precedent JSON files.
    3. Split all documents into chunks.
    4. Embed and store in ChromaDB.
    """
    logger.info("=== ドキュメント取り込みパイプラインを開始 ===")

    # Load documents
    legal_docs = load_legal_framework_documents()
    precedent_docs = load_precedent_documents()
    all_docs = legal_docs + precedent_docs

    if not all_docs:
        logger.error("取り込み可能なドキュメントが見つかりません。処理を中止します。")
        sys.exit(1)

    logger.info(
        "合計 %d 件のドキュメントを読み込み (法的文書: %d, 判例: %d)",
        len(all_docs),
        len(legal_docs),
        len(precedent_docs),
    )

    # Chunk documents
    chunks = split_documents(all_docs)
    logger.info("合計チャンク数: %d", len(chunks))

    # Embed and store
    embeddings = create_embeddings()
    store_documents(chunks, embeddings, reset=reset)

    logger.info("=== ドキュメント取り込みパイプライン完了 ===")


def main() -> None:
    """CLI entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="謎の国家 - ドキュメント取り込みパイプライン",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="既存のChromaDBコレクションをリセットしてから取り込みを実行",
    )
    args = parser.parse_args()

    run_ingestion(reset=args.reset)


if __name__ == "__main__":
    main()
