"""Tests for rag_system.config module.

Verifies that all configuration values are correctly defined,
environment variable overrides work, and paths are properly resolved.
"""

import importlib
import os
from pathlib import Path
from unittest import mock

import pytest


def _reload_config():
    """Reload the config module to pick up environment changes."""
    import rag_system.config as cfg

    return importlib.reload(cfg)


class TestProjectPaths:
    """Test that project path constants are correctly resolved."""

    def test_project_root_is_directory(self):
        from rag_system import config

        assert config.PROJECT_ROOT.is_dir()

    def test_project_root_is_absolute(self):
        from rag_system import config

        assert config.PROJECT_ROOT.is_absolute()

    def test_legal_framework_dir(self):
        from rag_system import config

        assert config.LEGAL_FRAMEWORK_DIR == config.PROJECT_ROOT / "legal_framework"

    def test_precedents_dir(self):
        from rag_system import config

        assert config.PRECEDENTS_DIR == config.PROJECT_ROOT / "precedents"

    def test_precedents_criminal_dir(self):
        from rag_system import config

        assert config.PRECEDENTS_CRIMINAL_DIR == config.PRECEDENTS_DIR / "criminal"

    def test_precedents_civil_dir(self):
        from rag_system import config

        assert config.PRECEDENTS_CIVIL_DIR == config.PRECEDENTS_DIR / "civil"

    def test_precedents_constitutional_dir(self):
        from rag_system import config

        expected = config.PRECEDENTS_DIR / "constitutional"
        assert config.PRECEDENTS_CONSTITUTIONAL_DIR == expected

    def test_precedents_metadata_path(self):
        from rag_system import config

        assert config.PRECEDENTS_METADATA_PATH == config.PRECEDENTS_DIR / "metadata.json"


class TestChromaDBPath:
    """Test CHROMA_DB_PATH default and environment override."""

    def test_default_chroma_db_path(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHROMA_DB_PATH", None)
            cfg = _reload_config()
            expected = str(cfg.PROJECT_ROOT / "chroma_db")
            assert cfg.CHROMA_DB_PATH == expected

    def test_chroma_db_path_env_override(self):
        custom_path = "/tmp/custom_chroma"
        with mock.patch.dict(os.environ, {"CHROMA_DB_PATH": custom_path}):
            cfg = _reload_config()
            assert cfg.CHROMA_DB_PATH == custom_path

    def test_chroma_db_path_is_string(self):
        from rag_system import config

        assert isinstance(config.CHROMA_DB_PATH, str)


class TestOllamaSettings:
    """Test Ollama / LLM configuration values."""

    def test_default_ollama_base_url(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OLLAMA_HOST", None)
            cfg = _reload_config()
            assert cfg.OLLAMA_BASE_URL == "http://localhost:11434"

    def test_ollama_base_url_env_override(self):
        custom_url = "http://remote-host:11434"
        with mock.patch.dict(os.environ, {"OLLAMA_HOST": custom_url}):
            cfg = _reload_config()
            assert cfg.OLLAMA_BASE_URL == custom_url

    def test_llm_model_name(self):
        from rag_system import config

        assert config.LLM_MODEL_NAME == "schroneko/llama-3.1-swallow-8b-instruct-v0.1"

    def test_llm_fallback_model(self):
        from rag_system import config

        assert config.LLM_FALLBACK_MODEL == "llama3.1:8b"

    def test_llm_temperature(self):
        from rag_system import config

        assert config.LLM_TEMPERATURE == 0.1
        assert isinstance(config.LLM_TEMPERATURE, float)

    def test_llm_num_ctx(self):
        from rag_system import config

        assert config.LLM_NUM_CTX == 4096
        assert isinstance(config.LLM_NUM_CTX, int)


class TestEmbeddingsModel:
    """Test embedding model configuration."""

    def test_embeddings_model_name(self):
        from rag_system import config

        expected = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        assert config.EMBEDDINGS_MODEL_NAME == expected


class TestDocumentChunking:
    """Test document chunking parameters."""

    def test_chunk_size(self):
        from rag_system import config

        assert config.CHUNK_SIZE == 1000
        assert isinstance(config.CHUNK_SIZE, int)

    def test_chunk_overlap(self):
        from rag_system import config

        assert config.CHUNK_OVERLAP == 200
        assert isinstance(config.CHUNK_OVERLAP, int)

    def test_chunk_overlap_less_than_size(self):
        from rag_system import config

        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

    def test_chunk_separators(self):
        from rag_system import config

        assert isinstance(config.CHUNK_SEPARATORS, list)
        assert len(config.CHUNK_SEPARATORS) > 0
        assert config.CHUNK_SEPARATORS == ["\n## ", "\n### ", "\n\n", "\n", " "]


class TestRetrieval:
    """Test retrieval configuration."""

    def test_retrieval_k(self):
        from rag_system import config

        assert config.RETRIEVAL_K == 5
        assert isinstance(config.RETRIEVAL_K, int)
        assert config.RETRIEVAL_K > 0


class TestChromaCollection:
    """Test ChromaDB collection configuration."""

    def test_chroma_collection_name(self):
        from rag_system import config

        assert config.CHROMA_COLLECTION_NAME == "nazono_kokka_legal"
        assert isinstance(config.CHROMA_COLLECTION_NAME, str)
