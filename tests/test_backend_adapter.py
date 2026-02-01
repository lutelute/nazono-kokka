"""Tests for rag_system.backend_adapter module.

Verifies BackendConfig dataclass creation, default values, validation,
and the create_backend() factory function using mocked dependencies.
Also confirms the langchain_ollama.OllamaLLM import path.
"""

from unittest import mock

import pytest
from langchain_classic.chains import RetrievalQA

from rag_system.backend_adapter import BackendConfig, create_backend
from rag_system.config import (
    LLM_MODEL_NAME,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    RETRIEVAL_K,
)


# ---------------------------------------------------------------------------
# Tests: BackendConfig dataclass
# ---------------------------------------------------------------------------


class TestBackendConfig:
    """Test the BackendConfig dataclass creation and defaults."""

    def test_required_fields(self):
        config = BackendConfig(name="test", model_name="llama3.1:8b")
        assert config.name == "test"
        assert config.model_name == "llama3.1:8b"

    def test_default_temperature(self):
        config = BackendConfig(name="test", model_name="llama3.1:8b")
        assert config.temperature == LLM_TEMPERATURE

    def test_default_num_ctx(self):
        config = BackendConfig(name="test", model_name="llama3.1:8b")
        assert config.num_ctx == LLM_NUM_CTX

    def test_default_retrieval_k(self):
        config = BackendConfig(name="test", model_name="llama3.1:8b")
        assert config.retrieval_k == RETRIEVAL_K

    def test_default_filter_fields_are_none(self):
        config = BackendConfig(name="test", model_name="llama3.1:8b")
        assert config.document_type is None
        assert config.case_type is None
        assert config.verdict is None

    def test_custom_temperature(self):
        config = BackendConfig(name="hot", model_name="llama3.1:8b", temperature=0.9)
        assert config.temperature == 0.9

    def test_custom_num_ctx(self):
        config = BackendConfig(name="wide", model_name="llama3.1:8b", num_ctx=8192)
        assert config.num_ctx == 8192

    def test_custom_retrieval_k(self):
        config = BackendConfig(name="many", model_name="llama3.1:8b", retrieval_k=10)
        assert config.retrieval_k == 10

    def test_filter_fields(self):
        config = BackendConfig(
            name="filtered",
            model_name="llama3.1:8b",
            document_type="precedent",
            case_type="criminal",
            verdict="有罪",
        )
        assert config.document_type == "precedent"
        assert config.case_type == "criminal"
        assert config.verdict == "有罪"

    def test_all_fields_set(self):
        config = BackendConfig(
            name="full",
            model_name="custom-model",
            temperature=0.5,
            num_ctx=2048,
            retrieval_k=3,
            document_type="legal_framework",
            case_type="civil",
            verdict="原告勝訴",
        )
        assert config.name == "full"
        assert config.model_name == "custom-model"
        assert config.temperature == 0.5
        assert config.num_ctx == 2048
        assert config.retrieval_k == 3
        assert config.document_type == "legal_framework"
        assert config.case_type == "civil"
        assert config.verdict == "原告勝訴"

    def test_is_dataclass_instance(self):
        import dataclasses

        config = BackendConfig(name="test", model_name="llama3.1:8b")
        assert dataclasses.is_dataclass(config)


# ---------------------------------------------------------------------------
# Tests: create_backend
# ---------------------------------------------------------------------------


class TestCreateBackend:
    """Test the create_backend factory function with mocked dependencies."""

    def test_calls_create_llm_with_config_params(self):
        config = BackendConfig(
            name="test",
            model_name="llama3.1:8b",
            temperature=0.5,
            num_ctx=2048,
        )

        with (
            mock.patch("rag_system.judge.create_llm") as mock_create_llm,
            mock.patch("rag_system.retriever.create_retriever") as mock_create_retriever,
            mock.patch("rag_system.judge.create_judicial_chain") as mock_create_chain,
        ):
            mock_llm = mock.MagicMock()
            mock_create_llm.return_value = mock_llm
            mock_retriever = mock.MagicMock()
            mock_create_retriever.return_value = mock_retriever
            mock_chain = mock.MagicMock(spec=RetrievalQA)
            mock_create_chain.return_value = mock_chain

            create_backend(config)

            mock_create_llm.assert_called_once_with(
                model_name="llama3.1:8b",
                temperature=0.5,
                num_ctx=2048,
            )

    def test_calls_create_retriever_with_config_params(self):
        config = BackendConfig(
            name="test",
            model_name="llama3.1:8b",
            retrieval_k=10,
            document_type="precedent",
            case_type="criminal",
            verdict="有罪",
        )

        with (
            mock.patch("rag_system.judge.create_llm") as mock_create_llm,
            mock.patch("rag_system.retriever.create_retriever") as mock_create_retriever,
            mock.patch("rag_system.judge.create_judicial_chain") as mock_create_chain,
        ):
            mock_create_llm.return_value = mock.MagicMock()
            mock_create_retriever.return_value = mock.MagicMock()
            mock_create_chain.return_value = mock.MagicMock(spec=RetrievalQA)

            create_backend(config)

            mock_create_retriever.assert_called_once_with(
                k=10,
                document_type="precedent",
                case_type="criminal",
                verdict="有罪",
            )

    def test_calls_create_judicial_chain_with_llm_and_retriever(self):
        config = BackendConfig(name="test", model_name="llama3.1:8b")

        with (
            mock.patch("rag_system.judge.create_llm") as mock_create_llm,
            mock.patch("rag_system.retriever.create_retriever") as mock_create_retriever,
            mock.patch("rag_system.judge.create_judicial_chain") as mock_create_chain,
        ):
            mock_llm = mock.MagicMock()
            mock_create_llm.return_value = mock_llm
            mock_retriever = mock.MagicMock()
            mock_create_retriever.return_value = mock_retriever
            mock_chain = mock.MagicMock(spec=RetrievalQA)
            mock_create_chain.return_value = mock_chain

            create_backend(config)

            mock_create_chain.assert_called_once_with(
                llm=mock_llm, retriever=mock_retriever
            )

    def test_returns_chain(self):
        config = BackendConfig(name="test", model_name="llama3.1:8b")

        with (
            mock.patch("rag_system.judge.create_llm"),
            mock.patch("rag_system.retriever.create_retriever"),
            mock.patch("rag_system.judge.create_judicial_chain") as mock_create_chain,
        ):
            mock_chain = mock.MagicMock(spec=RetrievalQA)
            mock_create_chain.return_value = mock_chain

            result = create_backend(config)

            assert result is mock_chain

    def test_default_config_uses_config_defaults(self):
        config = BackendConfig(name="default", model_name=LLM_MODEL_NAME)

        with (
            mock.patch("rag_system.judge.create_llm") as mock_create_llm,
            mock.patch("rag_system.retriever.create_retriever") as mock_create_retriever,
            mock.patch("rag_system.judge.create_judicial_chain") as mock_create_chain,
        ):
            mock_create_llm.return_value = mock.MagicMock()
            mock_create_retriever.return_value = mock.MagicMock()
            mock_create_chain.return_value = mock.MagicMock(spec=RetrievalQA)

            create_backend(config)

            mock_create_llm.assert_called_once_with(
                model_name=LLM_MODEL_NAME,
                temperature=LLM_TEMPERATURE,
                num_ctx=LLM_NUM_CTX,
            )
            mock_create_retriever.assert_called_once_with(
                k=RETRIEVAL_K,
                document_type=None,
                case_type=None,
                verdict=None,
            )

    def test_no_filter_params_passed_as_none(self):
        config = BackendConfig(name="no-filter", model_name="llama3.1:8b")

        with (
            mock.patch("rag_system.judge.create_llm"),
            mock.patch("rag_system.retriever.create_retriever") as mock_create_retriever,
            mock.patch("rag_system.judge.create_judicial_chain"),
        ):
            mock_create_retriever.return_value = mock.MagicMock()

            create_backend(config)

            call_kwargs = mock_create_retriever.call_args[1]
            assert call_kwargs["document_type"] is None
            assert call_kwargs["case_type"] is None
            assert call_kwargs["verdict"] is None


# ---------------------------------------------------------------------------
# Tests: langchain_ollama import
# ---------------------------------------------------------------------------


class TestOllamaImport:
    """Verify that langchain_ollama.OllamaLLM is importable."""

    def test_ollama_llm_import(self):
        from langchain_ollama import OllamaLLM

        assert OllamaLLM is not None

    def test_judge_module_uses_ollama_llm(self):
        from rag_system.judge import create_llm

        assert create_llm is not None
        # Verify the return type annotation references OllamaLLM
        # With `from __future__ import annotations`, annotations are strings
        annotations = create_llm.__annotations__
        assert "OllamaLLM" in str(annotations.get("return", ""))
