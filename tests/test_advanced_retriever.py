"""Tests for rag_system.advanced_retriever (legal-doc quota logic)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from rag_system.advanced_retriever import AdvancedRetriever, is_legal_framework


def _legal(name: str = "criminal_code") -> Document:
    return Document(
        page_content=f"法令本文 {name}",
        metadata={"source": f"legal_framework/{name}.md"},
    )


def _legal_typed(name: str = "civil_code") -> Document:
    return Document(
        page_content=f"法令本文 {name}",
        metadata={"document_type": "legal_framework", "source": f"{name}.md"},
    )


def _precedent(case_id: str) -> Document:
    return Document(
        page_content=f"判例本文 {case_id}",
        metadata={"source": f"precedents/criminal/{case_id}.json", "case_id": case_id},
    )


def _retriever(min_legal: int = 1) -> AdvancedRetriever:
    return AdvancedRetriever(vectorstore=object(), min_legal_docs=min_legal)


# ---------------------------------------------------------------------------
# is_legal_framework
# ---------------------------------------------------------------------------


class TestIsLegalFramework:
    def test_source_path(self):
        assert is_legal_framework(_legal())

    def test_document_type_metadata(self):
        assert is_legal_framework(_legal_typed())

    def test_precedent_is_not_legal(self):
        assert not is_legal_framework(_precedent("CRIM-2020-0001"))

    def test_empty_metadata(self):
        assert not is_legal_framework(Document(page_content="x", metadata={}))


# ---------------------------------------------------------------------------
# _apply_legal_quota
# ---------------------------------------------------------------------------


class TestLegalQuota:
    def test_swaps_in_legal_doc_when_missing(self):
        docs = [_precedent(f"CRIM-2020-{i:04d}") for i in range(5)]
        legal = _legal()
        staged = SimpleNamespace(dense=[(legal, 0.9)])

        result = _retriever()._apply_legal_quota(docs, staged)

        assert len(result) == 5
        assert any(is_legal_framework(d) for d in result)
        # 先頭の判例（最強候補）は保持され、末尾が入れ替わる
        assert result[0].metadata["case_id"] == "CRIM-2020-0000"
        assert is_legal_framework(result[-1])

    def test_no_change_when_quota_met(self):
        docs = [_legal(), *(_precedent(f"CRIM-2021-{i:04d}") for i in range(4))]
        staged = SimpleNamespace(dense=[(_legal("other"), 0.5)])

        result = _retriever()._apply_legal_quota(list(docs), staged)

        assert [d.page_content for d in result] == [d.page_content for d in docs]

    def test_gives_up_when_dense_has_no_legal(self):
        docs = [_precedent(f"CRIM-2022-{i:04d}") for i in range(3)]
        staged = SimpleNamespace(dense=[(_precedent("CRIM-2099-9999"), 0.4)])

        result = _retriever()._apply_legal_quota(list(docs), staged)

        assert [d.page_content for d in result] == [d.page_content for d in docs]

    def test_skips_duplicate_content(self):
        legal = _legal()
        docs = [legal, *(_precedent(f"CRIM-2023-{i:04d}") for i in range(3))]
        # dense に同一内容の法令しかない状態で min_legal=2
        staged = SimpleNamespace(dense=[(_legal(), 0.8)])

        result = _retriever(min_legal=2)._apply_legal_quota(list(docs), staged)

        # 重複は追加されず、法令は1件のまま
        assert sum(1 for d in result if is_legal_framework(d)) == 1

    def test_fills_two_when_min_two(self):
        docs = [_precedent(f"CIVIL-2020-{i:04d}") for i in range(5)]
        staged = SimpleNamespace(
            dense=[(_legal("criminal_code"), 0.9), (_legal_typed("civil_code"), 0.8)]
        )

        result = _retriever(min_legal=2)._apply_legal_quota(docs, staged)

        assert sum(1 for d in result if is_legal_framework(d)) == 2
        # 残った判例は先頭側
        assert result[0].metadata.get("case_id") == "CIVIL-2020-0000"

    def test_quota_disabled(self):
        docs = [_precedent("CRIM-2024-0001")]
        # min_legal_docs=0 では _get_relevant_documents がクォータを呼ばない
        r = _retriever(min_legal=0)
        assert r.min_legal_docs == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
