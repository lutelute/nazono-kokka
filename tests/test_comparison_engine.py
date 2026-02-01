"""Tests for rag_system.test_cases and rag_system.comparison_engine modules.

Verifies TestCaseManager CRUD operations (load, save, add, update, delete),
default cases loading, and ComparisonEngine with mocked backend execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from rag_system.backend_adapter import BackendConfig
from rag_system.comparison_engine import (
    ComparisonEngine,
    _count_case_id_hits,
    _count_keyword_hits,
    _count_statute_hits,
)
from rag_system.metrics import EvaluationResult
from rag_system.test_cases import TestCase, TestCaseManager, _generate_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_test_case() -> TestCase:
    """Return a sample TestCase instance."""
    return TestCase(
        id="TC-TEST001",
        category="criminal",
        query="窃盗罪の量刑基準は何ですか",
        expected_keywords=["窃盗", "量刑", "懲役"],
        expected_statutes=["刑法第235条"],
        expected_case_ids=["CRIMINAL-2019-0012"],
        difficulty="basic",
        description="窃盗罪の基本的な量刑基準テスト",
    )


@pytest.fixture()
def sample_test_case_b() -> TestCase:
    """Return a second sample TestCase for multi-case tests."""
    return TestCase(
        id="TC-TEST002",
        category="civil",
        query="損害賠償請求の要件は何ですか",
        expected_keywords=["損害賠償", "不法行為"],
        expected_statutes=["民法第709条"],
        expected_case_ids=[],
        difficulty="intermediate",
        description="民事損害賠償の要件テスト",
    )


@pytest.fixture()
def cases_json_path(tmp_path: Path, sample_test_case: TestCase) -> Path:
    """Create a temporary JSON file with one test case."""
    path = tmp_path / "cases.json"
    payload = {"test_cases": [sample_test_case.to_dict()]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@pytest.fixture()
def manager(tmp_path: Path) -> TestCaseManager:
    """Return a TestCaseManager using a temporary storage directory."""
    return TestCaseManager(storage_dir=tmp_path)


@pytest.fixture()
def sample_config() -> BackendConfig:
    """Return a sample BackendConfig."""
    return BackendConfig(name="test-backend", model_name="llama3.1:8b")


@pytest.fixture()
def sample_config_b() -> BackendConfig:
    """Return a second BackendConfig for multi-config tests."""
    return BackendConfig(name="test-creative", model_name="llama3.1:8b", temperature=0.7)


@pytest.fixture()
def mock_chain():
    """Return a mock chain that simulates a successful LLM invocation."""
    chain = mock.MagicMock()
    chain.invoke.return_value = {
        "result": "窃盗罪は刑法第235条に規定されています。量刑は懲役10年以下です。",
        "source_documents": [mock.MagicMock(), mock.MagicMock()],
    }
    return chain


# ---------------------------------------------------------------------------
# Tests: TestCase dataclass
# ---------------------------------------------------------------------------


class TestTestCaseDataclass:
    """Test the TestCase dataclass serialisation."""

    def test_to_dict(self, sample_test_case: TestCase):
        d = sample_test_case.to_dict()
        assert d["id"] == "TC-TEST001"
        assert d["category"] == "criminal"
        assert d["query"] == "窃盗罪の量刑基準は何ですか"
        assert d["expected_keywords"] == ["窃盗", "量刑", "懲役"]
        assert d["difficulty"] == "basic"

    def test_from_dict_full(self, sample_test_case: TestCase):
        d = sample_test_case.to_dict()
        restored = TestCase.from_dict(d)
        assert restored.id == sample_test_case.id
        assert restored.category == sample_test_case.category
        assert restored.query == sample_test_case.query
        assert restored.expected_keywords == sample_test_case.expected_keywords
        assert restored.expected_statutes == sample_test_case.expected_statutes
        assert restored.expected_case_ids == sample_test_case.expected_case_ids
        assert restored.difficulty == sample_test_case.difficulty
        assert restored.description == sample_test_case.description

    def test_from_dict_minimal(self):
        tc = TestCase.from_dict({"id": "TC-MIN", "category": "civil", "query": "テスト"})
        assert tc.id == "TC-MIN"
        assert tc.category == "civil"
        assert tc.query == "テスト"
        assert tc.expected_keywords == []
        assert tc.difficulty == "basic"

    def test_from_dict_generates_id_when_missing(self):
        tc = TestCase.from_dict({"category": "criminal", "query": "テスト"})
        assert tc.id.startswith("TC-")

    def test_roundtrip(self, sample_test_case: TestCase):
        restored = TestCase.from_dict(sample_test_case.to_dict())
        assert restored.to_dict() == sample_test_case.to_dict()


# ---------------------------------------------------------------------------
# Tests: _generate_id
# ---------------------------------------------------------------------------


class TestGenerateId:
    """Test the ID generation helper."""

    def test_format(self):
        generated = _generate_id()
        assert generated.startswith("TC-")
        assert len(generated) == 11  # "TC-" + 8 hex chars

    def test_uniqueness(self):
        ids = {_generate_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# Tests: TestCaseManager — Load / Save
# ---------------------------------------------------------------------------


class TestTestCaseManagerLoad:
    """Test TestCaseManager load operations."""

    def test_load_from_file(self, manager: TestCaseManager, cases_json_path: Path):
        cases = manager.load_from_file(cases_json_path)
        assert len(cases) == 1
        assert cases[0].id == "TC-TEST001"

    def test_load_from_file_populates_internal_dict(
        self, manager: TestCaseManager, cases_json_path: Path
    ):
        manager.load_from_file(cases_json_path)
        assert manager.get("TC-TEST001") is not None

    def test_load_from_nonexistent_file(self, manager: TestCaseManager, tmp_path: Path):
        cases = manager.load_from_file(tmp_path / "nonexistent.json")
        assert cases == []

    def test_load_from_invalid_json(self, manager: TestCaseManager, tmp_path: Path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json", encoding="utf-8")
        cases = manager.load_from_file(bad_file)
        assert cases == []

    def test_load_default_cases(self, manager: TestCaseManager, cases_json_path: Path):
        with mock.patch("rag_system.test_cases.DEFAULT_CASES_PATH", cases_json_path):
            cases = manager.load_default_cases()
        assert len(cases) == 1


class TestTestCaseManagerSave:
    """Test TestCaseManager save operations."""

    def test_save_to_file(
        self, manager: TestCaseManager, sample_test_case: TestCase, tmp_path: Path
    ):
        manager.add(sample_test_case)
        out_path = tmp_path / "output.json"
        manager.save_to_file(out_path)

        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(data["test_cases"]) == 1
        assert data["test_cases"][0]["id"] == "TC-TEST001"

    def test_save_creates_parent_dirs(
        self, manager: TestCaseManager, sample_test_case: TestCase, tmp_path: Path
    ):
        manager.add(sample_test_case)
        nested = tmp_path / "sub" / "dir" / "cases.json"
        manager.save_to_file(nested)
        assert nested.exists()

    def test_save_roundtrip(
        self,
        manager: TestCaseManager,
        sample_test_case: TestCase,
        sample_test_case_b: TestCase,
        tmp_path: Path,
    ):
        manager.add(sample_test_case)
        manager.add(sample_test_case_b)
        out_path = tmp_path / "roundtrip.json"
        manager.save_to_file(out_path)

        new_manager = TestCaseManager(storage_dir=tmp_path)
        loaded = new_manager.load_from_file(out_path)
        assert len(loaded) == 2
        ids = {tc.id for tc in loaded}
        assert "TC-TEST001" in ids
        assert "TC-TEST002" in ids


# ---------------------------------------------------------------------------
# Tests: TestCaseManager — CRUD
# ---------------------------------------------------------------------------


class TestTestCaseManagerCRUD:
    """Test TestCaseManager CRUD operations."""

    def test_add_and_get(self, manager: TestCaseManager, sample_test_case: TestCase):
        manager.add(sample_test_case)
        retrieved = manager.get("TC-TEST001")
        assert retrieved is not None
        assert retrieved.query == sample_test_case.query

    def test_add_overwrites_existing(self, manager: TestCaseManager):
        tc1 = TestCase(id="TC-DUP", category="criminal", query="original")
        tc2 = TestCase(id="TC-DUP", category="civil", query="overwritten")
        manager.add(tc1)
        manager.add(tc2)
        result = manager.get("TC-DUP")
        assert result is not None
        assert result.query == "overwritten"
        assert result.category == "civil"

    def test_list_cases(
        self,
        manager: TestCaseManager,
        sample_test_case: TestCase,
        sample_test_case_b: TestCase,
    ):
        manager.add(sample_test_case)
        manager.add(sample_test_case_b)
        cases = manager.list_cases()
        assert len(cases) == 2

    def test_list_cases_empty(self, manager: TestCaseManager):
        assert manager.list_cases() == []

    def test_get_nonexistent(self, manager: TestCaseManager):
        assert manager.get("TC-NOPE") is None

    def test_update_existing(self, manager: TestCaseManager, sample_test_case: TestCase):
        manager.add(sample_test_case)
        updated = manager.update("TC-TEST001", query="更新されたクエリ", difficulty="advanced")
        assert updated is not None
        assert updated.query == "更新されたクエリ"
        assert updated.difficulty == "advanced"
        # Original fields unchanged
        assert updated.category == "criminal"

    def test_update_ignores_id_field(self, manager: TestCaseManager, sample_test_case: TestCase):
        manager.add(sample_test_case)
        updated = manager.update("TC-TEST001", id="TC-HACKED")
        assert updated is not None
        assert updated.id == "TC-TEST001"

    def test_update_nonexistent(self, manager: TestCaseManager):
        result = manager.update("TC-NOPE", query="test")
        assert result is None

    def test_delete_existing(self, manager: TestCaseManager, sample_test_case: TestCase):
        manager.add(sample_test_case)
        result = manager.delete("TC-TEST001")
        assert result is True
        assert manager.get("TC-TEST001") is None

    def test_delete_nonexistent(self, manager: TestCaseManager):
        result = manager.delete("TC-NOPE")
        assert result is False

    def test_delete_reduces_count(
        self,
        manager: TestCaseManager,
        sample_test_case: TestCase,
        sample_test_case_b: TestCase,
    ):
        manager.add(sample_test_case)
        manager.add(sample_test_case_b)
        manager.delete("TC-TEST001")
        assert len(manager.list_cases()) == 1


# ---------------------------------------------------------------------------
# Tests: TestCaseManager — Import / Export
# ---------------------------------------------------------------------------


class TestTestCaseManagerImportExport:
    """Test TestCaseManager import and export operations."""

    def test_import_cases(self, manager: TestCaseManager, cases_json_path: Path):
        imported = manager.import_cases(cases_json_path)
        assert len(imported) == 1
        assert manager.get("TC-TEST001") is not None

    def test_export_cases(
        self, manager: TestCaseManager, sample_test_case: TestCase, tmp_path: Path
    ):
        manager.add(sample_test_case)
        export_path = tmp_path / "export.json"
        manager.export_cases(export_path)

        assert export_path.exists()
        data = json.loads(export_path.read_text(encoding="utf-8"))
        assert len(data["test_cases"]) == 1


# ---------------------------------------------------------------------------
# Tests: Scoring Helpers
# ---------------------------------------------------------------------------


class TestCountKeywordHits:
    """Test the _count_keyword_hits helper function."""

    def test_all_keywords_found(self):
        text = "窃盗罪の量刑は懲役10年以下です"
        assert _count_keyword_hits(text, ["窃盗", "量刑", "懲役"]) == 3

    def test_partial_hits(self):
        text = "窃盗罪について説明します"
        assert _count_keyword_hits(text, ["窃盗", "量刑", "懲役"]) == 1

    def test_no_hits(self):
        text = "関係のないテキスト"
        assert _count_keyword_hits(text, ["窃盗", "量刑"]) == 0

    def test_empty_keywords(self):
        assert _count_keyword_hits("any text", []) == 0

    def test_empty_text(self):
        assert _count_keyword_hits("", ["窃盗"]) == 0

    def test_case_insensitive(self):
        text = "The KEYWORD was found"
        assert _count_keyword_hits(text, ["keyword"]) == 1


class TestCountStatuteHits:
    """Test the _count_statute_hits helper function."""

    def test_all_statutes_found(self):
        text = "刑法第235条および民法第709条が適用される"
        assert _count_statute_hits(text, ["刑法第235条", "民法第709条"]) == 2

    def test_no_hits(self):
        text = "判例について"
        assert _count_statute_hits(text, ["刑法第235条"]) == 0

    def test_empty_list(self):
        assert _count_statute_hits("any text", []) == 0

    def test_exact_match(self):
        """Statute match is exact (not case-insensitive)."""
        text = "刑法第235条"
        assert _count_statute_hits(text, ["刑法第235条"]) == 1


class TestCountCaseIdHits:
    """Test the _count_case_id_hits helper function."""

    def test_hit(self):
        text = "判例CRIMINAL-2019-0012によると"
        assert _count_case_id_hits(text, ["CRIMINAL-2019-0012"]) == 1

    def test_no_hit(self):
        text = "関係のないテキスト"
        assert _count_case_id_hits(text, ["CRIMINAL-2019-0012"]) == 0

    def test_empty_list(self):
        assert _count_case_id_hits("text", []) == 0


# ---------------------------------------------------------------------------
# Tests: ComparisonEngine — initialisation
# ---------------------------------------------------------------------------


class TestComparisonEngineInit:
    """Test ComparisonEngine initialisation."""

    def test_default_results_dir(self):
        engine = ComparisonEngine()
        assert engine._results_dir.name == "results"

    def test_custom_results_dir(self, tmp_path: Path):
        engine = ComparisonEngine(results_dir=tmp_path / "custom")
        assert engine._results_dir == tmp_path / "custom"


# ---------------------------------------------------------------------------
# Tests: ComparisonEngine — run_comparison
# ---------------------------------------------------------------------------


class TestComparisonEngineRun:
    """Test ComparisonEngine.run_comparison with mocked backends."""

    def test_empty_configs_returns_empty(self, sample_test_case: TestCase):
        engine = ComparisonEngine()
        results = engine.run_comparison([], [sample_test_case])
        assert results == []

    def test_empty_test_cases_returns_empty(self, sample_config: BackendConfig):
        engine = ComparisonEngine()
        results = engine.run_comparison([sample_config], [])
        assert results == []

    def test_successful_execution(
        self,
        sample_config: BackendConfig,
        sample_test_case: TestCase,
        mock_chain,
        tmp_path: Path,
    ):
        engine = ComparisonEngine(results_dir=tmp_path / "results")

        with mock.patch.object(engine, "_create_chain", return_value=mock_chain):
            results = engine.run_comparison(
                [sample_config], [sample_test_case], save_results=False
            )

        assert len(results) == 1
        result = results[0]
        assert result.config_name == "test-backend"
        assert result.test_case_id == "TC-TEST001"
        assert "窃盗罪" in result.response_text
        assert result.source_count == 2
        assert result.elapsed_time > 0.0

    def test_keyword_scoring(
        self,
        sample_config: BackendConfig,
        sample_test_case: TestCase,
        mock_chain,
        tmp_path: Path,
    ):
        engine = ComparisonEngine(results_dir=tmp_path / "results")

        with mock.patch.object(engine, "_create_chain", return_value=mock_chain):
            results = engine.run_comparison(
                [sample_config], [sample_test_case], save_results=False
            )

        result = results[0]
        # Mock response contains "窃盗" and "量刑" but not directly "懲役" as keyword;
        # the response text is: 窃盗罪は刑法第235条に規定されています。量刑は懲役10年以下です。
        assert result.keyword_total == 3
        assert result.keyword_hits >= 2  # at least 窃盗 and 量刑

    def test_statute_scoring(
        self,
        sample_config: BackendConfig,
        sample_test_case: TestCase,
        mock_chain,
        tmp_path: Path,
    ):
        engine = ComparisonEngine(results_dir=tmp_path / "results")

        with mock.patch.object(engine, "_create_chain", return_value=mock_chain):
            results = engine.run_comparison(
                [sample_config], [sample_test_case], save_results=False
            )

        result = results[0]
        assert result.statute_total == 1
        assert result.statute_hits == 1  # "刑法第235条" is in the mock response

    def test_multiple_configs_and_cases(
        self,
        sample_config: BackendConfig,
        sample_config_b: BackendConfig,
        sample_test_case: TestCase,
        sample_test_case_b: TestCase,
        mock_chain,
        tmp_path: Path,
    ):
        engine = ComparisonEngine(results_dir=tmp_path / "results")

        with mock.patch.object(engine, "_create_chain", return_value=mock_chain):
            results = engine.run_comparison(
                [sample_config, sample_config_b],
                [sample_test_case, sample_test_case_b],
                save_results=False,
            )

        # 2 configs × 2 cases = 4 results
        assert len(results) == 4
        config_names = {r.config_name for r in results}
        assert config_names == {"test-backend", "test-creative"}

    def test_chain_creation_failure_produces_error_results(
        self,
        sample_config: BackendConfig,
        sample_test_case: TestCase,
        sample_test_case_b: TestCase,
        tmp_path: Path,
    ):
        engine = ComparisonEngine(results_dir=tmp_path / "results")

        with mock.patch.object(engine, "_create_chain", return_value=None):
            results = engine.run_comparison(
                [sample_config],
                [sample_test_case, sample_test_case_b],
                save_results=False,
            )

        assert len(results) == 2
        for r in results:
            assert "エラー" in r.response_text
            assert r.keyword_hits == 0
            assert r.statute_hits == 0

    def test_progress_callback_called(
        self,
        sample_config: BackendConfig,
        sample_test_case: TestCase,
        mock_chain,
        tmp_path: Path,
    ):
        engine = ComparisonEngine(results_dir=tmp_path / "results")
        callback = mock.MagicMock()

        with mock.patch.object(engine, "_create_chain", return_value=mock_chain):
            engine.run_comparison(
                [sample_config],
                [sample_test_case],
                progress_callback=callback,
                save_results=False,
            )

        callback.assert_called_once_with(1, 1, "実行中: test-backend / TC-TEST001")

    def test_progress_callback_on_chain_failure(
        self,
        sample_config: BackendConfig,
        sample_test_case: TestCase,
        tmp_path: Path,
    ):
        engine = ComparisonEngine(results_dir=tmp_path / "results")
        callback = mock.MagicMock()

        with mock.patch.object(engine, "_create_chain", return_value=None):
            engine.run_comparison(
                [sample_config],
                [sample_test_case],
                progress_callback=callback,
                save_results=False,
            )

        callback.assert_called_once_with(1, 1, "エラー: test-backend / TC-TEST001")


# ---------------------------------------------------------------------------
# Tests: ComparisonEngine — _execute_single error handling
# ---------------------------------------------------------------------------


class TestExecuteSingleErrors:
    """Test error handling in _execute_single."""

    def test_connection_error(
        self, sample_config: BackendConfig, sample_test_case: TestCase, tmp_path: Path
    ):
        engine = ComparisonEngine(results_dir=tmp_path / "results")
        chain = mock.MagicMock()
        chain.invoke.side_effect = ConnectionError("connection lost")

        result = engine._execute_single(chain, sample_config, sample_test_case)
        assert "エラー" in result.response_text
        assert result.keyword_hits == 0

    def test_unexpected_error(
        self, sample_config: BackendConfig, sample_test_case: TestCase, tmp_path: Path
    ):
        engine = ComparisonEngine(results_dir=tmp_path / "results")
        chain = mock.MagicMock()
        chain.invoke.side_effect = RuntimeError("unexpected")

        result = engine._execute_single(chain, sample_config, sample_test_case)
        assert "エラー" in result.response_text


# ---------------------------------------------------------------------------
# Tests: ComparisonEngine — _create_chain
# ---------------------------------------------------------------------------


class TestCreateChain:
    """Test _create_chain error handling."""

    def test_connection_error_returns_none(self, sample_config: BackendConfig):
        engine = ComparisonEngine()
        with mock.patch(
            "rag_system.comparison_engine.create_backend",
            side_effect=ConnectionError("no server"),
        ):
            assert engine._create_chain(sample_config) is None

    def test_file_not_found_returns_none(self, sample_config: BackendConfig):
        engine = ComparisonEngine()
        with mock.patch(
            "rag_system.comparison_engine.create_backend",
            side_effect=FileNotFoundError("no db"),
        ):
            assert engine._create_chain(sample_config) is None

    def test_generic_exception_returns_none(self, sample_config: BackendConfig):
        engine = ComparisonEngine()
        with mock.patch(
            "rag_system.comparison_engine.create_backend",
            side_effect=RuntimeError("boom"),
        ):
            assert engine._create_chain(sample_config) is None

    def test_success_returns_chain(self, sample_config: BackendConfig):
        engine = ComparisonEngine()
        mock_chain = mock.MagicMock()
        with mock.patch(
            "rag_system.comparison_engine.create_backend",
            return_value=mock_chain,
        ):
            result = engine._create_chain(sample_config)
        assert result is mock_chain


# ---------------------------------------------------------------------------
# Tests: ComparisonEngine — save / load results
# ---------------------------------------------------------------------------


class TestResultsPersistence:
    """Test result saving and loading."""

    def test_save_and_load_results(
        self,
        sample_config: BackendConfig,
        sample_test_case: TestCase,
        mock_chain,
        tmp_path: Path,
    ):
        results_dir = tmp_path / "results"
        engine = ComparisonEngine(results_dir=results_dir)

        with mock.patch.object(engine, "_create_chain", return_value=mock_chain):
            engine.run_comparison(
                [sample_config], [sample_test_case], save_results=True
            )

        files = engine.list_result_files()
        assert len(files) == 1

        loaded = engine.load_results(files[0])
        assert len(loaded) == 1
        assert loaded[0]["config_name"] == "test-backend"
        assert loaded[0]["test_case_id"] == "TC-TEST001"

    def test_list_result_files_empty_dir(self, tmp_path: Path):
        engine = ComparisonEngine(results_dir=tmp_path / "empty")
        assert engine.list_result_files() == []

    def test_load_results_nonexistent(self, tmp_path: Path):
        engine = ComparisonEngine(results_dir=tmp_path)
        assert engine.load_results(tmp_path / "nope.json") == []

    def test_load_results_invalid_json(self, tmp_path: Path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{bad json", encoding="utf-8")
        engine = ComparisonEngine(results_dir=tmp_path)
        assert engine.load_results(bad_file) == []

    def test_list_result_files_sorted_reverse(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "comparison_20240101_000000.json").write_text("{}", encoding="utf-8")
        (results_dir / "comparison_20240102_000000.json").write_text("{}", encoding="utf-8")
        (results_dir / "comparison_20240103_000000.json").write_text("{}", encoding="utf-8")

        engine = ComparisonEngine(results_dir=results_dir)
        files = engine.list_result_files()
        assert len(files) == 3
        assert files[0].name == "comparison_20240103_000000.json"
        assert files[-1].name == "comparison_20240101_000000.json"
