"""Tests for rag_system.metrics module.

Verifies EvaluationResult score calculations (keyword, statute, case_id,
overall) including edge cases such as zero totals, perfect scores, and
empty results.
"""

import pytest

from rag_system.metrics import (
    EvaluationResult,
    WEIGHT_CASE_ID,
    WEIGHT_KEYWORD,
    WEIGHT_STATUTE,
)


def _make_result(**overrides) -> EvaluationResult:
    """Create an EvaluationResult with sensible defaults, overriding as needed."""
    defaults = {
        "config_name": "test-config",
        "test_case_id": "TC-001",
        "response_text": "sample response",
    }
    defaults.update(overrides)
    return EvaluationResult(**defaults)


class TestScoreWeights:
    """Test that module-level score weight constants are correct."""

    def test_weight_keyword(self):
        assert WEIGHT_KEYWORD == 0.30

    def test_weight_statute(self):
        assert WEIGHT_STATUTE == 0.40

    def test_weight_case_id(self):
        assert WEIGHT_CASE_ID == 0.30

    def test_weights_sum_to_one(self):
        assert WEIGHT_KEYWORD + WEIGHT_STATUTE + WEIGHT_CASE_ID == pytest.approx(1.0)


class TestEvaluationResultDefaults:
    """Test default values of EvaluationResult fields."""

    def test_default_source_documents(self):
        result = _make_result()
        assert result.source_documents == []

    def test_default_elapsed_time(self):
        result = _make_result()
        assert result.elapsed_time == 0.0

    def test_default_hits_and_totals(self):
        result = _make_result()
        assert result.keyword_hits == 0
        assert result.keyword_total == 0
        assert result.statute_hits == 0
        assert result.statute_total == 0
        assert result.case_id_hits == 0
        assert result.case_id_total == 0

    def test_default_source_count(self):
        result = _make_result()
        assert result.source_count == 0


class TestKeywordScore:
    """Test keyword_score property calculations."""

    def test_zero_total_returns_one(self):
        result = _make_result(keyword_hits=0, keyword_total=0)
        assert result.keyword_score == 1.0

    def test_perfect_score(self):
        result = _make_result(keyword_hits=5, keyword_total=5)
        assert result.keyword_score == 1.0

    def test_partial_score(self):
        result = _make_result(keyword_hits=3, keyword_total=10)
        assert result.keyword_score == pytest.approx(0.3)

    def test_no_hits(self):
        result = _make_result(keyword_hits=0, keyword_total=5)
        assert result.keyword_score == 0.0

    def test_single_hit(self):
        result = _make_result(keyword_hits=1, keyword_total=4)
        assert result.keyword_score == pytest.approx(0.25)


class TestStatuteScore:
    """Test statute_score property calculations."""

    def test_zero_total_returns_one(self):
        result = _make_result(statute_hits=0, statute_total=0)
        assert result.statute_score == 1.0

    def test_perfect_score(self):
        result = _make_result(statute_hits=3, statute_total=3)
        assert result.statute_score == 1.0

    def test_partial_score(self):
        result = _make_result(statute_hits=2, statute_total=5)
        assert result.statute_score == pytest.approx(0.4)

    def test_no_hits(self):
        result = _make_result(statute_hits=0, statute_total=3)
        assert result.statute_score == 0.0


class TestCaseIdScore:
    """Test case_id_score property calculations."""

    def test_zero_total_returns_one(self):
        result = _make_result(case_id_hits=0, case_id_total=0)
        assert result.case_id_score == 1.0

    def test_perfect_score(self):
        result = _make_result(case_id_hits=2, case_id_total=2)
        assert result.case_id_score == 1.0

    def test_partial_score(self):
        result = _make_result(case_id_hits=1, case_id_total=3)
        assert result.case_id_score == pytest.approx(1.0 / 3.0)

    def test_no_hits(self):
        result = _make_result(case_id_hits=0, case_id_total=4)
        assert result.case_id_score == 0.0


class TestOverallScore:
    """Test overall_score weighted average calculation."""

    def test_all_zeros_totals_returns_one(self):
        """When all totals are zero, every component returns 1.0 → overall 1.0."""
        result = _make_result()
        assert result.overall_score == pytest.approx(1.0)

    def test_perfect_scores(self):
        result = _make_result(
            keyword_hits=5,
            keyword_total=5,
            statute_hits=3,
            statute_total=3,
            case_id_hits=2,
            case_id_total=2,
        )
        assert result.overall_score == pytest.approx(1.0)

    def test_all_zero_hits_nonzero_totals(self):
        result = _make_result(
            keyword_hits=0,
            keyword_total=5,
            statute_hits=0,
            statute_total=3,
            case_id_hits=0,
            case_id_total=2,
        )
        assert result.overall_score == pytest.approx(0.0)

    def test_weighted_calculation(self):
        result = _make_result(
            keyword_hits=2,
            keyword_total=4,
            statute_hits=3,
            statute_total=6,
            case_id_hits=1,
            case_id_total=2,
        )
        expected = (
            WEIGHT_KEYWORD * (2 / 4)
            + WEIGHT_STATUTE * (3 / 6)
            + WEIGHT_CASE_ID * (1 / 2)
        )
        assert result.overall_score == pytest.approx(expected)

    def test_mixed_zero_and_nonzero_totals(self):
        """When some totals are zero (score=1.0), overall reflects the mix."""
        result = _make_result(
            keyword_hits=0,
            keyword_total=0,
            statute_hits=1,
            statute_total=2,
            case_id_hits=0,
            case_id_total=0,
        )
        expected = (
            WEIGHT_KEYWORD * 1.0
            + WEIGHT_STATUTE * 0.5
            + WEIGHT_CASE_ID * 1.0
        )
        assert result.overall_score == pytest.approx(expected)

    def test_overall_score_is_float(self):
        result = _make_result(keyword_hits=1, keyword_total=2)
        assert isinstance(result.overall_score, float)


class TestEvaluationResultFields:
    """Test that custom field values are stored correctly."""

    def test_config_name(self):
        result = _make_result(config_name="custom-cfg")
        assert result.config_name == "custom-cfg"

    def test_test_case_id(self):
        result = _make_result(test_case_id="TC-999")
        assert result.test_case_id == "TC-999"

    def test_response_text(self):
        result = _make_result(response_text="LLM output here")
        assert result.response_text == "LLM output here"

    def test_source_documents(self):
        docs = [{"id": 1}, {"id": 2}]
        result = _make_result(source_documents=docs)
        assert result.source_documents == docs

    def test_elapsed_time(self):
        result = _make_result(elapsed_time=1.234)
        assert result.elapsed_time == pytest.approx(1.234)

    def test_source_count(self):
        result = _make_result(source_count=7)
        assert result.source_count == 7
