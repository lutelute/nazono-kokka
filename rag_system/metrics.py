"""Evaluation metrics for the RAG judicial system.

Provides the EvaluationResult dataclass for capturing and scoring
RAG pipeline responses against ground-truth test cases.

Scoring weights:
  - keyword_score:  30%
  - statute_score:  40%
  - case_id_score:  30%
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


# ---------------------------------------------------------------------------
# Score Weights
# ---------------------------------------------------------------------------

WEIGHT_KEYWORD = 0.30
WEIGHT_STATUTE = 0.40
WEIGHT_CASE_ID = 0.30


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Result of evaluating a single RAG response against a test case.

    Attributes:
        config_name:      Human-readable label for the RAG configuration used.
        test_case_id:     Identifier of the test case (e.g. ``"TC-001"``).
        response_text:    Raw text returned by the LLM.
        source_documents: List of retrieved source documents / metadata.
        elapsed_time:     Wall-clock time in seconds for the query.
        keyword_hits:     Number of expected keywords found in the response.
        keyword_total:    Total expected keywords for this test case.
        statute_hits:     Number of expected statute references found.
        statute_total:    Total expected statute references.
        case_id_hits:     Number of expected case IDs found.
        case_id_total:    Total expected case IDs.
        source_count:     Number of source documents retrieved.
    """

    config_name: str
    test_case_id: str
    response_text: str
    source_documents: List[Any] = field(default_factory=list)
    elapsed_time: float = 0.0
    keyword_hits: int = 0
    keyword_total: int = 0
    statute_hits: int = 0
    statute_total: int = 0
    case_id_hits: int = 0
    case_id_total: int = 0
    source_count: int = 0

    # -- Derived scores ----------------------------------------------------

    @property
    def keyword_score(self) -> float:
        """Ratio of keyword hits to expected keywords (0.0 – 1.0)."""
        if self.keyword_total == 0:
            return 1.0
        return self.keyword_hits / self.keyword_total

    @property
    def statute_score(self) -> float:
        """Ratio of statute hits to expected statutes (0.0 – 1.0)."""
        if self.statute_total == 0:
            return 1.0
        return self.statute_hits / self.statute_total

    @property
    def case_id_score(self) -> float:
        """Ratio of case-ID hits to expected case IDs (0.0 – 1.0)."""
        if self.case_id_total == 0:
            return 1.0
        return self.case_id_hits / self.case_id_total

    @property
    def overall_score(self) -> float:
        """Weighted average of the three component scores.

        Weights: keyword 30%, statute 40%, case_id 30%.
        """
        return (
            WEIGHT_KEYWORD * self.keyword_score
            + WEIGHT_STATUTE * self.statute_score
            + WEIGHT_CASE_ID * self.case_id_score
        )
