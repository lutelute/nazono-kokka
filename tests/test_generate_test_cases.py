"""Tests for scripts.generate_test_cases.

Validate the deterministic test-case generation logic and that produced cases
conform to the schema expected by ``rag_system.test_cases.TestCase``.
"""

from __future__ import annotations

from collections import Counter

import pytest

from scripts.generate_test_cases import (
    _clean_term,
    _difficulty,
    build_cases,
    load_groups,
)


def test_clean_term_strips_parentheses():
    assert _clean_term("不法行為（交通事故）") == "不法行為"
    assert _clean_term("売買契約") == "売買契約"
    assert _clean_term("過失致死傷") == "過失致死傷"


def test_difficulty_thresholds():
    assert _difficulty(40) == "basic"
    assert _difficulty(25) == "basic"
    assert _difficulty(24) == "intermediate"
    assert _difficulty(10) == "intermediate"
    assert _difficulty(9) == "advanced"


def test_build_cases_from_synthetic_groups():
    groups = {
        ("criminal", "窃盗"): {
            "count": 40,
            "statutes": Counter({"刑法第191条": 30, "刑法第190条": 5}),
            "principles": Counter({"財産権の保護": 20}),
            "case_ids": ["CRIM-2020-0005"],
        },
        ("civil", "売買契約"): {
            "count": 8,
            "statutes": Counter({"民法第147条": 6}),
            "principles": Counter({"契約不適合責任": 4}),
            "case_ids": ["CIVIL-2018-0005"],
        },
    }
    cases = build_cases(groups)

    # 窃盗 has count>=20 -> base + variant; 売買契約 only base.
    assert len(cases) == 3

    ids = [c["id"] for c in cases]
    assert ids == sorted(ids)  # sequential, ordered
    assert len(set(ids)) == len(ids)  # unique

    theft = cases[0]
    assert theft["category"] == "criminal"
    assert "窃盗" in theft["expected_keywords"]
    assert "刑法" in theft["expected_keywords"]
    # Strongest statute is the ground truth.
    assert theft["expected_statutes"] == ["刑法第191条"]
    assert theft["difficulty"] == "basic"

    sales = [c for c in cases if c["category"] == "civil"][0]
    assert sales["difficulty"] == "advanced"  # count 8 < 10
    assert sales["expected_statutes"] == ["民法第147条"]


def test_generated_cases_schema():
    """Every generated case must load cleanly into a TestCase."""
    from rag_system.test_cases import TestCase

    groups = load_groups()
    if not groups:
        pytest.skip("precedent corpus not present")
    cases = build_cases(groups)
    assert len(cases) >= 50

    for c in cases:
        tc = TestCase.from_dict(c)
        assert tc.id
        assert tc.category in {"criminal", "civil", "constitutional"}
        assert tc.query
        assert isinstance(tc.expected_keywords, list)
        assert isinstance(tc.expected_statutes, list)
