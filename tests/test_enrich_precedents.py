"""Tests for scripts.enrich_precedents (deterministic reasoning builder)."""

from __future__ import annotations

from scripts.enrich_precedents import build_detailed_reasoning


_BASE = {
    "case_type": "窃盗",
    "charges": ["刑法第191条違反（窃盗）"],
    "referenced_statutes": ["刑法第191条", "刑法第13条"],
    "legal_principles": ["累犯加重の適用", "比例原則"],
    "summary": "被告人は被害者宅に侵入し財物を窃取した事案である。",
    "reasoning": "裁判所は窃盗罪の成立を認めた。",
}


def test_detailed_reasoning_has_all_sections():
    data = dict(_BASE, verdict="有罪", sentence="拘禁刑1年")
    text = build_detailed_reasoning(data, "criminal")
    for section in ["【認定事実】", "【適用法令とその趣旨】", "【法的判断】", "【結論】"]:
        assert section in text
    # Much longer than the seed reasoning.
    assert len(text) > len(_BASE["reasoning"]) * 3


def test_detailed_reasoning_uses_case_metadata():
    data = dict(_BASE, verdict="有罪", sentence="拘禁刑1年")
    text = build_detailed_reasoning(data, "criminal")
    assert "刑法第191条" in text  # primary statute referenced
    assert "累犯加重の適用" in text  # principle referenced
    assert "拘禁刑1年" in text  # sentence appears in conclusion


def test_conclusion_varies_by_verdict():
    guilty = build_detailed_reasoning(
        dict(_BASE, verdict="有罪", sentence="拘禁刑1年"), "criminal"
    )
    acquitted = build_detailed_reasoning(
        dict(_BASE, verdict="無罪", sentence="無罪"), "criminal"
    )
    assert "無罪" in acquitted
    assert "無罪推定" in acquitted
    assert guilty != acquitted


def test_constitutional_unconstitutional_conclusion():
    data = {
        "case_type": "表現の自由",
        "charges": ["憲法第19条違反"],
        "referenced_statutes": ["憲法第19条"],
        "legal_principles": ["表現の自由の保障"],
        "summary": "国家が出版を事前差止めした事案である。",
        "reasoning": "裁判所は違憲と判断した。",
        "verdict": "違憲",
        "sentence": "違憲確認",
    }
    text = build_detailed_reasoning(data, "constitutional")
    assert "違憲" in text
    assert "憲法第19条" in text


def test_civil_dismissed_conclusion():
    data = dict(
        _BASE,
        case_type="売買契約",
        referenced_statutes=["民法第147条"],
        legal_principles=["契約不適合責任"],
        verdict="棄却",
        sentence="請求を棄却する",
    )
    text = build_detailed_reasoning(data, "civil")
    assert "棄却" in text
    assert "民法第147条" in text


def test_missing_fields_do_not_crash():
    text = build_detailed_reasoning({"summary": "概要のみ"}, "criminal")
    assert "【認定事実】" in text
    assert "【結論】" in text
