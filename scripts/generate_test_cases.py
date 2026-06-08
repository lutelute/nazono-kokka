"""Generate a broad test-case suite from the precedent corpus.

The hand-written ``test_cases/default_cases.json`` only has ~12 cases, which
is too few for the accuracy-comparison dashboard to say anything statistically
meaningful. This script derives one (or more) test case per ``case_type`` found
in the precedent database, using the *actual* referenced statutes and legal
principles of each group as ground truth — so the generated suite is grounded
in the data, deterministic, and needs no LLM.

Output: ``test_cases/generated_cases.json`` (importable from the テストケース
page, or loadable directly by the comparison engine).

Usage::

    python -m scripts.generate_test_cases            # write generated_cases.json
    python -m scripts.generate_test_cases --merge    # also merge with defaults
    python -m scripts.generate_test_cases --stats    # print summary only
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rag_system.config import (
    PRECEDENTS_CIVIL_DIR,
    PRECEDENTS_CONSTITUTIONAL_DIR,
    PRECEDENTS_CRIMINAL_DIR,
)

TEST_CASES_DIR = Path(__file__).resolve().parent.parent / "test_cases"
GENERATED_PATH = TEST_CASES_DIR / "generated_cases.json"
DEFAULT_PATH = TEST_CASES_DIR / "default_cases.json"

_CATEGORY_DIRS = [
    ("criminal", PRECEDENTS_CRIMINAL_DIR),
    ("civil", PRECEDENTS_CIVIL_DIR),
    ("constitutional", PRECEDENTS_CONSTITUTIONAL_DIR),
]

# Per-category query phrasing + generic keywords a correct answer should carry.
_CATEGORY_PROFILE = {
    "criminal": {
        "law": "刑法",
        "query": "{term}の構成要件と量刑の判断基準を、関連する条文と判例に基づいて説明せよ。",
        "extra_keywords": ["量刑"],
    },
    "civil": {
        "law": "民法",
        "query": "{term}をめぐる法的責任の有無と要件について、条文と判例に基づいて論じよ。",
        "extra_keywords": ["責任"],
    },
    "constitutional": {
        "law": "憲法",
        "query": "{term}に関する憲法上の保障の範囲と、合憲性の判断基準を述べよ。",
        "extra_keywords": ["憲法"],
    },
}

# A second, harder phrasing used to add variety for the most populous types.
_VARIANT_QUERY = {
    "criminal": "{term}の事案で、無罪となる場合と有罪となる場合の分かれ目を判例から整理せよ。",
    "civil": "{term}において損害賠償が認められるための要件を、判例の判断とともに具体的に述べよ。",
    "constitutional": "{term}が制限される場合の合憲性審査の枠組みを、判例を引用して説明せよ。",
}


def _clean_term(case_type: str) -> str:
    """Turn a raw case_type into a natural keyword/term.

    "不法行為（交通事故）" -> "不法行為", "わいせつ犯罪" -> "わいせつ犯罪".
    Parenthetical qualifiers are dropped so the term reads naturally.
    """
    term = re.sub(r"[（(].*?[）)]", "", case_type).strip()
    return term or case_type


def load_groups() -> dict[tuple[str, str], dict]:
    """Aggregate precedents by ``(category, case_type)``.

    Returns a mapping to ``{count, statutes: Counter, principles: Counter,
    case_ids: list}``.
    """
    groups: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "count": 0,
            "statutes": Counter(),
            "principles": Counter(),
            "case_ids": [],
        }
    )

    for category, dir_path in _CATEGORY_DIRS:
        if not dir_path.exists():
            continue
        for jp in sorted(dir_path.glob("*.json")):
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            case_type = data.get("case_type", "").strip()
            if not case_type:
                continue
            g = groups[(category, case_type)]
            g["count"] += 1
            for s in data.get("referenced_statutes", []) or []:
                g["statutes"][s] += 1
            for p in data.get("legal_principles", []) or []:
                g["principles"][p] += 1
            if cid := data.get("case_id"):
                g["case_ids"].append(cid)

    return groups


def _difficulty(count: int) -> str:
    if count >= 25:
        return "basic"
    if count >= 10:
        return "intermediate"
    return "advanced"


def _principle_keyword(principles: Counter) -> list[str]:
    """Pick the single most common legal-principle phrase as a keyword."""
    if not principles:
        return []
    top = principles.most_common(1)[0][0]
    # Keep it short enough to be a realistic substring check.
    return [top] if len(top) <= 12 else []


def build_cases(groups: dict[tuple[str, str], dict]) -> list[dict]:
    """Build the list of test-case dicts from aggregated groups."""
    cases: list[dict] = []
    # Sort by descending count so the most important types come first.
    ordered = sorted(groups.items(), key=lambda kv: -kv[1]["count"])

    seq = 0
    for (category, case_type), g in ordered:
        profile = _CATEGORY_PROFILE[category]
        term = _clean_term(case_type)

        # Top statute(s) are the strongest ground-truth signal.
        top_statutes = [s for s, _ in g["statutes"].most_common(2)]
        keywords = [term, profile["law"], *profile["extra_keywords"]]
        keywords += _principle_keyword(g["principles"])
        # De-dup while preserving order.
        keywords = list(dict.fromkeys(k for k in keywords if k))

        seq += 1
        cases.append(
            {
                "id": f"GEN-{seq:03d}",
                "category": category,
                "query": profile["query"].format(term=term),
                "expected_keywords": keywords,
                "expected_statutes": top_statutes[:1],
                "expected_case_ids": [],
                "difficulty": _difficulty(g["count"]),
                "description": f"{case_type}（{g['count']}件）に関する基本論点",
            }
        )

        # For populous types, add a harder variant for breadth.
        if g["count"] >= 20:
            seq += 1
            cases.append(
                {
                    "id": f"GEN-{seq:03d}",
                    "category": category,
                    "query": _VARIANT_QUERY[category].format(term=term),
                    "expected_keywords": list(
                        dict.fromkeys([term, profile["law"]])
                    ),
                    "expected_statutes": top_statutes[:1],
                    "expected_case_ids": [],
                    "difficulty": "advanced",
                    "description": f"{case_type}の応用論点（判例の分岐）",
                }
            )

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="判例コーパスからテストケースを生成")
    parser.add_argument("--merge", action="store_true",
                        help="既存の default_cases.json と統合した all_cases.json も書き出す")
    parser.add_argument("--stats", action="store_true",
                        help="生成内容のサマリのみ表示（ファイルは書かない）")
    args = parser.parse_args()

    groups = load_groups()
    cases = build_cases(groups)

    by_cat = Counter(c["category"] for c in cases)
    by_diff = Counter(c["difficulty"] for c in cases)
    print(f"生成テストケース数: {len(cases)}")
    print(f"  カテゴリ別: {dict(by_cat)}")
    print(f"  難易度別: {dict(by_diff)}")
    print(f"  元の case_type 数: {len(groups)}")

    if args.stats:
        for c in cases[:5]:
            print("  例:", c["id"], c["query"][:40], "→", c["expected_statutes"])
        return

    GENERATED_PATH.write_text(
        json.dumps({"test_cases": cases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"書き出し: {GENERATED_PATH}")

    if args.merge:
        defaults = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
        merged = defaults.get("test_cases", []) + cases
        all_path = TEST_CASES_DIR / "all_cases.json"
        all_path.write_text(
            json.dumps({"test_cases": merged}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"統合書き出し: {all_path}（計 {len(merged)} 件）")


if __name__ == "__main__":
    main()
