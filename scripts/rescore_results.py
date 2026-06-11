"""保存済み評価結果（incremental JSONL）を修正後の期待値で再スコアリングする。

期待値（expected_statutes 等）はプロンプトにも検索にも影響しないため、
テストケース修正の効果は LLM を再実行せず、保存済みの response_text に
対する文字列マッチの再計算だけで正確に得られる。

ヒット判定は ComparisonEngine と同一実装（_count_* を import）を使う。

Usage:
    .venv/bin/python scripts/rescore_results.py                  # 全構成
    .venv/bin/python scripts/rescore_results.py --fixed-only     # 修正10件の before/after のみ表示
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.comparison_engine import (
    _count_case_id_hits,
    _count_keyword_hits,
    _count_statute_hits,
)
from rag_system.metrics import WEIGHT_CASE_ID, WEIGHT_KEYWORD, WEIGHT_STATUTE

from scripts.fix_test_cases import STATUTE_FIXES

RESULTS = PROJECT_ROOT / "test_cases" / "results"


def score(hits: int, total: int) -> float:
    return 1.0 if total == 0 else hits / total


def rescore_row(row: dict, tc: dict, statutes: list[str]) -> dict:
    text = row["response_text"]
    kw = tc.get("expected_keywords", [])
    ci = tc.get("expected_case_ids", [])
    kw_h = _count_keyword_hits(text, kw)
    st_h = _count_statute_hits(text, statutes)
    ci_h = _count_case_id_hits(text, ci)
    overall = (
        WEIGHT_KEYWORD * score(kw_h, len(kw))
        + WEIGHT_STATUTE * score(st_h, len(statutes))
        + WEIGHT_CASE_ID * score(ci_h, len(ci))
    )
    return {
        "overall": round(overall, 4),
        "statute": f"{st_h}/{len(statutes)}",
        "keyword": f"{kw_h}/{len(kw)}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="保存済み結果の再スコアリング")
    parser.add_argument("--fixed-only", action="store_true", help="修正対象10件のみ表示")
    args = parser.parse_args()

    meta = {
        tc["id"]: tc
        for tc in json.loads(
            (PROJECT_ROOT / "test_cases" / "all_cases.json").read_text(encoding="utf-8")
        )["test_cases"]
    }

    out: dict[str, dict] = {}
    for path in sorted(RESULTS.glob("incremental_*.jsonl")):
        name = path.stem.replace("incremental_", "")
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue

        orig_scores, fixed_scores = [], []
        fixed_detail = []
        for r in rows:
            tc = meta[r["test_case_id"]]
            # 修正後の期待条文（パッチ未適用ファイルでも STATUTE_FIXES を優先適用）
            fixed_statutes = STATUTE_FIXES.get(tc["id"], tc.get("expected_statutes", []))
            orig = rescore_row(r, tc, tc.get("expected_statutes", []))
            fixed = rescore_row(r, tc, fixed_statutes)
            orig_scores.append(orig["overall"])
            fixed_scores.append(fixed["overall"])
            if tc["id"] in STATUTE_FIXES:
                fixed_detail.append(
                    {
                        "id": tc["id"],
                        "before": orig["overall"],
                        "after": fixed["overall"],
                        "statute_before": orig["statute"],
                        "statute_after": fixed["statute"],
                    }
                )

        avg = lambda xs: round(sum(xs) / len(xs), 4)
        out[name] = {
            "n": len(rows),
            "overall_before_fix": avg(orig_scores),
            "overall_after_fix": avg(fixed_scores),
            "fixed_cases": sorted(fixed_detail, key=lambda d: d["id"]),
        }

    for name, s in out.items():
        print(f"=== {name} (n={s['n']}) ===")
        print(f"  総合: 修正前 {s['overall_before_fix']} -> 修正後 {s['overall_after_fix']}")
        if args.fixed_only or True:
            for d in s["fixed_cases"]:
                arrow = "↑" if d["after"] > d["before"] else ("→" if d["after"] == d["before"] else "↓")
                print(
                    f"    {d['id']}: {d['before']:.2f} -> {d['after']:.2f} {arrow}"
                    f"  (statute {d['statute_before']} -> {d['statute_after']})"
                )

    save = RESULTS / "rescore_after_fix.json"
    save.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存: {save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
