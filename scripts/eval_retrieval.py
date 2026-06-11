"""検索段階のリコール評価スクリプト（LLM 不要）。

全テストケースについて dense / hybrid / hybrid+rerank の3方式で検索し、
期待条文・期待判例IDが検索結果 k 件に含まれる率（recall@k）を比較する。
LLM を使わないため Ollama の評価実行と並行して走らせられる。

Usage:
    .venv/bin/python scripts/eval_retrieval.py
    .venv/bin/python scripts/eval_retrieval.py --k 5 --cases all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.advanced_retriever import AdvancedRetriever
from rag_system.retriever import load_vectorstore, retrieve_advanced
from rag_system.test_cases import TEST_CASES_DIR, TestCaseManager

logging.disable(logging.INFO)

MODES = {
    "dense": {"use_hybrid": False, "use_rerank": False},
    "hybrid": {"use_hybrid": True, "use_rerank": False},
    "hybrid+rerank": {"use_hybrid": True, "use_rerank": True},
    "advanced(quota)": "advanced",  # AdvancedRetriever: hybrid+rerank+法令クォータ
}


def main() -> int:
    parser = argparse.ArgumentParser(description="検索リコール評価")
    parser.add_argument("--cases", default="all", choices=["all", "default", "generated"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="", help="この方式のみ実行 (例: 'advanced(quota)')")
    args = parser.parse_args()

    mgr = TestCaseManager()
    cases = mgr.load_from_file(TEST_CASES_DIR / f"{args.cases}_cases.json")
    if args.limit > 0:
        cases = cases[: args.limit]

    vs = load_vectorstore()
    print(f"=== 検索リコール評価: {len(cases)} 件 × {len(MODES)} 方式 (k={args.k}) ===", flush=True)

    modes = (
        {k: v for k, v in MODES.items() if k == args.only} if args.only else MODES
    )
    report: dict[str, dict] = {}

    for mode, kwargs in modes.items():
        t0 = time.time()
        statute_hits = statute_total = 0
        case_hits = case_total = 0
        legal_doc_present = 0  # 法令文書が1件以上含まれるケース数
        per_category: dict[str, list[float]] = defaultdict(list)

        adv = (
            AdvancedRetriever(vectorstore=vs, k=args.k)
            if kwargs == "advanced"
            else None
        )
        for i, tc in enumerate(cases):
            if adv is not None:
                docs = adv.invoke(tc.query)
            else:
                out = retrieve_advanced(tc.query, vectorstore=vs, k=args.k, **kwargs)
                docs = out.final
            texts = [d.page_content for d in docs]
            joined = "\n".join(texts)
            case_ids = {d.metadata.get("case_id", "") for d in docs}

            s_hit = sum(1 for s in tc.expected_statutes if s in joined)
            c_hit = sum(1 for c in tc.expected_case_ids if c in case_ids)
            statute_hits += s_hit
            statute_total += len(tc.expected_statutes)
            case_hits += c_hit
            case_total += len(tc.expected_case_ids)
            if any(
                d.metadata.get("document_type") == "legal_framework"
                or "legal_framework" in str(d.metadata.get("source", ""))
                for d in docs
            ):
                legal_doc_present += 1

            denom = len(tc.expected_statutes) + len(tc.expected_case_ids)
            if denom:
                per_category[tc.category].append((s_hit + c_hit) / denom)

            if (i + 1) % 50 == 0:
                print(f"  [{mode}] {i+1}/{len(cases)}", flush=True)

        elapsed = time.time() - t0
        report[mode] = {
            "statute_recall": round(statute_hits / statute_total, 4) if statute_total else None,
            "case_id_recall": round(case_hits / case_total, 4) if case_total else None,
            "legal_doc_presence": round(legal_doc_present / len(cases), 4),
            "by_category": {
                k: round(sum(v) / len(v), 4) for k, v in sorted(per_category.items())
            },
            "sec_per_query": round(elapsed / len(cases), 2),
        }
        print(f"[{mode}] 完了 ({elapsed:.0f}s): {json.dumps(report[mode], ensure_ascii=False)}", flush=True)

    out_path = TEST_CASES_DIR / "results" / "retrieval_recall.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged = report
    if out_path.exists():
        try:
            merged = json.loads(out_path.read_text(encoding="utf-8"))
            merged.update(report)
        except (json.JSONDecodeError, OSError):
            merged = report
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== サマリー (recall@%d) ===" % args.k, flush=True)
    hdr = f"{'方式':<16}{'statute':>10}{'case_id':>10}{'法令文書':>10}{'秒/件':>8}"
    print(hdr, flush=True)
    for mode, r in report.items():
        print(
            f"{mode:<16}{r['statute_recall']:>10}{r['case_id_recall']:>10}"
            f"{r['legal_doc_presence']:>10}{r['sec_per_query']:>8}",
            flush=True,
        )
    print(f"\n保存先: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
