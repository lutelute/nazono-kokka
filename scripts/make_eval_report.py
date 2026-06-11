"""評価結果群から Markdown 比較レポートを生成する。

入力（存在するものだけ使う）:
  - test_cases/results/retrieval_recall.json        検索段階の4方式比較
  - test_cases/results/incremental_*.jsonl          E2E評価の逐次保存（構成別）
  - test_cases/all_cases.json                       カテゴリ・難易度の参照

出力:
  - EVALUATION.md（リポジトリ直下）

Usage:
    .venv/bin/python scripts/make_eval_report.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "test_cases" / "results"

# dense+旧プロンプト構成（フリーズで5件のみ完了）のログ実測値
LEGACY_BASELINE = {
    "TC-001": 0.70,
    "TC-002": 0.60,
    "TC-003": 0.42,
    "TC-004": 0.55,
    "TC-005": 0.48,
}

# fix_test_cases.py の修正対象（評価解釈の注記用）
KNOWN_BAD_CASES = {
    "TC-002", "TC-003", "TC-004", "TC-005", "TC-006",
    "TC-007", "TC-008", "TC-009", "TC-011", "TC-012",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def frac_rate(rows: list[dict], key: str) -> float | None:
    hits = total = 0
    for r in rows:
        a, b = r[key].split("/")
        hits += int(a)
        total += int(b)
    return hits / total if total else None


def summarize(rows: list[dict], meta: dict[str, dict]) -> dict:
    scores = [r["overall_score"] for r in rows]
    by_cat: dict[str, list[float]] = defaultdict(list)
    by_diff: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        m = meta.get(r["test_case_id"])
        if m:
            by_cat[m["category"]].append(r["overall_score"])
            by_diff[m["difficulty"]].append(r["overall_score"])
    avg = lambda xs: sum(xs) / len(xs) if xs else None
    return {
        "n": len(rows),
        "overall": avg(scores),
        "keyword_rate": frac_rate(rows, "keyword"),
        "statute_rate": frac_rate(rows, "statute"),
        "by_category": {k: avg(v) for k, v in sorted(by_cat.items())},
        "by_difficulty": {k: avg(v) for k, v in sorted(by_diff.items())},
        "scores_by_id": {r["test_case_id"]: r["overall_score"] for r in rows},
    }


def fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def main() -> int:
    meta = {
        tc["id"]: tc
        for tc in json.loads(
            (PROJECT_ROOT / "test_cases" / "all_cases.json").read_text(encoding="utf-8")
        )["test_cases"]
    }

    configs: dict[str, dict] = {}
    for path in sorted(RESULTS.glob("incremental_*.jsonl")):
        name = path.stem.replace("incremental_", "")
        rows = load_jsonl(path)
        if rows:
            configs[name] = summarize(rows, meta)

    recall = {}
    recall_path = RESULTS / "retrieval_recall.json"
    if recall_path.exists():
        recall = json.loads(recall_path.read_text(encoding="utf-8"))

    lines: list[str] = []
    w = lines.append
    w(f"# RAG 評価レポート（{date.today().isoformat()}）")
    w("")
    w("謎の国家 RAG 司法システムの精度評価。139 テストケース（基本12＋生成127）を")
    w("keyword 30% / statute 40% / case_id 30% の重み付きスコアで評価する")
    w("（期待値が空の軸は満点扱い。case_id は 139 件中 138 件が空のため、実効は keyword＋statute）。")
    w("")
    w("**評価環境**: Ollama qwen2.5:7b-instruct / num_ctx=8192 / temperature 0.1 / k=5")
    w("")

    # ----- 検索段階 -----
    if recall:
        w("## 1. 検索段階の比較（LLM 不要・recall@5・139件）")
        w("")
        w("| 方式 | statute recall | 法令文書包含 | 秒/件 |")
        w("|---|---|---|---|")
        for mode, r in recall.items():
            w(
                f"| {mode} | {r['statute_recall']} | {r['legal_doc_presence']} | {r['sec_per_query']} |"
            )
        w("")
        w("- hybrid（BM25＋RRF）で statute recall が **0.64 → 0.84**。条文番号のような")
        w("  正確なキーワードは密ベクトルが苦手で、BM25 の語彙マッチが補完する。")
        w("- cross-encoder rerank でさらに 0.89 へ。ただし判例が優先され法令文書包含が")
        w("  0.35 まで低下 → **法令クォータ**（最低1件保証、`AdvancedRetriever`）で 0.81 に回復。")
        w("")

    # ----- E2E -----
    w("## 2. エンドツーエンド評価（139件）")
    w("")
    w("| 構成 | n | 総合 | keywordヒット率 | statuteヒット率 |")
    w("|---|---|---|---|---|")
    w(
        f"| dense＋旧プロンプト（参考: フリーズ前5件のみ） | 5 | "
        f"{fmt(sum(LEGACY_BASELINE.values())/len(LEGACY_BASELINE))} | — | — |"
    )
    for name, s in configs.items():
        w(
            f"| {name} | {s['n']} | **{fmt(s['overall'])}** | "
            f"{fmt(s['keyword_rate'])} | {fmt(s['statute_rate'])} |"
        )
    w("")

    for name, s in configs.items():
        w(f"### {name} の内訳")
        w("")
        cat = " / ".join(f"{k} {fmt(v)}" for k, v in s["by_category"].items())
        diff = " / ".join(f"{k} {fmt(v)}" for k, v in s["by_difficulty"].items())
        w(f"- カテゴリ別: {cat}")
        w(f"- 難易度別: {diff}")
        w("")

    # ----- 同一ケース対前比較 -----
    w("## 3. 同一ケース対前比較（TC-001〜005）")
    w("")
    header = "| ケース | dense＋旧プロンプト |"
    sep = "|---|---|"
    for name in configs:
        header += f" {name} |"
        sep += "---|"
    w(header)
    w(sep)
    for cid, legacy in LEGACY_BASELINE.items():
        row = f"| {cid} | {legacy:.2f} |"
        for s in configs.values():
            v = s["scores_by_id"].get(cid)
            row += f" {fmt(v, 2)} |"
        note = "（期待値修正対象）" if cid in KNOWN_BAD_CASES else ""
        w(row + (f" {note}" if note else ""))
    w("")
    w("TC-002〜005 は期待条文に実在日本法の条番号が混入したケース（§4）のため、")
    w("検索・プロンプトをいくら改善してもスコアが上がらない。改善効果は生成ケース群")
    w("（GEN-xxx）に現れている。")
    w("")

    # ----- テストケース品質 -----
    w("## 4. テストケースの品質問題と修正")
    w("")
    w("期待条文の突合（`scripts/fix_test_cases.py`）で、基本セット 12 件中 **10 件**に")
    w("実在日本法の条番号の混入を発見した。2 パターン：")
    w("")
    w("1. **コーパスに不存在**（7件）: 例 TC-003 刑法第246条（日本法の詐欺）— 謎の国家では第198条")
    w("2. **存在するが意味が違う**（3件）: 例 TC-011 憲法第81条 — 謎の国家では「地方公共団体の権能」。")
    w("   違憲審査権は第71条。LLM が資料から第71条を正しく転記していたのに減点されていた。")
    w("")
    w("修正は `scripts/fix_test_cases.py --apply` で適用する（GEN 系の期待値と整合確認済み）。")
    w("")

    # ----- 改善の経緯 -----
    w("## 5. 評価基盤の主な改善")
    w("")
    w("| 問題 | 対策 |")
    w("|---|---|")
    w("| qwen3.5:9b の thinking で 331 秒/件（思考 1,683 トークン） | `reasoning=False`、最終的に qwen2.5:7b-instruct を採用（指示追従が良くスコア 0.70 vs 0.30） |")
    w("| RAG プロンプト約 5-6k トークンが num_ctx=4096 で切り捨て | num_ctx=8192 に変更 |")
    w("| 実装済みの hybrid/rerank が評価チェーンで未使用 | `AdvancedRetriever` で RetrievalQA に接続 |")
    w("| rerank で法令文書が消える（包含 0.63→0.35） | 法令クォータ（最低1件保証）で 0.81 へ |")
    w("| 評価途中のフリーズ・異常終了で結果全損 | 逐次保存 JSONL ＋ `--resume` |")
    w("| Ollama(Metal)×PyTorch(MPS) の並行で Ollama がハング | 評価中は MPS 処理を並行させない運用 |")
    w("")

    # ----- 次の改善候補 -----
    w("## 6. 次の改善候補")
    w("")
    w("- テストケース修正の適用と対象 10 件の再評価（修正後スコアの確定）")
    w("- case_id 期待値の充実（現状 1/139 件のみ。判例 DB から関連判例を逆引きして生成すれば")
    w("  配点 30% が実質機能する）")
    w("- 検索クエリ拡張（質問文そのままでなく、論点語の抽出・追加）")
    w("- GEN-049 など期待条文とクエリ主題が緩く対応するケースの精査")
    w("")

    out = PROJECT_ROOT / "EVALUATION.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"レポート生成: {out}")
    for name, s in configs.items():
        print(f"  {name}: n={s['n']} overall={fmt(s['overall'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
