"""RAG 司法システムの一括評価スクリプト。

all_cases.json の全テストケースを指定モデルで実行し、
test_cases/results/ に結果 JSON を保存、カテゴリ別・難易度別サマリーを表示する。

Usage:
    .venv/bin/python scripts/run_evaluation.py                     # 全139件
    .venv/bin/python scripts/run_evaluation.py --cases default     # 基本12件
    .venv/bin/python scripts/run_evaluation.py --model qwen3:8b
    .venv/bin/python scripts/run_evaluation.py --limit 10          # 先頭10件のみ
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_ollama import OllamaLLM

from rag_system.backend_adapter import BackendConfig
from rag_system.comparison_engine import ComparisonEngine
from rag_system.config import LLM_NUM_CTX, LLM_TEMPERATURE, OLLAMA_BASE_URL
from rag_system.judge import create_judicial_chain
from rag_system.test_cases import TEST_CASES_DIR, TestCaseManager


class NoThinkEngine(ComparisonEngine):
    """qwen3 系の thinking を無効化した LLM でチェーンを構築するエンジン。

    qwen3.5 等は thinking がデフォルト有効で、1 件あたり 300 秒超かかる
    （回答 1 文に思考 1,600 トークン）。``reasoning=False`` で無効化すると
    同条件で数秒に短縮される。

    retriever_mode:
        "dense"    — 従来の密ベクトル検索のみ（k=5）
        "advanced" — hybrid(BM25+RRF) + cross-encoder rerank + 法令クォータ
                     （139件の retrieval recall 実測で statute recall 0.64→0.89）
    """

    def __init__(
        self,
        results_dir=None,
        retriever_mode: str = "dense",
        incremental_path: Path | None = None,
    ) -> None:
        super().__init__(results_dir)
        self._retriever_mode = retriever_mode
        self._vectorstore = None
        self._incremental_path = incremental_path

    def _execute_single(self, chain, config, test_case):
        """1件実行ごとに JSONL へ追記し、途中終了でも結果を残す。"""
        result = super()._execute_single(chain, config, test_case)
        if self._incremental_path is not None:
            row = {
                "config_name": result.config_name,
                "test_case_id": result.test_case_id,
                "overall_score": round(result.overall_score, 4),
                "keyword": f"{result.keyword_hits}/{result.keyword_total}",
                "statute": f"{result.statute_hits}/{result.statute_total}",
                "case_id": f"{result.case_id_hits}/{result.case_id_total}",
                "elapsed_sec": round(result.elapsed_time, 1),
                "response_text": result.response_text,
            }
            with open(self._incremental_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return result

    def _build_retriever(self):
        if self._retriever_mode != "advanced":
            return None  # create_judicial_chain がデフォルト(dense)を構築する
        from rag_system.advanced_retriever import AdvancedRetriever
        from rag_system.retriever import load_vectorstore

        if self._vectorstore is None:
            self._vectorstore = load_vectorstore()
        return AdvancedRetriever(vectorstore=self._vectorstore)

    def _create_chain(self, config: BackendConfig):
        try:
            llm = OllamaLLM(
                model=config.model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=config.temperature,
                num_ctx=config.num_ctx,
                reasoning=False,
            )
            return create_judicial_chain(llm=llm, retriever=self._build_retriever())
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "チェーン構築に失敗しました: %s", config.name
            )
            return None


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 一括評価")
    parser.add_argument("--model", default="qwen3.5:9b", help="Ollama モデル名")
    parser.add_argument(
        "--cases",
        default="all",
        choices=["all", "default", "generated"],
        help="使用するテストケースセット",
    )
    parser.add_argument("--limit", type=int, default=0, help="先頭 N 件に制限 (0=全件)")
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="コンテキストウィンドウ (RAGプロンプトは約5-6kトークンのため8192以上を推奨)",
    )
    parser.add_argument(
        "--retriever",
        default="dense",
        choices=["dense", "advanced"],
        help="検索方式: dense=従来 / advanced=hybrid+rerank+法令クォータ",
    )
    args = parser.parse_args()

    mgr = TestCaseManager()
    cases = mgr.load_from_file(TEST_CASES_DIR / f"{args.cases}_cases.json")
    if args.limit > 0:
        cases = cases[: args.limit]

    config = BackendConfig(
        name=f"{args.model.replace(':', '-')}+{args.retriever}",
        model_name=args.model,
        num_ctx=args.num_ctx,
    )
    incremental = TEST_CASES_DIR / "results" / f"incremental_{config.name}.jsonl"
    engine = NoThinkEngine(
        retriever_mode=args.retriever, incremental_path=incremental
    )

    print(f"=== 評価開始: {args.model} × {len(cases)} 件 ===", flush=True)
    t0 = time.time()

    def progress(step: int, total: int, message: str) -> None:
        elapsed = time.time() - t0
        eta = (elapsed / step) * (total - step) if step > 0 else 0
        print(
            f"[{step}/{total}] {message} (経過 {elapsed/60:.1f}分 / 残り目安 {eta/60:.1f}分)",
            flush=True,
        )

    results = engine.run_comparison(
        [config], cases, progress_callback=progress, save_results=True
    )

    total_time = time.time() - t0

    # ----- サマリー集計 -----
    case_by_id = {tc.id: tc for tc in cases}
    by_category: dict[str, list[float]] = defaultdict(list)
    by_difficulty: dict[str, list[float]] = defaultdict(list)
    scores = []
    errors = 0

    for r in results:
        tc = case_by_id.get(r.test_case_id)
        scores.append(r.overall_score)
        if r.response_text.startswith("エラー:"):
            errors += 1
        if tc is not None:
            by_category[tc.category].append(r.overall_score)
            by_difficulty[tc.difficulty].append(r.overall_score)

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "model": args.model,
        "retriever": args.retriever,
        "config_name": config.name,
        "n_cases": len(cases),
        "n_errors": errors,
        "total_time_min": round(total_time / 60, 1),
        "overall_score_avg": round(avg(scores), 4),
        "keyword_score_avg": round(avg([r.keyword_score for r in results]), 4),
        "statute_score_avg": round(avg([r.statute_score for r in results]), 4),
        "case_id_score_avg": round(avg([r.case_id_score for r in results]), 4),
        "by_category": {k: round(avg(v), 4) for k, v in sorted(by_category.items())},
        "by_difficulty": {k: round(avg(v), 4) for k, v in sorted(by_difficulty.items())},
        "elapsed_per_case_sec": round(avg([r.elapsed_time for r in results]), 1),
    }

    results_dir = TEST_CASES_DIR / "results"
    summary_path = results_dir / "latest_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 設定別にも残す（latest はあくまで直近実行のスナップショット）
    (results_dir / f"summary_{config.name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 評価サマリー ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"\nサマリー保存先: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
