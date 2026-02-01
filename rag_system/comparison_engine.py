"""精度比較エンジンモジュール。

複数の AI バックエンド設定に対してテストケースを一括実行し、
:class:`~rag_system.metrics.EvaluationResult` のリストとして結果を返す。
Streamlit UI 統合用のプログレスコールバックをサポートする。

Usage:
    from rag_system.comparison_engine import ComparisonEngine
    from rag_system.backend_adapter import BackendConfig
    from rag_system.test_cases import TestCaseManager

    engine = ComparisonEngine()
    mgr = TestCaseManager()
    cases = mgr.load_default_cases()
    configs = [
        BackendConfig(name="default", model_name="llama3.1:8b"),
        BackendConfig(name="creative", model_name="llama3.1:8b", temperature=0.7),
    ]
    results = engine.run_comparison(configs, cases)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

from rag_system.backend_adapter import BackendConfig, create_backend
from rag_system.config import PROJECT_ROOT
from rag_system.metrics import EvaluationResult
from rag_system.test_cases import TestCase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "test_cases" / "results"


# ---------------------------------------------------------------------------
# Progress Callback Type
# ---------------------------------------------------------------------------

# callback(current_step, total_steps, message)
ProgressCallback = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Scoring Helpers
# ---------------------------------------------------------------------------


def _count_keyword_hits(response_text: str, expected_keywords: list[str]) -> int:
    """応答テキスト中に含まれる期待キーワードの数を返す。

    Parameters
    ----------
    response_text:
        LLM の応答テキスト。
    expected_keywords:
        期待されるキーワードのリスト。

    Returns
    -------
    int
        ヒットしたキーワード数。
    """
    hits = 0
    text_lower = response_text.lower()
    for keyword in expected_keywords:
        if keyword.lower() in text_lower:
            hits += 1
    return hits


def _count_statute_hits(response_text: str, expected_statutes: list[str]) -> int:
    """応答テキスト中に含まれる期待条文参照の数を返す。

    Parameters
    ----------
    response_text:
        LLM の応答テキスト。
    expected_statutes:
        期待される条文参照のリスト (e.g. ``["刑法第235条"]``)。

    Returns
    -------
    int
        ヒットした条文数。
    """
    hits = 0
    for statute in expected_statutes:
        if statute in response_text:
            hits += 1
    return hits


def _count_case_id_hits(response_text: str, expected_case_ids: list[str]) -> int:
    """応答テキスト中に含まれる期待判例IDの数を返す。

    Parameters
    ----------
    response_text:
        LLM の応答テキスト。
    expected_case_ids:
        期待される判例IDのリスト (e.g. ``["CRIMINAL-2019-0012"]``)。

    Returns
    -------
    int
        ヒットした判例ID数。
    """
    hits = 0
    for case_id in expected_case_ids:
        if case_id in response_text:
            hits += 1
    return hits


# ---------------------------------------------------------------------------
# Comparison Engine
# ---------------------------------------------------------------------------


class ComparisonEngine:
    """複数 AI バックエンド設定に対するテストケース一括実行エンジン。

    Parameters
    ----------
    results_dir:
        比較結果 JSON の保存先ディレクトリ。デフォルトは
        ``test_cases/results/``。
    """

    def __init__(self, results_dir: Path | None = None) -> None:
        self._results_dir = results_dir or RESULTS_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_comparison(
        self,
        configs: list[BackendConfig],
        test_cases: list[TestCase],
        *,
        progress_callback: ProgressCallback | None = None,
        save_results: bool = True,
    ) -> list[EvaluationResult]:
        """複数設定 × テストケースの比較実行を行う。

        Parameters
        ----------
        configs:
            比較対象のバックエンド設定リスト。
        test_cases:
            実行するテストケースリスト。
        progress_callback:
            進捗通知用のコールバック。
            ``callback(current_step, total_steps, message)`` の形式で呼ばれる。
        save_results:
            ``True`` の場合、実行結果を JSON ファイルに自動保存する。

        Returns
        -------
        list[EvaluationResult]
            全設定 × 全テストケースの評価結果リスト。
        """
        if not configs:
            logger.warning("比較対象の設定が指定されていません")
            return []

        if not test_cases:
            logger.warning("テストケースが指定されていません")
            return []

        total_steps = len(configs) * len(test_cases)
        current_step = 0
        results: list[EvaluationResult] = []

        logger.info(
            "比較実行を開始: %d 設定 × %d テストケース = %d ステップ",
            len(configs),
            len(test_cases),
            total_steps,
        )

        for config in configs:
            chain = self._create_chain(config)
            if chain is None:
                # チェーン構築失敗 — このconfig のケースはすべてエラー結果とする
                for tc in test_cases:
                    current_step += 1
                    result = self._create_error_result(
                        config, tc, "バックエンドの構築に失敗しました"
                    )
                    results.append(result)
                    if progress_callback is not None:
                        progress_callback(
                            current_step,
                            total_steps,
                            f"エラー: {config.name} / {tc.id}",
                        )
                continue

            for tc in test_cases:
                current_step += 1
                if progress_callback is not None:
                    progress_callback(
                        current_step,
                        total_steps,
                        f"実行中: {config.name} / {tc.id}",
                    )

                result = self._execute_single(chain, config, tc)
                results.append(result)

                logger.info(
                    "完了: %s / %s — スコア %.2f (%.1f秒)",
                    config.name,
                    tc.id,
                    result.overall_score,
                    result.elapsed_time,
                )

        logger.info(
            "比較実行が完了しました: %d 件の結果",
            len(results),
        )

        if save_results and results:
            self._save_results(results, configs, test_cases)

        return results

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _create_chain(self, config: BackendConfig) -> Any:
        """バックエンド設定からチェーンを構築する。失敗時は ``None``。"""
        try:
            chain = create_backend(config)
            return chain
        except ConnectionError:
            logger.error(
                "バックエンド '%s' の構築に失敗しました: Ollamaサーバーに接続できません",
                config.name,
            )
            return None
        except FileNotFoundError:
            logger.error(
                "バックエンド '%s' の構築に失敗しました: ChromaDBが見つかりません",
                config.name,
            )
            return None
        except Exception:
            logger.exception(
                "バックエンド '%s' の構築中に予期しないエラーが発生しました",
                config.name,
            )
            return None

    def _execute_single(
        self,
        chain: Any,
        config: BackendConfig,
        test_case: TestCase,
    ) -> EvaluationResult:
        """単一のテストケースを実行し、評価結果を返す。"""
        start_time = time.time()

        try:
            raw_result = chain.invoke({"query": test_case.query})
            elapsed = time.time() - start_time

            response_text = raw_result.get("result", "")
            source_documents = raw_result.get("source_documents", [])

            return EvaluationResult(
                config_name=config.name,
                test_case_id=test_case.id,
                response_text=response_text,
                source_documents=source_documents,
                elapsed_time=elapsed,
                keyword_hits=_count_keyword_hits(
                    response_text, test_case.expected_keywords
                ),
                keyword_total=len(test_case.expected_keywords),
                statute_hits=_count_statute_hits(
                    response_text, test_case.expected_statutes
                ),
                statute_total=len(test_case.expected_statutes),
                case_id_hits=_count_case_id_hits(
                    response_text, test_case.expected_case_ids
                ),
                case_id_total=len(test_case.expected_case_ids),
                source_count=len(source_documents),
            )

        except ConnectionError:
            elapsed = time.time() - start_time
            logger.error(
                "テストケース '%s' 実行中にOllamaとの接続が切れました (%s)",
                test_case.id,
                config.name,
            )
            return self._create_error_result(
                config,
                test_case,
                "Ollamaサーバーとの接続に失敗しました",
                elapsed=elapsed,
            )
        except Exception:
            elapsed = time.time() - start_time
            logger.exception(
                "テストケース '%s' 実行中に予期しないエラーが発生しました (%s)",
                test_case.id,
                config.name,
            )
            return self._create_error_result(
                config,
                test_case,
                "予期しないエラーが発生しました",
                elapsed=elapsed,
            )

    def _create_error_result(
        self,
        config: BackendConfig,
        test_case: TestCase,
        error_message: str,
        *,
        elapsed: float = 0.0,
    ) -> EvaluationResult:
        """エラー発生時のEvaluationResultを生成する。"""
        return EvaluationResult(
            config_name=config.name,
            test_case_id=test_case.id,
            response_text=f"エラー: {error_message}",
            source_documents=[],
            elapsed_time=elapsed,
            keyword_hits=0,
            keyword_total=len(test_case.expected_keywords),
            statute_hits=0,
            statute_total=len(test_case.expected_statutes),
            case_id_hits=0,
            case_id_total=len(test_case.expected_case_ids),
            source_count=0,
        )

    def _save_results(
        self,
        results: list[EvaluationResult],
        configs: list[BackendConfig],
        test_cases: list[TestCase],
    ) -> None:
        """比較結果を JSON ファイルに保存する。"""
        self._results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{timestamp}.json"
        filepath = self._results_dir / filename

        payload = {
            "timestamp": timestamp,
            "configs": [
                {
                    "name": c.name,
                    "model_name": c.model_name,
                    "temperature": c.temperature,
                    "num_ctx": c.num_ctx,
                    "retrieval_k": c.retrieval_k,
                    "document_type": c.document_type,
                    "case_type": c.case_type,
                    "verdict": c.verdict,
                }
                for c in configs
            ],
            "test_case_ids": [tc.id for tc in test_cases],
            "results": [
                {
                    "config_name": r.config_name,
                    "test_case_id": r.test_case_id,
                    "response_text": r.response_text,
                    "elapsed_time": r.elapsed_time,
                    "keyword_hits": r.keyword_hits,
                    "keyword_total": r.keyword_total,
                    "statute_hits": r.statute_hits,
                    "statute_total": r.statute_total,
                    "case_id_hits": r.case_id_hits,
                    "case_id_total": r.case_id_total,
                    "source_count": r.source_count,
                    "keyword_score": r.keyword_score,
                    "statute_score": r.statute_score,
                    "case_id_score": r.case_id_score,
                    "overall_score": r.overall_score,
                }
                for r in results
            ],
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info("比較結果を保存しました: %s", filepath)
        except OSError as exc:
            logger.error("比較結果の保存に失敗しました: %s", exc)

    # ------------------------------------------------------------------
    # Result Loading
    # ------------------------------------------------------------------

    def load_results(self, filepath: Path) -> list[dict[str, Any]]:
        """保存済みの比較結果 JSON を読み込む。

        Parameters
        ----------
        filepath:
            結果 JSON ファイルのパス。

        Returns
        -------
        list[dict[str, Any]]
            結果レコードのリスト。読み込みに失敗した場合は空リスト。
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning("結果ファイルが見つかりません: %s", filepath)
            return []

        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("results", [])
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("結果ファイルの読み込みに失敗しました: %s", exc)
            return []

    def list_result_files(self) -> list[Path]:
        """保存済み結果ファイルの一覧を新しい順で返す。

        Returns
        -------
        list[Path]
            結果 JSON ファイルのリスト（降順ソート）。
        """
        if not self._results_dir.exists():
            return []
        files = sorted(
            self._results_dir.glob("comparison_*.json"),
            reverse=True,
        )
        return files
