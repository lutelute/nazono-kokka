"""Precedent generation script for 謎の国家.

Generates structured JSON case precedents using Ollama with a Japanese-enhanced
LLM. Precedents span criminal, civil, and constitutional categories and are
saved as individual JSON files with a master metadata index.

Usage:
    python scripts/generate_precedents.py
    python scripts/generate_precedents.py --category criminal
    python scripts/generate_precedents.py --count 100 --batch-size 5
    python scripts/generate_precedents.py --dry-run
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

try:
    import ollama
except ImportError:
    ollama = None

# ---------------------------------------------------------------------------
# Resolve project root so we can import rag_system.config
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from rag_system.config import (
    LLM_FALLBACK_MODEL,
    LLM_MODEL_NAME,
    OLLAMA_BASE_URL,
    PRECEDENTS_CIVIL_DIR,
    PRECEDENTS_CONSTITUTIONAL_DIR,
    PRECEDENTS_CRIMINAL_DIR,
    PRECEDENTS_DIR,
    PRECEDENTS_METADATA_PATH,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category Specifications
# ---------------------------------------------------------------------------

CATEGORY_SPECS: dict[str, dict] = {
    "criminal": {
        "dir": PRECEDENTS_CRIMINAL_DIR,
        "id_prefix": "CRIM",
        "case_types": [
            "窃盗", "強盗", "詐欺", "殺人", "傷害", "暴行",
            "公共秩序妨害", "汚職", "サイバー犯罪", "薬物犯罪",
            "環境犯罪", "器物損壊", "脅迫", "横領", "贈収賄",
            "テロ行為", "組織犯罪", "武器不法所持", "放火", "偽造",
        ],
        "verdicts": ["有罪", "無罪", "一部有罪"],
        "statutes_prefix": "刑法",
        "count_ratio": 0.50,
    },
    "civil": {
        "dir": PRECEDENTS_CIVIL_DIR,
        "id_prefix": "CIVIL",
        "case_types": [
            "契約不履行", "不法行為", "損害賠償", "所有権紛争",
            "相続紛争", "離婚", "親権", "賃貸借紛争", "債務不履行",
            "不当利得", "名誉毀損", "知的財産侵害", "消費者紛争",
            "労働紛争", "担保権実行",
        ],
        "verdicts": ["原告勝訴", "被告勝訴", "一部認容", "和解"],
        "statutes_prefix": "民法",
        "count_ratio": 0.35,
    },
    "constitutional": {
        "dir": PRECEDENTS_CONSTITUTIONAL_DIR,
        "id_prefix": "CONST",
        "case_types": [
            "基本的人権侵害", "表現の自由", "平等権", "信教の自由",
            "財産権", "教育の権利", "法の下の平等", "選挙権",
            "司法審査", "立法権限逸脱",
        ],
        "verdicts": ["合憲", "違憲", "一部違憲", "憲法判断回避"],
        "statutes_prefix": "憲法",
        "count_ratio": 0.15,
    },
}

# ---------------------------------------------------------------------------
# Precedent JSON Schema (template)
# ---------------------------------------------------------------------------

PRECEDENT_SCHEMA_KEYS = [
    "case_id",
    "case_type",
    "title",
    "date",
    "charges",
    "verdict",
    "sentence",
    "legal_principles",
    "summary",
    "reasoning",
    "referenced_statutes",
]

# ---------------------------------------------------------------------------
# Ollama Client Helpers
# ---------------------------------------------------------------------------


def _check_ollama_available() -> bool:
    """Check whether the Ollama server is reachable."""
    if ollama is None:
        logger.error(
            "ollama パッケージがインストールされていません。"
            "  pip install ollama  を実行してください。"
        )
        return False
    try:
        client = _get_client()
        client.list()
        return True
    except Exception as exc:
        logger.error(
            "Ollama サーバーに接続できません (%s)。\n"
            "  ollama serve  を実行してサーバーを起動してください。\n"
            "  詳細: %s",
            OLLAMA_BASE_URL,
            exc,
        )
        return False


def _get_client() -> "ollama.Client":
    """Return an Ollama client configured from project settings."""
    return ollama.Client(host=OLLAMA_BASE_URL)


def _resolve_model(client: "ollama.Client") -> str:
    """Return the best available model name, falling back if needed."""
    try:
        models = client.list()
        available = (
            {m.model for m in models.models}
            if hasattr(models, "models")
            else set()
        )
        if not available:
            available = {m.get("name", "") for m in models.get("models", [])}
    except Exception:
        available = set()

    if LLM_MODEL_NAME in available:
        return LLM_MODEL_NAME
    for name in available:
        if LLM_MODEL_NAME.split(":")[0] in name:
            logger.info("プライマリモデルの部分一致を使用: %s", name)
            return name

    if LLM_FALLBACK_MODEL in available:
        logger.warning(
            "プライマリモデル '%s' が見つかりません。フォールバック '%s' を使用します。",
            LLM_MODEL_NAME,
            LLM_FALLBACK_MODEL,
        )
        return LLM_FALLBACK_MODEL

    for name in available:
        if LLM_FALLBACK_MODEL.split(":")[0] in name:
            logger.warning("フォールバックモデルの部分一致を使用: %s", name)
            return name

    logger.warning(
        "利用可能なモデルが見つかりません。プライマリモデル '%s' を試みます。"
        "  ollama pull %s  でモデルをダウンロードしてください。",
        LLM_MODEL_NAME,
        LLM_MODEL_NAME,
    )
    return LLM_MODEL_NAME


def _generate(client: "ollama.Client", model: str, prompt: str) -> str:
    """Call Ollama generate and return the response text."""
    response = client.generate(model=model, prompt=prompt)
    if isinstance(response, dict):
        return response.get("response", "")
    return getattr(response, "response", str(response))


def _generate_with_retry(
    client: "ollama.Client",
    model: str,
    prompt: str,
    max_retries: int = 3,
) -> str:
    """Call LLM generation with retries on transient failures."""
    for attempt in range(1, max_retries + 1):
        try:
            return _generate(client, model, prompt)
        except Exception as exc:
            if attempt == max_retries:
                logger.error(
                    "生成に失敗しました（%d回試行後）: %s", max_retries, exc
                )
                raise
            wait = 2 ** attempt
            logger.warning(
                "生成エラー（試行 %d/%d）。%d秒後にリトライします: %s",
                attempt,
                max_retries,
                wait,
                exc,
            )
            time.sleep(wait)
    return ""  # unreachable but satisfies type checker


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------


def _build_precedent_prompt(
    category: str,
    case_type: str,
    case_id: str,
    verdict: str,
    statutes_prefix: str,
    year: int,
) -> str:
    """Build a prompt for generating a single case precedent as JSON."""
    return (
        f"あなたは架空の国家「謎の国家」の裁判記録を作成する法務官です。\n"
        f"以下の情報に基づいて、1件の判例をJSON形式で出力してください。\n\n"
        f"カテゴリ: {category}\n"
        f"事件種別: {case_type}\n"
        f"事件番号: {case_id}\n"
        f"判決: {verdict}\n"
        f"年度: {year}\n\n"
        f"以下の正確なJSONスキーマに従ってください。他のテキストは含めないでください。\n"
        f"JSONのみを出力してください。\n\n"
        f'{{\n'
        f'  "case_id": "{case_id}",\n'
        f'  "case_type": "{case_type}",\n'
        f'  "title": "（事件の名称を日本語で。例: 国家 v. 被告人名）",\n'
        f'  "date": "（{year}年のYYYY-MM-DD形式の日付）",\n'
        f'  "charges": ["（{statutes_prefix}の具体的な条文違反を1〜3項目）"],\n'
        f'  "verdict": "{verdict}",\n'
        f'  "sentence": "（具体的な判決内容・刑罰・賠償額など）",\n'
        f'  "legal_principles": ["（この判例で確立された法的原則を2〜4項目）"],\n'
        f'  "summary": "（事案の概要を100〜200文字で）",\n'
        f'  "reasoning": "（裁判所の判断理由を200〜400文字で）",\n'
        f'  "referenced_statutes": ["{statutes_prefix}第X条", "{statutes_prefix}第Y条"]\n'
        f'}}\n'
    )


# ---------------------------------------------------------------------------
# JSON Parsing & Validation
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict | None:
    """Attempt to extract a valid JSON object from LLM output.

    The LLM sometimes wraps JSON in markdown code fences or adds
    surrounding prose.  This function tries several strategies to
    extract a clean JSON dict.
    """
    # Strategy 1: direct parse
    text_stripped = text.strip()
    try:
        obj = json.loads(text_stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Strategy 2: find JSON within code fences
    import re

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Strategy 3: find first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start : brace_end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def _validate_precedent(data: dict) -> bool:
    """Validate that a precedent dict has the required fields."""
    required = {"case_id", "case_type", "title", "date", "verdict", "summary", "reasoning"}
    if not required.issubset(data.keys()):
        missing = required - data.keys()
        logger.warning("判例に必須フィールドが不足: %s", missing)
        return False
    # Ensure list fields are lists
    for key in ("charges", "legal_principles", "referenced_statutes"):
        if key in data and not isinstance(data[key], list):
            data[key] = [data[key]] if data[key] else []
    return True


def _build_fallback_precedent(
    case_id: str,
    category: str,
    case_type: str,
    verdict: str,
    statutes_prefix: str,
    year: int,
) -> dict:
    """Create a minimal valid precedent when LLM generation fails."""
    month = (hash(case_id) % 12) + 1
    day = (hash(case_id) % 28) + 1
    return {
        "case_id": case_id,
        "case_type": case_type,
        "title": f"国家 v. 被告人（{case_type}事件）",
        "date": f"{year}-{month:02d}-{day:02d}",
        "charges": [f"{statutes_prefix}第{(hash(case_id) % 100) + 1}条違反"],
        "verdict": verdict,
        "sentence": "裁判所の判断による",
        "legal_principles": ["法の適正手続", "比例原則"],
        "summary": f"本件は{case_type}に関する事案である。被告人の行為について{statutes_prefix}の関連条文に基づき審理が行われた。",
        "reasoning": f"裁判所は、{statutes_prefix}の関連規定および過去の判例を考慮し、本件における被告人の行為が構成要件に該当するかを慎重に検討した。証拠の評価および法的分析の結果、{verdict}の判断に至った。",
        "referenced_statutes": [
            f"{statutes_prefix}第{(hash(case_id) % 50) + 1}条",
            f"{statutes_prefix}第{(hash(case_id) % 50) + 51}条",
        ],
    }


# ---------------------------------------------------------------------------
# Precedent Generation
# ---------------------------------------------------------------------------


def generate_precedent(
    client: "ollama.Client",
    model: str,
    category: str,
    case_type: str,
    case_id: str,
    verdict: str,
    statutes_prefix: str,
    year: int,
) -> dict:
    """Generate a single case precedent as a structured dict.

    Falls back to a template-based precedent if LLM output cannot
    be parsed as valid JSON.
    """
    prompt = _build_precedent_prompt(
        category, case_type, case_id, verdict, statutes_prefix, year
    )

    try:
        raw = _generate_with_retry(client, model, prompt)
        data = _extract_json(raw)
        if data is not None and _validate_precedent(data):
            # Ensure case_id matches what we requested
            data["case_id"] = case_id
            return data
        logger.warning(
            "判例 %s のJSON解析に失敗。フォールバックを使用します。", case_id
        )
    except Exception as exc:
        logger.warning(
            "判例 %s の生成に失敗: %s。フォールバックを使用します。", case_id, exc
        )

    return _build_fallback_precedent(
        case_id, category, case_type, verdict, statutes_prefix, year
    )


def generate_category(
    client: "ollama.Client",
    model: str,
    category_key: str,
    count: int,
    batch_size: int,
) -> list[dict]:
    """Generate all precedents for a given category.

    Args:
        client: Ollama client instance.
        model: Model name to use.
        category_key: Key into CATEGORY_SPECS.
        count: Number of precedents to generate for this category.
        batch_size: Number of precedents between status log messages.

    Returns:
        List of precedent dicts.
    """
    spec = CATEGORY_SPECS[category_key]
    output_dir: Path = spec["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    case_types = spec["case_types"]
    verdicts = spec["verdicts"]
    statutes_prefix = spec["statutes_prefix"]
    id_prefix = spec["id_prefix"]

    precedents: list[dict] = []

    for i in range(count):
        seq = i + 1
        year = 2020 + (i % 5)
        case_type = case_types[i % len(case_types)]
        verdict = verdicts[i % len(verdicts)]
        case_id = f"{id_prefix}-{year}-{seq:04d}"

        precedent = generate_precedent(
            client, model, category_key, case_type, case_id,
            verdict, statutes_prefix, year,
        )
        precedents.append(precedent)

        # Save individual file
        filepath = output_dir / f"{case_id}.json"
        filepath.write_text(
            json.dumps(precedent, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if seq % batch_size == 0 or seq == count:
            logger.info(
                "[%s] %d/%d 件の判例を生成完了", category_key, seq, count
            )

        # Brief pause between generations
        time.sleep(0.3)

    return precedents


# ---------------------------------------------------------------------------
# Metadata Index
# ---------------------------------------------------------------------------


def build_metadata(all_precedents: dict[str, list[dict]]) -> dict:
    """Build the master metadata index from all generated precedents.

    Args:
        all_precedents: Mapping of category key to list of precedent dicts.

    Returns:
        Metadata dict suitable for writing to metadata.json.
    """
    total = sum(len(v) for v in all_precedents.values())
    entries = []
    for category_key, precedents in all_precedents.items():
        for p in precedents:
            entries.append({
                "case_id": p["case_id"],
                "category": category_key,
                "case_type": p.get("case_type", ""),
                "title": p.get("title", ""),
                "date": p.get("date", ""),
                "verdict": p.get("verdict", ""),
                "file": f"{category_key}/{p['case_id']}.json",
            })

    return {
        "total_count": total,
        "categories": {
            k: len(v) for k, v in all_precedents.items()
        },
        "precedents": entries,
    }


def save_metadata(metadata: dict) -> Path:
    """Write the master metadata index to disk."""
    PRECEDENTS_DIR.mkdir(parents=True, exist_ok=True)
    PRECEDENTS_METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("メタデータ保存完了: %s（計%d件）", PRECEDENTS_METADATA_PATH, metadata["total_count"])
    return PRECEDENTS_METADATA_PATH


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="謎の国家の判例データベースをOllamaを使用して生成するスクリプト",
    )
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_SPECS.keys()),
        default=None,
        help="生成するカテゴリを指定（省略時は全カテゴリを生成）",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="生成する判例の総数（デフォルト: 1000）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="進捗ログを出力する間隔（デフォルト: 10）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="接続確認のみ行い、実際の生成は行わない",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the precedent generation script."""
    args = parse_args()

    logger.info("=== 謎の国家 判例データベース生成スクリプト ===")

    # ---- Pre-flight checks ------------------------------------------------
    if not _check_ollama_available():
        sys.exit(1)

    client = _get_client()
    model = _resolve_model(client)
    logger.info("使用モデル: %s", model)

    if args.dry_run:
        logger.info("ドライラン完了。Ollamaサーバーおよびモデルの確認が完了しました。")
        return

    # ---- Determine categories and counts ----------------------------------
    total_count = args.count

    if args.category:
        categories = {args.category: total_count}
    else:
        categories = {}
        for key, spec in CATEGORY_SPECS.items():
            cat_count = max(1, int(total_count * spec["count_ratio"]))
            categories[key] = cat_count
        # Adjust to match total
        diff = total_count - sum(categories.values())
        if diff > 0:
            first_key = next(iter(categories))
            categories[first_key] += diff

    logger.info(
        "生成計画: 合計 %d 件 (%s)",
        sum(categories.values()),
        ", ".join(f"{k}={v}" for k, v in categories.items()),
    )

    # ---- Generate precedents ----------------------------------------------
    all_precedents: dict[str, list[dict]] = {}
    failed_categories: list[str] = []

    for category_key, count in categories.items():
        logger.info("--- カテゴリ '%s': %d件の判例生成開始 ---", category_key, count)
        try:
            precedents = generate_category(
                client, model, category_key, count, args.batch_size
            )
            all_precedents[category_key] = precedents
        except Exception as exc:
            logger.error(
                "カテゴリ '%s' の生成に失敗しました: %s", category_key, exc
            )
            failed_categories.append(category_key)

    # ---- Build and save metadata ------------------------------------------
    if all_precedents:
        metadata = build_metadata(all_precedents)
        save_metadata(metadata)

    # ---- Summary -----------------------------------------------------------
    total_generated = sum(len(v) for v in all_precedents.values())
    logger.info("=== 生成結果サマリー ===")
    logger.info("成功: %d 件の判例を生成", total_generated)
    for key, precedents in all_precedents.items():
        logger.info("  %s: %d 件", key, len(precedents))

    if failed_categories:
        logger.error(
            "失敗カテゴリ: %s", ", ".join(failed_categories)
        )
        sys.exit(1)
    else:
        logger.info("全カテゴリの判例生成が完了しました。")


if __name__ == "__main__":
    main()
