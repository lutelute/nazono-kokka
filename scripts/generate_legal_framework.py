"""Legal framework generation script for 謎の国家.

Generates comprehensive legal documents using Ollama with a Japanese-enhanced
LLM. Each document type (constitution, criminal code, civil code, etc.) is
generated in batches of articles and assembled into structured markdown files.

Usage:
    python scripts/generate_legal_framework.py
    python scripts/generate_legal_framework.py --document constitution
    python scripts/generate_legal_framework.py --batch-size 10
"""

import argparse
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
    LEGAL_FRAMEWORK_DIR,
    LLM_FALLBACK_MODEL,
    LLM_MODEL_NAME,
    OLLAMA_BASE_URL,
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
# Document Specifications
# ---------------------------------------------------------------------------

DOCUMENT_SPECS: dict[str, dict] = {
    "constitution": {
        "filename": "constitution.md",
        "title": "謎の国家 憲法",
        "description": "国家の最高法規。前文、基本原則、国民の権利と義務、統治機構、司法、地方自治、財政、緊急事態、最高法規性、改正手続、附則を含む。",
        "chapters": [
            {"name": "前文", "articles": "前文（序文として）"},
            {"name": "総則", "article_start": 1, "article_end": 10},
            {"name": "国民の権利及び義務", "article_start": 11, "article_end": 44},
            {"name": "国民議会", "article_start": 45, "article_end": 55},
            {"name": "内閣", "article_start": 56, "article_end": 62},
            {"name": "叡智の守護者", "article_start": 63, "article_end": 66},
            {"name": "司法", "article_start": 67, "article_end": 78},
            {"name": "地方自治", "article_start": 79, "article_end": 83},
            {"name": "財政", "article_start": 84, "article_end": 91},
            {"name": "緊急事態", "article_start": 92, "article_end": 95},
            {"name": "最高法規", "article_start": 96, "article_end": 98},
            {"name": "改正手続", "article_start": 99, "article_end": 102},
            {"name": "附則", "article_start": 103, "article_end": 112},
        ],
        "min_articles": 100,
    },
    "criminal_code": {
        "filename": "criminal_code.md",
        "title": "謎の国家 刑法",
        "description": "犯罪と刑罰を定める法律。総則（刑罰の種類、犯罪の成立要件、共犯、量刑、時効）、各則（国家に対する罪、公共の秩序に対する罪、個人に対する罪、財産に対する罪、性犯罪、汚職、サイバー犯罪、環境犯罪、テロ、組織犯罪、薬物、武器）、刑事手続を含む。",
        "chapters": [
            {"name": "総則 - 通則", "article_start": 1, "article_end": 5},
            {"name": "総則 - 刑罰", "article_start": 6, "article_end": 15},
            {"name": "総則 - 犯罪の成立", "article_start": 16, "article_end": 30},
            {"name": "総則 - 共犯", "article_start": 31, "article_end": 40},
            {"name": "総則 - 量刑", "article_start": 41, "article_end": 50},
            {"name": "総則 - 時効", "article_start": 51, "article_end": 60},
            {"name": "各則 - 国家に対する罪", "article_start": 61, "article_end": 80},
            {"name": "各則 - 公共の秩序に対する罪", "article_start": 81, "article_end": 100},
            {"name": "各則 - 個人に対する罪", "article_start": 101, "article_end": 130},
            {"name": "各則 - 財産に対する罪", "article_start": 131, "article_end": 160},
            {"name": "各則 - 性犯罪", "article_start": 161, "article_end": 175},
            {"name": "各則 - 汚職", "article_start": 176, "article_end": 190},
            {"name": "各則 - サイバー犯罪", "article_start": 191, "article_end": 210},
            {"name": "各則 - 環境犯罪", "article_start": 211, "article_end": 225},
            {"name": "各則 - テロ・組織犯罪", "article_start": 226, "article_end": 250},
            {"name": "各則 - 薬物・武器", "article_start": 251, "article_end": 270},
            {"name": "刑事手続", "article_start": 271, "article_end": 285},
        ],
        "min_articles": 200,
    },
    "civil_code": {
        "filename": "civil_code.md",
        "title": "謎の国家 民法",
        "description": "私法の一般法。総則（人、法人、物、法律行為、時効）、物権（所有権、用益物権、担保物権）、債権（契約、不法行為）、親族（婚姻、親子）、相続を含む。",
        "chapters": [
            {"name": "総則 - 通則", "article_start": 1, "article_end": 10},
            {"name": "総則 - 人", "article_start": 11, "article_end": 25},
            {"name": "総則 - 法人", "article_start": 26, "article_end": 40},
            {"name": "総則 - 物", "article_start": 41, "article_end": 50},
            {"name": "総則 - 法律行為", "article_start": 51, "article_end": 70},
            {"name": "総則 - 時効", "article_start": 71, "article_end": 80},
            {"name": "物権 - 所有権", "article_start": 81, "article_end": 100},
            {"name": "物権 - 用益物権", "article_start": 101, "article_end": 115},
            {"name": "物権 - 担保物権", "article_start": 116, "article_end": 135},
            {"name": "債権 - 総則", "article_start": 136, "article_end": 155},
            {"name": "債権 - 契約", "article_start": 156, "article_end": 190},
            {"name": "債権 - 不法行為", "article_start": 191, "article_end": 205},
            {"name": "親族", "article_start": 206, "article_end": 235},
            {"name": "相続", "article_start": 236, "article_end": 265},
        ],
        "min_articles": 200,
    },
    "cultural_regulations": {
        "filename": "cultural_regulations.md",
        "title": "謎の国家 文化規範法",
        "description": "謎の国家固有の文化・伝統を保護し発展させるための規範。叡智と神秘の文化、伝統文化、言語・文学、芸術、食文化、建築・景観、教育、知的財産、デジタル文化、文化行政、少数文化、祭礼、罰則を含む。",
        "chapters": [
            {"name": "総則", "article_start": 1, "article_end": 5},
            {"name": "叡智と神秘の文化", "article_start": 6, "article_end": 12},
            {"name": "伝統文化", "article_start": 13, "article_end": 18},
            {"name": "言語・文学", "article_start": 19, "article_end": 24},
            {"name": "芸術", "article_start": 25, "article_end": 30},
            {"name": "食文化", "article_start": 31, "article_end": 36},
            {"name": "建築・景観", "article_start": 37, "article_end": 42},
            {"name": "教育", "article_start": 43, "article_end": 48},
            {"name": "知的財産と文化", "article_start": 49, "article_end": 53},
            {"name": "デジタル文化", "article_start": 54, "article_end": 58},
            {"name": "文化行政", "article_start": 59, "article_end": 62},
            {"name": "少数文化の保護", "article_start": 63, "article_end": 65},
            {"name": "罰則", "article_start": 66, "article_end": 67},
        ],
        "min_articles": 50,
    },
    "ethical_guidelines": {
        "filename": "ethical_guidelines.md",
        "title": "謎の国家 倫理指針",
        "description": "統治と市民のための倫理的枠組み。統治者の倫理、公務員の倫理、市民の倫理、情報・技術倫理、教育・文化倫理、医療・福祉倫理、国際倫理、施行メカニズムを含む。",
        "chapters": [
            {"name": "総則", "article_start": 1, "article_end": 7},
            {"name": "統治倫理", "article_start": 8, "article_end": 18},
            {"name": "公務員倫理", "article_start": 19, "article_end": 28},
            {"name": "市民倫理", "article_start": 29, "article_end": 35},
            {"name": "情報・技術倫理", "article_start": 36, "article_end": 45},
            {"name": "教育・文化倫理", "article_start": 46, "article_end": 52},
            {"name": "医療・福祉倫理", "article_start": 53, "article_end": 60},
            {"name": "国際倫理", "article_start": 61, "article_end": 65},
            {"name": "施行メカニズム", "article_start": 66, "article_end": 70},
        ],
        "min_articles": 50,
    },
    "administrative_code": {
        "filename": "administrative_code.md",
        "title": "謎の国家 行政法",
        "description": "行政手続と政府構造を定める法律。総則、行政組織、行政行為、行政手続、行政立法、行政契約、行政強制、行政救済、情報公開、個人情報保護、電子政府、行政評価、地方行政、国際行政、罰則、附則を含む。",
        "chapters": [
            {"name": "総則", "article_start": 1, "article_end": 8},
            {"name": "行政組織", "article_start": 9, "article_end": 20},
            {"name": "行政行為", "article_start": 21, "article_end": 32},
            {"name": "行政手続", "article_start": 33, "article_end": 45},
            {"name": "行政立法", "article_start": 46, "article_end": 55},
            {"name": "行政契約", "article_start": 56, "article_end": 65},
            {"name": "行政強制", "article_start": 66, "article_end": 75},
            {"name": "行政救済", "article_start": 76, "article_end": 88},
            {"name": "情報公開", "article_start": 89, "article_end": 97},
            {"name": "個人情報保護", "article_start": 98, "article_end": 105},
            {"name": "電子政府", "article_start": 106, "article_end": 113},
            {"name": "行政評価", "article_start": 114, "article_end": 120},
            {"name": "地方行政", "article_start": 121, "article_end": 128},
            {"name": "国際行政", "article_start": 129, "article_end": 133},
            {"name": "罰則", "article_start": 134, "article_end": 138},
            {"name": "附則", "article_start": 139, "article_end": 140},
        ],
        "min_articles": 100,
    },
}

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
        available = {m.model for m in models.models} if hasattr(models, "models") else set()
        if not available:
            available = {m.get("name", "") for m in models.get("models", [])}
    except Exception:
        available = set()

    if LLM_MODEL_NAME in available:
        return LLM_MODEL_NAME
    # Check for partial match (tag may differ)
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


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------


def _build_chapter_prompt(
    doc_title: str,
    doc_description: str,
    chapter_info: dict,
) -> str:
    """Build a prompt for generating a single chapter of a legal document."""
    if "article_start" in chapter_info:
        article_range = (
            f"第{chapter_info['article_start']}条から"
            f"第{chapter_info['article_end']}条"
        )
        instruction = (
            f"以下の章の条文を日本語で作成してください。\n\n"
            f"文書: {doc_title}\n"
            f"文書概要: {doc_description}\n"
            f"章: {chapter_info['name']}\n"
            f"条文範囲: {article_range}\n\n"
            f"要件:\n"
            f"- 各条文は「第X条（見出し）」の形式で記述すること\n"
            f"- 各条文に具体的で詳細な法的内容を含めること\n"
            f"- 謎の国家という架空の国家に適した内容にすること\n"
            f"- Markdown形式で出力すること（章見出しは ### を使用）\n"
            f"- 条文の番号は{article_range}の範囲で連続させること\n\n"
            f"### {chapter_info['name']}\n"
        )
    else:
        instruction = (
            f"以下の文書の前文を日本語で作成してください。\n\n"
            f"文書: {doc_title}\n"
            f"文書概要: {doc_description}\n"
            f"章: {chapter_info['name']}\n\n"
            f"要件:\n"
            f"- 格調高い文体で書くこと\n"
            f"- 謎の国家の理念と建国の精神を含めること\n"
            f"- Markdown形式で出力すること\n\n"
        )
    return instruction


# ---------------------------------------------------------------------------
# Document Generation
# ---------------------------------------------------------------------------


def generate_document(
    client: "ollama.Client",
    model: str,
    doc_key: str,
    batch_size: int,
) -> str:
    """Generate a complete legal document by producing chapters in batches.

    Args:
        client: Ollama client instance.
        model: Model name to use for generation.
        doc_key: Key into DOCUMENT_SPECS.
        batch_size: Maximum number of articles per LLM call.

    Returns:
        Complete markdown document as a string.
    """
    spec = DOCUMENT_SPECS[doc_key]
    sections: list[str] = []

    # Title header
    sections.append(f"# {spec['title']}\n")

    total_chapters = len(spec["chapters"])
    for idx, chapter in enumerate(spec["chapters"], 1):
        logger.info(
            "[%s] 章 %d/%d を生成中: %s",
            doc_key,
            idx,
            total_chapters,
            chapter["name"],
        )

        # For chapters with many articles, split into sub-batches
        if "article_start" in chapter:
            start = chapter["article_start"]
            end = chapter["article_end"]
            total_in_chapter = end - start + 1

            if total_in_chapter > batch_size:
                # Split into sub-batches
                current = start
                while current <= end:
                    batch_end = min(current + batch_size - 1, end)
                    sub_chapter = {
                        "name": chapter["name"],
                        "article_start": current,
                        "article_end": batch_end,
                    }
                    prompt = _build_chapter_prompt(
                        spec["title"], spec["description"], sub_chapter
                    )
                    text = _generate_with_retry(client, model, prompt)
                    sections.append(text)
                    current = batch_end + 1
                    # Brief pause between batches to avoid overloading
                    time.sleep(0.5)
            else:
                prompt = _build_chapter_prompt(
                    spec["title"], spec["description"], chapter
                )
                text = _generate_with_retry(client, model, prompt)
                sections.append(text)
        else:
            # Preamble or special section
            prompt = _build_chapter_prompt(
                spec["title"], spec["description"], chapter
            )
            text = _generate_with_retry(client, model, prompt)
            sections.append(text)

        # Brief pause between chapters
        time.sleep(0.5)

    return "\n\n".join(sections)


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


def save_document(doc_key: str, content: str) -> Path:
    """Save generated content to the appropriate markdown file.

    Args:
        doc_key: Key into DOCUMENT_SPECS.
        content: Markdown content to write.

    Returns:
        Path to the saved file.
    """
    spec = DOCUMENT_SPECS[doc_key]
    LEGAL_FRAMEWORK_DIR.mkdir(parents=True, exist_ok=True)
    filepath = LEGAL_FRAMEWORK_DIR / spec["filename"]
    filepath.write_text(content, encoding="utf-8")
    logger.info("保存完了: %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="謎の国家の法的枠組みをOllamaを使用して生成するスクリプト",
    )
    parser.add_argument(
        "--document",
        choices=list(DOCUMENT_SPECS.keys()),
        default=None,
        help="生成する文書を指定（省略時は全文書を生成）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="1回のLLM呼び出しで生成する最大条文数（デフォルト: 20）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="接続確認のみ行い、実際の生成は行わない",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the legal framework generation script."""
    args = parse_args()

    logger.info("=== 謎の国家 法的枠組み生成スクリプト ===")

    # ---- Pre-flight checks ------------------------------------------------
    if not _check_ollama_available():
        sys.exit(1)

    client = _get_client()
    model = _resolve_model(client)
    logger.info("使用モデル: %s", model)

    if args.dry_run:
        logger.info("ドライラン完了。Ollamaサーバーおよびモデルの確認が完了しました。")
        return

    # ---- Determine which documents to generate ----------------------------
    if args.document:
        doc_keys = [args.document]
    else:
        doc_keys = list(DOCUMENT_SPECS.keys())

    # ---- Generate documents -----------------------------------------------
    generated: list[str] = []
    failed: list[str] = []

    for doc_key in doc_keys:
        logger.info("--- 文書生成開始: %s ---", doc_key)
        try:
            content = generate_document(client, model, doc_key, args.batch_size)
            save_document(doc_key, content)
            generated.append(doc_key)
        except Exception as exc:
            logger.error("文書 '%s' の生成に失敗しました: %s", doc_key, exc)
            failed.append(doc_key)

    # ---- Summary -----------------------------------------------------------
    logger.info("=== 生成結果サマリー ===")
    logger.info("成功: %d 文書 (%s)", len(generated), ", ".join(generated) if generated else "なし")
    if failed:
        logger.error("失敗: %d 文書 (%s)", len(failed), ", ".join(failed))
        sys.exit(1)
    else:
        logger.info("全文書の生成が完了しました。")


if __name__ == "__main__":
    main()
