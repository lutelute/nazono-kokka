"""CLI query interface for the RAG judicial system.

Provides both single-query mode (via ``--query``) and an interactive
REPL for submitting legal questions to the judicial reasoning chain.
Handles all edge cases: Ollama connectivity, missing models, empty
ChromaDB, SQLite version compatibility, and empty queries.

Usage:
    # Single query
    python rag_system/main.py --query "窃盗罪の量刑基準を示せ"

    # Interactive mode
    python rag_system/main.py
"""

import argparse
import logging
import os
import sqlite3
import sys

from rag_system.config import (
    CHROMA_DB_PATH,
    LLM_MODEL_NAME,
    OLLAMA_BASE_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimum SQLite version required by ChromaDB
# ---------------------------------------------------------------------------

SQLITE_MIN_VERSION = (3, 35, 0)


# ---------------------------------------------------------------------------
# Pre-flight Checks
# ---------------------------------------------------------------------------


def check_sqlite_version() -> bool:
    """Check that the SQLite version meets ChromaDB's minimum requirement.

    ChromaDB requires SQLite >= 3.35.0 for certain features.

    Returns
    -------
    bool
        ``True`` if the version is sufficient, ``False`` otherwise.
    """
    version_str = sqlite3.sqlite_version
    version_tuple = tuple(int(x) for x in version_str.split("."))

    if version_tuple < SQLITE_MIN_VERSION:
        min_str = ".".join(str(x) for x in SQLITE_MIN_VERSION)
        logger.error(
            "SQLiteバージョンが古すぎます: %s (最低 %s が必要)",
            version_str,
            min_str,
        )
        print(
            f"\nエラー: SQLiteバージョン {version_str} は"
            f"サポートされていません。\n"
            f"ChromaDB には SQLite {min_str} 以上が必要です。\n\n"
            "以下のいずれかの方法でアップグレードしてください:\n"
            "  1. Python を最新版に更新する\n"
            "  2. pysqlite3-binary をインストールする:\n"
            "     pip install pysqlite3-binary\n",
            file=sys.stderr,
        )
        return False

    logger.info("SQLiteバージョン確認: %s (OK)", version_str)
    return True


def check_chromadb_initialized() -> bool:
    """Check whether the ChromaDB vector store has been initialized.

    Returns
    -------
    bool
        ``True`` if the ChromaDB directory exists, ``False`` otherwise.
    """
    if not os.path.isdir(CHROMA_DB_PATH):
        logger.error(
            "ChromaDB ディレクトリが見つかりません: %s", CHROMA_DB_PATH
        )
        print(
            "\nエラー: ベクトルデータベースが初期化されていません。\n"
            "先に以下のコマンドでドキュメントを取り込んでください:\n\n"
            "  python rag_system/ingest.py\n",
            file=sys.stderr,
        )
        return False

    logger.info("ChromaDB ディレクトリ確認: %s (OK)", CHROMA_DB_PATH)
    return True


def check_ollama_server() -> bool:
    """Check whether the Ollama server is reachable.

    Returns
    -------
    bool
        ``True`` if the server responds, ``False`` otherwise.
    """
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(OLLAMA_BASE_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5):
            pass
        logger.info("Ollamaサーバー確認: %s (OK)", OLLAMA_BASE_URL)
        return True
    except (urllib.error.URLError, OSError):
        logger.error(
            "Ollamaサーバーに接続できません: %s", OLLAMA_BASE_URL
        )
        print(
            f"\nエラー: Ollamaサーバーに接続できません ({OLLAMA_BASE_URL})。\n"
            "以下のコマンドでOllamaを起動してください:\n\n"
            "  ollama serve\n",
            file=sys.stderr,
        )
        return False


def check_model_available() -> bool:
    """Check whether the configured LLM model is available on Ollama.

    Prints a warning if the model is not found but does not block
    execution (Ollama may auto-pull the model on first use).

    Returns
    -------
    bool
        ``True`` if the model is available, ``False`` otherwise.
    """
    import json
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags", method="GET"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            if any(LLM_MODEL_NAME in m for m in models):
                logger.info("LLMモデル確認: %s (OK)", LLM_MODEL_NAME)
                return True
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        pass

    logger.warning("モデル '%s' が見つかりません", LLM_MODEL_NAME)
    print(
        f"\n警告: モデル '{LLM_MODEL_NAME}' が見つかりません。\n"
        "以下のコマンドでモデルをダウンロードしてください:\n\n"
        f"  ollama pull {LLM_MODEL_NAME}\n\n"
        "フォールバックモデルで続行を試みます。\n",
        file=sys.stderr,
    )
    return False


def run_preflight_checks() -> bool:
    """Run all pre-flight checks before starting the judicial system.

    Returns
    -------
    bool
        ``True`` if all critical checks pass, ``False`` otherwise.
        Non-critical checks (model availability) issue warnings but
        do not cause failure.
    """
    logger.info("=== 起動前チェックを実行 ===")

    # Critical checks — abort if any fail
    if not check_sqlite_version():
        return False
    if not check_chromadb_initialized():
        return False
    if not check_ollama_server():
        return False

    # Non-critical — warn but continue
    check_model_available()

    logger.info("=== 起動前チェック完了 ===")
    return True


# ---------------------------------------------------------------------------
# Query Execution
# ---------------------------------------------------------------------------


def execute_query(query: str) -> None:
    """Execute a single judicial query and print the formatted result.

    Parameters
    ----------
    query:
        The legal question to adjudicate.
    """
    from rag_system.judge import (
        create_judicial_chain,
        format_judgment,
        judge,
    )

    if not query or not query.strip():
        print(
            "\nエラー: 有効な法的質問を入力してください。\n",
            file=sys.stderr,
        )
        return

    # Truncate extremely long queries to avoid exceeding context window
    max_query_length = 2000
    if len(query) > max_query_length:
        logger.warning(
            "クエリが長すぎるため切り詰めます (%d → %d 文字)",
            len(query),
            max_query_length,
        )
        query = query[:max_query_length]

    try:
        chain = create_judicial_chain()
    except ConnectionError as e:
        print(f"\nエラー: {e}\n", file=sys.stderr)
        return
    except FileNotFoundError as e:
        print(f"\nエラー: {e}\n", file=sys.stderr)
        return
    except Exception as e:
        logger.exception("司法推論チェーンの作成に失敗しました")
        print(
            f"\nエラー: 司法推論チェーンの作成に失敗しました: {e}\n",
            file=sys.stderr,
        )
        return

    print("\n処理中... しばらくお待ちください。\n")

    result = judge(chain, query)
    output = format_judgment(result)
    print(output)


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


def run_interactive() -> None:
    """Run the interactive judicial query REPL.

    Users can type legal questions and receive judicial decisions.
    Type ``quit``, ``exit``, or ``q`` to exit.  Type ``help`` for
    usage information.
    """
    from rag_system.judge import (
        create_judicial_chain,
        format_judgment,
        judge,
    )

    print("=" * 70)
    print("  謎の国家 — 司法判断システム")
    print("  RAG Judicial Decision System")
    print("=" * 70)
    print()
    print("法的質問を入力してください。")
    print("終了するには 'quit', 'exit', または 'q' と入力してください。")
    print("使い方を表示するには 'help' と入力してください。")
    print()

    # Initialize the chain once for reuse across queries
    try:
        chain = create_judicial_chain()
    except ConnectionError as e:
        print(f"\nエラー: {e}\n", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\nエラー: {e}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("司法推論チェーンの作成に失敗しました")
        print(
            f"\nエラー: 司法推論チェーンの作成に失敗しました: {e}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    max_query_length = 2000

    while True:
        try:
            query = input("\n【質問】> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nシステムを終了します。")
            break

        if not query:
            print("エラー: 有効な法的質問を入力してください。")
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\nシステムを終了します。")
            break

        if query.lower() == "help":
            _print_help()
            continue

        # Truncate extremely long queries
        if len(query) > max_query_length:
            logger.warning(
                "クエリが長すぎるため切り詰めます (%d → %d 文字)",
                len(query),
                max_query_length,
            )
            query = query[:max_query_length]
            print(
                f"警告: クエリが長すぎるため {max_query_length} 文字に"
                "切り詰めました。"
            )

        print("\n処理中... しばらくお待ちください。\n")

        result = judge(chain, query)
        output = format_judgment(result)
        print(output)


def _print_help() -> None:
    """Print usage instructions for the interactive REPL."""
    print()
    print("-" * 70)
    print("【使い方】")
    print("-" * 70)
    print()
    print("  法的質問を日本語で入力すると、関連する法令・判例を参照して")
    print("  司法判断を返します。")
    print()
    print("  質問例:")
    print("    - 窃盗罪の量刑基準を示せ")
    print("    - 文化保護政策について勧告せよ")
    print("    - この法律は憲法に違反するか")
    print("    - 契約不履行の損害賠償について")
    print()
    print("  コマンド:")
    print("    help  - この使い方を表示")
    print("    quit  - システムを終了 (exit, q でも可)")
    print()
    print("-" * 70)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the judicial query interface."""
    parser = argparse.ArgumentParser(
        description="謎の国家 — 司法判断システム (RAG Judicial Decision System)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="単発の法的質問を指定（省略時はインタラクティブモードで起動）",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="起動前チェックをスキップ",
    )
    args = parser.parse_args()

    # Run pre-flight checks unless explicitly skipped
    if not args.skip_checks:
        if not run_preflight_checks():
            sys.exit(1)

    if args.query is not None:
        execute_query(args.query)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
