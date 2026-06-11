"""Judicial reasoning chain for the RAG judicial system.

Uses LangChain RetrievalQA with a custom prompt template to produce
structured judicial decisions.  The chain retrieves relevant legal
documents and precedents from ChromaDB and instructs the LLM to:

1. Cite specific laws by article number
2. Reference relevant precedents by case ID
3. Provide structured legal reasoning
4. Deliver a verdict or recommendation

Usage:
    from rag_system.judge import create_judicial_chain, judge

    chain = create_judicial_chain()
    result = judge(chain, "窃盗罪の量刑基準を示せ")
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from rag_system.config import (
    LLM_FALLBACK_MODEL,
    LLM_MODEL_NAME,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
)
from rag_system.retriever import create_retriever, load_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judicial Prompt Template
# ---------------------------------------------------------------------------

JUDICIAL_PROMPT_TEMPLATE = """あなたは「謎の国家」の最高裁判所の裁判官です。
以下の法的資料と判例に基づいて、提出された法的質問に対して厳密かつ公正な司法判断を下してください。

【参照資料】
{context}

【法的質問】
{question}

【回答指示】
以下の形式に従って、構造化された司法判断を日本語で回答してください。

1. 【事案の概要】
   質問の法的論点を簡潔に整理してください。

2. 【適用法令】
   関連する法律の条文を具体的な条番号とともに引用してください。
   条番号は参照資料に記載されているものを「刑法第42条」「憲法第15条」のような
   表記で正確に転記し、参照資料にない条番号を創作してはいけません。

3. 【関連判例】
   参照資料に含まれる関連判例がある場合、判例番号（case_id）とともに引用してください。
   判例番号は「CRIM-2020-0005」「CIVIL-2021-0123」のような形式で、参照資料に
   記載されているIDを一字一句正確に転記してください。
   関連判例が見つからない場合は、その旨を明記してください。

4. 【法的推論】
   適用法令と関連判例に基づいて、段階的に法的推論を展開してください。
   比例原則、法的安定性、公共の利益等の法原則を考慮してください。

5. 【判断・勧告】
   最終的な司法判断または政策勧告を明確に述べてください。
   量刑が関わる場合は、具体的な刑罰の範囲を示してください。

回答:"""

JUDICIAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=JUDICIAL_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# LLM Initialization
# ---------------------------------------------------------------------------


def _check_ollama_connection(base_url: str) -> bool:
    """Check whether the Ollama server is reachable.

    Parameters
    ----------
    base_url:
        The Ollama server URL to check.

    Returns
    -------
    bool
        ``True`` if the server responds, ``False`` otherwise.
    """
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(base_url, method="GET")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _check_model_available(base_url: str, model_name: str) -> bool:
    """Check whether a specific model is available on the Ollama server.

    Parameters
    ----------
    base_url:
        The Ollama server URL.
    model_name:
        The model name to check.

    Returns
    -------
    bool
        ``True`` if the model is available, ``False`` otherwise.
    """
    import json
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            return any(model_name in m for m in models)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False


def create_llm(
    model_name: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
) -> OllamaLLM:
    """Create an Ollama LLM instance with connection validation.

    Checks that the Ollama server is reachable and the requested model
    is available.  Falls back to :data:`LLM_FALLBACK_MODEL` if the
    primary model is not found.

    Parameters
    ----------
    model_name:
        The Ollama model to use.  Defaults to :data:`LLM_MODEL_NAME`.
    base_url:
        The Ollama server URL.  Defaults to :data:`OLLAMA_BASE_URL`.
    temperature:
        Sampling temperature.  Defaults to :data:`LLM_TEMPERATURE`.
    num_ctx:
        Context window size.  Defaults to :data:`LLM_NUM_CTX`.

    Returns
    -------
    OllamaLLM
        A configured LLM instance.

    Raises
    ------
    ConnectionError
        If the Ollama server is not reachable.
    """
    if base_url is None:
        base_url = OLLAMA_BASE_URL
    if model_name is None:
        model_name = LLM_MODEL_NAME
    if temperature is None:
        temperature = LLM_TEMPERATURE
    if num_ctx is None:
        num_ctx = LLM_NUM_CTX

    # Check Ollama server connectivity
    if not _check_ollama_connection(base_url):
        raise ConnectionError(
            f"Ollamaサーバーに接続できません: {base_url}\n"
            "以下のコマンドでOllamaを起動してください:\n"
            "  ollama serve"
        )
    logger.info("Ollamaサーバーに接続しました: %s", base_url)

    # Check model availability, fall back if needed
    if not _check_model_available(base_url, model_name):
        logger.warning(
            "モデル '%s' が見つかりません。フォールバックモデル '%s' を試行します",
            model_name,
            LLM_FALLBACK_MODEL,
        )
        if _check_model_available(base_url, LLM_FALLBACK_MODEL):
            model_name = LLM_FALLBACK_MODEL
        else:
            logger.warning(
                "フォールバックモデルも見つかりません。"
                "以下のコマンドでモデルをダウンロードしてください:\n"
                "  ollama pull %s",
                LLM_MODEL_NAME,
            )
            # Proceed anyway — Ollama may auto-pull on first use
            model_name = LLM_MODEL_NAME

    logger.info("LLMモデルを使用: %s (temperature=%.2f)", model_name, temperature)

    return OllamaLLM(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
        validate_model_on_init=True,
    )


# ---------------------------------------------------------------------------
# Judicial Chain
# ---------------------------------------------------------------------------


def create_judicial_chain(
    llm: OllamaLLM | None = None,
    retriever: Any | None = None,
    **retriever_kwargs: Any,
) -> RetrievalQA:
    """Create a RetrievalQA chain configured for judicial reasoning.

    Parameters
    ----------
    llm:
        An existing Ollama LLM instance.  If ``None``, one is created
        with default settings.
    retriever:
        An existing LangChain retriever.  If ``None``, one is created
        via :func:`rag_system.retriever.create_retriever`.
    **retriever_kwargs:
        Additional keyword arguments forwarded to
        :func:`~rag_system.retriever.create_retriever` when
        ``retriever`` is ``None`` (e.g., ``k``, ``document_type``).

    Returns
    -------
    RetrievalQA
        A configured chain ready for judicial queries.
    """
    if llm is None:
        llm = create_llm()

    if retriever is None:
        vectorstore = load_vectorstore()
        retriever = create_retriever(vectorstore=vectorstore, **retriever_kwargs)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": JUDICIAL_PROMPT},
    )

    logger.info("司法推論チェーンを作成しました")
    return chain


# ---------------------------------------------------------------------------
# Query Execution
# ---------------------------------------------------------------------------


def judge(
    chain: RetrievalQA,
    query: str,
) -> dict[str, Any]:
    """Execute a judicial query and return the structured result.

    Parameters
    ----------
    chain:
        A configured :class:`RetrievalQA` chain.
    query:
        The legal question to adjudicate.

    Returns
    -------
    dict
        A dictionary containing:

        - ``query`` – the original query string
        - ``result`` – the LLM's judicial reasoning text
        - ``source_documents`` – the retrieved source documents used
    """
    if not query or not query.strip():
        logger.warning("空のクエリが指定されました")
        return {
            "query": query,
            "result": "エラー: 有効な法的質問を入力してください。",
            "source_documents": [],
        }

    logger.info("司法判断を実行: %s", query[:80])

    try:
        result = chain.invoke({"query": query})
    except ConnectionError:
        logger.error("Ollamaサーバーとの接続が切れました")
        return {
            "query": query,
            "result": (
                "エラー: Ollamaサーバーとの接続に失敗しました。\n"
                "サーバーが起動していることを確認してください: ollama serve"
            ),
            "source_documents": [],
        }
    except Exception:
        logger.exception("司法判断中にエラーが発生しました")
        return {
            "query": query,
            "result": "エラー: 司法判断の処理中に予期しないエラーが発生しました。",
            "source_documents": [],
        }

    source_documents = result.get("source_documents", [])
    logger.info(
        "司法判断完了 — 参照資料 %d 件",
        len(source_documents),
    )

    return {
        "query": query,
        "result": result.get("result", ""),
        "source_documents": source_documents,
    }


def format_judgment(result: dict[str, Any]) -> str:
    """Format a judicial result into a human-readable string.

    Parameters
    ----------
    result:
        The dictionary returned by :func:`judge`.

    Returns
    -------
    str
        A formatted string suitable for CLI display.
    """
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("【司法判断】")
    lines.append("=" * 70)
    lines.append("")
    lines.append(result.get("result", ""))
    lines.append("")

    source_docs = result.get("source_documents", [])
    if source_docs:
        lines.append("-" * 70)
        lines.append(f"【参照資料】（{len(source_docs)} 件）")
        lines.append("-" * 70)
        for i, doc in enumerate(source_docs, 1):
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            source = metadata.get("source", "不明")
            doc_type = metadata.get("document_type", "不明")
            case_id = metadata.get("case_id", "")

            label = f"  [{i}] {source}"
            if case_id:
                label += f" (判例: {case_id})"
            label += f" — 種別: {doc_type}"
            lines.append(label)

            # Show a snippet of the content
            content = doc.page_content if hasattr(doc, "page_content") else ""
            snippet = content[:200].replace("\n", " ")
            if len(content) > 200:
                snippet += "..."
            lines.append(f"       {snippet}")
            lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
