"""LangChain Agent for the RAG judicial system.

Integrates the archive tools (legal framework search, precedent search,
archive stats) into a ReAct-style LangChain agent that can autonomously
select and execute tools to answer legal questions.

Usage:
    from rag_system.agent import create_agent, run_agent

    agent = create_agent()
    result = run_agent(agent, "窃盗罪の量刑基準を示せ")
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from rag_system.config import (
    LLM_MODEL_NAME,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent System Prompt
# ---------------------------------------------------------------------------

AGENT_PROMPT_TEMPLATE = """あなたは「謎の国家」の法務AIアシスタントです。
ユーザーの法的質問に対して、利用可能なツールを活用して正確かつ包括的な回答を提供してください。

利用可能なツール:
{tools}

ツール名一覧: {tool_names}

【回答の手順】
1. 質問の法的論点を分析する
2. 必要に応じて法令検索ツールで関連条文を調べる
3. 必要に応じて判例検索ツールで関連判例を調べる
4. 書庫統計ツールでデータベースの概要を確認することもできる
5. 収集した情報に基づいて、構造化された回答を提供する

【重要な注意事項】
- 法令の条文番号を具体的に引用すること
- 判例がある場合はcase_idとともに引用すること
- 推測ではなく、検索結果に基づいた回答を行うこと

以下の形式で回答してください:

Question: 回答すべき質問
Thought: 次に何をすべきか考える
Action: 使用するツール名
Action Input: ツールへの入力
Observation: ツールの実行結果
... (Thought/Action/Action Input/Observationを必要に応じて繰り返す)
Thought: 最終的な回答を構成できる
Final Answer: 最終的な回答

それでは始めましょう。

Question: {input}
Thought: {agent_scratchpad}"""

AGENT_PROMPT = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    template=AGENT_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Agent Factory
# ---------------------------------------------------------------------------


def _get_tools() -> list[BaseTool]:
    """書庫アクセスツールのリストを取得する。

    Returns
    -------
    list[BaseTool]
        利用可能なツールのリスト。
    """
    from rag_system.tools import (
        archive_stats,
        legal_framework_search,
        precedent_search,
    )

    return [legal_framework_search, precedent_search, archive_stats]


def create_agent(
    *,
    model_name: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
    tools: list[BaseTool] | None = None,
    max_iterations: int = 10,
    handle_parsing_errors: bool = True,
) -> AgentExecutor:
    """ツール統合済みのLangChainエージェントを作成する。

    既存の :func:`~rag_system.judge.create_llm` を使用してLLMインスタンスを
    構築し、書庫アクセスツールをバインドしたReActエージェントを返す。

    Parameters
    ----------
    model_name:
        Ollama モデル名。デフォルトは :data:`~rag_system.config.LLM_MODEL_NAME`。
    temperature:
        サンプリング温度。デフォルトは :data:`~rag_system.config.LLM_TEMPERATURE`。
    num_ctx:
        コンテキストウィンドウサイズ。デフォルトは :data:`~rag_system.config.LLM_NUM_CTX`。
    tools:
        使用するツールのリスト。``None`` の場合、デフォルトの書庫ツール一式を使用。
    max_iterations:
        エージェントの最大反復回数。デフォルトは ``10``。
    handle_parsing_errors:
        パースエラーを自動的に処理するかどうか。デフォルトは ``True``。

    Returns
    -------
    AgentExecutor
        設定済みのエージェントエクゼキューター。

    Raises
    ------
    ConnectionError
        Ollama サーバーに接続できない場合。
    """
    from rag_system.judge import create_llm

    logger.info(
        "エージェントを構築: model='%s', temperature=%s, num_ctx=%s",
        model_name or LLM_MODEL_NAME,
        temperature or LLM_TEMPERATURE,
        num_ctx or LLM_NUM_CTX,
    )

    llm = create_llm(
        model_name=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
    )

    if tools is None:
        tools = _get_tools()

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=AGENT_PROMPT,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=max_iterations,
        handle_parsing_errors=handle_parsing_errors,
        verbose=False,
        return_intermediate_steps=True,
    )

    logger.info(
        "エージェントの構築が完了しました（ツール数: %d）",
        len(tools),
    )
    return executor


# ---------------------------------------------------------------------------
# Query Execution
# ---------------------------------------------------------------------------


def run_agent(
    agent: AgentExecutor,
    query: str,
) -> dict[str, Any]:
    """エージェントを使用して法的質問に回答する。

    Parameters
    ----------
    agent:
        :func:`create_agent` で作成されたエージェントエクゼキューター。
    query:
        ユーザーからの法的質問。

    Returns
    -------
    dict
        以下のキーを含む辞書:

        - ``query`` – 元のクエリ文字列
        - ``result`` – エージェントの最終回答テキスト
        - ``tool_calls`` – 実行されたツール呼び出しのリスト
          （各要素は ``{"tool": str, "input": str, "output": str}``）
    """
    if not query or not query.strip():
        logger.warning("空のクエリが指定されました")
        return {
            "query": query,
            "result": "エラー: 有効な法的質問を入力してください。",
            "tool_calls": [],
        }

    logger.info("エージェント実行: %s", query[:80])

    try:
        result = agent.invoke({"input": query})
    except ConnectionError:
        logger.error("Ollamaサーバーとの接続が切れました")
        return {
            "query": query,
            "result": (
                "エラー: Ollamaサーバーとの接続に失敗しました。\n"
                "サーバーが起動していることを確認してください: ollama serve"
            ),
            "tool_calls": [],
        }
    except Exception:
        logger.exception("エージェント実行中にエラーが発生しました")
        return {
            "query": query,
            "result": "エラー: エージェントの処理中に予期しないエラーが発生しました。",
            "tool_calls": [],
        }

    # 中間ステップからツール呼び出し情報を抽出
    tool_calls: list[dict[str, str]] = []
    intermediate_steps = result.get("intermediate_steps", [])
    for action, observation in intermediate_steps:
        tool_calls.append({
            "tool": action.tool,
            "input": str(action.tool_input),
            "output": str(observation),
        })

    output = result.get("output", "")
    logger.info(
        "エージェント実行完了 — ツール呼び出し %d 回",
        len(tool_calls),
    )

    return {
        "query": query,
        "result": output,
        "tool_calls": tool_calls,
    }
