"""RAG judicial system for 謎の国家 (Mystery Nation).

Provides LangChain tools and an agent for searching legal frameworks,
precedents, and retrieving archive statistics.

Usage:
    from rag_system import legal_framework_search, precedent_search
    from rag_system import archive_stats, create_agent
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_classic.agents import AgentExecutor
    from langchain_core.tools import BaseTool


def __getattr__(name: str) -> object:
    """Lazy-load public API symbols to avoid hard import errors at startup."""
    _tools_exports = {"legal_framework_search", "precedent_search", "archive_stats"}
    _agent_exports = {"create_agent", "run_agent"}

    if name in _tools_exports:
        from rag_system.tools import (
            archive_stats,
            legal_framework_search,
            precedent_search,
        )

        _mapping = {
            "legal_framework_search": legal_framework_search,
            "precedent_search": precedent_search,
            "archive_stats": archive_stats,
        }
        return _mapping[name]

    if name in _agent_exports:
        from rag_system.agent import create_agent, run_agent

        _mapping = {
            "create_agent": create_agent,
            "run_agent": run_agent,
        }
        return _mapping[name]

    raise AttributeError(f"module 'rag_system' has no attribute {name!r}")


__all__ = [
    "archive_stats",
    "create_agent",
    "legal_framework_search",
    "precedent_search",
    "run_agent",
]
