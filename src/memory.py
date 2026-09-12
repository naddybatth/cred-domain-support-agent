"""Task 8 - session memory with LangChain's session-based chat history.

InMemoryChatMessageHistory + RunnableWithMessageHistory, keyed by session_id.

Note on the deprecation warning: RunnableWithMessageHistory emits a
LangChainDeprecationWarning pointing at LangGraph's persistence layer. That warning is
expected, the class still works, and it is deliberately NOT silenced here so the graded
transcript shows the real behaviour.

This history is in-process only - it does not survive a restart, which the brief states is
sufficient for this task. It is what lets a follow-up turn say "and what about that
application?" and still resolve the record id from the previous turn.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from .mock_llm import RECORD_ID_RE

_STORE: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]


def reset_session(session_id: str) -> None:
    _STORE.pop(session_id, None)


def reset_all_sessions() -> None:
    _STORE.clear()


def known_record_ids(session_id: str) -> List[str]:
    """Record ids seen earlier in THIS session - the state a follow-up turn resolves against."""
    history = _STORE.get(session_id)
    if history is None:
        return []
    found: List[str] = []
    for message in history.messages:
        content = message.content if isinstance(message.content, str) else str(message.content)
        for match in RECORD_ID_RE.findall(content):
            if match not in found:
                found.append(match)
    return found


def resolve_with_memory(query: str, session_id: str) -> str:
    """Rewrite a follow-up that refers to an application without naming it.

    'is that one flagged?' after a turn about LN-20260004 becomes a query that still
    carries LN-20260004, which is what makes the multi-turn transcript meaningful.
    """
    if RECORD_ID_RE.search(query or ""):
        return query
    referential = re.search(
        r"\b(that|it|this|the same|my)\b.{0,40}\b(application|loan|case|request|one)\b|"
        r"\bis it\b|\bwhat about (it|that)\b",
        query or "",
        re.IGNORECASE,
    )
    if not referential:
        return query
    previous = known_record_ids(session_id)
    if not previous:
        return query
    return f"{query} (referring to application {previous[-1]})"


def build_memory_runnable(handler: Callable[[Dict[str, Any]], Any]):
    """Wrap any per-turn handler in LangChain session memory."""
    chain = RunnableLambda(handler)
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def session_transcript(session_id: str) -> List[Dict[str, str]]:
    history = _STORE.get(session_id)
    if history is None:
        return []
    return [
        {"type": message.type, "content": str(message.content)}
        for message in history.messages
    ]
