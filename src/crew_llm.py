"""CrewAI binding for the deterministic mock model.

This subclasses crewai.llms.base_llm.BaseLLM - CrewAI's own documented extension point
for a model that does not go through litellm - rather than trying to intercept calls from
outside the framework.

`supports_function_calling()` returns False, so CrewAI drives its text ReAct loop and this
class produces the `Thought / Action / Action Input` and `Thought / Final Answer` blocks
that crewai.agents.parser.parse() expects.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from crewai.llms.base_llm import BaseLLM

from .mock_llm import (
    MOCK,
    estimate_tokens,
    last_user_question,
    normalize_messages,
    observations,
    select_tool,
    tool_schemas,
)
from .tools import POLICY_MARKER, RECORD_MARKER

# Sentinel placed in the Composer task's expected_output. Presence of this token is how
# the mock knows a task wants the structured JSON envelope rather than prose.
JSON_ENVELOPE_SENTINEL = "RETURN_JSON_SUPPORT_ANSWER"


def _non_system_text(messages: Sequence[Dict[str, str]]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] != "system")


def _extract_marked_json(blob: str, marker: str) -> Optional[Dict[str, Any]]:
    """Pull the JSON payload that follows a marker, wherever it sits in the prompt."""
    index = blob.rfind(marker)
    if index == -1:
        return None
    tail = blob[index + len(marker):].lstrip()
    if not tail.startswith("{"):
        return None
    depth, in_string, escape = 0, False, False
    for position, character in enumerate(tail):
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(tail[: position + 1])
                except ValueError:
                    return None
    return None


def _compose_support_answer(blob: str) -> Dict[str, Any]:
    """Build the Task 9 envelope from the two specialists' marked outputs."""
    policy = _extract_marked_json(blob, POLICY_MARKER) or {}
    record = _extract_marked_json(blob, RECORD_MARKER) or {}

    policy_answer = (policy.get("answer") or "").strip()
    policy_refused = bool(policy.get("refused", False))
    top_similarity = float(policy.get("top_similarity") or 0.0)
    sources = policy.get("sources") or []

    parts: List[str] = []
    if policy_answer and not policy_refused:
        parts.append(policy_answer)

    record_found = bool(record.get("found"))
    if record_found:
        parts.append(
            f"Application {record['record_id']} ({record['category']}) is currently "
            f"'{record['status']}' for a sanctioned amount of "
            f"INR {record['loan_amount_inr']:,}, raised "
            f"{record['days_since_created']} day(s) ago."
        )
        if record.get("escalate_to_human"):
            parts.append(
                f"Escalation score {record['escalation_score']} is at or above the "
                f"{record['escalation_threshold']} threshold, so this application must "
                "be routed to a human reviewer before anything is communicated as final."
            )
    elif record and not record_found:
        parts.append(
            f"No loan application matching {record.get('record_id')} exists in the "
            "records this agent can see."
        )

    if not parts:
        parts.append(policy_answer or "I don't know based on the available policy documents.")

    has_policy = bool(policy_answer) and not policy_refused
    if policy_refused and not record_found:
        answer_type = "refusal"
    elif has_policy and record_found:
        answer_type = "policy_and_record"
    elif record_found:
        answer_type = "record"
    else:
        answer_type = "policy" if has_policy else "refusal"

    confidence = round(min(1.0, max(0.0, top_similarity)), 3)
    if answer_type in {"record", "policy_and_record"}:
        confidence = round(min(1.0, max(confidence, 0.9)), 3)

    return {
        "answer": " ".join(parts).strip(),
        "answer_type": answer_type,
        "sources": sources,
        "record_id": record.get("record_id") if record_found else None,
        "application_status": record.get("status") if record_found else None,
        "loan_amount_inr": record.get("loan_amount_inr") if record_found else None,
        "escalation_score": record.get("escalation_score") if record_found else None,
        "escalate_to_human": bool(record.get("escalate_to_human", False)),
        "refused": answer_type == "refusal",
        "guardrails_triggered": [],
        "confidence": confidence,
    }


class MockCrewLLM(BaseLLM):
    """Deterministic, offline, keyless LLM for CrewAI."""

    llm_type: str = "mock"
    model: str = "mock/deterministic-v1"
    temperature: float | None = 0.0
    provider: str = "mock"
    is_litellm: bool = False

    def __init__(self, **data: Any) -> None:
        data.setdefault("model", "mock/deterministic-v1")
        super().__init__(**data)
        object.__setattr__(self, "call_log", [])

    # CrewAI checks this via getattr; False selects the text ReAct loop.
    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        normalized = normalize_messages(messages)
        blob = _non_system_text(normalized)
        question = last_user_question(normalized)
        seen = observations(normalized)          # pitfall (1): template text can't match
        schemas = tool_schemas(normalized, tools)

        MOCK.call_count += 1
        MOCK.prompt_tokens += estimate_tokens(blob)

        # A task that asks for the structured envelope always answers with JSON.
        if JSON_ENVELOPE_SENTINEL in blob:
            payload = json.dumps(_compose_support_answer(blob))
            self.call_log.append({"kind": "compose", "tools": [s["name"] for s in schemas]})
            MOCK.completion_tokens += estimate_tokens(payload)
            return f"Thought: I now know the final answer\nFinal Answer: {payload}"

        # Tool step - only if this agent holds tools and no real tool result exists yet.
        if schemas and not seen:
            choice = select_tool(schemas, question)   # pitfall (2): schema-based dispatch
            if choice is not None:
                name, arguments = choice
                self.call_log.append({"kind": "action", "tool": name, "args": arguments})
                action = (
                    f"Thought: I need the {name} tool to gather grounded evidence "
                    "before answering.\n"
                    f"Action: {name}\n"
                    f"Action Input: {json.dumps(arguments)}"
                )
                MOCK.completion_tokens += estimate_tokens(action)
                return action

        # Final answer: hand back the real tool result verbatim so the next agent in the
        # crew receives the marked payload rather than a paraphrase of it.
        if seen:
            final = seen[-1].strip()
        else:
            final = (
                "I don't know based on the available policy documents. No tool evidence "
                "was gathered for this task."
            )
        self.call_log.append({"kind": "final", "chars": len(final)})
        MOCK.completion_tokens += estimate_tokens(final)
        return f"Thought: I now know the final answer\nFinal Answer: {final}"
