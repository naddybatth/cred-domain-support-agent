"""The one request path every entry point shares: CLI, FastAPI HTTP and the WebSocket.

Order of operations per request, and which task each step belongs to:

  1. runtime token/cost budget check   (Task 15, runtime layer) - rejects before any call
  2. input guardrails: PII mask + injection detect (Task 10)
  3. session memory resolves a referential follow-up (Task 8)
  4. response cache lookup on the normalized query (Task 16)
  5. CrewAI crew: retrieval -> lookup -> composer (Task 7), validated against the
     Pydantic envelope (Task 9)
  6. output groundedness guardrail (Task 10)
  7. Autogen review team approves or revises the draft (Task 14)
  8. one structured JSON-Lines log entry with a trace id (Task 12)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .cache import GENERATION_CACHE
from .config import COLLECTION_SENTENCE, TOP_K, similarity_threshold
from .governance import BUDGET, BudgetExceededError
from .guardrails import apply_input_guardrails, check_groundedness
from .logging_utils import new_trace_id, request_span
from .memory import get_session_history, resolve_with_memory
from .mock_llm import MOCK
from .review import review_draft
from .schemas import SupportAnswer

INJECTION_REFUSAL = (
    "This request contains an instruction that tries to override the agent's rules, so it "
    "was not executed. Loan policy answers and application lookups are available; "
    "instructions to ignore policy, change roles or approve an application are not."
)


def _blocked_answer(reason: str, triggered: List[str]) -> SupportAnswer:
    return SupportAnswer(
        answer=reason,
        answer_type="refusal",
        sources=[],
        refused=True,
        guardrails_triggered=triggered,
        confidence=0.0,
    )


def handle_query(
    query: str,
    session_id: str = "default",
    endpoint: str = "cli",
    use_cache: bool = True,
    run_review: bool = True,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    trace_id = trace_id or new_trace_id()
    result: Dict[str, Any] = {"trace_id": trace_id}

    with request_span(endpoint, query, session_id=session_id, trace_id=trace_id) as span:
        # 1. runtime budget - refuse oversized input before any model or tool call
        try:
            usage = BUDGET.check_request(query)
        except BudgetExceededError as error:
            response = _blocked_answer(str(error), ["runtime_budget_exceeded"])
            span["extra"] = {
                "refused": True,
                "guardrails": response.guardrails_triggered,
                "budget": error.usage,
                "cache_hit": False,
            }
            result.update({"response": response, "budget": error.usage, "cache_hit": False,
                           "review": None})
            return result

        # 2. input guardrails
        guard = apply_input_guardrails(query)
        safe_query = guard.text
        if guard.blocked:
            response = _blocked_answer(INJECTION_REFUSAL, guard.triggered)
            span["extra"] = {
                "refused": True,
                "guardrails": guard.triggered,
                "budget": usage,
                "cache_hit": False,
            }
            result.update({"response": response, "budget": usage, "cache_hit": False,
                           "review": None, "guardrails": guard.triggered})
            return result

        # 3. session memory
        history = get_session_history(session_id)
        resolved_query = resolve_with_memory(safe_query, session_id)

        # 4 + 5. cache around the crew run
        def _run() -> Dict[str, Any]:
            from .crew import run_crew

            return run_crew(resolved_query)

        if use_cache:
            crew_result, cache_hit, elapsed = GENERATION_CACHE.get_or_compute(
                f"{session_id}::{resolved_query}", _run
            )
        else:
            crew_result, cache_hit, elapsed = _run(), False, 0.0

        response: SupportAnswer = crew_result["response"].model_copy(deep=True)
        # the retrieved context for the groundedness check and the review stage
        from .rag import retrieve

        hits = retrieve(resolved_query, collection_name=COLLECTION_SENTENCE, k=TOP_K)
        context = [h.text for h in hits]
        top_similarity = hits[0].similarity if hits else 0.0

        # A record answer is grounded in the SYSTEM OF RECORD, not in policy prose.
        # Judging it only against retrieved policy chunks would mark a perfectly faithful
        # record statement as ungrounded, so the record payload joins the evidence set.
        if response.record_id:
            from .tools import check_loan_application_status

            record = check_loan_application_status(response.record_id)
            if record.get("found"):
                context = context + [
                    f"Record {record['record_id']} is a {record['category']} with status "
                    f"{record['status']} for a sanctioned amount of "
                    f"INR {record['loan_amount_inr']}, raised "
                    f"{record['days_since_created']} days ago, "
                    f"flagged_for_fraud_review={record['flagged_for_fraud_review']}, "
                    f"escalation score {record['escalation_score']} against a threshold of "
                    f"{record['escalation_threshold']}, so it must be routed to a human "
                    f"reviewer: {record['escalate_to_human']}."
                ]

        # 6. output guardrail
        grounded = check_groundedness(
            response.answer,
            context,
            top_similarity,
            similarity_threshold(collection=COLLECTION_SENTENCE),
        )
        if grounded.triggered and response.answer_type not in {"record", "policy_and_record"}:
            response.answer = grounded.text
            response.refused = True
            response.answer_type = "refusal"
            response.guardrails_triggered = list(
                dict.fromkeys(response.guardrails_triggered + grounded.triggered)
            )
            response.confidence = 0.0

        # 7. Autogen review
        review: Optional[Dict[str, Any]] = None
        if run_review and not response.refused:
            review = review_draft(response.answer, context)
            verdict = review["verdict"]
            response.answer = verdict.final_answer
            if not verdict.approved:
                response.guardrails_triggered = list(
                    dict.fromkeys(response.guardrails_triggered + ["autogen_review_revised"])
                )

        # memory: record the turn AFTER the answer is final
        history.add_user_message(resolved_query)
        history.add_ai_message(response.answer)

        span["extra"] = {
            "refused": response.refused,
            "guardrails": response.guardrails_triggered,
            "answer_type": response.answer_type,
            "record_id": response.record_id,
            "cache_hit": cache_hit,
            "top_similarity": top_similarity,
            "budget": usage,
            "llm_calls": MOCK.call_count,
            "review_approved": None if review is None else review["verdict"].approved,
        }

        result.update(
            {
                "response": response,
                "cache_hit": cache_hit,
                "compute_seconds": round(elapsed, 6),
                "context": context,
                "top_similarity": top_similarity,
                "retrieved_docs": list(dict.fromkeys(h.doc_id for h in hits)),
                "budget": usage,
                "review": review,
                "tool_calls": crew_result.get("tool_calls", []),
                "resolved_query": resolved_query,
                "guardrails": guard.triggered,
            }
        )

    result["latency_ms"] = span.get("latency_ms")
    return result
