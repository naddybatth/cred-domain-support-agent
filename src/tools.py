"""Task 6 - the record-lookup tool with a designed escalation score, plus the RAG tool.

Escalation score (stated exactly, as the brief requires):

    escalation_score = 0.6 * flagged_for_fraud_review + 0.4 * min(days_since_created / 30, 1)

Why these two terms and these weights:
  * flagged_for_fraud_review is a hard compliance signal, so it carries the majority
    weight - a flagged application can never sit below 0.60 no matter how fresh it is.
  * days_since_created is a normalized recency (really staleness) signal. An application
    that has been sitting for weeks is an ageing-SLA problem even when nothing is flagged,
    but it must never on its own outrank a fraud flag, so it is capped at 0.40.

Recommended escalation threshold = 0.55, justified against this dataset's own
distribution (printed by scripts/justify_threshold.py):

    max escalation_score reachable by an UNflagged application : 0.4000
    min escalation_score observed on a flagged application      : 0.7067
    escalation_score p50 / p80 / p90                            : 0.2533 / 0.7067 / 0.8400
    records at or above 0.55                                    : 10/48 = 20.83%

0.55 sits inside the empty band between those two clusters - above anything staleness
alone can produce and below the lowest flagged score actually observed - so it lands
between the p50 and p80 of the score distribution and selects exactly the fraud-flagged
population without ever firing on age alone.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .config import (
    COLLECTION_SENTENCE,
    ESCALATION_THRESHOLD,
    FRAUD_WEIGHT,
    MAX_AGE_DAYS,
    RECENCY_WEIGHT,
    similarity_threshold,
    TOP_K,
)
from .dataset import BY_ID

# Markers let the Composer agent identify each specialist's contribution in its prompt
# without any name matching. See src/crew.py.
POLICY_MARKER = "POLICY_CONTEXT:"
RECORD_MARKER = "RECORD_JSON:"


def compute_escalation_score(flagged: bool, days_since_created: int) -> float:
    recency = min(days_since_created / MAX_AGE_DAYS, 1.0)
    score = FRAUD_WEIGHT * (1.0 if flagged else 0.0) + RECENCY_WEIGHT * recency
    return round(min(1.0, max(0.0, score)), 4)


def check_loan_application_status(record_id: str) -> Dict[str, Any]:
    """Look up one loan application and return its status, amount and escalation score."""
    record = BY_ID.get((record_id or "").strip().upper())
    if record is None:
        return {
            "found": False,
            "record_id": record_id,
            "error": "No loan application with that id exists in the dataset.",
        }
    score = compute_escalation_score(
        record["flagged_for_fraud_review"], record["days_since_created"]
    )
    return {
        "found": True,
        "record_id": record["record_id"],
        "category": record["category"],
        "status": record["status"],
        "loan_amount_inr": record["loan_amount_inr"],
        "days_since_created": record["days_since_created"],
        "flagged_for_fraud_review": record["flagged_for_fraud_review"],
        "escalation_score": score,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "escalate_to_human": score >= ESCALATION_THRESHOLD,
        "formula": (
            f"{FRAUD_WEIGHT} * flagged + {RECENCY_WEIGHT} * "
            f"min(days_since_created/{MAX_AGE_DAYS}, 1)"
        ),
    }


def rag_policy_lookup(query: str) -> Dict[str, Any]:
    """Answer a lending-policy question using only the indexed knowledge base."""
    from .rag import answer  # imported lazily so dataset-only runs need no chroma

    return answer(
        query,
        collection_name=COLLECTION_SENTENCE,
        k=TOP_K,
        threshold=similarity_threshold(collection=COLLECTION_SENTENCE),
    )


# ---------------------------------------------------------------------------
# CrewAI tool wrappers
# ---------------------------------------------------------------------------
def build_crew_tools():
    """Return (rag_tool, lookup_tool) as CrewAI BaseTool instances.

    The RAG tool is deliberately named `rag_lookup`. The brief warns that a dispatcher
    matching the substring "lookup" in a tool name would silently misclassify it; this
    repo's dispatcher (src/mock_llm.select_tool) never reads names at all, so the name is
    kept as a live regression test of that behaviour.
    """
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class RagArgs(BaseModel):
        query: str = Field(description="A natural-language Cred lending-policy question.")

    class LookupArgs(BaseModel):
        record_id: str = Field(description="A loan application id such as LN-20260004.")

    class RagLookupTool(BaseTool):
        name: str = "rag_lookup"
        description: str = (
            "Search the Cred lending-policy knowledge base and return grounded policy "
            "text with its source documents. Use for any question about eligibility, "
            "EMI, fees, KYC, disputes, closure, rates, prepayment, minimum balance, "
            "credit score, joint accounts or NRI accounts."
        )
        args_schema: type[BaseModel] = RagArgs

        def _run(self, query: str) -> str:
            result = rag_policy_lookup(query)
            payload = {
                "answer": result["answer"],
                "refused": result["refused"],
                "top_similarity": result["top_similarity"],
                "sources": [
                    {
                        "doc_id": h["doc_id"],
                        "doc_title": h["doc_title"],
                        "chunk_id": h["chunk_id"],
                        "similarity": h["similarity"],
                    }
                    for h in result["hits"]
                ],
                "context": [h["text"] for h in result["hits"]],
            }
            return f"{POLICY_MARKER} {json.dumps(payload)}"

    class LoanStatusTool(BaseTool):
        name: str = "check_loan_application_status"
        description: str = (
            "Look up one loan application by its record id and return its status, "
            "sanctioned amount and a designed escalation score."
        )
        args_schema: type[BaseModel] = LookupArgs

        def _run(self, record_id: str) -> str:
            return f"{RECORD_MARKER} {json.dumps(check_loan_application_status(record_id))}"

    return RagLookupTool(), LoanStatusTool()


def escalation_distribution() -> List[float]:
    from .dataset import LOAN_APPLICATIONS

    return sorted(
        compute_escalation_score(r["flagged_for_fraud_review"], r["days_since_created"])
        for r in LOAN_APPLICATIONS
    )
