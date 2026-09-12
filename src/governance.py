"""Task 15 - the four-layer AI-governance model, applied to this system.

Layers, and what this repo actually enforces at each:

  1. Data layer        - only fabricated records ever enter the system; fixed-format PII
                         is masked before the model and before the log (Tasks 10, 12).
  2. Model layer       - MOCK_LLM is deterministic and offline; CrewAI telemetry is
                         disabled so no prompt text leaves the process (see README).
  3. Application layer - PRINCIPLE OF LEAST AUTONOMY. Only the Lookup Agent may call
                         check_loan_application_status. Enforced by ToolAccessPolicy
                         below, which is checked at crew-assembly time, so a mis-wiring
                         raises instead of silently granting a tool.
  4. Runtime layer     - a per-request token/cost budget that rejects an oversized
                         request instead of silently exceeding the cap.

Risk classification: HIGH.
Justification: the scheme places medical data, hiring decisions and financial data in the
High band. This agent reads loan-application records - sanctioned amounts, approval
status and fraud flags - and its answers are read by support staff who relay them to
members. A wrong status or a wrongly reassuring policy answer can cause a member to act
on an incorrect view of a credit decision, and an unmasked PAN or account number in a log
is a reportable data incident. It is squarely financial data, so it is High risk, not the
Medium band that ordinary customer-support ticketing sits in. That classification is why
the system has no payment, approval or disbursal capability at all: the agent can read a
record and recommend escalation, and a human makes every decision that moves money.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

from .config import MAX_REQUEST_TOKENS, MAX_TOTAL_TOKENS, RISK_LEVEL, TOKEN_COST_PER_1K_INR
from .mock_llm import estimate_tokens

RISK_SCHEME = {
    "Low": "summarization, transcription",
    "Medium": "code generation, customer-support tickets",
    "High": "medical data, hiring decisions, financial data",
}


class ToolPermissionError(PermissionError):
    """Raised when an agent is wired to a tool it is not allowed to hold."""


@dataclass(frozen=True)
class ToolAccessPolicy:
    """Application-layer least-autonomy policy: agent role -> tools it may hold."""

    allowed: Dict[str, frozenset] = field(
        default_factory=lambda: {
            "Policy Retrieval Specialist": frozenset({"rag_lookup"}),
            "Loan Application Lookup Specialist": frozenset({"check_loan_application_status"}),
            "Member Response Composer": frozenset(),  # composes only; holds no tools
        }
    )

    # The single privileged tool in this system.
    privileged_tool: str = "check_loan_application_status"

    def assert_allowed(self, role: str, tool_names: Sequence[str]) -> None:
        permitted = self.allowed.get(role)
        if permitted is None:
            raise ToolPermissionError(
                f"Agent role {role!r} is not registered in the tool-access policy. "
                "Register it explicitly before giving it any tool."
            )
        for name in tool_names:
            if name not in permitted:
                raise ToolPermissionError(
                    f"Least-autonomy violation: agent {role!r} may not hold tool "
                    f"{name!r}. Permitted for this role: {sorted(permitted) or 'none'}."
                )

    def agents_allowed_privileged_tool(self) -> List[str]:
        return [
            role for role, tools in self.allowed.items() if self.privileged_tool in tools
        ]


POLICY = ToolAccessPolicy()


# ---------------------------------------------------------------------------
# Runtime layer - per-request token / cost budget
# ---------------------------------------------------------------------------
class BudgetExceededError(RuntimeError):
    def __init__(self, message: str, usage: Dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass
class RequestBudget:
    max_request_tokens: int = MAX_REQUEST_TOKENS
    max_total_tokens: int = MAX_TOTAL_TOKENS

    def check_request(self, text: str) -> Dict[str, Any]:
        """Reject an oversized request BEFORE any model or tool call is made."""
        tokens = estimate_tokens(text or "")
        usage = {
            "request_tokens": tokens,
            "max_request_tokens": self.max_request_tokens,
            "estimated_cost_inr": round(tokens / 1000 * TOKEN_COST_PER_1K_INR, 6),
        }
        if tokens > self.max_request_tokens:
            raise BudgetExceededError(
                f"Request rejected: {tokens} estimated prompt tokens exceeds the "
                f"per-request cap of {self.max_request_tokens}. The request was not "
                "sent to any model or tool.",
                usage,
            )
        return usage

    def check_total(self, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
        total = prompt_tokens + completion_tokens
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
            "max_total_tokens": self.max_total_tokens,
            "estimated_cost_inr": round(total / 1000 * TOKEN_COST_PER_1K_INR, 6),
        }
        if total > self.max_total_tokens:
            raise BudgetExceededError(
                f"Response abandoned: {total} total tokens exceeds the per-request cap "
                f"of {self.max_total_tokens}.",
                usage,
            )
        return usage


BUDGET = RequestBudget()


def risk_classification() -> Dict[str, Any]:
    return {
        "risk_level": RISK_LEVEL,
        "scheme": RISK_SCHEME,
        "justification": (
            "The agent reads loan-application records - sanctioned amount, approval "
            "status and fraud flag - and its output is relayed to members by support "
            "staff. That is financial data, which the scheme places in the High band, "
            "above the Medium band used for ordinary customer-support ticketing. The "
            "consequence of the classification is visible in the architecture: the "
            "system has no approval, disbursal or payment capability, every money "
            "decision stays with a human, and fixed-format PII is masked before it "
            "reaches either the model or the log."
        ),
    }
