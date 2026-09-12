"""Task 10 - input-side PII masking and prompt-injection detection, output-side grounding.

Scope note, taken straight from the brief: PAN, Aadhaar and bank-account numbers are
fixed-format and ARE masked here. Applicant names and income figures are free text with
no reliable pattern to match under a keyless MOCK_LLM-only masker, and are explicitly out
of scope for masking - we simply never put real ones into the system.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .mock_llm import tokenize

# --- fixed-format PII ------------------------------------------------------
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
AADHAAR_RE = re.compile(r"\b(?!0|1)\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
ACCOUNT_RE = re.compile(r"\b(?:a/?c|acct|account)\s*(?:no\.?|number|#)?\s*[:\-]?\s*(\d{9,18})\b",
                        re.IGNORECASE)
BARE_ACCOUNT_RE = re.compile(r"\b\d{11,18}\b")

# --- prompt injection ------------------------------------------------------
INJECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("override_instructions", re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all)\b"
        r".{0,20}\b(instruction|instructions|prompt|rule|rules|context)\b", re.IGNORECASE)),
    ("role_hijack", re.compile(
        r"\byou are now\b|\bact as\b.{0,30}\b(admin|developer|root|system)\b|"
        r"\bdeveloper mode\b|\bjailbreak\b", re.IGNORECASE)),
    ("system_prompt_exfiltration", re.compile(
        r"\b(reveal|print|show|repeat|output|dump)\b.{0,30}"
        r"\b(system prompt|your instructions|your prompt|initial prompt)\b", re.IGNORECASE)),
    ("privilege_escalation", re.compile(
        r"\b(approve|disburse|sanction|release)\b.{0,40}\b(without|bypass|skip)\b"
        r".{0,30}\b(review|check|verification|approval|human)\b", re.IGNORECASE)),
    ("tool_coercion", re.compile(
        r"\b(call|invoke|run|execute)\b.{0,25}\b(tool|function|endpoint)\b.{0,25}"
        r"\b(directly|yourself|as admin)\b", re.IGNORECASE)),
]


@dataclass
class GuardrailResult:
    text: str
    triggered: List[str] = field(default_factory=list)
    detail: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return "prompt_injection" in self.triggered


def mask_pii(text: str) -> GuardrailResult:
    """Redact every fixed-format PII field. Returns the masked text plus what fired."""
    triggered: List[str] = []
    detail: Dict[str, List[str]] = {}
    masked = text

    pans = PAN_RE.findall(masked)
    if pans:
        masked = PAN_RE.sub("[PAN_REDACTED]", masked)
        triggered.append("pii_pan")
        detail["pan"] = pans

    # account-number-with-label first, so "account number 123456789012" is not
    # swallowed by the Aadhaar pattern
    accounts = [m.group(1) for m in ACCOUNT_RE.finditer(masked)]
    if accounts:
        masked = ACCOUNT_RE.sub("account number [ACCOUNT_REDACTED]", masked)
        triggered.append("pii_bank_account")
        detail["bank_account"] = accounts

    aadhaars = ["".join(m.group(0).split()) for m in AADHAAR_RE.finditer(masked)]
    if aadhaars:
        masked = AADHAAR_RE.sub("[AADHAAR_REDACTED]", masked)
        triggered.append("pii_aadhaar")
        detail["aadhaar"] = aadhaars

    leftovers = BARE_ACCOUNT_RE.findall(masked)
    if leftovers:
        masked = BARE_ACCOUNT_RE.sub("[ACCOUNT_REDACTED]", masked)
        if "pii_bank_account" not in triggered:
            triggered.append("pii_bank_account")
        detail.setdefault("bank_account", []).extend(leftovers)

    return GuardrailResult(text=masked, triggered=triggered, detail=detail)


def detect_injection(text: str) -> GuardrailResult:
    hits = [name for name, pattern in INJECTION_PATTERNS if pattern.search(text)]
    triggered = ["prompt_injection"] if hits else []
    return GuardrailResult(text=text, triggered=triggered, detail={"patterns": hits})


def apply_input_guardrails(text: str) -> GuardrailResult:
    """Runs both input-side guardrails. The masked text is what the model AND the log see."""
    pii = mask_pii(text)
    injection = detect_injection(pii.text)
    return GuardrailResult(
        text=pii.text,
        triggered=pii.triggered + injection.triggered,
        detail={**pii.detail, **injection.detail},
    )


def check_groundedness(
    answer: str, context_chunks: Sequence[str], top_similarity: float, threshold: float
) -> GuardrailResult:
    """Output-side guardrail: refuse when the retrieved context does not support the answer.

    Two independent conditions, either of which refuses:
      1. retrieval confidence below the calibrated threshold
      2. less than half the answer's content words appear in the retrieved context
    """
    triggered: List[str] = []
    if top_similarity < threshold:
        triggered.append("low_retrieval_confidence")

    answer_terms = set(tokenize(answer))
    context_terms: set = set()
    for chunk in context_chunks:
        context_terms |= set(tokenize(chunk))
    overlap = len(answer_terms & context_terms) / len(answer_terms) if answer_terms else 0.0
    if answer_terms and overlap < 0.5:
        triggered.append("ungrounded_answer")

    if triggered:
        return GuardrailResult(
            text=(
                "I don't know based on the available policy documents. The retrieved "
                "context does not support an answer to this question, so I am escalating "
                "it to a human agent rather than guessing."
            ),
            triggered=["output_groundedness"] + triggered,
            detail={"context_overlap": [f"{overlap:.3f}"],
                    "top_similarity": [f"{top_similarity:.4f}"]},
        )
    return GuardrailResult(text=answer, triggered=[], detail={"context_overlap": [f"{overlap:.3f}"]})
