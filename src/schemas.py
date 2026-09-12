"""Task 9 (and the Task 14 verdict) - Pydantic contracts every response must satisfy."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class RetrievedSource(BaseModel):
    doc_id: str
    doc_title: str
    chunk_id: str
    similarity: float = Field(ge=-1.0, le=1.0)


class SupportAnswer(BaseModel):
    """The response_format every CrewAI crew response is validated against."""

    answer: str = Field(min_length=1, description="Grounded answer shown to the member")
    answer_type: Literal["policy", "record", "policy_and_record", "refusal"]
    sources: List[RetrievedSource] = Field(default_factory=list)
    record_id: Optional[str] = None
    application_status: Optional[str] = None
    loan_amount_inr: Optional[int] = Field(default=None, ge=0)
    escalation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    escalate_to_human: bool = False
    refused: bool = False
    guardrails_triggered: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("record_id")
    @classmethod
    def _record_id_shape(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not value.startswith("LN-"):
            raise ValueError("record_id must look like LN-20260001")
        return value


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="default", min_length=1, max_length=64)


class AskResponse(BaseModel):
    trace_id: str
    latency_ms: float
    response: SupportAnswer


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(min_length=3, max_length=80)
    title: str = Field(min_length=3, max_length=200)
    body: str = Field(min_length=40)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    chunks_indexed_fixed: int
    chunks_indexed_sentence: int


class ReviewVerdict(BaseModel):
    """Task 14 - the Autogen Final-Editor's structured output."""

    approved: bool
    final_answer: str
    reason: str
