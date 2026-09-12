"""Task 14 - Autogen review stage after the CrewAI crew's draft answer.

A 2-agent RoundRobinGroupChat (Policy-Compliance-Reviewer -> Final-Editor) inspects the
crew's draft plus the retrieved context, and either approves it unchanged or revises it.

Three constructor details the brief warns about, all handled here:
  * the bound is `max_turns` - RoundRobinGroupChat has no `max_iterations` parameter;
  * MaxMessageTermination counts the initiating task message as message 1, so the
    equivalent bound is MaxMessageTermination(3), not (2). Both are wired below and the
    termination condition is combined with max_turns=2;
  * the Final-Editor carries output_content_type=ReviewVerdict, which means the TEAM must
    also be constructed with custom_message_types=[StructuredMessage[ReviewVerdict]] or
    the run raises ValueError: Message type ... is not registered.

The model client is a deterministic offline stand-in implementing autogen_core.models.
ChatCompletionClient, so this stage needs no API key and no network.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional, Sequence

from autogen_core import CancellationToken
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from pydantic import BaseModel

from .mock_llm import estimate_tokens, split_sentences, tokenize
from .schemas import ReviewVerdict

REVIEWER_NAME = "Policy_Compliance_Reviewer"
EDITOR_NAME = "Final_Editor"

# A sentence in the draft whose content words are not supported by the retrieved context
# is what the reviewer is looking for.
SUPPORT_RATIO_FLOOR = 0.5


def _message_text(message: LLMMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)


def find_ungrounded_sentences(draft: str, context: Sequence[str]) -> List[str]:
    """Deterministic compliance check: which draft sentences the context does not support."""
    context_terms: set = set()
    for chunk in context:
        context_terms |= set(tokenize(chunk))
    unsupported: List[str] = []
    for sentence in split_sentences(draft):
        terms = set(tokenize(sentence))
        if not terms:
            continue
        support = len(terms & context_terms) / len(terms)
        if support < SUPPORT_RATIO_FLOOR:
            unsupported.append(sentence)
    return unsupported


def strip_ungrounded(draft: str, ungrounded: Sequence[str]) -> str:
    kept = [s for s in split_sentences(draft) if s not in set(ungrounded)]
    if not kept:
        return (
            "I don't know based on the available policy documents. Every claim in the "
            "draft answer was unsupported by the retrieved context, so the draft was "
            "withheld and this question is being escalated to a human agent."
        )
    return " ".join(kept)


class MockReviewClient(ChatCompletionClient):
    """Deterministic offline ChatCompletionClient for the review team."""

    def __init__(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self.call_count = 0

    # -- the two pieces of state the review actually runs on --------------
    def set_case(self, draft: str, context: Sequence[str]) -> None:
        self._draft = draft
        self._context = list(context)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Any = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        self.call_count += 1
        prompt = "\n".join(_message_text(m) for m in messages)
        prompt_tokens = estimate_tokens(prompt)
        self._prompt_tokens += prompt_tokens

        ungrounded = find_ungrounded_sentences(self._draft, self._context)

        # The Final-Editor is the agent given a structured output type, so a request that
        # asks for JSON of a BaseModel is the editor's turn. Everything else is the
        # reviewer's prose turn. No agent-name matching is used.
        wants_structured = isinstance(json_output, type) and issubclass(json_output, BaseModel)

        if wants_structured:
            if ungrounded:
                verdict = ReviewVerdict(
                    approved=False,
                    final_answer=strip_ungrounded(self._draft, ungrounded),
                    reason=(
                        f"Revised. {len(ungrounded)} sentence(s) in the draft were not "
                        "supported by the retrieved policy context and were removed: "
                        + " | ".join(s[:90] for s in ungrounded)
                    ),
                )
            else:
                verdict = ReviewVerdict(
                    approved=True,
                    final_answer=self._draft,
                    reason=(
                        "Approved unchanged. Every sentence in the draft is supported by "
                        "the retrieved policy context and no compliance issue was found."
                    ),
                )
            content = verdict.model_dump_json()
        else:
            if ungrounded:
                content = (
                    f"COMPLIANCE REVIEW: {len(ungrounded)} unsupported claim(s) found. "
                    "The following sentence(s) are not grounded in the retrieved policy "
                    "context and must be removed before this answer reaches a member: "
                    + " | ".join(s[:120] for s in ungrounded)
                )
            else:
                content = (
                    "COMPLIANCE REVIEW: every sentence in the draft traces back to the "
                    "retrieved policy context. No unsupported claim, no unmasked PII, no "
                    "promise of approval or payment. Recommend approval unchanged."
                )

        completion_tokens = estimate_tokens(content)
        self._completion_tokens += completion_tokens
        return CreateResult(
            finish_reason="stop",
            content=content,
            usage=RequestUsage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            ),
            cached=False,
        )

    async def create_stream(  # pragma: no cover - the team uses create()
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Any = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        result = await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield result

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._prompt_tokens, completion_tokens=self._completion_tokens
        )

    def total_usage(self) -> RequestUsage:
        return self.actual_usage()

    def count_tokens(self, messages: Sequence[LLMMessage], **kwargs: Any) -> int:
        return sum(estimate_tokens(_message_text(m)) for m in messages)

    def remaining_tokens(self, messages: Sequence[LLMMessage], **kwargs: Any) -> int:
        return max(0, 8192 - self.count_tokens(messages))

    @property
    def capabilities(self) -> ModelInfo:
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False,
            function_calling=False,
            json_output=True,
            structured_output=True,
            family=ModelFamily.UNKNOWN,
            multiple_system_messages=True,
        )


def build_review_team(client: MockReviewClient):
    reviewer = AssistantAgent(
        name=REVIEWER_NAME,
        model_client=client,
        description="Checks a draft member answer against the retrieved policy context.",
        system_message=(
            "You are a lending-policy compliance reviewer. You read a draft answer and "
            "the policy context it was built from, and you name every sentence that the "
            "context does not support. You never add new policy claims of your own."
        ),
    )
    editor = AssistantAgent(
        name=EDITOR_NAME,
        model_client=client,
        description="Issues the final structured verdict on the draft.",
        system_message=(
            "You are the final editor. Given the draft and the reviewer's findings, you "
            "either approve the draft unchanged or return a revised answer with every "
            "unsupported claim removed. You always return the structured verdict."
        ),
        # structured Pydantic output on the agent
        output_content_type=ReviewVerdict,
    )

    team = RoundRobinGroupChat(
        [reviewer, editor],
        max_turns=2,  # the real parameter name; there is no max_iterations
        # MaxMessageTermination counts the initiating task message as message 1, so 3 is
        # what lets BOTH agents speak. Kept alongside max_turns as a belt-and-braces bound.
        termination_condition=MaxMessageTermination(3),
        # REQUIRED because the editor emits StructuredMessage[ReviewVerdict]; without it
        # the run raises ValueError: Message type ... is not registered.
        custom_message_types=[StructuredMessage[ReviewVerdict]],
    )
    return team


async def review_draft_async(draft: str, context: Sequence[str]) -> Dict[str, Any]:
    client = MockReviewClient()
    client.set_case(draft, context)
    team = build_review_team(client)

    task = (
        "Review the DRAFT ANSWER below against the RETRIEVED POLICY CONTEXT. Remove any "
        "claim the context does not support.\n\n"
        f"DRAFT ANSWER:\n{draft}\n\n"
        "RETRIEVED POLICY CONTEXT:\n" + "\n---\n".join(context)
    )
    result = await team.run(task=task)

    verdict: Optional[ReviewVerdict] = None
    transcript: List[Dict[str, str]] = []
    for message in result.messages:
        source = getattr(message, "source", "task")
        content = getattr(message, "content", "")
        if isinstance(content, ReviewVerdict):
            verdict = content
            transcript.append({"source": source, "content": content.model_dump_json()})
        else:
            transcript.append({"source": source, "content": str(content)})

    if verdict is None:  # never expected; keeps the caller total
        verdict = ReviewVerdict(
            approved=False,
            final_answer=draft,
            reason="Review stage produced no structured verdict; draft held for a human.",
        )

    return {
        "verdict": verdict,
        "transcript": transcript,
        "stop_reason": result.stop_reason,
        "model_calls": client.call_count,
        "usage": {
            "prompt_tokens": client.actual_usage().prompt_tokens,
            "completion_tokens": client.actual_usage().completion_tokens,
        },
    }


def review_draft(draft: str, context: Sequence[str]) -> Dict[str, Any]:
    """Synchronous wrapper so the CLI and the FastAPI paths can both call the review stage.

    If an event loop is already running on this thread (which is the case inside the
    WebSocket endpoint), asyncio.run() would raise, so the coroutine is executed on its
    own loop in a worker thread instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(review_draft_async(draft, context))

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, review_draft_async(draft, context)).result()
