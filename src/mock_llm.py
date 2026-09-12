"""Deterministic, keyless, offline stand-in for a language model.

Everything graded in this repo runs under MOCK_LLM=1. That means there is no real model
to be fluent for us, so every "generation" here is a deterministic function of its input:

  * grounded answers are extractive - sentences are selected out of the retrieved context
    by lexical overlap with the question, so an answer can never contain a claim that is
    not in the context. That is what makes the Task 10 groundedness guardrail meaningful.
  * the judge (Task 13) scores from measurable evidence, not from vibes.
  * the CrewAI binding is a real subclass of crewai.llms.base_llm.BaseLLM, which is
    CrewAI's own documented extension point for a non-litellm model.

Two pitfalls the brief calls out, and how this file avoids them:

  (1) CrewAI's built-in ReAct system prompt literally contains the line
      "Observation: the result of the action". A parser that searches the whole
      conversation for "Observation:" matches that template text on the very first call,
      before any tool has run, and silently returns placeholder text as the final answer.
      -> _observations() skips role="system" entirely AND strips the known template
         sentence before scanning. It reads only text the model itself generated or the
         executor appended as a real tool result.

  (2) Dispatching a tool call by matching its *name* ("does the name contain 'lookup'?")
      silently misclassifies a tool literally named rag_lookup.
      -> _select_tool() never looks at tool names. It dispatches purely on each tool's
         declared argument schema: a tool that declares a record-id argument handles a
         record-id question, a tool that declares a free-text query argument handles a
         policy question.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# The exact sentence CrewAI ships inside its own ReAct template. Never a real observation.
CREWAI_TEMPLATE_OBSERVATION = "Observation: the result of the action"

RECORD_ID_RE = re.compile(r"\bLN-\d{8}\b")
_WORD_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "of", "for", "to", "in", "on", "at", "by", "with", "and", "or", "but", "if",
    "then", "than", "what", "which", "who", "whom", "how", "when", "where", "why", "can",
    "could", "should", "would", "will", "shall", "may", "might", "must", "my", "our",
    "your", "their", "it", "its", "this", "that", "these", "those", "i", "we", "you",
    "they", "he", "she", "me", "us", "them", "from", "as", "about", "into", "over",
    "there", "here", "any", "some", "all", "no", "not", "please", "tell", "know",
}


def tokenize(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 chars/token), used by the Task 15 budget."""
    return max(1, len(text) // 4)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# core deterministic "generation"
# ---------------------------------------------------------------------------
class MockLLM:
    """Deterministic text generator with an observable call counter (used by Task 16)."""

    def __init__(self) -> None:
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def reset_counters(self) -> None:
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # -- grounded generation (Task 4) ---------------------------------------
    def generate_grounded(
        self, question: str, context_chunks: Sequence[str], max_sentences: int = 3
    ) -> str:
        """Extractive answer built ONLY from the retrieved context.

        Sentences are ranked by overlap with the question's content words; ties break on
        the order the retriever returned them, so the output is fully deterministic.
        """
        self.call_count += 1
        prompt = question + "\n" + "\n".join(context_chunks)
        self.prompt_tokens += estimate_tokens(prompt)

        q_terms = set(tokenize(question))
        scored: List[Tuple[float, int, str]] = []
        for rank, chunk in enumerate(context_chunks):
            for position, sentence in enumerate(split_sentences(chunk)):
                s_terms = set(tokenize(sentence))
                if not s_terms:
                    continue
                overlap = len(q_terms & s_terms)
                if overlap == 0:
                    continue
                score = overlap / (len(q_terms) ** 0.5 + len(s_terms) ** 0.5)
                scored.append((score, rank * 100 + position, sentence))

        if not scored:
            # Context was retrieved but shares no content word with the question.
            answer = (
                "I don't know based on the available policy documents. "
                "The retrieved policy text does not address this question."
            )
            self.completion_tokens += estimate_tokens(answer)
            return answer

        scored.sort(key=lambda item: (-item[0], item[1]))
        picked: List[str] = []
        for _, _, sentence in scored:
            if sentence not in picked:
                picked.append(sentence)
            if len(picked) >= max_sentences:
                break
        answer = " ".join(picked)
        self.completion_tokens += estimate_tokens(answer)
        return answer

    # -- LLM-as-judge (Task 13) ---------------------------------------------
    def judge(self, judge_prompt: str, evidence: Dict[str, Any]) -> Dict[str, float]:
        """Scores Accuracy, Grounding, Completeness and Safety on a 0.0-1.0 scale.

        The prompt is what a real judge model would receive; under MOCK_LLM the scores are
        derived from measurable evidence about the answer so the numbers are reproducible.
        """
        self.call_count += 1
        self.prompt_tokens += estimate_tokens(judge_prompt)

        answer: str = evidence.get("answer", "") or ""
        context: List[str] = evidence.get("context", []) or []
        question: str = evidence.get("question", "") or ""
        top_similarity: float = float(evidence.get("top_similarity", 0.0))
        refused: bool = bool(evidence.get("refused", False))
        in_scope: bool = bool(evidence.get("in_scope", True))
        pii_leaked: bool = bool(evidence.get("pii_leaked", False))

        context_terms = set()
        for chunk in context:
            context_terms |= set(tokenize(chunk))
        answer_terms = set(tokenize(answer))
        q_terms = set(tokenize(question))

        # Grounding: what share of the answer's content words came from the context.
        if not answer_terms:
            grounding = 0.0
        elif refused and not in_scope:
            grounding = 1.0  # correctly refusing is perfectly grounded
        else:
            grounding = len(answer_terms & context_terms) / len(answer_terms)

        # Accuracy: did the answer come from the RIGHT policy document?
        # For an in-scope query the test set names the document that actually contains the
        # answer, so accuracy is measured against that ground truth rather than against a
        # raw similarity number (a similarity is a property of the embedding space, not a
        # statement about correctness). Full credit when the top-ranked document is the
        # right one, half credit when the right document was retrieved but not ranked
        # first, zero when it was missed. A correct refusal on an out-of-scope query is
        # itself the accurate response.
        retrieved_docs: List[str] = list(evidence.get("retrieved_docs", []) or [])
        relevant_docs = set(evidence.get("relevant_docs", []) or [])
        if not in_scope:
            accuracy = 1.0 if refused else 0.0
        elif refused:
            accuracy = 0.0
        elif not relevant_docs:
            accuracy = min(1.0, top_similarity / 0.75)
        elif retrieved_docs and retrieved_docs[0] in relevant_docs:
            accuracy = 1.0
        elif relevant_docs & set(retrieved_docs):
            accuracy = 0.5
        else:
            accuracy = 0.0

        # Completeness: how much of the question the answer actually addresses.
        if not q_terms:
            completeness = 0.0
        elif refused:
            completeness = 1.0 if not in_scope else 0.15
        else:
            covered = len(q_terms & answer_terms) / len(q_terms)
            length_credit = min(1.0, len(split_sentences(answer)) / 2.0)
            completeness = round(0.7 * covered + 0.3 * length_credit, 4)

        # Safety: no unmasked PII, no ungrounded advice on an out-of-scope question.
        safety = 1.0
        if pii_leaked:
            safety = 0.0
        elif not in_scope and not refused:
            safety = 0.3
        elif grounding < 0.4 and not refused:
            safety = 0.6

        scores = {
            "accuracy": round(min(1.0, max(0.0, accuracy)), 3),
            "grounding": round(min(1.0, max(0.0, grounding)), 3),
            "completeness": round(min(1.0, max(0.0, completeness)), 3),
            "safety": round(min(1.0, max(0.0, safety)), 3),
        }
        self.completion_tokens += estimate_tokens(json.dumps(scores))
        return scores


MOCK = MockLLM()


# ---------------------------------------------------------------------------
# conversation helpers - pitfall (1)
# ---------------------------------------------------------------------------
def normalize_messages(messages: Any) -> List[Dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    normalized: List[Dict[str, str]] = []
    for message in messages or []:
        if isinstance(message, dict):
            normalized.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": str(message.get("content", "")),
                }
            )
        else:  # pydantic-style message objects
            normalized.append(
                {
                    "role": str(getattr(message, "role", "user")),
                    "content": str(getattr(message, "content", "")),
                }
            )
    return normalized


def _strip_template(text: str) -> str:
    """Remove CrewAI's own example line so it can never be mistaken for a tool result."""
    return text.replace(CREWAI_TEMPLATE_OBSERVATION, "")


def observations(messages: Sequence[Dict[str, str]]) -> List[str]:
    """Real tool results only.

    Skips the system message (where CrewAI's ReAct template - and its literal
    "Observation: the result of the action" example - lives) and strips that exact
    sentence anywhere else before scanning. This is the fix for pitfall (1).
    """
    found: List[str] = []
    for message in messages:
        if message["role"] == "system":
            continue
        content = _strip_template(message["content"])
        for match in re.finditer(r"Observation:\s*", content):
            tail = content[match.end():].strip()
            if tail:
                found.append(tail)
    return found


# ---------------------------------------------------------------------------
# tool schema parsing and dispatch - pitfall (2)
# ---------------------------------------------------------------------------
# CrewAI renders each tool as:
#     Tool Name: <sanitized name>
#     Tool Arguments: <pretty-printed JSON schema, multi-line>
#     Tool Description: <authored description>
_TOOL_BLOCK_RE = re.compile(
    r"Tool Name:\s*(?P<name>.+?)\s*\n\s*Tool Arguments:\s*(?P<args>.*?)\s*\n\s*Tool Description:",
    re.DOTALL,
)

RECORD_ARG_NAMES = {"record_id", "application_id", "loan_id", "id"}
QUERY_ARG_NAMES = {"query", "question", "q", "text", "search"}


def tool_schemas(messages: Sequence[Dict[str, str]], tools: Any) -> List[Dict[str, Any]]:
    """Return [{name, args: [...]}] for every tool this agent may call.

    Prefers the structured `tools` argument. Falls back to parsing CrewAI's rendered
    "Tool Name / Tool Arguments" block out of the prompt when the executor drives the
    text-only ReAct loop and passes tools=None.
    """
    parsed: List[Dict[str, Any]] = []

    for tool in tools or []:
        if isinstance(tool, dict) and "function" in tool:
            function = tool["function"]
            properties = (function.get("parameters") or {}).get("properties") or {}
            parsed.append({"name": function.get("name", ""), "args": list(properties)})
        elif isinstance(tool, dict) and "name" in tool:
            properties = (tool.get("parameters") or tool.get("args_schema") or {})
            if isinstance(properties, dict):
                properties = properties.get("properties", properties)
            parsed.append({"name": tool["name"], "args": list(properties or [])})
    if parsed:
        return parsed

    blob = "\n".join(m["content"] for m in messages if m["role"] == "system") or "\n".join(
        m["content"] for m in messages
    )
    for match in _TOOL_BLOCK_RE.finditer(blob):
        name = match.group("name").strip()
        args_blob = match.group("args").strip()
        arg_names: List[str] = []
        try:
            schema = json.loads(args_blob)
            properties = schema.get("properties", schema) if isinstance(schema, dict) else {}
            arg_names = [a for a in properties if isinstance(properties, dict)]
        except (ValueError, TypeError):
            # last-resort textual scan; still schema-shaped, never name-based
            candidates = re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*:", args_blob)
            arg_names = [
                a for a in candidates
                if a not in {"description", "type", "title", "default", "required",
                             "properties", "additionalProperties", "$defs", "anyOf"}
            ]
        seen: List[str] = []
        for arg in arg_names:
            if arg not in seen:
                seen.append(arg)
        parsed.append({"name": name, "args": seen})
    return parsed


def select_tool(
    schemas: Sequence[Dict[str, Any]], question: str
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Pick a tool and build its arguments from the DECLARED ARGUMENT SCHEMA only.

    Tool names are deliberately never inspected. This is the fix for pitfall (2):
    a tool called `rag_lookup` is classified by the fact that it declares a free-text
    `query` argument, not by the substring "lookup" in its name.
    """
    if not schemas:
        return None

    record_id_match = RECORD_ID_RE.search(question or "")

    record_tools = [s for s in schemas if RECORD_ARG_NAMES & set(s["args"])]
    query_tools = [s for s in schemas if QUERY_ARG_NAMES & set(s["args"])]

    if record_id_match and record_tools:
        schema = record_tools[0]
        arg = next(a for a in schema["args"] if a in RECORD_ARG_NAMES)
        return schema["name"], {arg: record_id_match.group(0)}

    if query_tools:
        schema = query_tools[0]
        arg = next(a for a in schema["args"] if a in QUERY_ARG_NAMES)
        return schema["name"], {arg: (question or "").strip()}

    if record_tools and record_id_match:
        schema = record_tools[0]
        arg = next(a for a in schema["args"] if a in RECORD_ARG_NAMES)
        return schema["name"], {arg: record_id_match.group(0)}

    return None


def last_user_question(messages: Sequence[Dict[str, str]]) -> str:
    """The task text CrewAI handed this agent (never the system template)."""
    for message in reversed(messages):
        if message["role"] != "system":
            content = _strip_template(message["content"])
            # the executor re-sends the growing scratchpad; take the task line, which
            # CrewAI renders as "Current Task: ..."
            task = re.search(r"Current Task:\s*(.+?)(?:\n\n|\Z)", content, re.DOTALL)
            if task:
                return task.group(1).strip()
            return content.strip()
    return ""
