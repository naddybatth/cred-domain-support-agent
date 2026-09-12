"""Task 7 - the CrewAI crew, assembled under the Task 15 least-autonomy policy."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .crew_llm import JSON_ENVELOPE_SENTINEL, MockCrewLLM
from .governance import POLICY, ToolPermissionError
from .schemas import SupportAnswer
from .tools import build_crew_tools

RETRIEVAL_ROLE = "Policy Retrieval Specialist"
LOOKUP_ROLE = "Loan Application Lookup Specialist"
COMPOSER_ROLE = "Member Response Composer"


def _guarded_agent(agent_class, role: str, tools: List[Any], **kwargs):
    """Every agent is constructed through here, so the policy cannot be bypassed."""
    POLICY.assert_allowed(role, [t.name for t in tools])
    return agent_class(role=role, tools=tools, **kwargs)


def build_crew(verbose: bool = False):
    from crewai import Agent, Crew, Process, Task

    rag_tool, lookup_tool = build_crew_tools()
    llm = MockCrewLLM()

    retrieval_agent = _guarded_agent(
        Agent,
        RETRIEVAL_ROLE,
        [rag_tool],
        goal=(
            "Find the Cred lending-policy text that actually answers the member's "
            "question, and return it with its source documents untouched."
        ),
        backstory=(
            "You are a lending-policy librarian. You never answer from memory. You "
            "return exactly what the knowledge base says, or you say you don't know."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=verbose,
        max_iter=3,
    )

    lookup_agent = _guarded_agent(
        Agent,
        LOOKUP_ROLE,
        [lookup_tool],
        goal=(
            "Look up the referenced loan application and report its status, amount and "
            "escalation score exactly as the system of record returns them."
        ),
        backstory=(
            "You are the only agent in this crew permitted to read application records. "
            "You report what the record says and never infer a status."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=verbose,
        max_iter=3,
    )

    composer_agent = _guarded_agent(
        Agent,
        COMPOSER_ROLE,
        [],  # holds no tools at all - least autonomy
        goal=(
            "Combine the policy evidence and the record evidence into one accurate "
            "member-facing answer in the required JSON envelope."
        ),
        backstory=(
            "You write the reply support staff will read out. You only use evidence the "
            "other two specialists supplied, and you never invent a number."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=verbose,
        max_iter=2,
    )

    return retrieval_agent, lookup_agent, composer_agent, llm


def _tasks_for(query: str, agents, needs_record: bool):
    from crewai import Task

    retrieval_agent, lookup_agent, composer_agent = agents
    tasks = [
        Task(
            description=(
                f"{query}\n\n"
                "Use your policy-search tool on the member question above and return the "
                "tool's payload unchanged."
            ),
            expected_output="The raw POLICY_CONTEXT payload returned by the tool.",
            agent=retrieval_agent,
        )
    ]
    if needs_record:
        tasks.append(
            Task(
                description=(
                    f"{query}\n\n"
                    "Use your record-lookup tool on the application id in the member "
                    "question above and return the tool's payload unchanged."
                ),
                expected_output="The raw RECORD_JSON payload returned by the tool.",
                agent=lookup_agent,
            )
        )
    tasks.append(
        Task(
            description=(
                f"{query}\n\n"
                "Combine the evidence gathered by the other specialists into the final "
                "member-facing answer. Use only that evidence."
            ),
            expected_output=(
                f"{JSON_ENVELOPE_SENTINEL}: return ONLY a JSON object with the keys "
                "answer, answer_type, sources, record_id, application_status, "
                "loan_amount_inr, escalation_score, escalate_to_human, refused, "
                "guardrails_triggered, confidence."
            ),
            agent=composer_agent,
            context=tasks[:],
        )
    )
    return tasks


def run_crew(query: str, verbose: bool = False) -> Dict[str, Any]:
    """Run the crew via .kickoff() and validate the result against SupportAnswer."""
    from crewai import Crew, Process

    from .mock_llm import RECORD_ID_RE

    retrieval_agent, lookup_agent, composer_agent, llm = build_crew(verbose=verbose)
    needs_record = bool(RECORD_ID_RE.search(query or ""))
    tasks = _tasks_for(query, (retrieval_agent, lookup_agent, composer_agent), needs_record)

    crew = Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=verbose,
        memory=False,
    )

    result = crew.kickoff()
    raw = str(result).strip()

    try:
        payload = json.loads(raw)
    except ValueError:
        start, end = raw.find("{"), raw.rfind("}")
        payload = json.loads(raw[start : end + 1]) if start != -1 and end != -1 else {}

    # Task 9 - validate in code, not only by declaring a schema.
    validated = SupportAnswer.model_validate(payload)

    return {
        "query": query,
        "response": validated,
        "raw": raw,
        "tool_calls": [c for c in llm.call_log if c["kind"] == "action"],
        "llm_steps": llm.call_log,
        "used_record_lookup": needs_record,
    }


def demonstrate_least_autonomy() -> Dict[str, Any]:
    """Task 15 - prove the privileged tool cannot be wired to another agent."""
    from crewai import Agent

    rag_tool, lookup_tool = build_crew_tools()
    try:
        _guarded_agent(
            Agent,
            RETRIEVAL_ROLE,
            [rag_tool, lookup_tool],  # deliberate mis-wiring
            goal="try to hold the privileged tool",
            backstory="deliberate violation used as a test",
            llm=MockCrewLLM(),
        )
    except ToolPermissionError as error:
        return {"blocked": True, "error": str(error),
                "allowed_holders": POLICY.agents_allowed_privileged_tool()}
    return {"blocked": False, "error": None,
            "allowed_holders": POLICY.agents_allowed_privileged_tool()}
