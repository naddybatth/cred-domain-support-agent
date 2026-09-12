# Part 4 - Resilience & Governance (Tasks 14-16)

## Task 14 - Autogen RoundRobinGroupChat review stage


### RoundRobinGroupChat(max_turns=2, MaxMessageTermination(3), custom_message_types=[StructuredMessage[ReviewVerdict]])

```
CASE 1 - review APPROVES the draft unchanged
  draft   : A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outstanding principal if closed within the first twelve months.
  verdict : {
  "approved": true,
  "final_answer": "A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outstanding principal if closed within the first twelve months.",
  "reason": "Approved unchanged. Every sentence in the draft is supported by the retrieved policy context and no compliance issue was found."
}
  stop    : Maximum number of messages 3 reached, current message count: 3
  turns   : 3 messages (task message + reviewer + editor)
  transcript:
    [user] Review the DRAFT ANSWER below against the RETRIEVED POLICY CONTEXT. Remove any claim the context does not support.

DRAFT ANSWER:
A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outstanding p
    [Policy_Compliance_Reviewer] COMPLIANCE REVIEW: every sentence in the draft traces back to the retrieved policy context. No unsupported claim, no unmasked PII, no promise of approval or payment. Recommend approval unchanged.
    [Final_Editor] {"approved":true,"final_answer":"A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outstanding principal if closed within the first twelve months.","reason":"Approved unchanged. Every sentence 

CASE 2 - review REVISES the draft (an ungrounded claim was injected on purpose)
  draft   : A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outstanding principal if closed within the first twelve months. Cred will also waive your entire outstanding balance if you simply email support before the weekend.
  verdict : {
  "approved": false,
  "final_answer": "A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outstanding principal if closed within the first twelve months.",
  "reason": "Revised. 1 sentence(s) in the draft were not supported by the retrieved policy context and were removed: Cred will also waive your entire outstanding balance if you simply email support before th"
}
  stop    : Maximum number of messages 3 reached, current message count: 3
  transcript:
    [user] Review the DRAFT ANSWER below against the RETRIEVED POLICY CONTEXT. Remove any claim the context does not support.

DRAFT ANSWER:
A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outstanding p
    [Policy_Compliance_Reviewer] COMPLIANCE REVIEW: 1 unsupported claim(s) found. The following sentence(s) are not grounded in the retrieved policy context and must be removed before this answer reaches a member: Cred will also waive your entire outsta
    [Final_Editor] {"approved":false,"final_answer":"A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outstanding principal if closed within the first twelve months.","reason":"Revised. 1 sentence(s) in the draf
```


## Task 15 - four-layer governance


### governance demonstration

```
APPLICATION LAYER - principle of least autonomy
  agents permitted to call check_loan_application_status: ['Loan Application Lookup Specialist']
  attempt to wire it to the Retrieval Agent was blocked : True
  error: Least-autonomy violation: agent 'Policy Retrieval Specialist' may not hold tool 'check_loan_application_status'. Permitted for this role: ['rag_lookup'].

  How the guard works: every agent in src/crew.py is constructed through
  _guarded_agent(), which calls ToolAccessPolicy.assert_allowed(role, tools)
  before the Agent object exists. The policy maps each agent role to the exact
  set of tool names it may hold, and the Composer maps to the empty set. A
  mis-wiring therefore raises ToolPermissionError at crew-assembly time rather
  than silently granting a second agent read access to application records. The
  check is on construction, not on invocation, so there is no window in which a
  wrongly-wired agent exists and could be kicked off.

RISK CLASSIFICATION
  scheme      : {
                "Low": "summarization, transcription",
                "Medium": "code generation, customer-support tickets",
                "High": "medical data, hiring decisions, financial data"
}
  risk level  : High
  justification:
  The agent reads loan-application records - sanctioned amount, approval status and fraud flag - and its output is relayed to members by support staff. That is financial data, which the scheme places in the High band, above the Medium band used for ordinary customer-support ticketing. The consequence of the classification is visible in the architecture: the system has no approval, disbursal or payment capability, every money decision stays with a human, and fixed-format PII is masked before it reaches either the model or the log.

RUNTIME LAYER - per-request token / cost budget
  normal request accepted : {'request_tokens': 6, 'max_request_tokens': 512, 'estimated_cost_inr': 0.0009}
  oversized request REJECTED: Request rejected: 1590 estimated prompt tokens exceeds the per-request cap of 512. The request was not sent to any model or tool.
  usage: {'request_tokens': 1590, 'max_request_tokens': 512, 'estimated_cost_inr': 0.2385}
  through the full pipeline -> refused=True guardrails=['runtime_budget_exceeded']
  The rejection happens before any model or tool call, so the budget is never
  silently exceeded.
```


## Task 16 - response caching with before/after evidence


### cache hit, before and after

```
query: What is the average monthly balance requirement in a metro branch?

CALL 1 (cold)  cache_hit=False  wall=94.5 ms  model+tool calls made=4
CALL 2 (warm)  cache_hit=True  wall=4.5 ms  model+tool calls made=0

cache stats: {'hits': 1, 'misses': 1, 'saved_llm_or_tool_calls': 1, 'hit_rate': 0.5}
identical answer returned: True

Evidence reading: the warm call added zero crew LLM steps beyond the small fixed
cost of the guardrail and groundedness re-check, and returned in a fraction of the
cold-call wall time, because the whole crew run was served from the cache keyed on
the normalized query text.
```

