# Part 3 - Evaluation, Observability & FastAPI Deployment (Tasks 11-13)

## Task 11 - FastAPI: 2 HTTP endpoints + 1 WebSocket


### TestClient exercise of every endpoint

```
GET /health -> 200 {'status': 'ok', 'mock_llm': True, 'embedding_backend': 'sentence-transformers:sentence-transformers/all-MiniLM-L6-v2'}

POST /ask -> 200
{
  "trace_id": "e8725a50bc8a4c16",
  "latency_ms": 43.834,
  "response": {
    "answer": "A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outstanding principal if closed within the first twelve months, 3 percent between months thirteen and twenty-four, and 2 percent thereafter. A floating-rate loan taken by an individual borrower for a non-business purpose carries no foreclosure or part-prepayment charge at any point in the tenure. Business Loans carry a flat foreclosure charge of 4 percent of the outstanding principal unless the closure is funded from the borrower's own verified sources.",
    "answer_type": "policy",
    "sources": [
      {
        "doc_id": "kb08_prepayment_penalty",
        "doc_title": "Prepayment and Foreclosure Penalty Rules",
        "chunk_id": "kb08_prepayment_penalty::sent::0",
        "similarity": 0.7354
      },
      {
        "doc_id": "kb08_prepayment_penalty",
        "doc_title": "Prepayment and Foreclosure Penalty Rules",
        "chunk_id": "kb08_prepayment_penalty::sent::1",
        "similarity": 0.7303
      },
      {
        "doc_id": "kb08_prepayment_penalty",
        "doc_title": "Prepayment and Foreclosure Penalty Rules",
        "chunk_id": "kb08_prepayment_penalty::sent::2",
        "similarity": 0.5938

POST /add-document -> 200 {'trace_id': 'e0d36e2b5c9e465e', 'doc_id': 'kb13_standing_instructions', 'chunks_indexed_fixed': 37, 'chunks_indexed_sentence': 49}
verified the new document is retrievable: A standing instruction may be registered for any recurring debit and is executed on the due date if the balance is sufficient. A failed standing instruction is 
demo document removed after the test: True

WS /ws/chat connect -> ready session=ws-4c912199b1244ab4
WS turn 1 -> record_id=LN-20260004 status=Rejected
WS turn 2 (multi-turn, same socket) -> record_id=LN-20260004
WS client disconnected mid-conversation (context manager exit).
server still serving other clients after the disconnect: GET /health -> 200, POST /ask -> 200
```


## Task 12 - structured JSON-Lines logging with trace ids


### logs/requests.jsonl

```
{"answer_type": "policy", "budget": {"estimated_cost_inr": 0.00225, "max_request_tokens": 512, "request_tokens": 15}, "cache_hit": false, "endpoint": "POST /ask", "guardrails": [], "latency_ms": 43.834, "level": "INFO", "llm_calls": 34, "record_id": null, "refused": false, "request": "What is the foreclosure charge on a fixed rate personal loan?", "review_approved": true, "service": "cred-domain-support-agent", "session_id": "http-demo", "status": "ok", "top_similarity": 0.7354, "trace_id": "e8725a50bc8a4c16", "ts": 1789297912.457}
{"collection_sizes": {"kb_fixed_overlap": 37, "kb_sentence": 49}, "doc_id": "kb13_standing_instructions", "endpoint": "POST /add-document", "latency_ms": 45.891, "level": "INFO", "request": "Standing Instruction Rules", "service": "cred-domain-support-agent", "session_id": "default", "status": "ok", "trace_id": "e0d36e2b5c9e465e", "ts": 1789297912.502}
{"answer_type": "policy", "budget": {"estimated_cost_inr": 0.00225, "max_request_tokens": 512, "request_tokens": 15}, "cache_hit": false, "endpoint": "POST /ask", "guardrails": [], "latency_ms": 46.069, "level": "INFO", "llm_calls": 38, "record_id": null, "refused": false, "request": "What happens if a standing instruction fails on the due date?", "review_approved": true, "service": "cred-domain-support-agent", "session_id": "add-doc-verify", "status": "ok", "top_similarity": 0.7287, "trace_id": "ee52368bf99541bc", "ts": 1789297912.55}
{"answer_type": "record", "budget": {"estimated_cost_inr": 0.00165, "max_request_tokens": 512, "request_tokens": 11}, "cache_hit": false, "endpoint": "WS /ws/chat", "guardrails": [], "latency_ms": 49.89, "level": "INFO", "llm_calls": 43, "record_id": "LN-20260004", "refused": false, "request": "What is the status of application LN-20260004?", "review_approved": true, "service": "cred-domain-support-agent", "session_id": "ws-4c912199b1244ab4", "status": "ok", "top_similarity": 0.2831, "trace_id": "973930e67b954180", "ts": 1789297912.597}
{"answer_type": "record", "budget": {"estimated_cost_inr": 0.00165, "max_request_tokens": 512, "request_tokens": 11}, "cache_hit": false, "endpoint": "WS /ws/chat", "guardrails": [], "latency_ms": 50.205, "level": "INFO", "llm_calls": 48, "record_id": "LN-20260004", "refused": false, "request": "Is that application flagged for fraud review?", "review_approved": true, "service": "cred-domain-support-agent", "session_id": "ws-4c912199b1244ab4", "status": "ok", "top_similarity": 0.3994, "trace_id": "7dd1acbdf3cb4a63", "ts": 1789297912.648}
{"answer_type": "refusal", "budget": {"estimated_cost_inr": 0.0009, "max_request_tokens": 512, "request_tokens": 6}, "cache_hit": false, "endpoint": "POST /ask", "guardrails": ["output_groundedness", "low_retrieval_confidence", "ungrounded_answer"], "latency_ms": 52.668, "level": "INFO", "llm_calls": 51, "record_id": null, "refused": true, "request": "What is the EMI formula?", "review_approved": null, "service": "cred-domain-support-agent", "session_id": "after-disconnect", "status": "ok", "top_similarity": 0.4065, "trace_id": "0e1a1c1f75684bdb", "ts": 1789297912.7}

PII control: the request field is the MASKED text. Proof - a PAN was posted to /ask and the log line below contains [PAN_REDACTED], never the PAN itself.
{
  "answer_type": "refusal",
  "budget": {
    "estimated_cost_inr": 0.00165,
    "max_request_tokens": 512,
    "request_tokens": 11
  },
  "cache_hit": false,
  "endpoint": "POST /ask",
  "guardrails": [
    "output_groundedness",
    "low_retrieval_confidence",
    "ungrounded_answer"
  ],
  "latency_ms": 39.887,
  "level": "INFO",
  "llm_calls": 54,
  "record_id": null,
  "refused": true,
  "request": "My PAN is [PAN_REDACTED] - what is the EMI formula?",
  "review_approved": null,
  "service": "cred-domain-support-agent",
  "session_id": "pii-log-demo",
  "status": "ok",
  "top_similarity": 0.2825,
  "trace_id": "9c81dff436034374",
  "ts": 1789297912.755
}

raw PAN present in log line? False
```


## Task 13 - LLM-as-judge evaluation over 15 queries


### scores per query and the four averages

```
==============================================================================================================
TASK 13 - LLM-AS-JUDGE EVALUATION (MOCK_LLM), 15 queries x 4 properties
==============================================================================================================
id      acc   grnd   cmpl   safe     sim  refused  topic
--------------------------------------------------------------------------------------------------------------
Q01    1.00   1.00   0.80   1.00   0.684    False  loan eligibility criteria by loan type
Q02    1.00   1.00   0.72   1.00   0.602    False  EMI calculation rules
Q03    1.00   1.00   0.77   1.00   0.784    False  credit-card fee structure
Q04    1.00   1.00   0.86   1.00   0.556    False  KYC document requirements
Q05    1.00   1.00   0.80   1.00   0.649    False  fraud-dispute resolution process
Q06    1.00   1.00   0.70   1.00   0.637    False  account-closure process
Q07    1.00   1.00   0.77   1.00   0.654    False  interest-rate slabs
Q08    1.00   1.00   1.00   1.00   0.735    False  prepayment-penalty rules
Q09    1.00   1.00   0.88   1.00   0.752    False  minimum-balance requirements
Q10    1.00   1.00   0.88   1.00   0.724    False  credit-score impact factors
Q11    1.00   1.00   0.93   1.00   0.850    False  joint-account rules
Q12    1.00   1.00   1.00   1.00   0.739    False  NRI-account eligibility
Q13    0.00   0.75   0.45   1.00   0.397    False  edge case - record lookup plus policy
Q14    1.00   1.00   1.00   1.00   0.127     True  OUT OF SCOPE - unrelated domain
Q15    1.00   1.00   1.00   1.00   0.332     True  OUT OF SCOPE - plausible but not in the knowledge ba
--------------------------------------------------------------------------------------------------------------
AVG    0.93   0.98   0.84   1.00

Average accuracy     : 0.9333
Average grounding    : 0.9833
Average completeness : 0.8373
Average safety       : 1.0
```


### full per-query detail (JSON)

```
[
  {
    "id": "Q01",
    "topic": "loan eligibility criteria by loan type",
    "query": "What is the minimum monthly income needed to qualify for a personal loan?",
    "in_scope": true,
    "refused": false,
    "answer_type": "policy",
    "top_similarity": 0.6844,
    "retrieved_docs": [
      "kb01_loan_eligibility",
      "kb07_interest_rate_slabs"
    ],
    "relevant_docs": [
      "kb01_loan_eligibility"
    ],
    "answer": "A Personal Loan applicant must be a salaried or self-employed resident between 21 and 58 years of age with a minimum net monthly income of INR 25,000 and at least six months in the current job. Personal Loan rates run from 10.75 percent to 18.00 percent per annum and are set from the applicant's credit score band, employer category and existing obligations. Auto Loan eligibility is assessed on the on-road price of the vehicle and we fund up to 85 percent of it for new vehicles and 70 percent for used vehicles under five years old.",
    "accuracy": 1.0,
    "grounding": 1.0,
    "completeness": 0.8,
    "safety": 1.0
  },
  {
    "id": "Q02",
    "topic": "EMI calculation rules",
    "query": "How is the EMI calculated on a reducing balance loan?",
    "in_scope": true,
    "refused": false,
    "answer_type": "policy",
    "top_similarity": 0.6017,
    "retrieved_docs": [
      "kb02_emi_calculation",
      "kb09_minimum_balance"
    ],
    "relevant_docs": [
      "kb02_emi_calculation"
    ],
    "answer": "The equated monthly instalment is computed with the standard reducing-balance formula EMI = P x r x (1+r)^n / ((1+r)^n - 1), where P is the sanctioned principal, r is the monthly interest rate (annual rate divided by twelve, expressed as a decimal) and n is the tenure in months. A change in the floating reference rate is absorbed by extending or shortening the tenure by default, and the EMI amount itself is revised only when the applicant asks in writing. A shortfall attracts a charge of 6 percent of the gap between the required and the maintained balance, subject to a floor of INR 150 and a ceiling of INR 600 per month.",
    "accuracy": 1.0,
    "grounding": 1.0,
    "completeness": 0.72,
    "safety": 1.0
  },
  {
    "id": "Q03",
    "topic": "credit-card fee structure",
    "query": "What is the cash advance fee when I withdraw cash on my credit card?",
    "in_scope": true,
    "refused": false,
    "answer_type": "policy",
    "top_similarity": 0.7842,
    "retrieved_docs": [
      "kb03_credit_card_fees",
      "kb06_account_closure"
    ],
    "relevant_docs": [
      "kb03_credit_card_fees"
    ],
    "answer": "Cash withdrawal at an ATM using the card attracts a cash advance fee of 2.5 percent of the withdrawn amount subject to a floor of INR 500, and interest on the withdrawn amount runs from the transaction date with no interest-free period. The joining fee is INR 500 plus applicable taxes and the annual fee of INR 500 is waived for the following year whenever the cardholder spends INR 1,50,000 or more in the preceding twelve billing cycles.",
    "accuracy": 1.0,
    "grounding": 1.0,
    "completeness": 0.767,
    "safety": 1.0
  },
  {
    "id": "Q04",
    "topic": "KYC document requirements",
    "query": "Which documents count as proof of address for KYC?",
    "in_scope": true,
    "refused": false,
    "answer_type": "policy",
    "top_similarity": 0.5563,
    "retrieved_docs": [
      "kb04_kyc_documents"
    ],
    "relevant_docs": [
      "kb04_kyc_documents"
    ],
    "answer": "Accepted address documents are the Aadhaar letter, passport, a utility bill not older than two months, or a registered rent agreement. Re-KYC is required every two years for high-risk customers, every eight years for medium-risk and every ten years for low-risk customers. Self-employed applicants additionally submit their business registration proof and the latest two years of income tax returns.",
    "accuracy": 1.0,
    "grounding": 1.0,
    "completeness": 0.86,
    "safety": 1.0
  },
  {
    "id": "Q05",
    "topic": "fraud-dispute resolution process",
    "query": "How long do I have to report an unauthorised transaction to have zero liability?",
    "in_scope": true,
    "refused": false,
    "answer_type": "policy",
    "top_similarity": 0.6495,
    "retrieved_docs": [
      "kb05_fraud_dispute"
    ],
    "relevant_docs": [
      "kb05_fraud_dispute"
    ],
    "answer": "A customer must report an unauthorised transaction through the in-app dispute form, the 24x7 helpline or a written complaint, and the reporting timestamp is what governs liability. Where the delay in reporting is four to seven working days, the customer's liability is capped at the transaction value or a slab limit, whichever is lower. Where the loss arises from a deficiency on our side or from a third-party breach that the customer reported within three working days, the customer bears zero liability and the disputed amount is credited on a provisional basis within ten working days.",
    "accuracy": 1.0,
    "grounding": 1.0,
    "completeness": 0.8,
    "safety": 1.0
  },
  {
    "id": "Q06",
    "topic": "account-closure process",
    "query": "Is there a charge for closing my savings account after eighteen months?",
    "in_scope": true,
    "refused": false,
    "answer_type": "policy",
    "top_similarity": 0.6375,
    "retrieved_docs": [
      "kb06_account_closure",
      "kb09_minimum_balance"
    ],
    "relevant_docs": [
      "kb06_account_closure"
    ],
    "answer": "No closure charge applies if the account has been open for more than twelve months; accounts closed between fourteen days and twelve months attract a closure charge of INR 500 plus taxes, and closure within the first fourteen days is free. Basic savings bank deposit accounts, salary accounts with a credit in the last three months, and accounts of minors and senior citizens above 70 are exempt from the requirement. An account closure request must be signed by all account holders and submitted wit
```

