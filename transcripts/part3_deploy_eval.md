# Part 3 - Evaluation, Observability & FastAPI Deployment (Tasks 11-13)

## Task 11 - FastAPI: 2 HTTP endpoints + 1 WebSocket


### TestClient exercise of every endpoint

```
GET /health -> 200 {'status': 'ok', 'mock_llm': True, 'embedding_backend': 'hashed-tfidf-4096d (offline fallback)'}

POST /ask -> 200
{
  "trace_id": "2543e640edbe4fb9",
  "latency_ms": 110.385,
  "response": {
    "answer": "A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outstanding principal if closed within the first twelve months, 3 percent between months thirteen and twenty-four, and 2 percent thereafter. A floating-rate loan taken by an individual borrower for a non-business purpose carries no foreclosure or part-prepayment charge at any point in the tenure. Business Loans carry a flat foreclosure charge of 4 percent of the outstanding principal unless the closure is funded from the borrower's own verified sources.",
    "answer_type": "policy",
    "sources": [
      {
        "doc_id": "kb08_prepayment_penalty",
        "doc_title": "Prepayment and Foreclosure Penalty Rules",
        "chunk_id": "kb08_prepayment_penalty::sent::0",
        "similarity": 0.481
      },
      {
        "doc_id": "kb08_prepayment_penalty",
        "doc_title": "Prepayment and Foreclosure Penalty Rules",
        "chunk_id": "kb08_prepayment_penalty::sent::1",
        "similarity": 0.4064
      },
      {
        "doc_id": "kb08_prepayment_penalty",
        "doc_title": "Prepayment and Foreclosure Penalty Rules",
        "chunk_id": "kb08_prepayment_penalty::sent::2",
        "similarity": 0.247


POST /add-document -> 200 {'trace_id': 'd777aab43fb4424a', 'doc_id': 'kb13_standing_instructions', 'chunks_indexed_fixed': 37, 'chunks_indexed_sentence': 49}

WS /ws/chat connect -> ready session=ws-41fd9929c10449ad
WS turn 1 -> record_id=LN-20260004 status=Rejected
WS turn 2 (multi-turn, same socket) -> record_id=LN-20260004
WS client disconnected mid-conversation (context manager exit).
server still serving other clients after the disconnect: GET /health -> 200, POST /ask -> 200
```


## Task 12 - structured JSON-Lines logging with trace ids


### logs/requests.jsonl

```
{"answer_type": "refusal", "budget": {"estimated_cost_inr": 0.00165, "max_request_tokens": 512, "request_tokens": 11}, "cache_hit": false, "endpoint": "guard-demo", "guardrails": ["output_groundedness", "low_retrieval_confidence", "ungrounded_answer"], "latency_ms": 115.177, "level": "INFO", "llm_calls": 30, "record_id": null, "refused": true, "request": "What is the best recipe for hyderabadi biryani?", "review_approved": null, "service": "cred-domain-support-agent", "session_id": "guard", "status": "ok", "top_similarity": 0.041, "trace_id": "0f816cd8fdb84df0", "ts": 1789227750.778}
{"answer_type": "policy", "budget": {"estimated_cost_inr": 0.00225, "max_request_tokens": 512, "request_tokens": 15}, "cache_hit": false, "endpoint": "POST /ask", "guardrails": [], "latency_ms": 110.385, "level": "INFO", "llm_calls": 34, "record_id": null, "refused": false, "request": "What is the foreclosure charge on a fixed rate personal loan?", "review_approved": true, "service": "cred-domain-support-agent", "session_id": "http-demo", "status": "ok", "top_similarity": 0.481, "trace_id": "2543e640edbe4fb9", "ts": 1789227750.918}
{"collection_sizes": {"kb_fixed_overlap": 37, "kb_sentence": 49}, "doc_id": "kb13_standing_instructions", "endpoint": "POST /add-document", "latency_ms": 165.285, "level": "INFO", "request": "Standing Instruction Rules", "service": "cred-domain-support-agent", "session_id": "default", "status": "ok", "trace_id": "d777aab43fb4424a", "ts": 1789227751.033}
{"answer_type": "record", "budget": {"estimated_cost_inr": 0.00165, "max_request_tokens": 512, "request_tokens": 11}, "cache_hit": false, "endpoint": "WS /ws/chat", "guardrails": [], "latency_ms": 143.383, "level": "INFO", "llm_calls": 39, "record_id": "LN-20260004", "refused": false, "request": "What is the status of application LN-20260004?", "review_approved": true, "service": "cred-domain-support-agent", "session_id": "ws-41fd9929c10449ad", "status": "ok", "top_similarity": 0.0529, "trace_id": "92ede53354ab4613", "ts": 1789227751.206}
{"answer_type": "record", "budget": {"estimated_cost_inr": 0.00165, "max_request_tokens": 512, "request_tokens": 11}, "cache_hit": false, "endpoint": "WS /ws/chat", "guardrails": [], "latency_ms": 121.171, "level": "INFO", "llm_calls": 44, "record_id": "LN-20260004", "refused": false, "request": "Is that application flagged for fraud review?", "review_approved": true, "service": "cred-domain-support-agent", "session_id": "ws-41fd9929c10449ad", "status": "ok", "top_similarity": 0.0642, "trace_id": "99281675da0549a5", "ts": 1789227751.351}
{"answer_type": "refusal", "budget": {"estimated_cost_inr": 0.0009, "max_request_tokens": 512, "request_tokens": 6}, "cache_hit": false, "endpoint": "POST /ask", "guardrails": ["output_groundedness", "low_retrieval_confidence", "ungrounded_answer"], "latency_ms": 92.895, "level": "INFO", "llm_calls": 47, "record_id": null, "refused": true, "request": "What is the EMI formula?", "review_approved": null, "service": "cred-domain-support-agent", "session_id": "after-disconnect", "status": "ok", "top_similarity": 0.1316, "trace_id": "426b15126ab74131", "ts": 1789227751.48}

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
  "latency_ms": 105.541,
  "level": "INFO",
  "llm_calls": 50,
  "record_id": null,
  "refused": true,
  "request": "My PAN is [PAN_REDACTED] - what is the EMI formula?",
  "review_approved": null,
  "service": "cred-domain-support-agent",
  "session_id": "pii-log-demo",
  "status": "ok",
  "top_similarity": 0.0861,
  "trace_id": "5614a0d9cc0e4de3",
  "ts": 1789227751.579
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
Q01    1.00   1.00   0.80   1.00   0.241    False  loan eligibility criteria by loan type
Q02    1.00   1.00   0.86   1.00   0.152    False  EMI calculation rules
Q03    1.00   1.00   0.88   1.00   0.252    False  credit-card fee structure
Q04    1.00   1.00   0.72   1.00   0.177    False  KYC document requirements
Q05    1.00   1.00   0.80   1.00   0.301    False  fraud-dispute resolution process
Q06    1.00   1.00   0.60   1.00   0.206    False  account-closure process
Q07    1.00   1.00   0.77   1.00   0.352    False  interest-rate slabs
Q08    1.00   1.00   1.00   1.00   0.481    False  prepayment-penalty rules
Q09    1.00   1.00   0.88   1.00   0.319    False  minimum-balance requirements
Q10    1.00   1.00   1.00   1.00   0.238    False  credit-score impact factors
Q11    1.00   1.00   0.93   1.00   0.599    False  joint-account rules
Q12    1.00   1.00   1.00   1.00   0.262    False  NRI-account eligibility
Q13    0.50   0.81   0.45   1.00   0.040    False  edge case - record lookup plus policy
Q14    1.00   1.00   1.00   1.00   0.041     True  OUT OF SCOPE - unrelated domain
Q15    1.00   1.00   1.00   1.00   0.143     True  OUT OF SCOPE - plausible but not in the knowledge ba
--------------------------------------------------------------------------------------------------------------
AVG    0.97   0.99   0.85   1.00

Average accuracy     : 0.9667
Average grounding    : 0.9875
Average completeness : 0.8462
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
    "top_similarity": 0.2407,
    "retrieved_docs": [
      "kb01_loan_eligibility",
      "kb08_prepayment_penalty"
    ],
    "relevant_docs": [
      "kb01_loan_eligibility"
    ],
    "answer": "A Personal Loan applicant must be a salaried or self-employed resident between 21 and 58 years of age with a minimum net monthly income of INR 25,000 and at least six months in the current job. A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outstanding principal if closed within the first twelve months, 3 percent between months thirteen and twenty-four, and 2 percent thereafter. Part-prepayment on a Personal Loan is permitted only after six instalments have been paid, is capped at 25 percent of the outstanding principal in a financial year, and by default reduces the tenure rather than the instalment.",
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
    "top_similarity": 0.1519,
    "retrieved_docs": [
      "kb02_emi_calculation",
      "kb01_loan_eligibility"
    ],
    "relevant_docs": [
      "kb02_emi_calculation"
    ],
    "answer": "The equated monthly instalment is computed with the standard reducing-balance formula EMI = P x r x (1+r)^n / ((1+r)^n - 1), where P is the sanctioned principal, r is the monthly interest rate (annual rate divided by twelve, expressed as a decimal) and n is the tenure in months. An Education Loan needs a confirmed admission letter from a recognised institution and a co-applicant who is an earning parent or guardian. A Business Loan applicant must show two consecutive years of filed income tax returns and a business vintage of at least 24 months.",
    "accuracy": 1.0,
    "grounding": 1.0,
    "completeness": 0.86,
    "safety": 1.0
  },
  {
    "id": "Q03",
    "topic": "credit-card fee structure",
    "query": "What is the cash advance fee when I withdraw cash on my credit card?",
    "in_scope": true,
    "refused": false,
    "answer_type": "policy",
    "top_similarity": 0.252,
    "retrieved_docs": [
      "kb03_credit_card_fees",
      "kb06_account_closure"
    ],
    "relevant_docs": [
      "kb03_credit_card_fees"
    ],
    "answer": "Cash withdrawal at an ATM using the card attracts a cash advance fee of 2.5 percent of the withdrawn amount subject to a floor of INR 500, and interest on the withdrawn amount runs from the transaction date with no interest-free period. A credit-card account cannot be closed while any amount remains outstanding or while a dispute is open on it. The joining fee is INR 500 plus applicable taxes and the annual fee of INR 500 is waived for the following year whenever the cardholder spends INR 1,50,000 or more in the preceding twelve billing cycles.",
    "accuracy": 1.0,
    "grounding": 1.0,
    "completeness": 0.883,
    "safety": 1.0
  },
  {
    "id": "Q04",
    "topic": "KYC document requirements",
    "query": "Which documents count as proof of address for KYC?",
    "in_scope": true,
    "refused": false,
    "answer_type": "policy",
    "top_similarity": 0.1767,
    "retrieved_docs": [
      "kb04_kyc_documents"
    ],
    "relevant_docs": [
      "kb04_kyc_documents"
    ],
    "answer": "Every applicant must submit one officially valid document for proof of identity and one for proof of address before an account or a loan can be activated. Accepted address documents are the Aadhaar letter, passport, a utility bill not older than two months, or a registered rent agreement. Self-employed applicants additionally submit their business registration proof and the latest two years of income tax returns.",
    "accuracy": 1.0,
    "grounding": 1.0,
    "completeness": 0.72,
    "safety": 1.0
  },
  {
    "id": "Q05",
    "topic": "fraud-dispute resolution process",
    "query": "How long do I have to report an unauthorised transaction to have zero liability?",
    "in_scope": true,
    "refused": false,
    "answer_type": "policy",
    "top_similarity": 0.3015,
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
    "top_similarity": 0.206,
    "retrieved_docs": [
      "kb06_account_closure",
      "kb08_prepayment_penalty"
    ],
    "relevant_docs": [
      "kb06_account_closure"
    ],
    "answer": "No closure charge applies if the account has been open for more than twelve months; accounts closed between fourteen days and twelve months attract a closure charge of INR 500 plus taxes, and closure within the first fourteen days is free. A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outstanding principal if close
```

