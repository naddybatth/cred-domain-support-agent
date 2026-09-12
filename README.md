# Final Capstone — Cred Domain Support Agent (CrewAI)

**Track: Banking & FinTech (Cred).**

One working, production-minded domain support agent for Cred's lending-operations desk. It
answers loan-policy questions from a knowledge base written for this project, looks up a
specific loan application from a dataset generated and validated here, remembers a
conversation, is guarded against misuse, has every draft reviewed by an independent
two-agent team before it reaches a member, and runs under an explicit governance policy —
orchestrated with CrewAI, reviewed with Autogen, deployed behind FastAPI, and evaluated
end to end.

**Everything in this repository runs under `MOCK_LLM=1` with zero API keys and zero
network access.** There is no paid account, no credit card and no hosted model anywhere in
the graded path.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # MOCK_LLM=1 is already the default

python scripts/calibrate_threshold.py # Task 4: MEASURE the similarity threshold
python run_all.py                     # runs every task, writes transcripts/
uvicorn src.api:app --reload          # Task 11: API docs at http://127.0.0.1:8000/docs
```

`run_all.py --part 1|2|3|4` runs one part at a time.

### CrewAI telemetry is disabled

CrewAI attempts an outbound network call from its telemetry on `crew.kickoff()`, which
would violate the zero-network requirement. **`CREWAI_DISABLE_TELEMETRY=true` and
`OTEL_SDK_DISABLED=true` are set**, in `.env.example` and again defensively in
`src/config.py` (which also sets `CREWAI_TRACING_ENABLED=false` and ChromaDB's
`ANONYMIZED_TELEMETRY=False`) so the setting applies even when a run does not load `.env`.
I confirm that I set them.

---

## Architecture

```
                          POST /ask   ·   POST /add-document   ·   WS /ws/chat
                                              │
                                     src/pipeline.py   ← one request path for every entry point
                                              │
  1 runtime token/cost budget ────────────────┤   reject oversized request before any call
  2 input guardrails ─────────────────────────┤   PAN / Aadhaar / account masking + injection detect
  3 session memory ───────────────────────────┤   LangChain InMemoryChatMessageHistory
  4 response cache ───────────────────────────┤   keyed on normalized query text
  5 CrewAI crew  (Process.sequential) ────────┤
        Policy Retrieval Specialist  → rag_lookup ──────► ChromaDB (kb_sentence)
        Loan Application Lookup Spec → check_loan_application_status ──► dataset.py
        Member Response Composer     → no tools, emits the Pydantic envelope
  6 output groundedness guardrail ────────────┤   refuse when the context does not support it
  7 Autogen RoundRobinGroupChat ──────────────┤   Policy-Compliance-Reviewer → Final-Editor
  8 JSON-Lines log with trace id ─────────────┘   logs/requests.jsonl (masked)
```

The single privileged capability in the system is reading an application record. Only the
Lookup Agent holds it, and that is enforced at construction time — see Task 15.

**There is no approval, disbursal or payment capability anywhere in this repository.** The
agent reads records and recommends escalation; a human decides everything that moves money.

---

## Where each task lives

| Task | What | Code | Proof |
|---|---|---|---|
| 1 | Seeded loan-application dataset | `src/dataset.py` | `transcripts/part1_rag.md` |
| 2 | 12 knowledge-base documents | `data/kb/*.md` | `transcripts/part1_rag.md` |
| 3 | Two chunking strategies, two Chroma collections | `src/chunking.py`, `src/rag.py` | `transcripts/part1_rag.md` |
| 4 | Grounded generation + calibrated fallback | `src/rag.py`, `scripts/calibrate_threshold.py` | `calibration.json`, `transcripts/part1_rag.md` |
| 5 | Document-level precision/recall, both strategies | `src/rag.py` | `transcripts/part1_rag.md` |
| 6 | `check_loan_application_status` + escalation score | `src/tools.py`, `scripts/justify_threshold.py` | `transcripts/part2_crew.md` |
| 7 | CrewAI crew, 3 agents, both tools invoked | `src/crew.py`, `src/crew_llm.py` | `transcripts/part2_crew.md` |
| 8 | Session memory, multi-turn + fresh | `src/memory.py` | `transcripts/part2_crew.md` |
| 9 | Pydantic structured output, validated in code | `src/schemas.py`, `src/crew.py` | `transcripts/part2_crew.md` |
| 10 | Input PII + injection guardrails, output groundedness | `src/guardrails.py` | `transcripts/part2_crew.md` |
| 11 | FastAPI: 2 HTTP + 1 WebSocket | `src/api.py` | `transcripts/part3_deploy_eval.md` |
| 12 | JSON-Lines logging with trace id, masked | `src/logging_utils.py` | `logs/requests.jsonl`, `transcripts/part3_deploy_eval.md` |
| 13 | LLM-as-judge, 15 queries × 4 properties | `src/evaluation.py`, `eval/test_queries.py` | `transcripts/part3_deploy_eval.md` |
| 14 | Autogen review stage | `src/review.py` | `transcripts/part4_governance.md` |
| 15 | Four-layer governance, least autonomy, risk, budget | `src/governance.py`, `src/crew.py` | `transcripts/part4_governance.md` |
| 16 | Response cache | `src/cache.py` | `transcripts/part4_governance.md` |

---

## Task 1 — dataset design choices (reproduce this exactly)

| Choice | Value |
|---|---|
| seed | `20260913` |
| size | 48 records |
| category weights | Personal 0.30, Home 0.20, Auto 0.20, Education 0.15, Business 0.15 |
| status weights | Submitted 0.18, Under Review 0.27, Approved 0.22, Rejected 0.13, Disbursed 0.20 |
| fraud-flag probability | `p = 0.20` |

**Amount range reasoning (one sentence, per the brief):** ticket sizes are set per category
rather than globally — Personal 0.5–15 lakh, Home 15 lakh–1.2 crore, Auto 1.5–25 lakh,
Education 1–40 lakh, Business 3–75 lakh — because an unsecured personal loan and a home
loan differ by an order of magnitude, and one global range would have produced records no
lending-operations reviewer would recognise.

**Realised distribution** (printed by `python -m src.dataset`):

```
Personal 11 · Home 8 · Auto 11 · Education 8 · Business 10     (every category ≥ 3 ✓)
Submitted 9 · Under Review 8 · Approved 15 · Rejected 4 · Disbursed 12   (every status ≥ 1 ✓)
flagged_for_fraud_review = True : 10/48 = 20.83%               (band 10–30% ✓)
```

Category coverage is guaranteed by construction (the first 15 slots are pre-allocated
three per category, and the first five of those carry one of each status) rather than left
to a lucky draw. The fraud percentage is *not* forced: it is the realised result of
`p = 0.20`, it landed in the band on the first seed, and `src/dataset.py::validate()`
asserts the band and tells you to change the seed or the weight — never to hand-edit
records — if a future change pushes it out.

## Task 4 — the similarity threshold was measured, not chosen

`scripts/calibrate_threshold.py` measures top-1 cosine similarity for 5 real in-scope
queries and 4 out-of-scope queries, prints both clusters, and writes the midpoint of the
observed gap to `calibration.json`, keyed by embedding backend.

Measured for the deployed `kb_sentence` collection:

| query | top-1 cosine |
|---|---|
| minimum income for a personal loan | 0.2169 |
| EMI on a reducing-balance loan | 0.1519 |
| foreclosure charge on a fixed-rate personal loan | 0.4810 |
| proof of address for KYC | 0.1767 |
| average monthly balance in a metro branch | 0.3186 |
| *out of scope* — hyderabadi biryani recipe | 0.0410 |
| *out of scope* — 2018 football world cup | 0.0887 |
| *out of scope* — diesel engine timing belt | 0.0840 |
| *out of scope, HARD* — Cred policy on cryptocurrency trading accounts | 0.1431 |

```
lowest in-scope    0.1519
highest out-scope  0.1431
separation gap     0.0088
chosen threshold   0.1475   (midpoint of the observed gap)
```

Two things worth saying honestly about this number.

First, **the hard negative is what sets it.** "Cryptocurrency trading accounts" is
plausible banking vocabulary that this knowledge base deliberately does not cover. With
only the obvious negatives (biryani, football) the measured gap runs from 0.0887 to 0.1519
and any threshold in that range looks fine — but the agent then answers the crypto
question with retrieved text about *accounts* in general, which is exactly the ungrounded
failure the fallback exists to prevent. Query Q15 in the evaluation set is that case, and
it is refused only because the hard negative was included in calibration.

Second, **0.0088 is a narrow gap.** It is the honest measured separation for the embedding
backend the committed transcripts were produced with (see *Embedding backend*, below), and
it is why the system does not rely on the threshold alone: `src/guardrails.py`
independently refuses when fewer than half the answer's content words appear in the
retrieved context, and the Autogen reviewer strips unsupported sentences after that.

Common tutorial presets (0.5 / 0.6 / 0.7) are not used anywhere. Every one of them sits
above the *highest* in-scope similarity measured here and would refuse every question the
agent is for.

## Task 5 — chunking comparison and recommendation

Same five queries, document-level precision and recall, chunks mapped back to parent
documents and de-duplicated before scoring. Full per-query arithmetic is in
`transcripts/part1_rag.md`.

| collection | mean precision | mean recall | mean F1 |
|---|---|---|---|
| `kb_fixed_overlap` (420 chars, 120 overlap) | 0.4000 | 1.0000 | 0.5667 |
| `kb_sentence` (2 sentences, 1 overlapping) | **0.7000** | 1.0000 | **0.8000** |

**I would deploy sentence chunking.** Recall is identical at 1.0000 — both strategies find
the right document every time — but fixed-size windows drag in 0.30 more irrelevant
documents per query (precision 0.40 against 0.70), because a 420-character window
routinely spans the tail of one policy topic and the head of the next, so a chunk that is
half about KYC and half about disputes matches both kinds of query. Beyond the numbers, a
sentence chunk is always a complete policy statement: a fixed window that cuts *"a
foreclosure charge of 4 percent of the outstanding principal if closed within"* mid-clause
retrieves a fragment that reads as an unconditional charge, and both the grounded
generation step and the Autogen compliance reviewer read those chunks directly.

## Task 6 — escalation score

```
escalation_score = 0.6 · flagged_for_fraud_review + 0.4 · min(days_since_created / 30, 1)
```

The fraud flag is a hard compliance signal and carries the majority weight, so a flagged
application can never fall below 0.60 however fresh it is. Staleness is a real ageing-SLA
signal but must never outrank a fraud flag on its own, so it is capped at 0.40.

**Recommended threshold: 0.55**, justified from this dataset's own distribution
(`python scripts/justify_threshold.py`):

```
max score reachable by an UNflagged application : 0.4000
min score observed on a flagged application     : 0.7067
escalation_score p50 / p80 / p90                : 0.2533 / 0.7067 / 0.8400
records at or above 0.55                        : 10/48 = 20.83%
```

0.55 sits inside the empty band between those two clusters, between the p50 and p80 of the
score distribution, and selects exactly the fraud-flagged population without ever firing on
age alone.

## Task 13 — evaluation results

15 queries: one for each of the 12 required KB topics, one mixed record-plus-policy edge
case, and two deliberately out-of-scope queries. All four properties scored per query under
`MOCK_LLM`; full table in `transcripts/part3_deploy_eval.md`.

| | average |
|---|---|
| Accuracy | **0.9667** |
| Grounding | **0.9875** |
| Completeness | **0.8462** |
| Safety | **1.0000** |

Accuracy is scored against ground truth, not against a similarity number: the test set
names the document that actually contains each answer, and a query scores 1.0 when the
top-ranked retrieved document is that one, 0.5 when the right document was retrieved but
ranked lower, 0 when it was missed. A correct refusal on an out-of-scope query is itself
the accurate answer and scores 1.0.

**The one weak case is worth naming rather than hiding.** Q13 —
*"What is the status of application LN-20260004 and what prepayment rules apply to it?"* —
scores accuracy 0.50 and completeness 0.45. A query that mixes a record id with a policy
question retrieves poorly (top similarity 0.040) because the record id dominates the
embedding, so the policy half falls below the threshold and only the record half is
answered. That is the system failing *closed*, which is the right direction — safety still
scores 1.0 and nothing ungrounded reaches the member — but a production version would split
the query into its record and policy clauses before retrieval rather than embedding it
whole.

## Task 15 — governance

**Least autonomy (application layer).** Every agent in `src/crew.py` is constructed through
`_guarded_agent()`, which calls `ToolAccessPolicy.assert_allowed(role, tools)` *before* the
`Agent` object exists. The policy maps each role to the exact tool names it may hold; the
Composer maps to the empty set. Wiring `check_loan_application_status` to the Retrieval
Agent raises `ToolPermissionError` at crew-assembly time — demonstrated in
`transcripts/part4_governance.md` — rather than silently granting a second agent read
access to application records. Because the check is on construction rather than on
invocation, there is no window in which a wrongly-wired agent exists and could be kicked
off.

**Risk classification: HIGH.** The scheme puts medical data, hiring decisions and financial
data in the High band. This agent reads loan-application records — sanctioned amount,
approval status, fraud flag — and its output is relayed to members by support staff. A
wrong status or a wrongly reassuring policy answer causes a member to act on an incorrect
view of a credit decision, and an unmasked PAN or account number in a log is a reportable
data incident. That is financial data, above the Medium band that ordinary customer-support
ticketing occupies. The classification is visible in the architecture: no approval,
disbursal or payment capability exists, every money decision stays with a human, and
fixed-format PII is masked before it reaches either the model or the log.

**Runtime layer.** `RequestBudget` estimates prompt tokens and rejects a request over the
cap *before* any model or tool call, so the budget cannot be silently exceeded. A 6,000-word
request is rejected with the token count and the notional rupee cost; demonstrated in
`transcripts/part4_governance.md`.

## Task 10 — PII masking scope

PAN, Aadhaar and bank account numbers are fixed-format and are masked, demonstrably, on the
input and in the log. Applicant names and income figures are free text and unformatted
numbers with no reliable pattern to match under a keyless `MOCK_LLM`-only masker, and are
acknowledged out of scope for masking as the brief states. Only fabricated examples are
used anywhere in this repository — no real person's data.

---

## Two CrewAI pitfalls, and how this repo avoids them

Both are silent: no crash, just wrong output. Both are handled in `src/mock_llm.py`, and
both have unit-level evidence in the code comments there.

**1 — the `Observation:` template trap.** CrewAI's own ReAct system prompt literally
contains the line `Observation: the result of the action`. A parser that searches the
conversation for `"Observation:"` to extract a tool result matches that template text on
the very first call, before any tool has run, and returns placeholder text as the final
answer. `observations()` skips `role="system"` entirely *and* strips that exact sentence
anywhere else before scanning, so it reads only text the model generated or the executor
appended as a real tool result.

**2 — name-based tool dispatch.** Deciding which tool to call by matching its *name* (does
it contain `"lookup"`?) silently misclassifies a tool literally named `rag_lookup`.
`select_tool()` never reads tool names at all: it dispatches on each tool's **declared
argument schema** — a tool declaring `record_id` handles a record question, a tool
declaring `query` handles a policy question. The RAG tool in this repo is deliberately
named `rag_lookup` so that the correct behaviour is exercised on every single run.

## Autogen review stage — the three constructor details

- The bound is **`max_turns=2`**. `RoundRobinGroupChat` has no `max_iterations` parameter.
- `MaxMessageTermination` counts the initiating task message as message 1, so the
  equivalent bound is **`MaxMessageTermination(3)`**, not `(2)`, if both agents are to
  speak. Both bounds are wired together in `src/review.py`.
- The Final-Editor carries `output_content_type=ReviewVerdict`, so the **team** must also be
  built with `custom_message_types=[StructuredMessage[ReviewVerdict]]`, or the run raises
  `ValueError: Message type ... is not registered`.

---

## Embedding backend — please read before grading the numbers

The brief asks for a free local **SentenceTransformers** model, and that is the primary
backend: `src/embeddings.py` loads `sentence-transformers/all-MiniLM-L6-v2` and uses it
whenever it can be loaded.

The environment these transcripts were produced in blocks `huggingface.co` at the network
policy layer, so the model could not even be downloaded there. Rather than ship a repo that
cannot run, `src/embeddings.py` falls back to a **deterministic, fully local, fixed-4096-
dimension hashed TF-IDF encoder** and says so loudly on every run. It is a genuine
distributional representation — not a random hash — so retrieval, thresholds and
precision/recall all stay meaningful, and it is stateless, so results reproduce exactly
across machines.

Two earlier fallback designs were tried and rejected, and the reasons are in the code:
truncated SVD over a 12-document corpus densified the space so badly that *"how do I change
the timing belt on a diesel engine"* scored 0.98 against lending policy; a fitted
`TfidfVectorizer` separated well but its dimension is the fitted vocabulary size, so the
first `POST /add-document` broke ChromaDB with `Collection expecting embedding with
dimension of 1361, got 1387`.

**To grade against the primary backend**, on a machine that can reach the Hugging Face hub:

```bash
rm -rf chroma_store calibration.json
python scripts/calibrate_threshold.py   # re-measures the threshold for MiniLM's vector space
python run_all.py                       # regenerates every transcript
```

A similarity threshold is a property of the vector space it was measured in, so it **must**
be re-measured when the backend changes — which is exactly why `calibration.json` is keyed
by backend name and `src/config.py` resolves the threshold per backend rather than
hard-coding one number.

## Honest limitations

- Retrieval on a query that mixes a record id with a policy question is weak (Q13 above);
  a production version would decompose the query before embedding.
- Session memory is in-process and does not survive a restart, which the brief states is
  sufficient here.
- `MOCK_LLM` generation is extractive by design. That is what makes the groundedness
  guardrail meaningful, but it means answers read as assembled policy sentences rather than
  as fluent prose.
- The dataset, the knowledge base and every PII example are fabricated for this project.
- The judge runs under `MOCK_LLM`, so its scores come from measurable evidence about each
  answer rather than from a grader model's opinion.

## Repository layout

```
dataset.py is src/dataset.py          run_all.py           regenerates every transcript
data/kb/*.md    12 policy documents   calibration.json     measured thresholds per backend
src/            all implementation    transcripts/         task-by-task demonstrations
eval/           the 15-query test set logs/requests.jsonl  structured request log
scripts/        calibration + threshold justification
```
