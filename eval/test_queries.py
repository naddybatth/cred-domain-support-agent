"""Task 13 - the 15-query evaluation set.

Coverage rule from the brief: at least one query touching every required KB topic, plus
at least two deliberately out-of-scope or edge-case queries.

`relevant_docs` is the ground truth used for the Task 5 document-level precision/recall.
`in_scope=False` marks a query the system is expected to refuse.
"""
from __future__ import annotations

from typing import Any, Dict, List

TEST_QUERIES: List[Dict[str, Any]] = [
    {
        "id": "Q01",
        "topic": "loan eligibility criteria by loan type",
        "query": "What is the minimum monthly income needed to qualify for a personal loan?",
        "relevant_docs": ["kb01_loan_eligibility"],
        "in_scope": True,
    },
    {
        "id": "Q02",
        "topic": "EMI calculation rules",
        "query": "How is the EMI calculated on a reducing balance loan?",
        "relevant_docs": ["kb02_emi_calculation"],
        "in_scope": True,
    },
    {
        "id": "Q03",
        "topic": "credit-card fee structure",
        "query": "What is the cash advance fee when I withdraw cash on my credit card?",
        "relevant_docs": ["kb03_credit_card_fees"],
        "in_scope": True,
    },
    {
        "id": "Q04",
        "topic": "KYC document requirements",
        "query": "Which documents count as proof of address for KYC?",
        "relevant_docs": ["kb04_kyc_documents"],
        "in_scope": True,
    },
    {
        "id": "Q05",
        "topic": "fraud-dispute resolution process",
        "query": "How long do I have to report an unauthorised transaction to have zero liability?",
        "relevant_docs": ["kb05_fraud_dispute"],
        "in_scope": True,
    },
    {
        "id": "Q06",
        "topic": "account-closure process",
        "query": "Is there a charge for closing my savings account after eighteen months?",
        "relevant_docs": ["kb06_account_closure"],
        "in_scope": True,
    },
    {
        "id": "Q07",
        "topic": "interest-rate slabs",
        "query": "What interest rate applies to a home loan for a credit score above 780?",
        "relevant_docs": ["kb07_interest_rate_slabs"],
        "in_scope": True,
    },
    {
        "id": "Q08",
        "topic": "prepayment-penalty rules",
        "query": "What is the foreclosure charge on a fixed rate personal loan?",
        "relevant_docs": ["kb08_prepayment_penalty"],
        "in_scope": True,
    },
    {
        "id": "Q09",
        "topic": "minimum-balance requirements",
        "query": "What is the average monthly balance requirement in a metro branch?",
        "relevant_docs": ["kb09_minimum_balance"],
        "in_scope": True,
    },
    {
        "id": "Q10",
        "topic": "credit-score impact factors",
        "query": "Does closing an old credit card lower my credit score?",
        "relevant_docs": ["kb10_credit_score_factors"],
        "in_scope": True,
    },
    {
        "id": "Q11",
        "topic": "joint-account rules",
        "query": "Under an either or survivor mandate can one holder operate the joint account alone?",
        "relevant_docs": ["kb11_joint_account"],
        "in_scope": True,
    },
    {
        "id": "Q12",
        "topic": "NRI-account eligibility",
        "query": "Is the interest on an NRE account taxable in India?",
        "relevant_docs": ["kb12_nri_account"],
        "in_scope": True,
    },
    {
        "id": "Q13",
        "topic": "edge case - record lookup plus policy",
        "query": "What is the status of application LN-20260004 and what prepayment rules apply to it?",
        "relevant_docs": ["kb08_prepayment_penalty"],
        "in_scope": True,
    },
    {
        "id": "Q14",
        "topic": "OUT OF SCOPE - unrelated domain",
        "query": "What is the best recipe for hyderabadi biryani?",
        "relevant_docs": [],
        "in_scope": False,
    },
    {
        "id": "Q15",
        "topic": "OUT OF SCOPE - plausible but not in the knowledge base",
        "query": "What is Cred's policy on cryptocurrency trading accounts for members?",
        "relevant_docs": [],
        "in_scope": False,
    },
]

# The subset used for the Task 4 demonstration and the Task 5 chunking comparison.
PR_QUERIES = [(q["query"], q["relevant_docs"]) for q in TEST_QUERIES[:5]]
OUT_OF_SCOPE_QUERY = TEST_QUERIES[13]["query"]


def topics_covered() -> List[str]:
    return sorted({q["topic"] for q in TEST_QUERIES})
