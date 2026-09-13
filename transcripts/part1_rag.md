# Part 1 - Dataset Design & RAG Core (Tasks 1-5)

*embedding backend: `sentence-transformers:sentence-transformers/all-MiniLM-L6-v2`*


## Task 1 - dataset generation and validation


### python -m src.dataset

```
====================================================================
TASK 1 - LOAN APPLICATION DATASET
====================================================================
seed=20260913  size=48

Count per category (requirement: every category >= 3)
  Personal Loan     11
  Home Loan          8
  Auto Loan         11
  Education Loan     8
  Business Loan     10

Count per status (requirement: every status >= 1)
  Submitted          9
  Under Review       8
  Approved          15
  Rejected           4
  Disbursed         12

flagged_for_fraud_review = True : 10/48 = 20.83%   (band 10-30%: PASS)
loan_amount_inr range in generated data: INR 218,000 - INR 11,714,000

First 5 records:
   {'record_id': 'LN-20260001', 'category': 'Personal Loan', 'status': 'Submitted', 'loan_amount_inr': 250000, 'days_since_created': 19, 'flagged_for_fraud_review': False}
   {'record_id': 'LN-20260002', 'category': 'Personal Loan', 'status': 'Under Review', 'loan_amount_inr': 448000, 'days_since_created': 29, 'flagged_for_fraud_review': False}
   {'record_id': 'LN-20260003', 'category': 'Personal Loan', 'status': 'Approved', 'loan_amount_inr': 831000, 'days_since_created': 12, 'flagged_for_fraud_review': False}
   {'record_id': 'LN-20260004', 'category': 'Home Loan', 'status': 'Rejected', 'loan_amount_inr': 2352000, 'days_since_created': 11, 'flagged_for_fraud_review': False}
   {'record_id': 'LN-20260005', 'category': 'Home Loan', 'status': 'Disbursed', 'loan_amount_inr': 5005000, 'days_since_created': 29, 'flagged_for_fraud_review': False}
```


## Task 2 - knowledge base


### 12 authored documents

```
kb01_loan_eligibility         153 words  Loan Eligibility Criteria by Loan Type
kb02_emi_calculation          131 words  EMI Calculation Rules
kb03_credit_card_fees         131 words  Credit Card Fee Structure
kb04_kyc_documents            120 words  KYC Document Requirements
kb05_fraud_dispute            136 words  Fraud Dispute Resolution Process
kb06_account_closure          135 words  Account Closure Process
kb07_interest_rate_slabs      129 words  Interest Rate Slabs
kb08_prepayment_penalty       138 words  Prepayment and Foreclosure Penalty Rules
kb09_minimum_balance          154 words  Minimum Balance Requirements
kb10_credit_score_factors     142 words  Credit Score Impact Factors
kb11_joint_account            143 words  Joint Account Rules
kb12_nri_account              154 words  NRI Account Eligibility
```


## Task 3 - two chunking strategies, two ChromaDB collections


### index build

```
fixed_overlap chunks produced : 36
sentence chunks produced      : 48
collection kb_fixed_overlap     count after upsert: 37
collection kb_sentence          count after upsert: 49

sanity query 'foreclosure charge on a personal loan':
  kb_fixed_overlap
    0.6070  kb08_prepayment_penalty  percent between months thirteen and twenty-four, and 2 percent thereafter. Part-pr...
    0.5974  kb08_prepayment_penalty  A floating-rate loan taken by an individual borrower for a non-business purpose ca...
  kb_sentence
    0.6804  kb08_prepayment_penalty  A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outst...
    0.6187  kb08_prepayment_penalty  A floating-rate loan taken by an individual borrower for a non-business purpose ca...
```


## Task 4 - calibrated grounded generation


### calibration.json (measured, written by scripts/calibrate_threshold.py)

```
{
  "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2": {
    "backend": "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2",
    "measurements": {
      "kb_fixed_overlap": {
        "chosen_threshold": 0.4506,
        "highest_out_of_scope": 0.3266,
        "in_scope": {
          "How is the EMI calculated on a reducing balance loan?": 0.5747,
          "What is the average monthly balance requirement in a metro branch?": 0.7338,
          "What is the foreclosure charge on a fixed rate personal loan?": 0.7172,
          "What is the minimum income needed for a personal loan?": 0.6804,
          "Which documents count as proof of address for KYC?": 0.582
        },
        "lowest_in_scope": 0.5747,
        "out_of_scope": {
          "How do I change the timing belt on a diesel engine?": 0.1645,
          "What is Cred's policy on cryptocurrency trading accounts for members?": 0.3266,
          "What is the best recipe for hyderabadi biryani?": 0.1635,
          "Who won the football world cup in 2018?": 0.094
        },
        "separation_gap": 0.2481
      },
      "kb_sentence": {
        "chosen_threshold": 0.4443,
        "highest_out_of_scope": 0.3322,
        "in_scope": {
          "How is the EMI calculated on a reducing balance loan?": 0.6017,
          "What is the average monthly balance requirement in a metro branch?": 0.7516,
          "What is the foreclosure charge on a fixed rate personal loan?": 0.7354,
          "What is the minimum income needed for a personal loan?": 0.6914,
          "Which documents count as proof of address for KYC?": 0.5563
        },
        "lowest_in_scope": 0.5563,
        "out_of_scope": {
          "How do I change the timing belt on a diesel engine?": 0.0772,
          "What is Cred's policy on cryptocurrency trading accounts for members?": 0.3322,
          "What is the best recipe for hyderabadi biryani?": 0.127,
          "Who won the football world cup in 2018?": 0.0995
        },
        "separation_gap": 0.2241
      }
    },
    "thresholds": {
      "kb_fixed_overlap": 0.4506,
      "kb_sentence": 0.4443
    }
  }
}
```


### 5 in-scope queries + 1 out-of-scope fallback

```
threshold in force for kb_sentence: 0.4443

Q: What is the minimum monthly income needed to qualify for a personal loan?
   top_similarity=0.6844 refused=False
   A: A Personal Loan applicant must be a salaried or self-employed resident between 21 and 58 years of age with a minimum net monthly income of INR 25,000 and at least six months in the current job. Personal Loan rates run from 10.75 percent to 18.00 percent per annum and are set from the applicant's cre
   sources: ['kb01_loan_eligibility', 'kb07_interest_rate_slabs']

Q: How is the EMI calculated on a reducing balance loan?
   top_similarity=0.6017 refused=False
   A: The equated monthly instalment is computed with the standard reducing-balance formula EMI = P x r x (1+r)^n / ((1+r)^n - 1), where P is the sanctioned principal, r is the monthly interest rate (annual rate divided by twelve, expressed as a decimal) and n is the tenure in months. A change in the floa
   sources: ['kb02_emi_calculation', 'kb09_minimum_balance']

Q: What is the cash advance fee when I withdraw cash on my credit card?
   top_similarity=0.7842 refused=False
   A: Cash withdrawal at an ATM using the card attracts a cash advance fee of 2.5 percent of the withdrawn amount subject to a floor of INR 500, and interest on the withdrawn amount runs from the transaction date with no interest-free period. The joining fee is INR 500 plus applicable taxes and the annual
   sources: ['kb03_credit_card_fees', 'kb06_account_closure']

Q: Which documents count as proof of address for KYC?
   top_similarity=0.5563 refused=False
   A: Accepted address documents are the Aadhaar letter, passport, a utility bill not older than two months, or a registered rent agreement. Re-KYC is required every two years for high-risk customers, every eight years for medium-risk and every ten years for low-risk customers. Self-employed applicants ad
   sources: ['kb04_kyc_documents']

Q: How long do I have to report an unauthorised transaction to have zero liability?
   top_similarity=0.6495 refused=False
   A: A customer must report an unauthorised transaction through the in-app dispute form, the 24x7 helpline or a written complaint, and the reporting timestamp is what governs liability. Where the delay in reporting is four to seven working days, the customer's liability is capped at the transaction value
   sources: ['kb05_fraud_dispute']

OUT-OF-SCOPE Q: What is the best recipe for hyderabadi biryani?
   top_similarity=0.1270 refused=True
   A: I don't know based on the available policy documents. This question is outside the Cred lending-policy knowledge base, so I am not going to answer it from memory. Please route it to a human agent.
```


## Task 5 - document-level precision / recall for BOTH strategies


### per-query arithmetic, both collections

```
collection: kb_fixed_overlap
  Q: What is the minimum monthly income needed to qualify for a persona
     retrieved docs : ['kb01_loan_eligibility', 'kb07_interest_rate_slabs']
     relevant  docs : ['kb01_loan_eligibility']
     true positives : ['kb01_loan_eligibility']
     precision = 1/2 = 0.5000
     recall    = 1/1 = 1.0000
     f1        = 0.6667
  Q: How is the EMI calculated on a reducing balance loan?
     retrieved docs : ['kb02_emi_calculation']
     relevant  docs : ['kb02_emi_calculation']
     true positives : ['kb02_emi_calculation']
     precision = 1/1 = 1.0000
     recall    = 1/1 = 1.0000
     f1        = 1.0000
  Q: What is the cash advance fee when I withdraw cash on my credit car
     retrieved docs : ['kb03_credit_card_fees', 'kb06_account_closure']
     relevant  docs : ['kb03_credit_card_fees']
     true positives : ['kb03_credit_card_fees']
     precision = 1/2 = 0.5000
     recall    = 1/1 = 1.0000
     f1        = 0.6667
  Q: Which documents count as proof of address for KYC?
     retrieved docs : ['kb04_kyc_documents']
     relevant  docs : ['kb04_kyc_documents']
     true positives : ['kb04_kyc_documents']
     precision = 1/1 = 1.0000
     recall    = 1/1 = 1.0000
     f1        = 1.0000
  Q: How long do I have to report an unauthorised transaction to have z
     retrieved docs : ['kb05_fraud_dispute']
     relevant  docs : ['kb05_fraud_dispute']
     true positives : ['kb05_fraud_dispute']
     precision = 1/1 = 1.0000
     recall    = 1/1 = 1.0000
     f1        = 1.0000
  MEAN precision=0.8000 recall=1.0000 f1=0.8667

collection: kb_sentence
  Q: What is the minimum monthly income needed to qualify for a persona
     retrieved docs : ['kb01_loan_eligibility', 'kb07_interest_rate_slabs']
     relevant  docs : ['kb01_loan_eligibility']
     true positives : ['kb01_loan_eligibility']
     precision = 1/2 = 0.5000
     recall    = 1/1 = 1.0000
     f1        = 0.6667
  Q: How is the EMI calculated on a reducing balance loan?
     retrieved docs : ['kb02_emi_calculation', 'kb09_minimum_balance']
     relevant  docs : ['kb02_emi_calculation']
     true positives : ['kb02_emi_calculation']
     precision = 1/2 = 0.5000
     recall    = 1/1 = 1.0000
     f1        = 0.6667
  Q: What is the cash advance fee when I withdraw cash on my credit car
     retrieved docs : ['kb03_credit_card_fees', 'kb06_account_closure']
     relevant  docs : ['kb03_credit_card_fees']
     true positives : ['kb03_credit_card_fees']
     precision = 1/2 = 0.5000
     recall    = 1/1 = 1.0000
     f1        = 0.6667
  Q: Which documents count as proof of address for KYC?
     retrieved docs : ['kb04_kyc_documents']
     relevant  docs : ['kb04_kyc_documents']
     true positives : ['kb04_kyc_documents']
     precision = 1/1 = 1.0000
     recall    = 1/1 = 1.0000
     f1        = 1.0000
  Q: How long do I have to report an unauthorised transaction to have z
     retrieved docs : ['kb05_fraud_dispute']
     relevant  docs : ['kb05_fraud_dispute']
     true positives : ['kb05_fraud_dispute']
     precision = 1/1 = 1.0000
     recall    = 1/1 = 1.0000
     f1        = 1.0000
  MEAN precision=0.7000 recall=1.0000 f1=0.8000
```


### chunk integrity check on the same query

```
  kb_fixed_overlap
    answer ends mid-sentence: False
    answer: Accepted address documents are the Aadhaar Accepted address documents are the Aadhaar letter, passport, a utility bill not older than two months, or a registered rent agreement. Every applicant must submit one officially valid document for 
  kb_sentence
    answer ends mid-sentence: False
    answer: Accepted address documents are the Aadhaar letter, passport, a utility bill not older than two months, or a registered rent agreement. Re-KYC is required every two years for high-risk customers, every eight years for medium-risk and every t

  fixed windows that end mid-sentence: 23/36
    [kb01_loan_eligibility::fixed::0] ... the sanctioned amount is capped at 80 percent of the assessed property value. Auto Loan
    [kb01_loan_eligibility::fixed::1] ...n letter from a recognised institution and a co-applicant who is an earning parent or gu
    [kb02_emi_calculation::fixed::0] ...so the interest component of each instalment falls and the principal component rises acr
  sentence chunks that end mid-sentence: 0/48
```


**Recommendation: deploy `kb_sentence` (sentence chunking).**

On the numbers, `kb_fixed_overlap` scores mean precision 0.8000 / recall 1.0000 / F1 0.8667 against `kb_sentence` at 0.7000 / 1.0000 / 0.8000, so `kb_fixed_overlap` is ahead on F1. Recall is identical at 1.0000 for both - every query retrieves the right document either way - so the whole difference is precision, and it comes partly from granularity: 36 fixed chunks against 48 sentence chunks means the top-k spreads across fewer parent documents.

What precision does not measure is whether a retrieved chunk is a complete policy statement. The check above shows 23 of 36 fixed windows end mid-sentence, and on the KYC query that truncation reaches the generated answer: the list of accepted address documents is cut after its first item, so a member is told Aadhaar is the address proof and the passport, utility bill and rent agreement are silently dropped. Nothing downstream catches it - the fragment's words are in the retrieved context, so both the groundedness guardrail and the Autogen reviewer pass it. Sentence chunking cannot produce that failure because it never splits mid-clause.

For a High-risk financial system, a strategy that is slightly cleaner on average but can hand a member a truncated policy rule is the worse deployment. The next improvement would be chunking on sentence boundaries with a character-count target, which should take the precision of fixed windows with the integrity of sentence chunks.

