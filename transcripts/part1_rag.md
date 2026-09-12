# Part 1 - Dataset Design & RAG Core (Tasks 1-5)

*embedding backend: `hashed-tfidf-4096d (offline fallback)`*


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
collection kb_fixed_overlap     count after upsert: 36
collection kb_sentence          count after upsert: 48

sanity query 'foreclosure charge on a personal loan':
  kb_fixed_overlap
    0.4653  kb08_prepayment_penalty  A floating-rate loan taken by an individual borrower for a non-business purpose ca...
    0.2460  kb08_prepayment_penalty  percent between months thirteen and twenty-four, and 2 percent thereafter. Part-pr...
  kb_sentence
    0.3877  kb08_prepayment_penalty  A floating-rate loan taken by an individual borrower for a non-business purpose ca...
    0.3527  kb08_prepayment_penalty  A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outst...
```


## Task 4 - calibrated grounded generation


### calibration.json (measured, written by scripts/calibrate_threshold.py)

```
{
  "hashed-tfidf-4096d (offline fallback)": {
    "backend": "hashed-tfidf-4096d (offline fallback)",
    "measurements": {
      "kb_fixed_overlap": {
        "chosen_threshold": 0.137,
        "highest_out_of_scope": 0.12,
        "in_scope": {
          "How is the EMI calculated on a reducing balance loan?": 0.154,
          "What is the average monthly balance requirement in a metro branch?": 0.3091,
          "What is the foreclosure charge on a fixed rate personal loan?": 0.5259,
          "What is the minimum income needed for a personal loan?": 0.242,
          "Which documents count as proof of address for KYC?": 0.2256
        },
        "lowest_in_scope": 0.154,
        "out_of_scope": {
          "How do I change the timing belt on a diesel engine?": 0.1197,
          "What is Cred's policy on cryptocurrency trading accounts for members?": 0.12,
          "What is the best recipe for hyderabadi biryani?": 0.0392,
          "Who won the football world cup in 2018?": 0.0902
        },
        "separation_gap": 0.034
      },
      "kb_sentence": {
        "chosen_threshold": 0.1475,
        "highest_out_of_scope": 0.1431,
        "in_scope": {
          "How is the EMI calculated on a reducing balance loan?": 0.1519,
          "What is the average monthly balance requirement in a metro branch?": 0.3186,
          "What is the foreclosure charge on a fixed rate personal loan?": 0.481,
          "What is the minimum income needed for a personal loan?": 0.2169,
          "Which documents count as proof of address for KYC?": 0.1767
        },
        "lowest_in_scope": 0.1519,
        "out_of_scope": {
          "How do I change the timing belt on a diesel engine?": 0.084,
          "What is Cred's policy on cryptocurrency trading accounts for members?": 0.1431,
          "What is the best recipe for hyderabadi biryani?": 0.041,
          "Who won the football world cup in 2018?": 0.0887
        },
        "separation_gap": 0.0088
      }
    },
    "thresholds": {
      "kb_fixed_overlap": 0.137,
      "kb_sentence": 0.1475
    }
  }
}
```


### 5 in-scope queries + 1 out-of-scope fallback

```
threshold in force for kb_sentence: 0.1475

Q: What is the minimum monthly income needed to qualify for a personal loan?
   top_similarity=0.2407 refused=False
   A: A Personal Loan applicant must be a salaried or self-employed resident between 21 and 58 years of age with a minimum net monthly income of INR 25,000 and at least six months in the current job. A fixed-rate Personal Loan attracts a foreclosure charge of 4 percent of the outstanding principal if clos
   sources: ['kb01_loan_eligibility', 'kb08_prepayment_penalty']

Q: How is the EMI calculated on a reducing balance loan?
   top_similarity=0.1519 refused=False
   A: The equated monthly instalment is computed with the standard reducing-balance formula EMI = P x r x (1+r)^n / ((1+r)^n - 1), where P is the sanctioned principal, r is the monthly interest rate (annual rate divided by twelve, expressed as a decimal) and n is the tenure in months. An Education Loan ne
   sources: ['kb01_loan_eligibility', 'kb02_emi_calculation']

Q: What is the cash advance fee when I withdraw cash on my credit card?
   top_similarity=0.2520 refused=False
   A: Cash withdrawal at an ATM using the card attracts a cash advance fee of 2.5 percent of the withdrawn amount subject to a floor of INR 500, and interest on the withdrawn amount runs from the transaction date with no interest-free period. A credit-card account cannot be closed while any amount remains
   sources: ['kb03_credit_card_fees', 'kb06_account_closure']

Q: Which documents count as proof of address for KYC?
   top_similarity=0.1767 refused=False
   A: Every applicant must submit one officially valid document for proof of identity and one for proof of address before an account or a loan can be activated. Accepted address documents are the Aadhaar letter, passport, a utility bill not older than two months, or a registered rent agreement. Self-emplo
   sources: ['kb04_kyc_documents']

Q: How long do I have to report an unauthorised transaction to have zero liability?
   top_similarity=0.3015 refused=False
   A: A customer must report an unauthorised transaction through the in-app dispute form, the 24x7 helpline or a written complaint, and the reporting timestamp is what governs liability. Where the delay in reporting is four to seven working days, the customer's liability is capped at the transaction value
   sources: ['kb05_fraud_dispute']

OUT-OF-SCOPE Q: What is the best recipe for hyderabadi biryani?
   top_similarity=0.0410 refused=True
   A: I don't know based on the available policy documents. This question is outside the Cred lending-policy knowledge base, so I am not going to answer it from memory. Please route it to a human agent.
```


## Task 5 - document-level precision / recall for BOTH strategies


### per-query arithmetic, both collections

```
collection: kb_fixed_overlap
  Q: What is the minimum monthly income needed to qualify for a persona
     retrieved docs : ['kb01_loan_eligibility', 'kb07_interest_rate_slabs', 'kb08_prepayment_penalty']
     relevant  docs : ['kb01_loan_eligibility']
     true positives : ['kb01_loan_eligibility']
     precision = 1/3 = 0.3333
     recall    = 1/1 = 1.0000
     f1        = 0.5000
  Q: How is the EMI calculated on a reducing balance loan?
     retrieved docs : ['kb01_loan_eligibility', 'kb02_emi_calculation', 'kb08_prepayment_penalty']
     relevant  docs : ['kb02_emi_calculation']
     true positives : ['kb02_emi_calculation']
     precision = 1/3 = 0.3333
     recall    = 1/1 = 1.0000
     f1        = 0.5000
  Q: What is the cash advance fee when I withdraw cash on my credit car
     retrieved docs : ['kb03_credit_card_fees', 'kb04_kyc_documents', 'kb06_account_closure']
     relevant  docs : ['kb03_credit_card_fees']
     true positives : ['kb03_credit_card_fees']
     precision = 1/3 = 0.3333
     recall    = 1/1 = 1.0000
     f1        = 0.5000
  Q: Which documents count as proof of address for KYC?
     retrieved docs : ['kb04_kyc_documents', 'kb11_joint_account']
     relevant  docs : ['kb04_kyc_documents']
     true positives : ['kb04_kyc_documents']
     precision = 1/2 = 0.5000
     recall    = 1/1 = 1.0000
     f1        = 0.6667
  Q: How long do I have to report an unauthorised transaction to have z
     retrieved docs : ['kb05_fraud_dispute', 'kb09_minimum_balance']
     relevant  docs : ['kb05_fraud_dispute']
     true positives : ['kb05_fraud_dispute']
     precision = 1/2 = 0.5000
     recall    = 1/1 = 1.0000
     f1        = 0.6667
  MEAN precision=0.4000 recall=1.0000 f1=0.5667

collection: kb_sentence
  Q: What is the minimum monthly income needed to qualify for a persona
     retrieved docs : ['kb01_loan_eligibility', 'kb08_prepayment_penalty']
     relevant  docs : ['kb01_loan_eligibility']
     true positives : ['kb01_loan_eligibility']
     precision = 1/2 = 0.5000
     recall    = 1/1 = 1.0000
     f1        = 0.6667
  Q: How is the EMI calculated on a reducing balance loan?
     retrieved docs : ['kb01_loan_eligibility', 'kb02_emi_calculation']
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


**Recommendation.** Sentence chunking scores mean precision 0.7000 / recall 1.0000 / F1 0.8000, against fixed-size-with-overlap at 0.4000 / 1.0000 / 0.5667 on the same five queries. I would deploy **kb_sentence**: it is at least as accurate on these numbers, and because it never splits a sentence, a retrieved chunk is always a complete policy statement, which is what the grounded-generation step and the Autogen compliance reviewer both read. A fixed window that cuts 'a foreclosure charge of 4 percent of the outstanding principal if closed within' mid-clause retrieves a fragment that can be quoted misleadingly even when the similarity score looks healthy.

