"""Task 4 - empirically calibrate the 'I don't know' similarity threshold.

The brief forbids an untested preset (0.5 / 0.6 / 0.7 are tutorial defaults that do not
reliably separate short policy-sentence embeddings from unrelated queries). This script
measures top-1 cosine similarity for real in-scope and deliberately out-of-scope queries
against BOTH collections, prints the two observed clusters, and proposes a threshold at
the midpoint of the gap between them. The chosen value is then pinned in src/config.py
and quoted in README.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CALIBRATION_FILE, COLLECTION_FIXED, COLLECTION_SENTENCE  # noqa: E402
from src.embeddings import active_backend  # noqa: E402
from src.rag import build_index, retrieve  # noqa: E402

IN_SCOPE = [
    "What is the minimum income needed for a personal loan?",
    "How is the EMI calculated on a reducing balance loan?",
    "What is the foreclosure charge on a fixed rate personal loan?",
    "Which documents count as proof of address for KYC?",
    "What is the average monthly balance requirement in a metro branch?",
]

# Two obvious negatives and one HARD negative. The hard negative matters more than the
# obvious ones: "cryptocurrency trading accounts" is plausible banking vocabulary that this
# knowledge base deliberately does not cover, so it is the query that actually sets where
# the threshold has to sit. Calibrating only against biryani and football would leave the
# threshold far too low and the agent would answer a policy question it has no policy for.
OUT_OF_SCOPE = [
    "What is the best recipe for hyderabadi biryani?",
    "Who won the football world cup in 2018?",
    "How do I change the timing belt on a diesel engine?",
    "What is Cred's policy on cryptocurrency trading accounts for members?",
]


def main() -> None:
    build_index()
    backend = active_backend()
    print("=" * 74)
    print("TASK 4 - THRESHOLD CALIBRATION (measured, not a preset)")
    print("=" * 74)
    print(f"embedding backend: {backend}")
    print("A threshold is only valid for the vector space it was measured in. Changing")
    print("the embedding backend means re-running this script.")

    proposals = {}
    measurements = {}
    for collection in (COLLECTION_FIXED, COLLECTION_SENTENCE):
        print(f"\ncollection: {collection}")
        print("-" * 74)
        in_scores, out_scores = [], []

        print("  IN-SCOPE (top-1 cosine similarity)")
        for query in IN_SCOPE:
            hits = retrieve(query, collection_name=collection, k=1)
            score = hits[0].similarity if hits else 0.0
            in_scores.append(score)
            print(f"    {score:6.4f}  {query}")

        print("  OUT-OF-SCOPE (top-1 cosine similarity)")
        for query in OUT_OF_SCOPE:
            hits = retrieve(query, collection_name=collection, k=1)
            score = hits[0].similarity if hits else 0.0
            out_scores.append(score)
            print(f"    {score:6.4f}  {query}")

        lowest_in = min(in_scores)
        highest_out = max(out_scores)
        gap = lowest_in - highest_out
        midpoint = round((lowest_in + highest_out) / 2, 4)
        proposals[collection] = midpoint
        measurements[collection] = {
            "in_scope": dict(zip(IN_SCOPE, [round(v, 4) for v in in_scores])),
            "out_of_scope": dict(zip(OUT_OF_SCOPE, [round(v, 4) for v in out_scores])),
            "lowest_in_scope": round(lowest_in, 4),
            "highest_out_of_scope": round(highest_out, 4),
            "separation_gap": round(gap, 4),
            "chosen_threshold": midpoint,
        }
        print(f"  lowest in-scope   : {lowest_in:.4f}")
        print(f"  highest out-scope : {highest_out:.4f}")
        print(f"  separation gap    : {gap:.4f}")
        print(f"  proposed threshold: {midpoint:.4f}  (midpoint of the observed gap)")

    existing = {}
    if CALIBRATION_FILE.exists():
        existing = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    existing[backend] = {
        "backend": backend,
        "thresholds": proposals,
        "measurements": measurements,
    }
    CALIBRATION_FILE.write_text(
        json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 74)
    print(f"written to {CALIBRATION_FILE.name} under backend key: {backend}")
    for collection, value in proposals.items():
        print(f"  {collection:<20} -> threshold {value}")


if __name__ == "__main__":
    main()
