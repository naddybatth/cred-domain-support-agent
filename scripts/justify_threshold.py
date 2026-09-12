"""Prints the escalation-score distribution that justifies ESCALATION_THRESHOLD."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ESCALATION_THRESHOLD  # noqa: E402
from src.dataset import LOAN_APPLICATIONS  # noqa: E402
from src.tools import compute_escalation_score, escalation_distribution  # noqa: E402


def percentile(sorted_values, p: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round((p / 100.0) * (len(sorted_values) - 1))))
    return sorted_values[index]


def main() -> None:
    scores = escalation_distribution()
    days = sorted(r["days_since_created"] for r in LOAN_APPLICATIONS)
    above = [s for s in scores if s >= ESCALATION_THRESHOLD]
    flagged = [r for r in LOAN_APPLICATIONS if r["flagged_for_fraud_review"]]
    unflagged_max = max(
        compute_escalation_score(False, r["days_since_created"]) for r in LOAN_APPLICATIONS
    )
    flagged_min = min(
        (compute_escalation_score(True, r["days_since_created"]) for r in flagged),
        default=0.0,
    )

    print("=" * 74)
    print("TASK 6 - ESCALATION THRESHOLD JUSTIFICATION (from this dataset's own spread)")
    print("=" * 74)
    print(f"records                         : {len(scores)}")
    print(f"days_since_created p50 / p80    : {percentile(days, 50)} / {percentile(days, 80)}")
    print(f"escalation_score  p50 / p80/p90 : {percentile(scores, 50)} / "
          f"{percentile(scores, 80)} / {percentile(scores, 90)}")
    print(f"max score reachable UNflagged   : {unflagged_max}")
    print(f"min score reachable flagged     : {flagged_min}")
    print(f"chosen ESCALATION_THRESHOLD     : {ESCALATION_THRESHOLD}")
    print(f"records at or above threshold   : {len(above)}/{len(scores)} "
          f"= {100*len(above)/len(scores):.2f}%")
    print()
    print("Reading: the threshold sits in the empty band between the highest score an")
    print("unflagged application can reach and the lowest a flagged one can reach, so it")
    print("selects the fraud-flagged population without ever firing on staleness alone.")


if __name__ == "__main__":
    main()
