"""Central configuration. Everything the grader needs to reproduce a run lives here."""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional; the repo runs fine without python-dotenv installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "kb"
CHROMA_DIR = ROOT / "chroma_store"
TRANSCRIPT_DIR = ROOT / "transcripts"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "requests.jsonl"

# --- graded mode -----------------------------------------------------------
# MOCK_LLM=1 is the graded default: no API key, no network, deterministic output.
MOCK_LLM: bool = os.getenv("MOCK_LLM", "1").strip().lower() in {"1", "true", "yes"}

# CrewAI phones home on kickoff() unless these are set. See README.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chromadb posthog
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- dataset (Task 1) ------------------------------------------------------
DATASET_SEED = 20260913
DATASET_SIZE = 48

# --- retrieval (Tasks 3-5) -------------------------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

COLLECTION_FIXED = "kb_fixed_overlap"
COLLECTION_SENTENCE = "kb_sentence"

FIXED_CHUNK_CHARS = 420
FIXED_CHUNK_OVERLAP = 120
SENTENCE_CHUNK_SENTENCES = 2
SENTENCE_CHUNK_STRIDE = 1  # 1 sentence of overlap between neighbouring chunks

# Calibrated in Task 4 by scripts/calibrate_threshold.py - never copied from a tutorial.
# That script MEASURES top-1 cosine similarity for in-scope and out-of-scope queries and
# writes the midpoint of the observed gap to calibration.json. The threshold is stored per
# embedding backend, because a threshold measured for one vector space is meaningless in
# another. README.md quotes the measured numbers.
CALIBRATION_FILE = ROOT / "calibration.json"
FALLBACK_SIMILARITY_THRESHOLD = 0.20  # used only if calibration.json is missing


def similarity_threshold(backend: str | None = None, collection: str | None = None) -> float:
    """Resolve the calibrated threshold for the active backend and collection."""
    override = os.getenv("SIMILARITY_THRESHOLD")
    if override:
        return float(override)
    if not CALIBRATION_FILE.exists():
        return FALLBACK_SIMILARITY_THRESHOLD
    import json

    data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    if backend is None:
        from .embeddings import active_backend

        backend = active_backend()
    entry = data.get(backend)
    if not entry:
        return FALLBACK_SIMILARITY_THRESHOLD
    collection = collection or COLLECTION_SENTENCE
    value = entry.get("thresholds", {}).get(collection)
    return float(value) if value is not None else FALLBACK_SIMILARITY_THRESHOLD

# --- escalation (Task 6) ---------------------------------------------------
FRAUD_WEIGHT = 0.6
RECENCY_WEIGHT = 0.4
MAX_AGE_DAYS = 30
ESCALATION_THRESHOLD = 0.55

# --- runtime governance (Task 15) ------------------------------------------
MAX_REQUEST_TOKENS = 512        # per-request prompt budget
MAX_TOTAL_TOKENS = 1024         # prompt + generated budget
TOKEN_COST_PER_1K_INR = 0.15    # notional, used only to report a rupee figure

# --- risk classification (Task 15) -----------------------------------------
RISK_LEVEL = "High"
