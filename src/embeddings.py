"""Embedding backend.

PRIMARY (and what the brief asks for): a free, local SentenceTransformers model
(all-MiniLM-L6-v2). It runs on CPU, needs no API key, and is downloaded once from the
Hugging Face hub on first use, after which it is served from the local cache.

FALLBACK: some sandboxes block the Hugging Face hub outright, which makes the primary
backend impossible to even download. Rather than let the whole pipeline fail there, this
module falls back to a deterministic, fully local TF-IDF + truncated-SVD encoder fitted on
the knowledge base itself. It is a genuine distributional embedding - not a random hash -
so retrieval, thresholds and precision/recall all remain meaningful, and it is seeded so
every run reproduces exactly.

Which backend produced a given run is always reported: active_backend() is printed in
every transcript and in the calibration output, because the calibrated similarity
threshold is backend-specific and must be re-measured if the backend changes.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Sequence

import numpy as np

from .config import EMBED_MODEL_NAME

FORCE_FALLBACK = os.getenv("EMBED_BACKEND", "").lower() == "fallback"
HASH_FEATURES = 4096

_BACKEND_NAME = {"value": "uninitialised"}


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class _SentenceTransformerBackend:
    name = f"sentence-transformers:{EMBED_MODEL_NAME}"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(EMBED_MODEL_NAME)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in np.asarray(vectors)]


class _TfidfBackend:
    """Deterministic offline fallback: stateless hashed TF-IDF.

    Two earlier attempts were rejected, and why:

      * truncated SVD over the KB corpus - on a corpus this small SVD densified the space
        so much that an unrelated query ("how do I change the timing belt on a diesel
        engine") scored 0.98 against lending policy, which would have destroyed the
        out-of-scope fallback that Task 4 depends on;
      * a fitted TfidfVectorizer - separation was good, but the vector dimension is the
        fitted vocabulary size, so the moment POST /add-document adds a thirteenth policy
        document the vocabulary grows and ChromaDB rejects the new vectors with
        "Collection expecting embedding with dimension of 1361, got 1387".

    HashingVectorizer fixes both: a fixed 4096-dimensional hashed feature space that never
    changes as documents are added, with sublinear term frequencies and L2 normalisation,
    and no fitted state at all - so it is reproducible across machines and processes.
    """

    name = f"hashed-tfidf-{HASH_FEATURES}d (offline fallback)"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vectorizer = HashingVectorizer(
            n_features=HASH_FEATURES,
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            alternate_sign=False,
            norm="l2",
        )

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        matrix = self._vectorizer.transform(list(texts)).toarray()
        return _l2_normalize(np.asarray(matrix, dtype=np.float64)).tolist()


@lru_cache(maxsize=1)
def _backend():
    if not FORCE_FALLBACK:
        try:
            backend = _SentenceTransformerBackend()
            _BACKEND_NAME["value"] = backend.name
            return backend
        except Exception as error:  # noqa: BLE001 - any download/import failure
            print(
                f"[embeddings] SentenceTransformers unavailable ({type(error).__name__}); "
                "falling back to the deterministic offline TF-IDF encoder. "
                "Re-run with network access to the Hugging Face hub to use the primary "
                "backend, and re-run scripts/calibrate_threshold.py afterwards."
            )
    backend = _TfidfBackend()
    _BACKEND_NAME["value"] = backend.name
    return backend


def embed(texts: Sequence[str]) -> List[List[float]]:
    return _backend().encode(texts)


def embedding_dim() -> int:
    return len(embed(["dimension probe"])[0])


def active_backend() -> str:
    _backend()
    return _BACKEND_NAME["value"]
