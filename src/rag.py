"""Tasks 3-5 - embeddings, two ChromaDB collections, grounded generation, P/R comparison."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

from .chunking import Chunk, Document, fixed_size_chunks, load_documents, sentence_chunks
from .config import (
    CHROMA_DIR,
    COLLECTION_FIXED,
    COLLECTION_SENTENCE,
    similarity_threshold,
    TOP_K,
)
from .mock_llm import MOCK

REFUSAL = (
    "I don't know based on the available policy documents. "
    "This question is outside the Cred lending-policy knowledge base, so I am not going "
    "to answer it from memory. Please route it to a human agent."
)


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    similarity: float


from .embeddings import active_backend, embed  # noqa: F401  (re-exported)


@lru_cache(maxsize=1)
def _client():
    import chromadb
    from chromadb.config import Settings

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )


def _collection(name: str):
    # cosine space, because every similarity number reported in this repo is a cosine.
    return _client().get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def index_chunks(name: str, chunks: Sequence[Chunk]) -> int:
    """Task 3 - embed every chunk and upsert it into its own collection."""
    collection = _collection(name)
    vectors = embed([c.text for c in chunks])
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[c.as_metadata() for c in chunks],
    )
    return collection.count()


def build_index(force: bool = False) -> Dict[str, int]:
    """Build BOTH collections. Idempotent - upsert means a rerun is safe."""
    documents = load_documents()
    counts = {
        COLLECTION_FIXED: index_chunks(COLLECTION_FIXED, fixed_size_chunks(documents)),
        COLLECTION_SENTENCE: index_chunks(COLLECTION_SENTENCE, sentence_chunks(documents)),
    }
    return counts


def retrieve(query: str, collection_name: str = COLLECTION_SENTENCE, k: int = TOP_K) -> List[Hit]:
    collection = _collection(collection_name)
    result = collection.query(
        query_embeddings=embed([query]),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits: List[Hit] = []
    for chunk_id, text, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append(
            Hit(
                chunk_id=chunk_id,
                doc_id=str(metadata.get("doc_id", "")),
                doc_title=str(metadata.get("doc_title", "")),
                text=text,
                # chroma cosine distance = 1 - cosine similarity
                similarity=round(1.0 - float(distance), 4),
            )
        )
    return hits


def answer(
    query: str,
    collection_name: str = COLLECTION_SENTENCE,
    k: int = TOP_K,
    threshold: float | None = None,
) -> Dict[str, Any]:
    """Task 4 - grounded generation with a calibrated 'I don't know' fallback."""
    if threshold is None:
        threshold = similarity_threshold(collection=collection_name)
    hits = retrieve(query, collection_name=collection_name, k=k)
    top_similarity = hits[0].similarity if hits else 0.0

    if not hits or top_similarity < threshold:
        return {
            "query": query,
            "answer": REFUSAL,
            "refused": True,
            "top_similarity": top_similarity,
            "threshold": threshold,
            "collection": collection_name,
            "hits": [h.__dict__ for h in hits],
            "source_documents": sorted({h.doc_id for h in hits}) if hits else [],
        }

    generated = MOCK.generate_grounded(query, [h.text for h in hits])
    return {
        "query": query,
        "answer": generated,
        "refused": generated.startswith("I don't know"),
        "top_similarity": top_similarity,
        "threshold": threshold,
        "collection": collection_name,
        "hits": [h.__dict__ for h in hits],
        "source_documents": sorted({h.doc_id for h in hits}),
    }


# ---------------------------------------------------------------------------
# Task 5 - document-level precision / recall
# ---------------------------------------------------------------------------
def precision_recall(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]
) -> Dict[str, Any]:
    """Chunks are mapped to parent documents and de-duplicated BEFORE scoring."""
    retrieved = sorted(set(retrieved_doc_ids))
    relevant = sorted(set(relevant_doc_ids))
    true_positives = sorted(set(retrieved) & set(relevant))
    precision = len(true_positives) / len(retrieved) if retrieved else 0.0
    recall = len(true_positives) / len(relevant) if relevant else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "retrieved_docs": retrieved,
        "relevant_docs": relevant,
        "true_positives": true_positives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "precision_arithmetic": f"{len(true_positives)}/{len(retrieved) or 0}",
        "recall_arithmetic": f"{len(true_positives)}/{len(relevant) or 0}",
    }


def compare_strategies(
    labelled_queries: Sequence[Tuple[str, List[str]]], k: int = TOP_K
) -> Dict[str, Any]:
    """Run the SAME queries through both collections and report per-query arithmetic."""
    report: Dict[str, Any] = {}
    for collection_name in (COLLECTION_FIXED, COLLECTION_SENTENCE):
        rows = []
        for query, relevant in labelled_queries:
            hits = retrieve(query, collection_name=collection_name, k=k)
            scored = precision_recall([h.doc_id for h in hits], relevant)
            scored["query"] = query
            scored["top_similarity"] = hits[0].similarity if hits else 0.0
            rows.append(scored)
        report[collection_name] = {
            "per_query": rows,
            "mean_precision": round(sum(r["precision"] for r in rows) / len(rows), 4),
            "mean_recall": round(sum(r["recall"] for r in rows) / len(rows), 4),
            "mean_f1": round(sum(r["f1"] for r in rows) / len(rows), 4),
        }
    return report


if __name__ == "__main__":
    print(json.dumps(build_index(), indent=2))
