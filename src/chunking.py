"""Task 3 (part one) - two independent chunking strategies over the same knowledge base.

Both strategies keep a pointer back to the parent document, because Task 5 scores
precision and recall at the *document* level and therefore has to map chunks up.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from .config import (
    FIXED_CHUNK_CHARS,
    FIXED_CHUNK_OVERLAP,
    KB_DIR,
    SENTENCE_CHUNK_SENTENCES,
    SENTENCE_CHUNK_STRIDE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    strategy: str
    text: str

    def as_metadata(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("text")
        return data


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str


def load_documents(kb_dir=KB_DIR) -> List[Document]:
    """Read the 12 authored markdown policy documents off disk, in stable order."""
    documents: List[Document] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        lines = raw.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        documents.append(Document(doc_id=path.stem, title=title, body=body))
    if not documents:
        raise RuntimeError(f"no knowledge-base documents found under {kb_dir}")
    return documents


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def fixed_size_chunks(
    documents: List[Document],
    size: int = FIXED_CHUNK_CHARS,
    overlap: int = FIXED_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Strategy A: fixed character window with a sliding overlap.

    Cheap and uniform, but a window boundary can cut a policy sentence in half, which is
    exactly the failure mode Task 5 is asking us to measure rather than assume.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than the window")
    chunks: List[Chunk] = []
    step = size - overlap
    for doc in documents:
        body = doc.body
        start, index = 0, 0
        while start < len(body):
            window = body[start : start + size].strip()
            if window:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::fixed::{index}",
                        doc_id=doc.doc_id,
                        doc_title=doc.title,
                        strategy="fixed_overlap",
                        text=window,
                    )
                )
                index += 1
            if start + size >= len(body):
                break
            start += step
    return chunks


def sentence_chunks(
    documents: List[Document],
    sentences_per_chunk: int = SENTENCE_CHUNK_SENTENCES,
    stride_overlap: int = SENTENCE_CHUNK_STRIDE,
) -> List[Chunk]:
    """Strategy B: groups of whole sentences, overlapping by `stride_overlap` sentences.

    Never splits mid-clause, so a retrieved chunk is always a readable policy statement.
    """
    step = max(1, sentences_per_chunk - stride_overlap)
    chunks: List[Chunk] = []
    for doc in documents:
        sentences = split_sentences(doc.body)
        index, cursor = 0, 0
        while cursor < len(sentences):
            window = sentences[cursor : cursor + sentences_per_chunk]
            text = " ".join(window).strip()
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::sent::{index}",
                        doc_id=doc.doc_id,
                        doc_title=doc.title,
                        strategy="sentence",
                        text=text,
                    )
                )
                index += 1
            if cursor + sentences_per_chunk >= len(sentences):
                break
            cursor += step
    return chunks


def main() -> None:
    documents = load_documents()
    fixed = fixed_size_chunks(documents)
    sentence = sentence_chunks(documents)
    print("=" * 68)
    print("TASK 3 - CHUNKING")
    print("=" * 68)
    print(f"documents            : {len(documents)}")
    print(f"fixed_overlap chunks : {len(fixed)}  "
          f"(window={FIXED_CHUNK_CHARS} chars, overlap={FIXED_CHUNK_OVERLAP})")
    print(f"sentence chunks      : {len(sentence)}  "
          f"({SENTENCE_CHUNK_SENTENCES} sentences, {SENTENCE_CHUNK_STRIDE} overlapping)")
    print()
    print("sample fixed_overlap chunk:")
    print("  ", fixed[0].chunk_id, "->", fixed[0].text[:110], "...")
    print("sample sentence chunk:")
    print("  ", sentence[0].chunk_id, "->", sentence[0].text[:110], "...")


if __name__ == "__main__":
    main()
