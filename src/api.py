"""Task 11 + 12 - FastAPI deployment: 2 HTTP endpoints, 1 WebSocket, structured logging.

Run with:  uvicorn src.api:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from .config import KB_DIR, MOCK_LLM
from .embeddings import active_backend
from .logging_utils import new_trace_id, request_span
from .pipeline import handle_query
from .schemas import (
    AddDocumentRequest,
    AddDocumentResponse,
    AskRequest,
    AskResponse,
)

app = FastAPI(
    title="Cred Domain Support Agent",
    version="1.0.0",
    description=(
        "Banking & FinTech capstone. RAG over a hand-authored lending-policy knowledge "
        "base, a CrewAI crew with a record-lookup tool, guardrails, an Autogen review "
        "stage and a governance layer. Runs entirely under MOCK_LLM with no API key."
    ),
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "mock_llm": MOCK_LLM,
        "embedding_backend": active_backend(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Ask the support agent one question. Every call writes one JSON-Lines log entry."""
    result = handle_query(
        request.query,
        session_id=request.session_id,
        endpoint="POST /ask",
    )
    return AskResponse(
        trace_id=result["trace_id"],
        latency_ms=result.get("latency_ms") or 0.0,
        response=result["response"],
    )


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(request: AddDocumentRequest) -> AddDocumentResponse:
    """Add a policy document to the knowledge base and re-index BOTH collections."""
    from .chunking import Document, fixed_size_chunks, sentence_chunks
    from .config import COLLECTION_FIXED, COLLECTION_SENTENCE
    from .rag import index_chunks

    trace_id = new_trace_id()
    with request_span("POST /add-document", request.title, trace_id=trace_id) as span:
        safe_id = "".join(c for c in request.doc_id if c.isalnum() or c in "_-")
        if not safe_id:
            raise HTTPException(status_code=400, detail="doc_id must be alphanumeric")

        path = KB_DIR / f"{safe_id}.md"
        path.write_text(f"# {request.title}\n{request.body.strip()}\n", encoding="utf-8")

        document = Document(doc_id=safe_id, title=request.title, body=request.body.strip())
        fixed = index_chunks(COLLECTION_FIXED, fixed_size_chunks([document]))
        sentence = index_chunks(COLLECTION_SENTENCE, sentence_chunks([document]))
        span["extra"] = {"doc_id": safe_id, "collection_sizes":
                         {COLLECTION_FIXED: fixed, COLLECTION_SENTENCE: sentence}}

    return AddDocumentResponse(
        trace_id=trace_id,
        doc_id=safe_id,
        chunks_indexed_fixed=fixed,
        chunks_indexed_sentence=sentence,
    )


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """Real-time multi-turn chat.

    A client that disconnects mid-conversation is caught explicitly; the server keeps
    running and other clients are unaffected.
    """
    await websocket.accept()
    session_id = f"ws-{new_trace_id()}"
    try:
        await websocket.send_json(
            {"type": "ready", "session_id": session_id,
             "message": "Connected to the Cred support agent. Send {\"query\": \"...\"}."}
        )
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = {"query": raw}

            try:
                request = AskRequest(
                    query=payload.get("query", ""),
                    session_id=payload.get("session_id", session_id),
                )
            except ValidationError as error:
                await websocket.send_json({"type": "error", "detail": error.errors()})
                continue

            # handle_query is synchronous and does real work; run it off the event
            # loop so one slow turn cannot stall the socket or the rest of the server.
            result = await run_in_threadpool(
                handle_query,
                request.query,
                session_id=request.session_id,
                endpoint="WS /ws/chat",
            )
            await websocket.send_json(
                {
                    "type": "answer",
                    "trace_id": result["trace_id"],
                    "session_id": request.session_id,
                    "response": result["response"].model_dump(),
                }
            )
    except WebSocketDisconnect:
        # Expected whenever a client closes the tab mid-conversation. Swallowed on
        # purpose: the event loop and every other connection stay alive.
        print(f"[ws] client disconnected cleanly, session={session_id}; server still up")
    except Exception as error:  # noqa: BLE001 - never take the server down for one client
        print(f"[ws] session={session_id} failed: {type(error).__name__}: {error}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
