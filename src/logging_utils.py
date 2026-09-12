"""Task 12 - ELK-style structured JSON-Lines request logging with a trace id.

The logged request text is the guardrail-MASKED text, never the raw input. The same
masking that protects what the model sees protects what reaches disk, so a fixed-format
PII field never lands in the log file in the clear.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from .config import LOG_DIR, LOG_FILE
from .guardrails import mask_pii


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def write_entry(entry: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


@contextmanager
def request_span(
    endpoint: str,
    request_text: str,
    session_id: str = "default",
    trace_id: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Emit exactly one JSON-Lines entry per request, with timing and a trace id."""
    trace_id = trace_id or new_trace_id()
    started = time.perf_counter()
    started_wall = time.time()
    span: Dict[str, Any] = {"trace_id": trace_id, "extra": {}}
    status = "ok"
    error: Optional[str] = None
    try:
        yield span
    except Exception as exc:  # noqa: BLE001 - we log then re-raise
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        span["latency_ms"] = latency_ms
        entry = {
            "trace_id": trace_id,
            "ts": round(started_wall, 3),
            "level": "INFO" if status == "ok" else "ERROR",
            "service": "cred-domain-support-agent",
            "endpoint": endpoint,
            "session_id": session_id,
            # masked, always - this is the PII control for Task 12
            "request": mask_pii(request_text).text,
            "status": status,
            "latency_ms": latency_ms,
        }
        if error:
            entry["error"] = error
        entry.update(span.get("extra", {}))
        write_entry(entry)


def read_log(limit: int = 50) -> list[Dict[str, Any]]:
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines[-limit:]]
