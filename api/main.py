import sys
import json as _json
import datetime
import pathlib
import threading
from pathlib import Path

# ensure project root is on path so rag/ and ingestion/ are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.query_engine import (
    query,
    query_prepare,
    generate_response_stream,
    correct_section,
    get_db_table,
)

VECTORDB_DIR   = Path("/home/justin/rag-civil/vectordb")
FRONTEND_DIR   = Path(__file__).parent.parent / "frontend"
REQUEST_LOG    = Path("/home/justin/rag-civil/code_requests.log")
ANALYTICS_FILE = Path("/home/justin/rag-civil/analytics.json")

app = FastAPI(title="Civil RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_analytics_lock = threading.Lock()


# ── request / response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    filter_agency:       str | None = None
    filter_jurisdiction: str | None = None
    filter_state:        str | None = None
    filter_locality:     str | None = None

class CorrectRequest(BaseModel):
    source_file: str
    page: int
    chunk_index: int
    new_section: str

class CodeRequest(BaseModel):
    name:     str = Field(default="", max_length=80)
    document: str = Field(max_length=200)
    notes:    str = Field(default="", max_length=500)

class AnalyticsEvent(BaseModel):
    event:       str            # 'prompt_submitted' | 'fail_response' | 'manual_pulled'
    source_file: str | None = None


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.post("/query")
def run_query(req: QueryRequest):
    try:
        result = query(
            user_query=req.query,
            filter_agency=req.filter_agency or None,
            filter_jurisdiction=req.filter_jurisdiction or None,
            filter_state=req.filter_state or None,
            filter_locality=req.filter_locality or None,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
def run_query_stream(req: QueryRequest):
    """Server-sent events endpoint: yields source blocks one at a time as the LLM generates."""
    try:
        prep = query_prepare(
            user_query=req.query,
            filter_agency=req.filter_agency or None,
            filter_jurisdiction=req.filter_jurisdiction or None,
            filter_state=req.filter_state or None,
            filter_locality=req.filter_locality or None,
        )
    except Exception as e:
        def _err():
            yield f'data: {_json.dumps({"type": "error", "message": str(e)})}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    if prep.get("empty"):
        def _no_chunks():
            yield f'data: {_json.dumps({"type": "meta", "source_groups": [], "chunks": []})}\n\n'
            yield f'data: {_json.dumps({"type": "fail", "text": "The provided standards do not address this query."})}\n\n'
            yield 'data: {"type": "done", "raw": ""}\n\n'
        return StreamingResponse(_no_chunks(), media_type="text/event-stream")

    def _generate():
        yield f'data: {_json.dumps({"type": "meta", "source_groups": prep["source_groups"], "chunks": prep["chunks"]})}\n\n'
        for event in generate_response_stream(req.query, prep["context"]):
            yield f'data: {_json.dumps(event)}\n\n'

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/correct")
def run_correct(req: CorrectRequest):
    ok = correct_section(
        source_file=req.source_file,
        page=req.page,
        chunk_index=req.chunk_index,
        new_section=req.new_section,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Correction failed")
    return {"ok": True}


@app.get("/filters")
def get_filters():
    try:
        db = lancedb.connect(str(VECTORDB_DIR))
        table = db.open_table("civil_engineering_codes")
        rows = table.search().limit(999999).to_list()

        # Agencies with jurisdiction context for grouped dropdown
        seen_agencies: dict[str, dict] = {}
        for r in rows:
            ag = r.get("agency", "")
            if ag and ag not in seen_agencies:
                seen_agencies[ag] = {
                    "name":         ag,
                    "jurisdiction": r.get("jurisdiction", ""),
                    "state":        r.get("state", ""),
                }
        agencies = sorted(seen_agencies.values(), key=lambda x: (x["jurisdiction"], x["name"]))

        # Unique state codes
        states = sorted({r["state"] for r in rows if r.get("state")})

        # Localities with state + human-readable display name
        seen_local: dict[tuple, dict] = {}
        for r in rows:
            loc = r.get("locality", "")
            st  = r.get("state", "")
            if loc and st:
                key = (st, loc)
                if key not in seen_local:
                    seen_local[key] = {
                        "state":   st,
                        "code":    loc,
                        "display": r.get("agency", loc),
                    }
        localities = sorted(seen_local.values(), key=lambda x: (x["state"], x["display"]))

        return {"agencies": agencies, "states": states, "localities": localities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/request")
def submit_request(req: CodeRequest):
    """Receive a request to add a document/standard. Logs to file; no user input is executed."""
    doc = req.document.strip()
    if not doc:
        raise HTTPException(status_code=422, detail="document field is required")

    entry = (
        f"[{datetime.datetime.now().isoformat(timespec='seconds')}] "
        f"name={repr(req.name.strip())} | "
        f"doc={repr(doc)} | "
        f"notes={repr(req.notes.strip())}\n"
    )
    print(entry, flush=True)
    REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with REQUEST_LOG.open("a", encoding="utf-8") as f:
        f.write(entry)

    return {"ok": True}


@app.get("/standards/list")
def get_standards_list():
    """Return one entry per unique source file with agency/jurisdiction metadata."""
    try:
        db = lancedb.connect(str(VECTORDB_DIR))
        table = db.open_table("civil_engineering_codes")
        rows = table.search().limit(999999).to_list()
        seen = {}
        for r in rows:
            sf = r.get("source_file", "")
            if sf and sf not in seen:
                seen[sf] = {
                    "source_file":  sf,
                    "agency":       r.get("agency", ""),
                    "jurisdiction": r.get("jurisdiction", ""),
                    "state":        r.get("state", ""),
                }
        return sorted(seen.values(), key=lambda x: x["source_file"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/event")
def log_analytics_event(ev: AnalyticsEvent):
    """Record an anonymous usage event. Thread-safe; persists to analytics.json."""
    with _analytics_lock:
        if ANALYTICS_FILE.exists():
            data = _json.loads(ANALYTICS_FILE.read_text())
        else:
            data = {"prompts_submitted": 0, "failed_responses": 0, "manual_pulls": {}}

        if ev.event == "prompt_submitted":
            data["prompts_submitted"] = data.get("prompts_submitted", 0) + 1
        elif ev.event == "fail_response":
            data["failed_responses"] = data.get("failed_responses", 0) + 1
        elif ev.event == "manual_pulled" and ev.source_file:
            pulls = data.setdefault("manual_pulls", {})
            pulls[ev.source_file] = pulls.get(ev.source_file, 0) + 1

        ANALYTICS_FILE.write_text(_json.dumps(data, indent=2))
    return {"ok": True}


@app.get("/analytics")
def get_analytics():
    """Return accumulated anonymous usage statistics."""
    if not ANALYTICS_FILE.exists():
        return {"prompts_submitted": 0, "failed_responses": 0, "manual_pulls": {}}
    return _json.loads(ANALYTICS_FILE.read_text())


# ── serve frontend ─────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
