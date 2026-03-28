import sys
import json as _json
import datetime
import pathlib
import threading
from pathlib import Path

# ensure project root is on path so rag/ and ingestion/ are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
from fastapi import FastAPI, HTTPException, Request
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
    embed_query,
    _get_reranker,
)
from rag.env_config import WARM_ON_STARTUP, ENVIRONMENT, IS_PRODUCTION, VECTORDB_DIR

_PROJECT_ROOT  = Path(__file__).parent.parent
FRONTEND_DIR   = _PROJECT_ROOT / "frontend"
REQUEST_LOG    = _PROJECT_ROOT / "code_requests.log"
ANALYTICS_FILE = _PROJECT_ROOT / "analytics.json"
RATE_LIMIT_LOG = _PROJECT_ROOT / "rate_limit.log"
DAILY_QUERY_LIMIT = 20

# Public API URL — set CIVIL_API_URL to override (e.g. the RunPod proxy URL).
# If unset, the frontend falls back to same-origin requests (local/dev mode).
import os as _os
_PUBLIC_API_URL = _os.environ.get("CIVIL_API_URL", "")

app = FastAPI(title="Civil RAG API")

_CORS_ORIGINS = (
    ["https://civilsmartdictionary.com", "https://www.civilsmartdictionary.com"]
    if IS_PRODUCTION else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_analytics_lock = threading.Lock()
_rate_lock      = threading.Lock()
_daily_counts: dict[str, tuple[int, str]] = {}  # ip -> (count, date_str)

import time as _time
_FILTERS_TTL = 3600  # 1 hour — refresh after new documents are ingested
_filters_cache: dict | None = None
_filters_cache_ts: float = 0.0
_standards_cache: list | None = None
_standards_cache_ts: float = 0.0


def _startup_warmup():
    """Background thread: pre-load all heavy resources before the first request."""
    _get_reranker()
    print("[startup] Cross-encoder ready", flush=True)

    try:
        get_db_table()  # always run: ensures FTS index is built before first query
        print("[startup] LanceDB connection ready", flush=True)
    except Exception as e:
        print(f"[startup] LanceDB warm-up failed: {e}", flush=True)

    if WARM_ON_STARTUP:

        try:
            embed_query("warmup")
            print("[startup] Ollama embedding model ready", flush=True)
        except Exception as e:
            print(f"[startup] Ollama warm-up failed: {e}", flush=True)


@app.on_event("startup")
async def startup_tasks():
    print(f"[startup] Environment: {ENVIRONMENT}", flush=True)
    threading.Thread(target=_startup_warmup, daemon=True, name="startup-warmup").start()


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
    document: str = Field(max_length=200)
    notes:    str = Field(default="", max_length=500)
    email:    str = Field(default="", max_length=120)

class AnalyticsEvent(BaseModel):
    event:       str            # 'prompt_submitted' | 'fail_response' | 'manual_pulled'
    source_file: str | None = None


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Liveness + readiness probe. Returns 200 if both Ollama and LanceDB are reachable."""
    import urllib.request
    result = {"ollama": "ok", "lancedb": "ok", "rows": None}

    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/", timeout=2) as r:
            if r.status != 200:
                result["ollama"] = f"unexpected status {r.status}"
    except Exception as e:
        print(f"[health] ollama check failed: {e}", flush=True)
        result["ollama"] = "error"

    try:
        tbl = get_db_table()
        result["rows"] = tbl.count_rows()
    except Exception as e:
        print(f"[health] lancedb check failed: {e}", flush=True)
        result["lancedb"] = "error"

    result["status"] = "ok" if result["ollama"] == "ok" and result["lancedb"] == "ok" else "degraded"
    status_code = 200 if result["status"] == "ok" else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(content=result, status_code=status_code)


@app.get("/config")
def get_config():
    """Return runtime configuration for the frontend.

    The frontend calls this on load to discover the backend URL instead of
    having it hardcoded. Set CIVIL_API_URL on the pod to expose the proxy URL.
    """
    return {"api_url": _PUBLIC_API_URL, "environment": ENVIRONMENT}


_RATE_LIMIT_WHITELIST = {"127.0.0.1", "::1"}


def _real_ip(request: Request) -> str:
    """Return the real client IP, reading X-Forwarded-For when behind a proxy (RunPod, nginx)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host


def _check_daily_limit(ip: str) -> None:
    """Raise 429 if this IP has exceeded the daily query limit. Logs cap hits."""
    if ip in _RATE_LIMIT_WHITELIST:
        return
    today = datetime.date.today().isoformat()
    with _rate_lock:
        count, date = _daily_counts.get(ip, (0, today))
        if date != today:
            count = 0
        if count >= DAILY_QUERY_LIMIT:
            entry = f"{datetime.datetime.now().isoformat()} | {ip} | daily cap hit\n"
            print(entry.strip(), flush=True)
            RATE_LIMIT_LOG.parent.mkdir(parents=True, exist_ok=True)
            with RATE_LIMIT_LOG.open("a") as f:
                f.write(entry)
            raise HTTPException(
                status_code=429,
                detail="Daily query limit reached. The service resets at midnight UTC.",
            )
        _daily_counts[ip] = (count + 1, today)


@app.post("/query")
def run_query(req: QueryRequest, request: Request):
    _check_daily_limit(_real_ip(request))
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
        print(f"[error] /query failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Query failed. Please retry.")


@app.post("/query/stream")
def run_query_stream(req: QueryRequest, request: Request):
    """Server-sent events endpoint: yields source blocks one at a time as the LLM generates."""
    _check_daily_limit(_real_ip(request))
    try:
        prep = query_prepare(
            user_query=req.query,
            filter_agency=req.filter_agency or None,
            filter_jurisdiction=req.filter_jurisdiction or None,
            filter_state=req.filter_state or None,
            filter_locality=req.filter_locality or None,
        )
    except Exception as e:
        print(f"[error] /query/stream failed: {e}", flush=True)
        def _err():
            yield f'data: {_json.dumps({"type": "error", "message": "Query failed. Please retry."})}\n\n'
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
    global _filters_cache, _filters_cache_ts
    if _filters_cache and (_time.time() - _filters_cache_ts) < _FILTERS_TTL:
        return _filters_cache
    try:
        db = lancedb.connect(str(VECTORDB_DIR))
        table = db.open_table("civil_engineering_codes")
        rows = table.search().limit(999999).to_list()

        # Agencies — include locality so frontend cascade can match LOCAL agencies
        seen_agencies: dict[str, dict] = {}
        for r in rows:
            ag = r.get("agency", "")
            if ag and ag not in seen_agencies:
                seen_agencies[ag] = {
                    "name":         ag,
                    "jurisdiction": r.get("jurisdiction", ""),
                    "state":        r.get("state", ""),
                    "locality":     r.get("locality", ""),
                }
        agencies = sorted(seen_agencies.values(), key=lambda x: (x["jurisdiction"], x["name"]))

        # Unique state codes (kept for backward compat)
        states = sorted({r["state"] for r in rows if r.get("state")})

        # Localities (kept for backward compat)
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

        # Scopes — unique (jurisdiction, state, locality) combos for the scope dropdown
        seen_scopes: dict[tuple, dict] = {}
        for r in rows:
            key = (r.get("jurisdiction", ""), r.get("state", ""), r.get("locality", ""))
            if key[0] and key not in seen_scopes:
                seen_scopes[key] = {
                    "jurisdiction": key[0],
                    "state":        key[1],
                    "locality":     key[2],
                }
        scopes = sorted(seen_scopes.values(),
                        key=lambda x: (x["jurisdiction"], x["state"], x["locality"]))

        result = {"agencies": agencies, "states": states, "localities": localities, "scopes": scopes}
        _filters_cache = result
        _filters_cache_ts = _time.time()
        return result
    except Exception as e:
        print(f"[error] /filters failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Failed to load filters.")


@app.post("/request")
def submit_request(req: CodeRequest):
    """Receive a request to add a document/standard. Logs to file; no user input is executed."""
    doc = req.document.strip()
    if not doc:
        raise HTTPException(status_code=422, detail="document field is required")

    entry = (
        f"[{datetime.datetime.now().isoformat(timespec='seconds')}] "
        f"doc={repr(doc)} | "
        f"notes={repr(req.notes.strip())} | "
        f"email={repr(req.email.strip())}\n"
    )
    print(entry, flush=True)
    REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with REQUEST_LOG.open("a", encoding="utf-8") as f:
        f.write(entry)

    return {"ok": True}


@app.get("/standards", response_class=FileResponse)
def standards_page():
    return FileResponse(str(FRONTEND_DIR / "standards.html"))


@app.get("/standards/list")
def get_standards_list():
    """Return one entry per unique source file with agency/jurisdiction metadata."""
    global _standards_cache, _standards_cache_ts
    if _standards_cache and (_time.time() - _standards_cache_ts) < _FILTERS_TTL:
        return _standards_cache
    try:
        db = lancedb.connect(str(VECTORDB_DIR))
        table = db.open_table("civil_engineering_codes")
        cols = ['source_file', 'agency', 'jurisdiction', 'state', 'locality', 'file_link']
        rows = table.search().select(cols).limit(999999).to_list()
        seen = {}
        for r in rows:
            sf = r.get("source_file", "")
            if sf and sf not in seen:
                seen[sf] = {
                    "source_file":  sf,
                    "agency":       r.get("agency", ""),
                    "jurisdiction": r.get("jurisdiction", ""),
                    "state":        r.get("state", ""),
                    "locality":     r.get("locality", ""),
                    "file_link":    r.get("file_link", ""),
                }
        result = sorted(seen.values(), key=lambda x: (x["agency"], x["source_file"]))
        _standards_cache = result
        _standards_cache_ts = _time.time()
        return result
    except Exception as e:
        print(f"[error] /standards/list failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Failed to load standards list.")


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
