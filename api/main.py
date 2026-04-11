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
import os as _os
DAILY_QUERY_LIMIT  = int(_os.environ.get("CIVIL_DAILY_LIMIT", "20"))

# Public API URL — set CIVIL_API_URL to override (e.g. the RunPod proxy URL).
# If unset, the frontend falls back to same-origin requests (local/dev mode).
_PUBLIC_API_URL    = _os.environ.get("CIVIL_API_URL", "")
_CORRECT_TOKEN     = _os.environ.get("CIVIL_CORRECT_TOKEN", "")  # if set, /correct requires this token

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

_analytics_lock  = threading.Lock()
_rate_lock       = threading.Lock()


_filters_lock    = threading.Lock()
_standards_lock  = threading.Lock()
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
    page:        int = Field(ge=0, le=99999)
    chunk_index: int = Field(ge=0, le=99999)
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
    """Return the real client IP.

    Only trusts X-Forwarded-For when the immediate connection comes from a known
    trusted proxy (localhost / private RunPod proxy). Takes the rightmost IP added
    by that proxy — not the leftmost, which is trivially forged by the client.
    """
    direct = request.client.host if request.client else "unknown"
    if direct in _RATE_LIMIT_WHITELIST:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[-1].strip()
    return direct


def _check_daily_limit(ip: str) -> None:
    """Raise 429 if this IP has exceeded the daily query limit. Logs cap hits."""
    if ip in _RATE_LIMIT_WHITELIST:
        return
    today = datetime.date.today().isoformat()
    with _rate_lock:
        # Prune stale entries from previous days to prevent unbounded growth
        stale = [k for k, (_, d) in _daily_counts.items() if d != today]
        for k in stale:
            del _daily_counts[k]
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


def _undo_daily_count(ip: str) -> None:
    """Decrement the daily count for an IP — called when the backend itself failed
    so a service outage doesn't silently burn through a user's daily quota."""
    if ip in _RATE_LIMIT_WHITELIST:
        return
    today = datetime.date.today().isoformat()
    with _rate_lock:
        count, date = _daily_counts.get(ip, (0, today))
        if date == today and count > 0:
            _daily_counts[ip] = (count - 1, today)


def _undo_record_query(
    ip: str,
    filter_agency: str | None,
    filter_jurisdiction: str | None,
    filter_state: str | None,
    filter_locality: str | None,
) -> None:
    """Reverse the analytics written by _record_query — called when the query fails
    so a backend outage doesn't inflate hourly and filter-usage counts."""
    if ip in _RATE_LIMIT_WHITELIST:
        return
    today = datetime.date.today().isoformat()
    hour  = datetime.datetime.utcnow().strftime("%H")
    with _analytics_lock:
        if not ANALYTICS_FILE.exists():
            return
        try:
            data = _json.loads(ANALYTICS_FILE.read_text())
        except Exception:
            return

        by_day = data.get("queries_by_day", {})
        if by_day.get(today, 0) > 0:
            by_day[today] -= 1

        by_hour = data.get("queries_by_hour", {})
        if by_hour.get(hour, 0) > 0:
            by_hour[hour] -= 1

        fu = data.get("filter_usage", {})
        if any((filter_agency, filter_jurisdiction, filter_state, filter_locality)):
            if filter_agency and fu.get("agency", 0) > 0:
                fu["agency"] -= 1
            if filter_jurisdiction and fu.get("jurisdiction", 0) > 0:
                fu["jurisdiction"] -= 1
            if filter_state and fu.get("state", 0) > 0:
                fu["state"] -= 1
            if filter_locality and fu.get("locality", 0) > 0:
                fu["locality"] -= 1
        else:
            if fu.get("unfiltered", 0) > 0:
                fu["unfiltered"] -= 1

        tmp = ANALYTICS_FILE.with_suffix(".tmp")
        tmp.write_text(_json.dumps(data, indent=2))
        tmp.replace(ANALYTICS_FILE)


@app.post("/query")
def run_query(req: QueryRequest, request: Request):
    ip = _real_ip(request)
    _check_daily_limit(ip)
    _record_query(ip, req.filter_agency, req.filter_jurisdiction, req.filter_state, req.filter_locality)
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
        _undo_daily_count(ip)
        _undo_record_query(ip, req.filter_agency, req.filter_jurisdiction, req.filter_state, req.filter_locality)
        print(f"[error] /query failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Query failed. Please retry.")


@app.post("/query/stream")
def run_query_stream(req: QueryRequest, request: Request):
    """Server-sent events endpoint: yields source blocks one at a time as the LLM generates."""
    ip = _real_ip(request)
    _check_daily_limit(ip)
    _record_query(ip, req.filter_agency, req.filter_jurisdiction, req.filter_state, req.filter_locality)
    try:
        prep = query_prepare(
            user_query=req.query,
            filter_agency=req.filter_agency or None,
            filter_jurisdiction=req.filter_jurisdiction or None,
            filter_state=req.filter_state or None,
            filter_locality=req.filter_locality or None,
        )
    except Exception as e:
        _undo_daily_count(ip)
        _undo_record_query(ip, req.filter_agency, req.filter_jurisdiction, req.filter_state, req.filter_locality)
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
        try:
            yield f'data: {_json.dumps({"type": "meta", "source_groups": prep["source_groups"], "chunks": prep["chunks"]})}\n\n'
            for event in generate_response_stream(req.query, prep["context"]):
                yield f'data: {_json.dumps(event)}\n\n'
        except Exception as e:
            _undo_daily_count(ip)
            _undo_record_query(ip, req.filter_agency, req.filter_jurisdiction, req.filter_state, req.filter_locality)
            print(f"[error] _generate stream failed: {e}", flush=True)
            yield f'data: {_json.dumps({"type": "error", "message": "Stream interrupted. Please retry."})}\n\n'

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/correct")
def run_correct(req: CorrectRequest, request: Request):
    if _CORRECT_TOKEN and request.headers.get("X-Correct-Token") != _CORRECT_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    ok = correct_section(
        source_file=req.source_file,
        page=req.page,
        chunk_index=req.chunk_index,
        new_section=req.new_section,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Correction failed")
    with _analytics_lock:
        data = {"prompts_submitted": 0, "failed_responses": 0, "manual_pulls": {}}
        if ANALYTICS_FILE.exists():
            try:
                data = _json.loads(ANALYTICS_FILE.read_text())
            except Exception:
                pass
        corrections = data.setdefault("correction_count", {})
        corrections[req.source_file] = corrections.get(req.source_file, 0) + 1
        tmp = ANALYTICS_FILE.with_suffix(".tmp")
        tmp.write_text(_json.dumps(data, indent=2))
        tmp.replace(ANALYTICS_FILE)
    return {"ok": True}


@app.get("/filters")
def get_filters():
    global _filters_cache, _filters_cache_ts
    # Fast path: return cached value without blocking on I/O
    with _filters_lock:
        if _filters_cache and (_time.time() - _filters_cache_ts) < _FILTERS_TTL:
            return _filters_cache

    # Slow path: build outside the lock so concurrent requests don't serialize
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
        with _filters_lock:
            if not _filters_cache or (_time.time() - _filters_cache_ts) >= _FILTERS_TTL:
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
    with _standards_lock:
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
        with _standards_lock:
            if not _standards_cache or (_time.time() - _standards_cache_ts) >= _FILTERS_TTL:
                _standards_cache = result
                _standards_cache_ts = _time.time()
        return result
    except Exception as e:
        print(f"[error] /standards/list failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Failed to load standards list.")


def _record_query(
    ip: str,
    filter_agency: str | None,
    filter_jurisdiction: str | None,
    filter_state: str | None,
    filter_locality: str | None,
) -> None:
    """Record per-query stats: day/hour bucket and which filters were active.
    Skips recording for whitelisted IPs (localhost / test runners)."""
    if ip in _RATE_LIMIT_WHITELIST:
        return
    today = datetime.date.today().isoformat()
    hour  = datetime.datetime.utcnow().strftime("%H")
    with _analytics_lock:
        data = {"prompts_submitted": 0, "failed_responses": 0, "manual_pulls": {}}
        if ANALYTICS_FILE.exists():
            try:
                data = _json.loads(ANALYTICS_FILE.read_text())
            except Exception:
                pass

        # queries_by_day: {"2026-04-07": N, ...}
        by_day = data.setdefault("queries_by_day", {})
        by_day[today] = by_day.get(today, 0) + 1

        # queries_by_hour: {"00": N, "01": N, ...}
        by_hour = data.setdefault("queries_by_hour", {})
        by_hour[hour] = by_hour.get(hour, 0) + 1

        # filter_usage: counts of each filter dimension being set
        fu = data.setdefault("filter_usage", {
            "agency": 0, "jurisdiction": 0, "state": 0, "locality": 0, "unfiltered": 0,
        })
        if any((filter_agency, filter_jurisdiction, filter_state, filter_locality)):
            if filter_agency:        fu["agency"]       = fu.get("agency", 0)       + 1
            if filter_jurisdiction:  fu["jurisdiction"] = fu.get("jurisdiction", 0) + 1
            if filter_state:         fu["state"]        = fu.get("state", 0)        + 1
            if filter_locality:      fu["locality"]     = fu.get("locality", 0)     + 1
        else:
            fu["unfiltered"] = fu.get("unfiltered", 0) + 1

        tmp = ANALYTICS_FILE.with_suffix(".tmp")
        tmp.write_text(_json.dumps(data, indent=2))
        tmp.replace(ANALYTICS_FILE)


@app.post("/analytics/event")
def log_analytics_event(ev: AnalyticsEvent):
    """Record an anonymous usage event. Thread-safe; persists to analytics.json."""
    with _analytics_lock:
        data = {"prompts_submitted": 0, "failed_responses": 0, "manual_pulls": {}}
        if ANALYTICS_FILE.exists():
            try:
                data = _json.loads(ANALYTICS_FILE.read_text())
            except Exception:
                pass  # corrupted file — start fresh

        if ev.event == "prompt_submitted":
            data["prompts_submitted"] = data.get("prompts_submitted", 0) + 1
        elif ev.event == "fail_response":
            data["failed_responses"] = data.get("failed_responses", 0) + 1
        elif ev.event == "manual_pulled" and ev.source_file:
            pulls = data.setdefault("manual_pulls", {})
            pulls[ev.source_file] = pulls.get(ev.source_file, 0) + 1

        tmp = ANALYTICS_FILE.with_suffix(".tmp")
        tmp.write_text(_json.dumps(data, indent=2))
        tmp.replace(ANALYTICS_FILE)
    return {"ok": True}


@app.get("/analytics")
def get_analytics(request: Request):
    """Return analytics dashboard (HTML for browsers, JSON for API clients)."""
    with _analytics_lock:
        if not ANALYTICS_FILE.exists():
            data = {"prompts_submitted": 0, "failed_responses": 0, "manual_pulls": {}, "queries_by_day": {}}
        else:
            try:
                data = _json.loads(ANALYTICS_FILE.read_text())
            except Exception:
                data = {"prompts_submitted": 0, "failed_responses": 0, "manual_pulls": {}, "queries_by_day": {}}
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        from fastapi.responses import HTMLResponse
        return HTMLResponse(_build_dashboard_html(data))
    return data


def _build_dashboard_html(data: dict) -> str:
    """Generate a dark-themed bar chart dashboard for query analytics."""
    by_day = data.get("queries_by_day", {})
    today = datetime.date.today()
    days   = [(today - datetime.timedelta(days=i)).isoformat() for i in range(29, -1, -1)]
    counts = [by_day.get(d, 0) for d in days]
    labels = _json.dumps(days)
    values = _json.dumps(counts)
    total  = sum(by_day.values()) if by_day else 0
    week_keys = {(today - datetime.timedelta(days=i)).isoformat() for i in range(7)}
    week_total = sum(v for k, v in by_day.items() if k in week_keys)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Civil Smart Dictionary \u2014 Analytics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a1a; color: #ccc; font-family: monospace; padding: 2rem; }}
  h1   {{ color: #fff; font-size: 1.1rem; margin-bottom: 1.2rem; }}
  .stats {{ display: flex; gap: 1.5rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .stat  {{ background: #252525; padding: 1rem 1.5rem; border-radius: 6px; min-width: 120px; }}
  .stat .n {{ font-size: 2rem; color: #fff; line-height: 1; }}
  .stat .l {{ font-size: 0.72rem; color: #888; margin-top: 0.3rem; }}
  .chart-wrap {{ max-width: 900px; }}
</style></head>
<body>
<h1>civilsmartdictionary.com \u2014 queries</h1>
<div class="stats">
  <div class="stat"><div class="n">{week_total}</div><div class="l">last 7 days</div></div>
  <div class="stat"><div class="n">{total}</div><div class="l">all time</div></div>
</div>
<div class="chart-wrap"><canvas id="chart"></canvas></div>
<script>
new Chart(document.getElementById('chart'), {{
  type: 'bar',
  data: {{
    labels: {labels},
    datasets: [{{ label: 'queries', data: {values},
      backgroundColor: '#4a7c59', borderRadius: 3 }}]
  }},
  options: {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#888', maxRotation: 45, font: {{ size: 10 }} }}, grid: {{ color: '#2a2a2a' }} }},
      y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#2a2a2a' }}, beginAtZero: true }}
    }}
  }}
}});
</script>
</body></html>"""


# ── serve frontend ─────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
