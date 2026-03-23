import sys
import json as _json
import datetime
import pathlib
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

VECTORDB_DIR  = Path("/home/justin/rag-civil/vectordb")
FRONTEND_DIR  = Path(__file__).parent.parent / "frontend"
REQUEST_LOG   = Path("/home/justin/rag-civil/code_requests.log")

app = FastAPI(title="Civil RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    filter_agency: str | None = None
    filter_jurisdiction: str | None = None
    filter_state: str | None = None

class CorrectRequest(BaseModel):
    source_file: str
    page: int
    chunk_index: int
    new_section: str

class CodeRequest(BaseModel):
    name:     str = Field(default="", max_length=80)
    document: str = Field(max_length=200)
    notes:    str = Field(default="", max_length=500)


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.post("/query")
def run_query(req: QueryRequest):
    try:
        result = query(
            user_query=req.query,
            filter_agency=req.filter_agency or None,
            filter_jurisdiction=req.filter_jurisdiction or None,
            filter_state=req.filter_state or None,
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
        )
    except Exception as e:
        def _err():
            yield f'data: {_json.dumps({"type": "error", "message": str(e)})}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    if prep.get("empty"):
        def _no_chunks():
            yield f'data: {_json.dumps({"type": "meta", "source_groups": [], "chunks": []})}\n\n'
            yield f'data: {_json.dumps({"type": "text", "text": "The provided standards do not address this query."})}\n\n'
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
        agencies      = sorted({r["agency"]       for r in rows if r.get("agency")})
        jurisdictions = sorted({r["jurisdiction"]  for r in rows if r.get("jurisdiction")})
        return {"agencies": agencies, "jurisdictions": jurisdictions}
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


# ── serve frontend ─────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
