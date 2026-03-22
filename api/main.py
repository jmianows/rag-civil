import sys
from pathlib import Path

# ensure project root is on path so rag/ and ingestion/ are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag.query_engine import query, correct_section, get_db_table

VECTORDB_DIR = Path("/home/justin/rag-civil/vectordb")
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="Civil RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request/response models ---

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


# --- endpoints ---

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

        agencies = sorted({r["agency"] for r in rows if r.get("agency")})
        jurisdictions = sorted({r["jurisdiction"] for r in rows if r.get("jurisdiction")})

        return {"agencies": agencies, "jurisdictions": jurisdictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- serve frontend ---

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
