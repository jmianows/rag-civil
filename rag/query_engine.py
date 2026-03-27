import ollama
import lancedb
import re
import os
import itertools
import threading
import requests
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from rag.env_config import OLLAMA_KEEP_ALIVE, RERANKER_DEVICE, LLM_MODEL, VECTORDB_DIR

_ollama_session = requests.Session()
_ollama_session.headers.update({"Connection": "keep-alive"})

# ── Hybrid retrieval: detection patterns and agency term map ───────────────────

_SECTION_RE = re.compile(
    r'\b(?:\d{4}\.\d[\d\.\(\)a-zA-Z]*'   # CFR: 1926.502, 1910.146(d)
    r'|[A-Z]\d+\.\d+'                     # MUTCD: 2A.15, 6K.07
    r'|\d+-\d+\.\d[\d\.\(\)]*'            # WSDOT: 6-02.3(25)
    r')'
)

# Uppercase key → agency value as stored in DB metadata
_AGENCY_TERMS: dict[str, str] = {
    'OSHA':            'OSHA',
    '29 CFR':          'OSHA',
    'CFR 1926':        'OSHA',
    'MUTCD':           'FHWA',
    'FHWA':            'FHWA',
    'FEDERAL HIGHWAY': 'FHWA',
    'USACE':           'USACE',
    'ARMY CORPS':      'USACE',
    'EM 1110':         'USACE',
    'EPA':             'EPA',
    'NPDES':           'EPA',
    'CGP':             'EPA',
    'WSDOT':           'WSDOT',
    'ADA':             'ADA',
    'PROWAG':          'ADA',
    'AASHTO':          'AASHTO',
}


def _detect_intent(query: str) -> tuple[bool, str | None]:
    """Returns (has_section_number, auto_detected_agency_or_None)."""
    has_section = bool(_SECTION_RE.search(query))
    q_upper = query.upper()
    for term, agency in _AGENCY_TERMS.items():
        if term in q_upper:
            return has_section, agency
    return has_section, None


_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("  [reranker] Loading cross-encoder model...", flush=True)
        device = RERANKER_DEVICE
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                print("  [reranker] CUDA requested but unavailable, falling back to CPU", flush=True)
                device = "cpu"
        except ImportError:
            device = "cpu"
        _reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device,
            max_length=512,
        )
    return _reranker


def rerank_chunks(query: str, chunks: list, top_k: int = 7) -> list:
    """Re-rank retrieved chunks by (query, chunk) relevance using a cross-encoder."""
    if len(chunks) <= top_k:
        return chunks
    reranker = _get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top = ranked[:top_k]
    for score, c in top:
        c.rerank_score = float(score)
    return [c for _, c in top]


def _vector_search(table, embedding: list[float], n: int, where: str | None) -> list[dict]:
    s = table.search(embedding)
    if where:
        s = s.where(where)
    return s.limit(n).to_list()


def _fts_search(table, query_text: str, n: int) -> list[dict]:
    return table.search(query_text, query_type='fts').limit(n).to_list()


def _rrf_merge(lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score each chunk by 1/(rank+k) summed across all lists."""
    scores: dict[str, float] = {}
    rows:   dict[str, dict]  = {}
    for result_list in lists:
        for rank, row in enumerate(result_list):
            cid = row['id']
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rank + k)
            rows[cid] = row
    return [rows[cid] for cid in sorted(scores, key=lambda c: -scores[c])]

EMBED_MODEL   = "mxbai-embed-large"
N_RESULTS     = 7

# ── Ollama load balancer ───────────────────────────────────────────────────────
# Set OLLAMA_HOSTS env var to a comma-separated list of hosts to round-robin across.
# Defaults to a single local instance. Example:
#   OLLAMA_HOSTS=http://localhost:11434,http://localhost:11435
_raw_hosts   = os.environ.get("OLLAMA_HOSTS", "http://localhost:11434").split(",")
OLLAMA_HOSTS = [h.strip().rstrip("/") for h in _raw_hosts if h.strip()]
_host_cycle  = itertools.cycle(OLLAMA_HOSTS)
_host_lock   = threading.Lock()

def _next_ollama_host() -> str:
    """Return the next Ollama host in round-robin order. Thread-safe."""
    with _host_lock:
        return next(_host_cycle)
RERANK_POOL   = 20   # candidate pool fetched before cross-encoder re-ranking
CONTEXT_WINDOW_BEFORE = 1
CONTEXT_WINDOW_AFTER  = 3
MAX_CHUNK_CHARS = 1000 
#This sets up the configuration and a clean dataclass to carry retrieved chunk data through the pipeline.
@dataclass
class RetrievedChunk:
    text:                  str
    source_file:           str
    agency:                str
    jurisdiction:          str
    state:                 str
    section:               str
    doc_page:              str
    page:                  int
    chunk_index:           int
    distance:              float
    llm_corrected_section:  bool = False
    llm_corrected_doc_page: bool = False
    locality:               str = ""
    file_link:              str = ""
    rerank_score:           float = 0.0


def _ensure_fts_index(table) -> None:
    """Build FTS index if not already present. No-op when already indexed."""
    try:
        table.search("test", query_type='fts').limit(1).to_list()
    except Exception:
        print("  [FTS] Building full-text index...", flush=True)
        table.create_fts_index('text', replace=False)


def _ensure_file_link_column(table) -> None:
    existing = [f.name for f in table.schema]
    if "file_link" not in existing:
        table.add_columns({"file_link": "cast('' as string)"})
        print("  [DB] Added file_link column.", flush=True)


def _ensure_doc_page_flag_column(table) -> None:
    existing = [f.name for f in table.schema]
    if "llm_corrected_doc_page" not in existing:
        table.add_columns({"llm_corrected_doc_page": "cast(false as boolean)"})
        print("  [DB] Added llm_corrected_doc_page column.", flush=True)


_db_table = None
_db_lock  = threading.Lock()

def get_db_table() -> lancedb.table.LanceTable:
    global _db_table
    if _db_table is None:
        with _db_lock:
            if _db_table is None:
                db = lancedb.connect(str(VECTORDB_DIR))
                t  = db.open_table("civil_engineering_codes")
                _ensure_fts_index(t)
                _ensure_file_link_column(t)
                _ensure_doc_page_flag_column(t)
                _db_table = t
    return _db_table

def invalidate_db_table():
    global _db_table
    _db_table = None


_ollama_clients: dict[str, ollama.Client] = {}
_ollama_clients_lock = threading.Lock()

def _get_ollama_client(host: str) -> ollama.Client:
    """Return a cached ollama.Client for the given host, creating one if needed."""
    if host not in _ollama_clients:
        with _ollama_clients_lock:
            if host not in _ollama_clients:
                _ollama_clients[host] = ollama.Client(host=host)
    return _ollama_clients[host]


def embed_query(query: str) -> list[float]:
    host = _next_ollama_host()
    response = _get_ollama_client(host).embeddings(
        model=EMBED_MODEL,
        prompt=f"Represent this sentence for searching relevant passages: {query}"
    )
    return response["embedding"]


def retrieve_chunks(
    query: str,
    table: lancedb.table.LanceTable,
    n_results: int = N_RESULTS,
    filter_agency: str = None,
    filter_jurisdiction: str = None,
    filter_state: str = None,
    filter_locality: str = None,
) -> list[RetrievedChunk]:

    embedding = embed_query(query)
    has_section, auto_agency = _detect_intent(query)

    # User-explicit filters always win; auto-detection only fires when no UI filter is set
    user_filter_active = any([filter_agency, filter_jurisdiction, filter_state, filter_locality])

    def _sf(v: str) -> str:
        """Sanitize a user-supplied filter value for safe interpolation into a LanceDB WHERE string."""
        return v.replace("'", "").replace("\\", "") if v else v

    # Build user-filter WHERE clause (hard filter, unchanged from before)
    clauses = []
    if filter_agency:       clauses.append(f"agency = '{_sf(filter_agency)}'")
    if filter_jurisdiction: clauses.append(f"jurisdiction = '{_sf(filter_jurisdiction)}'")
    if filter_state:        clauses.append(f"state = '{_sf(filter_state)}'")
    if filter_locality:     clauses.append(f"locality = '{_sf(filter_locality)}'")
    user_where = " AND ".join(clauses) if clauses else None

    # Auto-agency boost: a separate filtered vector search whose results get an RRF bonus
    auto_where = f"agency = '{_sf(auto_agency)}'" if (auto_agency and not user_filter_active) else None

    # Fetch larger pools from each search before merging
    pool_size = n_results * 3

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_vector  = ex.submit(_vector_search, table, embedding, pool_size, user_where)
        f_fts     = ex.submit(_fts_search,    table, query,     pool_size) if has_section    else None
        f_boosted = ex.submit(_vector_search, table, embedding, pool_size, auto_where)       if auto_where else None

        vector_rows  = f_vector.result()
        fts_rows     = f_fts.result()     if f_fts     else []
        boosted_rows = f_boosted.result() if f_boosted else []

    # Chunks appearing in multiple lists earn cumulative RRF score and rise to the top
    lists_to_merge = [l for l in [vector_rows, fts_rows, boosted_rows] if l]
    merged = _rrf_merge(lists_to_merge)[:n_results]

    chunks = []
    for r in merged:
        chunks.append(RetrievedChunk(
            text=r["text"],
            source_file=r["source_file"],
            agency=r["agency"],
            jurisdiction=r["jurisdiction"],
            state=r.get("state", ""),
            locality=r.get("locality", ""),
            section=r["section"],
            llm_corrected_section=r.get("llm_corrected_section", False),
            llm_corrected_doc_page=r.get("llm_corrected_doc_page", False),
            doc_page=r.get("doc_page", "UNKNOWN"),
            page=r["page"],
            chunk_index=r["chunk_index"],
            distance=r.get("_distance", r.get("_score", 0.0)),
            file_link=r.get("file_link", ""),
        ))

    # Drop chunks with implausibly high distance — these are FTS results with no
    # vector similarity (e.g. keyword match with dist=15-32 vs normal 0.3-0.7)
    chunks = [c for c in chunks if c.distance <= 1.0]

    return chunks

def expand_context(
    chunk: RetrievedChunk,
    table: lancedb.table.LanceTable,
    window_before: int = 1,
    window_after: int = 2,
) -> str:
    source_file  = chunk.source_file
    page         = chunk.page
    chunk_index  = chunk.chunk_index

    neighbors = []

    # fetch chunks before
    for offset in range(-window_before, 0):
        neighbor_id = f"{source_file}__p{page}__c{chunk_index + offset}"
        try:
            result = table.search() \
                .where(f"id = '{neighbor_id}'") \
                .limit(1) \
                .to_list()
            if result:
                neighbors.append((chunk_index + offset, result[0]["text"]))
        except Exception:
            continue

    # always include the chunk itself
    neighbors.append((chunk_index, chunk.text))

    # fetch chunks after — look across page boundaries too
    for offset in range(1, window_after + 1):
        neighbor_id = f"{source_file}__p{page}__c{chunk_index + offset}"
        try:
            result = table.search() \
                .where(f"id = '{neighbor_id}'") \
                .limit(1) \
                .to_list()
            if result:
                neighbors.append((chunk_index + offset, result[0]["text"]))
                continue
        except Exception:
            pass

        # if not found on same page try next page chunk 0
        if offset == 1:
            next_page_id = f"{source_file}__p{page + 1}__c0"
            try:
                result = table.search() \
                    .where(f"id = '{next_page_id}'") \
                    .limit(1) \
                    .to_list()
                if result:
                    neighbors.append((page + 1, result[0]["text"]))
            except Exception:
                pass

    neighbors.sort(key=lambda x: x[0])
    texts = [t for _, t in neighbors]

    if len(texts) == 1:
        return texts[0]

    combined = texts[0]
    for i in range(1, len(texts)):
        combined = remove_overlap(combined, texts[i])

    return combined

def remove_overlap(text_a: str, text_b: str, min_overlap: int = 20) -> str:
    max_check = min(len(text_a), len(text_b), 200)

    for length in range(max_check, min_overlap - 1, -1):
        suffix = text_a[-length:]
        if text_b.startswith(suffix):
            return text_a + text_b[length:]

    return text_a + " " + text_b

def group_chunks(
    chunks: list[RetrievedChunk],
    table: lancedb.table.LanceTable,
) -> list[dict]:
    # expand each chunk with neighboring context
    expanded = []
    for chunk in chunks:
        text = expand_context(chunk, table)
        expanded.append({
            "chunk":    chunk,
            "text":     text,
        })

    # group chunks that are from the same source file and close in section
    groups = []
    used = set()

    for i, item in enumerate(expanded):
        if i in used:
            continue

        group = [item]
        used.add(i)

        for j, other in enumerate(expanded):
            if j in used:
                continue

            same_file = (
                item["chunk"].source_file == other["chunk"].source_file
            )
            same_section = (
                item["chunk"].section == other["chunk"].section
                and item["chunk"].section != "UNKNOWN"
            )

            if same_file and same_section:
                group.append(other)
                used.add(j)

        groups.append(group)

    return groups
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
SYSTEM_PROMPT = """You are a civil engineering code and standards lookup tool.

Your job is to read the retrieved sections and present their contents accurately.

Content rules:
- Present ONLY information that appears in the retrieved sections. Never add outside knowledge.
- Copy language verbatim. Do not paraphrase or reword. Truncate long passages with "..." only at natural sentence breaks.
- Preserve exact wording of shall, should, and may — these carry distinct legal meaning.
- Present all relevant language: shall statements, should guidance, may provisions, and numeric values. Do not skip a source because it only contains guidance rather than hard requirements.
- If the query names a specific regulation or section number (e.g., "OSHA 1926.502"), prioritize retrieved sections from that regulation over summaries of it in other documents.
- Show most local jurisdiction first, then expand to federal below.
- Never provide engineering advice, opinions, or recommendations.

Source rules:
- Work through each retrieved source one at a time.
- For each source: if the source addresses the topic of the query in any way — even partially, even as guidance — extract the relevant bullets and tag it. Only skip a source if it is entirely off-topic.
- Never interleave content from different sources.

Output format — follow this exactly:
1. Write all relevant bullets from SOURCE N as a bullet list.
2. On a new line by itself, write [[SRC_N]] — where N matches the source number from "--- SOURCE N ---" in the context.
3. Leave one blank line, then repeat for the next source.
- [[SRC_N]] must appear on its own line, never inline or mid-sentence.
- [[SRC_N]] must appear AFTER the last bullet for that source, never before.
- Only emit [[SRC_N]] if you have written at least one bullet for that source. A tag with no bullets is forbidden.
- No preamble, summaries, or conclusions. Do not restate the question.

FAIL rule — read carefully:
- If you have written ANY bullet points above, you are done. Do NOT emit [[FAIL]] under any circumstances.
- Only emit [[FAIL]] if every single retrieved section is entirely unrelated to the query topic — the documents do not address it at all. If any section touches on the subject, even without an exact value, answer with what is there.
- Output EXACTLY this and nothing else:
[[FAIL]]"""

def deduplicate_lines(text: str) -> str:
    """Remove exact-duplicate lines from text (preserves first occurrence)."""
    seen: set[str] = set()
    out = []
    for line in text.splitlines():
        key = line.strip()
        if key not in seen:
            seen.add(key)
            out.append(line)
    return "\n".join(out)


def format_context(groups: list[dict]) -> tuple[str, list[dict]]:
    context_blocks = []
    source_groups = []

    for i, group in enumerate(groups, start=1):
        combined_texts = [item["text"] for item in group]
        combined = combined_texts[0]
        for j in range(1, len(combined_texts)):
            combined = remove_overlap(combined, combined_texts[j])
        combined = deduplicate_lines(combined)

        chunk = group[0]["chunk"]
        section_display = chunk.section
        if chunk.llm_corrected_section:
            section_display = f"{chunk.section} [llm corrected]"

        if len(combined) > MAX_CHUNK_CHARS:
            combined = combined[:MAX_CHUNK_CHARS] + "..."

        block = f"""--- SOURCE {i} ---
{combined}

[SRC_{i}] Document: {chunk.source_file} {section_display}, page {chunk.doc_page}
---"""

        context_blocks.append(block)
        source_groups.append({
            "idx":                   i,
            "source_file":           chunk.source_file,
            "agency":                chunk.agency,
            "jurisdiction":          chunk.jurisdiction,
            "state":                 chunk.state,
            "locality":              chunk.locality,
            "section":               chunk.section,
            "doc_page":              chunk.doc_page,
            "page":                  chunk.page,
            "chunk_index":           chunk.chunk_index,
            "distance":              chunk.distance,
            "llm_corrected_section": chunk.llm_corrected_section,
            "file_link":             chunk.file_link,
            "text":                  combined,
        })

    return "\n\n".join(context_blocks), source_groups

def strip_thinking(text: str) -> str:
    # remove any thinking blocks Qwen3 emits despite think=False
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

_SRC_TAG_RE = re.compile(r'\[\[SRC_\d+\]\]')

def strip_spurious_fail(text: str) -> str:
    """Remove [[FAIL]] block if any [[SRC_N]] citation is already present (format violation guard)."""
    if _SRC_TAG_RE.search(text) and "[[FAIL]]" in text:
        text = re.sub(
            r'\n*The provided standards do not address[^\n]*\n?.*?\[\[FAIL\]\]',
            '',
            text,
            flags=re.DOTALL,
        ).strip()
        text = text.replace("[[FAIL]]", "").strip()
    return text

def generate_response(query: str, context: str) -> str:
    import json

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nRetrieved sections:\n\n{context}",
            }
        ],
        "stream":     False,
        "think":      False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature":    0.1,
            "num_ctx":        4096,
            "num_predict":    600,
            "repeat_penalty": 1.3,
            "top_k":          20,
            "num_gpu":        999,
        }
    }

    response = _ollama_session.post(
        _next_ollama_host() + "/api/chat",
        json=payload
    )
    response.raise_for_status()

    data = response.json()
    content = data["message"]["content"].strip()
    return strip_spurious_fail(strip_thinking(content))
#if the parser finds a section it will go back, correct it, and then flag it did so for us.
def correct_section(
    source_file: str,
    page: int,
    chunk_index: int,
    new_section: str,
) -> bool:
    from ingestion.retag import log_correction
    table = get_db_table()
    chunk_id = f"{_sf(source_file)}__p{page}__c{chunk_index}"

    try:
        result = table.search() \
            .where(f"id = '{chunk_id}'") \
            .limit(1) \
            .to_list()
        old_section = result[0]["section"] if result else "UNKNOWN"
    except Exception:
        old_section = "UNKNOWN"

    try:
        table.update(
            where=f"id = '{chunk_id}'",
            values={
                "section":               new_section,
                "llm_corrected_section": True,
            }
        )
    except Exception as e:
        print(f"  Failed to update {chunk_id}: {e}")
        return False

    log_correction(source_file, page, chunk_index, old_section, new_section)
    return True
def enrich_section(text: str, source_file: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"You are reading a chunk from {source_file}.\nIdentify the section number this chunk belongs to.\nReturn ONLY the section number (e.g. '1926.502', '1310.02(13)(b)', '6-02.3').\nIf you cannot determine a section number return UNKNOWN.\nDo not explain your answer.\n\nText:\n{text[:500]}",
            }
        ],
        "stream":     False,
        "think":      False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_predict": 20,
        }
    }

    response = _ollama_session.post(
        _next_ollama_host() + "/api/chat",
        json=payload
    )

    data = response.json()
    result = data["message"]["content"].strip()
    result = strip_thinking(result)

    import re
    if re.match(r'^[\d][\d\.\-\(\)\[\]a-zA-Z]*$', result) and len(result) >= 3:
        return result
    return "UNKNOWN"

def enrich_doc_page(text: str, source_file: str) -> str:
    """Ask the LLM for the document page number/range of this chunk."""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content":
            f"You are reading a chunk from {source_file}.\n"
            f"Identify the document page number this chunk appears on.\n"
            f"Return ONLY the page number or range (e.g. '142', '142-143').\n"
            f"If you cannot determine the page number return UNKNOWN.\n"
            f"Do not explain.\n\nText:\n{text[:500]}",
        }],
        "stream": False, "think": False, "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0, "num_predict": 10},
    }
    response = _ollama_session.post(_next_ollama_host() + "/api/chat", json=payload)
    result = strip_thinking(response.json()["message"]["content"].strip())
    if re.match(r'^\d[\d\-]*$', result) and len(result) >= 1:
        return result
    return "UNKNOWN"


def _enrich_and_persist(chunks_sec: list, chunks_page: list, table) -> None:
    """Background: enrich UNKNOWN sections and doc_pages, persist to DB and corrections log."""
    from ingestion.retag import log_correction
    with ThreadPoolExecutor(max_workers=4) as ex:
        sec_futs  = [(c, ex.submit(enrich_section,  c.text, c.source_file)) for c in chunks_sec]
        page_futs = [(c, ex.submit(enrich_doc_page, c.text, c.source_file)) for c in chunks_page]

    for chunk, fut in sec_futs:
        try:
            result = fut.result()
            if result == "UNKNOWN":
                continue
            cid = f"{chunk.source_file}__p{chunk.page}__c{chunk.chunk_index}"
            table.update(where=f"id = '{cid}'",
                         values={"section": result, "llm_corrected_section": True})
            log_correction(chunk.source_file, chunk.page, chunk.chunk_index,
                           chunk.section, result, field="section")
            print(f"  [enrich] section {cid} → {result}", flush=True)
        except Exception as e:
            print(f"  [enrich] warn section: {e}", flush=True)

    for chunk, fut in page_futs:
        try:
            result = fut.result()
            if result == "UNKNOWN":
                continue
            cid = f"{chunk.source_file}__p{chunk.page}__c{chunk.chunk_index}"
            table.update(where=f"id = '{cid}'",
                         values={"doc_page": result, "llm_corrected_doc_page": True})
            log_correction(chunk.source_file, chunk.page, chunk.chunk_index,
                           chunk.doc_page, result, field="doc_page")
            print(f"  [enrich] doc_page {cid} → {result}", flush=True)
        except Exception as e:
            print(f"  [enrich] warn doc_page: {e}", flush=True)


def query_prepare(
    user_query: str,
    filter_agency: str = None,
    filter_jurisdiction: str = None,
    filter_state: str = None,
    filter_locality: str = None,
) -> dict:
    """Run retrieval + formatting pipeline without calling the LLM.
    Returns {context, source_groups, chunks} or {empty: True, ...} if no results."""
    print("  [1/4] Connecting to database...")
    table = get_db_table()

    print("  [2/4] Embedding query and retrieving chunks...")
    chunks = retrieve_chunks(
        query=user_query,
        table=table,
        n_results=RERANK_POOL,
        filter_agency=filter_agency,
        filter_jurisdiction=filter_jurisdiction,
        filter_state=filter_state,
        filter_locality=filter_locality,
    )
    chunks = rerank_chunks(user_query, chunks, top_k=N_RESULTS)
    print(f"  Retrieved {len(chunks)} chunks (re-ranked from {RERANK_POOL})")

    RERANK_FLOOR = 2.3
    if not chunks or chunks[0].rerank_score < RERANK_FLOOR:
        top = round(chunks[0].rerank_score, 2) if chunks else None
        print(f"  [threshold] Top rerank score {top} below floor {RERANK_FLOOR} — declining")
        return {"empty": True, "source_groups": [], "chunks": [
            {"source_file": c.source_file, "agency": c.agency, "jurisdiction": c.jurisdiction,
             "state": c.state, "locality": c.locality, "section": c.section,
             "doc_page": c.doc_page, "page": c.page, "chunk_index": c.chunk_index,
             "distance": c.distance, "rerank_score": c.rerank_score,
             "llm_corrected_section": c.llm_corrected_section, "file_link": c.file_link}
            for c in chunks
        ], "context": ""}

    print("  [3/4] Skipping live enrichment — use ingestion/enrich_file.py", flush=True)

    print("  [4/4] Building context...")
    groups = group_chunks(chunks, table)
    context, source_groups = format_context(groups)

    return {
        "context":       context,
        "source_groups": source_groups,
        "chunks": [
            {
                "source_file":           c.source_file,
                "agency":                c.agency,
                "jurisdiction":          c.jurisdiction,
                "state":                 c.state,
                "locality":              c.locality,
                "section":               c.section,
                "doc_page":              c.doc_page,
                "page":                  c.page,
                "chunk_index":           c.chunk_index,
                "distance":              c.distance,
                "rerank_score":          c.rerank_score,
                "llm_corrected_section": c.llm_corrected_section,
                "file_link":             c.file_link,
            }
            for c in chunks
        ],
    }


def generate_response_stream(user_query: str, context: str):
    """Generator: streams LLM response, yields source_block events + final done event.
    Detects [[SRC_N]] flags and yields a source_block when each one completes."""
    import json as _json
    import re as _re

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Query: {user_query}\n\nRetrieved sections:\n\n{context}"},
        ],
        "stream":     True,
        "think":      False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature":    0.1,
            "num_ctx":        4096,
            "num_predict":    600,
            "repeat_penalty": 1.3,
            "top_k":          20,
            "num_gpu":        999,
        },
    }

    flag_re       = _re.compile(r'\[\[SRC_(\d+)\]\]|\[\[FAIL\]\]')
    buffer        = ""
    full_text     = ""
    has_src_block = False
    GUARD         = 12   # len("[[SRC_99]]") == 11

    with _ollama_session.post(_next_ollama_host() + "/api/chat", json=payload, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data  = _json.loads(line)
            token = data.get("message", {}).get("content", "")
            buffer    += token
            full_text += token

            m = flag_re.search(buffer)
            if m:
                text_before = strip_thinking(buffer[:m.start()])
                buffer = buffer[m.end():]
                if m.group(1) is not None:
                    # [[SRC_N]] — yield preceding text, then emit citation inline
                    if text_before:
                        yield {"type": "text", "text": text_before}
                    yield {"type": "source_block", "n": int(m.group(1))}
                    has_src_block = True
                else:
                    # [[FAIL]] — package message text into the fail event; suppress if citations seen
                    if not has_src_block:
                        yield {"type": "fail", "text": "My current knowledge base can't find this. Think I should? Request manuals to add using the button at top right!"}
            else:
                # Flush only complete newline-terminated lines from the safe zone.
                # GUARD chars are kept buffered to avoid splitting a [[SRC_N]] marker.
                safe_end = max(0, len(buffer) - GUARD)
                while True:
                    nl = buffer.find('\n', 0, safe_end)
                    if nl == -1:
                        break
                    line = strip_thinking(buffer[:nl+1])
                    buffer = buffer[nl+1:]
                    safe_end = max(0, len(buffer) - GUARD)
                    if line.strip():
                        yield {"type": "text", "text": line + '\n'}

            if data.get("done"):
                break

    # Flush remaining buffer
    remainder = strip_thinking(buffer)
    if remainder.strip():
        yield {"type": "text", "text": remainder}

    print(f"\n── LLM RAW OUTPUT ──\n{full_text}\n── END ──\n", flush=True)
    yield {"type": "done", "raw": full_text}


def query(
    user_query: str,
    filter_agency: str = None,
    filter_jurisdiction: str = None,
    filter_state: str = None,
    filter_locality: str = None,
) -> dict:
    prep = query_prepare(
        user_query=user_query,
        filter_agency=filter_agency,
        filter_jurisdiction=filter_jurisdiction,
        filter_state=filter_state,
        filter_locality=filter_locality,
    )
    if prep.get("empty"):
        return {
            "query":         user_query,
            "response":      "The provided standards do not address this query.",
            "chunks":        [],
            "source_groups": [],
        }

    print("  [5/5] Generating response...")
    response = generate_response(user_query, prep["context"])

    return {
        "query":         user_query,
        "response":      response,
        "chunks":        prep["chunks"],
        "source_groups": prep["source_groups"],
    }


if __name__ == "__main__":
    import sys
    import json

    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "stormwater requirements for construction sites"

    print(f"\nQuery: {user_query}\n")
    print("Retrieving and generating response...\n")

    result = query(user_query)

    print(result["response"])
    print("\n--- Retrieved chunks ---")
    for c in result["chunks"]:
        print(f"  {c['source_file']} | {c['agency']} | Section {c['section']} | Page {c['doc_page']}")