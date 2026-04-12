import ollama
import lancedb
import re
import os
import json
import time
import datetime as _dt
import itertools
import threading
import requests
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from rag.env_config import OLLAMA_KEEP_ALIVE, RERANKER_DEVICE, LLM_MODEL, VECTORDB_DIR, RERANK_FLOOR

_QUERY_LOG      = Path(__file__).parent.parent / "query_log.jsonl"
_query_log_lock = threading.Lock()

def _log_query_response(query: str, raw: str) -> None:
    entry = json.dumps({
        "ts":       _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query":    query,
        "response": raw,
    }, ensure_ascii=False)
    try:
        with _query_log_lock:
            with _QUERY_LOG.open("a", encoding="utf-8") as f:
                f.write(entry + "\n")
    except Exception as e:
        print(f"  [warn] query log write failed: {e}", flush=True)

_ollama_session = requests.Session()
_ollama_session.headers.update({"Connection": "keep-alive"})
_ollama_session.timeout = (10, 120)  # (connect, read) seconds

# ── Hybrid retrieval: detection patterns and agency term map ───────────────────

_SECTION_RE = re.compile(
    r'\b(?:\d{4}\.\d[\d\.\(\)a-zA-Z]*'   # CFR: 1926.502, 1910.146(d)
    r'|[A-Z]\d+\.\d+'                     # MUTCD: 2A.15, 6K.07
    r'|\d+-\d+\.\d[\d\.\(\)]*'            # WSDOT: 6-02.3(25)
    r')'
)

# Pre-compiled regexes used in text processing functions
_THINK_RE         = re.compile(r'<think>.*?</think>', re.DOTALL)
_SPURIOUS_FAIL_RE = re.compile(r'\n*The provided standards do not address[^\n]*\n?.*?\^\^\^FAIL\^\^\^', re.DOTALL)
_SRC_TAG_RE       = re.compile(r'\^\^\^\d+\^\^\^')
_SECTION_VAL_RE   = re.compile(r'^[\d][\d\.\-\(\)\[\]a-zA-Z]*$')
_FLAG_RE          = re.compile(r'\^\^\^(\d+)\^\^\^|\^\^\^FAIL\^\^\^')

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
_reranker_lock = threading.Lock()

def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:  # double-checked
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
    """Re-rank retrieved chunks by (query, chunk) relevance using a cross-encoder.
    Always scores all chunks, slices to top_k, then drops any below RERANK_FLOOR."""
    reranker = _get_reranker()
    pairs = [(query, c.text[:300]) for c in chunks]
    scores = reranker.predict(pairs)
    for score, c in zip(scores, chunks):
        c.rerank_score = float(score)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top = [c for _, c in ranked[:top_k]]
    return [c for c in top if c.rerank_score >= RERANK_FLOOR]


def _vector_search(table, embedding: list[float], n: int, where: str | None) -> list[dict]:
    s = table.search(embedding)
    if where:
        s = s.where(where)
    return s.limit(n).to_list()


def _fts_search(table, query_text: str, n: int, where: str | None = None) -> list[dict]:
    s = table.search(query_text, query_type='fts')
    if where:
        s = s.where(where)
    return s.limit(n).to_list()


def _rrf_merge(lists: list[list[dict]], k: int = 60, weights: list[float] | None = None) -> list[dict]:
    """Reciprocal Rank Fusion: score each chunk by w/(rank+k) summed across all lists.
    weights[i] scales list i's contribution; defaults to 1.0 for each list."""
    scores: dict[str, float] = {}
    rows:   dict[str, dict]  = {}
    for i, result_list in enumerate(lists):
        w = weights[i] if weights and i < len(weights) else 1.0
        for rank, row in enumerate(result_list):
            cid = row['id']
            scores[cid] = scores.get(cid, 0.0) + w / (rank + k)
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
CONTEXT_WINDOW_AFTER  = 2
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
    except Exception as e:
        if "permission" in str(e).lower():
            raise
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


def _sf(v: str) -> str:
    """Sanitize a string value for safe interpolation into a LanceDB WHERE clause."""
    return v.replace("'", "''").replace("\\", "\\\\") if v else v


def retrieve_chunks(
    query: str,
    table: lancedb.table.LanceTable,
    n_results: int = N_RESULTS,
    filter_agency: str = None,
    filter_jurisdiction: str = None,
    filter_state: str = None,
    filter_locality: str = None,
) -> list[RetrievedChunk]:
    _tr0 = time.monotonic()
    embedding = embed_query(query)
    print(f"  [time]   embed_query: {time.monotonic()-_tr0:.2f}s", flush=True)

    has_section, auto_agency = _detect_intent(query)

    # User-explicit filters always win; auto-detection only fires when no UI filter is set
    user_filter_active = any((filter_agency, filter_jurisdiction, filter_state, filter_locality))

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

    _tr1 = time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_vector  = ex.submit(_vector_search, table, embedding, pool_size, user_where)
        f_fts     = ex.submit(_fts_search,    table, query,     pool_size, user_where) if has_section    else None
        f_boosted = ex.submit(_vector_search, table, embedding, pool_size, auto_where)       if auto_where else None

        vector_rows  = f_vector.result()
        fts_rows     = f_fts.result()     if f_fts     else []
        boosted_rows = f_boosted.result() if f_boosted else []

    print(f"  [time]   vector+fts search: {time.monotonic()-_tr1:.2f}s", flush=True)

    # Chunks appearing in multiple lists earn cumulative RRF score and rise to the top.
    # Boosted list (agency-filtered) gets 0.5x weight — nudges without overwhelming main results.
    lists_to_merge = [l for l in [vector_rows, fts_rows, boosted_rows] if l]
    weights = [1.0] * len(lists_to_merge)
    if boosted_rows:
        weights[-1] = 0.5
    merged = _rrf_merge(lists_to_merge, weights=weights)[:n_results]

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
            distance=r.get("_distance", -1.0),  # -1.0 sentinel = FTS-only, no vector distance
            file_link=r.get("file_link", ""),
        ))

    # Drop vector results with implausibly high distance (> 1.0).
    # FTS-only results carry sentinel distance=-1.0 and are always kept.
    filtered = [c for c in chunks if c.distance < 0 or c.distance <= 1.0]
    dropped = len(chunks) - len(filtered)
    if dropped:
        print(f"  [retrieve] dropped {dropped}/{len(chunks)} chunks (vector distance > 1.0)", flush=True)
    chunks = filtered

    return chunks


def remove_overlap(text_a: str, text_b: str, min_overlap: int = 20) -> str:
    max_check = min(len(text_a), len(text_b), 200)

    for length in range(max_check, min_overlap - 1, -1):
        suffix = text_a[-length:]
        if text_b.startswith(suffix):
            return text_a + text_b[length:]

    return text_a + " " + text_b


def _expand_from_cache(
    chunk: RetrievedChunk,
    # keyed by (source_file, chunk_index) → (text, section)
    ci_map: dict[tuple[str, int], tuple[str, str]],
) -> str:
    """Build expanded context for a chunk using a pre-fetched (sf, ci) → (text, section) map.

    New chunker uses a global chunk_index across the whole document, so neighbors
    are found by chunk_index offset alone — page is irrelevant for adjacency.
    Neighbors are only included if they share the same section as the retrieved chunk,
    so expansion never bleeds across section boundaries.
    """
    sf      = chunk.source_file
    ci      = chunk.chunk_index
    section = chunk.section

    neighbors = []
    for offset in range(-CONTEXT_WINDOW_BEFORE, 0):
        entry = ci_map.get((sf, ci + offset))
        if entry and entry[1] == section:
            neighbors.append((ci + offset, entry[0]))

    neighbors.append((ci, chunk.text))

    for offset in range(1, CONTEXT_WINDOW_AFTER + 1):
        entry = ci_map.get((sf, ci + offset))
        if entry and entry[1] == section:
            neighbors.append((ci + offset, entry[0]))

    neighbors.sort(key=lambda x: x[0])
    texts = [t for _, t in neighbors]

    if len(texts) == 1:
        return texts[0]

    combined = texts[0]
    for i in range(1, len(texts)):
        combined = remove_overlap(combined, texts[i])
    return combined


def group_chunks(
    chunks: list[RetrievedChunk],
    table: lancedb.table.LanceTable,
) -> list[dict]:
    # Fetch neighbors by source_file + chunk_index range — no page needed.
    # The new chunker uses a global chunk_index so adjacent indices are always
    # adjacent chunks regardless of which physical page they land on.
    # One query per distinct source_file keeps the number of DB round-trips small.
    from collections import defaultdict
    by_file: dict[str, tuple[int, int]] = {}  # sf → (min_ci, max_ci)
    for chunk in chunks:
        sf = chunk.source_file
        ci = chunk.chunk_index
        lo = ci - CONTEXT_WINDOW_BEFORE
        hi = ci + CONTEXT_WINDOW_AFTER
        if sf in by_file:
            prev_lo, prev_hi = by_file[sf]
            by_file[sf] = (min(prev_lo, lo), max(prev_hi, hi))
        else:
            by_file[sf] = (lo, hi)

    ci_map: dict[tuple[str, int], tuple[str, str]] = {}
    for sf, (lo, hi) in by_file.items():
        try:
            rows = (
                table.search()
                .where(
                    f"source_file = '{_sf(sf)}'"
                    f" AND chunk_index >= {lo}"
                    f" AND chunk_index <= {hi}"
                )
                .limit(hi - lo + 1)
                .to_list()
            )
            for r in rows:
                ci_map[(sf, r["chunk_index"])] = (r["text"], r.get("section", "UNKNOWN"))
        except Exception as e:
            print(f"  [warn] group_chunks: neighbor fetch failed for {sf}: {e}", flush=True)

    texts = [_expand_from_cache(chunk, ci_map) for chunk in chunks]
    expanded = [{"chunk": chunk, "text": text} for chunk, text in zip(chunks, texts)]

    # Group chunks from the same file and same section together so they appear
    # as one combined source block in the LLM context.
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
                and item["chunk"].section not in ("UNKNOWN", "")
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
- Copy language verbatim. Do not paraphrase, reword, or truncate.
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
2. Immediately after the last bullet for that source, start a NEW LINE and write ^^^N^^^ — alone, nothing else on that line.
3. Leave one blank line, then move to the next source and repeat.

Critical constraints:
- Each ^^^N^^^ closes exactly ONE source block. NEVER collect multiple citations together at the end.
- ^^^N^^^ must be alone on its own line — not appended to a bullet, not grouped with other tags.
- ^^^N^^^ must appear AFTER all bullets for that source, never before.
- Only emit ^^^N^^^ if you wrote at least one bullet for that source. A tag with no bullets is forbidden.
- No preamble, summaries, or conclusions. Do not restate the question.

Example output (SOURCE 2 and SOURCE 5 are relevant, SOURCE 4 is off-topic and skipped):
- Minimum lane width shall be 12 feet on arterials.
- Shoulder width shall not be less than 4 feet.
^^^2^^^

- Curb ramp slopes shall not exceed 1:12.
- Cross slope shall not exceed **1:50**.
^^^5^^^

Bold rule:
- You may bold the single most important numeric value — the direct answer to the query — using standard Markdown: **value**. Example: cross slope shall not exceed **1:50**.
- Use at most once per response. Only bold a specific value (number + units, ratio, or percentage) that is the direct answer. Do not bold full sentences or ranges.
- If there is no single clear numeric answer, omit bold entirely.
- Do not bold section numbers, code references, or headings (e.g. 10.8.1.B, §1926.502).

FAIL rule — read carefully:
- If you have written ANY bullet points above, you are done. Do NOT emit ^^^FAIL^^^ under any circumstances.
- Only emit ^^^FAIL^^^ if every single retrieved section is entirely unrelated to the query topic — the documents do not address it at all. If any section touches on the subject, even without an exact value, answer with what is there.
- Output EXACTLY this and nothing else:
^^^FAIL^^^"""

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
    return _THINK_RE.sub('', text).strip()


def strip_spurious_fail(text: str) -> str:
    """Remove ^^^FAIL^^^ block if any ^^^N^^^ citation is already present (format violation guard)."""
    if _SRC_TAG_RE.search(text) and "^^^FAIL^^^" in text:
        text = _SPURIOUS_FAIL_RE.sub('', text).strip()
        text = text.replace("^^^FAIL^^^", "").strip()
    return text

_LLM_OPTIONS_FULL = {
    "temperature":    0.1,
    "num_ctx":        8192,
    "num_predict":    1500,
    "repeat_penalty": 1.5,
    "repeat_last_n":  256,
    "top_k":          20,
    "num_gpu":        999,
}


def _make_ollama_payload(messages: list, stream: bool = False, options: dict | None = None) -> dict:
    payload: dict = {
        "model":      LLM_MODEL,
        "messages":   messages,
        "stream":     stream,
        "think":      False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    if options:
        payload["options"] = options
    return payload


def generate_response(query: str, context: str) -> str:
    payload = _make_ollama_payload(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Query: {query}\n\nRetrieved sections:\n\n{context}"},
        ],
        options=_LLM_OPTIONS_FULL,
    )

    for _attempt in range(3):
        try:
            response = _ollama_session.post(
                _next_ollama_host() + "/api/chat",
                json=payload
            )
            response.raise_for_status()
            break
        except Exception as _e:
            if _attempt == 2:
                raise
            print(f"  [ollama] attempt {_attempt + 1} failed: {_e}, retrying in 2s...", flush=True)
            time.sleep(2)

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
    new_section = new_section.strip()
    if not new_section or len(new_section) < 3 or len(new_section) > 60 or not _SECTION_VAL_RE.match(new_section):
        print(f"  [correct] rejected invalid section: {new_section!r}", flush=True)
        return False
    from ingestion.retag import log_correction
    table = get_db_table()
    # Build ID from raw source_file to match stored IDs; _sf() only in SQL strings
    chunk_id = f"{source_file}__p{page}__c{chunk_index}"
    safe_id  = _sf(chunk_id)

    try:
        result = table.search() \
            .where(f"id = '{safe_id}'") \
            .limit(1) \
            .to_list()
        old_section = result[0]["section"] if result else "UNKNOWN"
    except Exception:
        old_section = "UNKNOWN"

    try:
        table.update(
            where=f"id = '{safe_id}'",
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
def _chunk_to_dict(c) -> dict:
    return {
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


def query_prepare(
    user_query: str,
    filter_agency: str = None,
    filter_jurisdiction: str = None,
    filter_state: str = None,
    filter_locality: str = None,
) -> dict:
    """Run retrieval + formatting pipeline without calling the LLM.
    Returns {context, source_groups, chunks} or {empty: True, ...} if no results."""
    _t0 = time.monotonic()

    print(f"  [query] {user_query!r}", flush=True)
    print("  [1/3] Connecting to database...")
    table = get_db_table()
    print(f"  [time] db connect: {time.monotonic()-_t0:.2f}s", flush=True)

    _t1 = time.monotonic()
    print("  [2/3] Embedding query and retrieving chunks...")
    chunks = retrieve_chunks(
        query=user_query,
        table=table,
        n_results=RERANK_POOL,
        filter_agency=filter_agency,
        filter_jurisdiction=filter_jurisdiction,
        filter_state=filter_state,
        filter_locality=filter_locality,
    )
    print(f"  [time] embed+search: {time.monotonic()-_t1:.2f}s", flush=True)

    _t2 = time.monotonic()
    chunks = rerank_chunks(user_query, chunks, top_k=N_RESULTS)
    print(f"  [time] rerank: {time.monotonic()-_t2:.2f}s", flush=True)
    print(f"  Retrieved {len(chunks)} chunks (re-ranked from {RERANK_POOL})")

    if not chunks or chunks[0].rerank_score < RERANK_FLOOR:
        top = round(chunks[0].rerank_score, 2) if chunks else None
        print(f"  [threshold] Top rerank score {top} below floor {RERANK_FLOOR} — declining")
        return {"empty": True, "source_groups": [], "chunks": [_chunk_to_dict(c) for c in chunks], "context": ""}

    _t3 = time.monotonic()
    print("  [3/3] Building context...")
    groups = group_chunks(chunks, table)
    context, source_groups = format_context(groups)
    print(f"  [time] context build: {time.monotonic()-_t3:.2f}s", flush=True)
    print(f"  [time] TOTAL pre-LLM: {time.monotonic()-_t0:.2f}s", flush=True)

    return {
        "context":       context,
        "source_groups": source_groups,
        "chunks":        [_chunk_to_dict(c) for c in chunks],
    }


def generate_response_stream(user_query: str, context: str):
    """Generator: streams LLM response, yields source_block events + final done event.
    Detects ^^^N^^^ flags and yields a source_block when each one completes."""
    payload = _make_ollama_payload(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Query: {user_query}\n\nRetrieved sections:\n\n{context}"},
        ],
        stream=True,
        options=_LLM_OPTIONS_FULL,
    )

    buffer        = ""
    full_text     = ""
    has_src_block = False
    GUARD         = 11   # len("^^^FAIL^^^") == 10

    _llm_start = time.monotonic()
    _first_token_logged = False
    with _ollama_session.post(_next_ollama_host() + "/api/chat", json=payload, stream=True) as resp:
        resp.raise_for_status()
        try:
          for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] invalid JSON from Ollama mid-stream: {e}", flush=True)
                continue
            token = data.get("message", {}).get("content", "")
            if token and not _first_token_logged:
                print(f"  [time] LLM first token: {time.monotonic()-_llm_start:.2f}s — streaming to client", flush=True)
                _first_token_logged = True
            buffer    += token
            full_text += token

            m = _FLAG_RE.search(buffer)
            if m:
                text_before = strip_thinking(buffer[:m.start()])
                buffer = buffer[m.end():]
                if m.group(1) is not None:
                    # ^^^N^^^ — yield preceding text, then emit citation inline
                    if text_before:
                        yield {"type": "text", "text": text_before}
                    yield {"type": "source_block", "n": int(m.group(1))}
                    has_src_block = True
                else:
                    # ^^^FAIL^^^ — package message text into the fail event; suppress if citations seen
                    if not has_src_block:
                        yield {"type": "fail", "text": "My current knowledge base can't find this. Think I should? Request manuals to add using the button at top right!"}
            else:
                # Flush safe zone (everything except last GUARD chars) to avoid splitting ^^^N^^^.
                # Prefer flushing at newlines; fall back to flushing in chunks for responsiveness.
                safe_end = max(0, len(buffer) - GUARD)
                while True:
                    nl = buffer.find('\n', 0, safe_end)
                    if nl != -1:
                        flushed = strip_thinking(buffer[:nl+1])
                        buffer = buffer[nl+1:]
                        safe_end = max(0, len(buffer) - GUARD)
                        if flushed.strip():
                            yield {"type": "text", "text": flushed + '\n'}
                    elif safe_end > 80:
                        chunk = strip_thinking(buffer[:safe_end])
                        buffer = buffer[safe_end:]
                        if chunk.strip():
                            yield {"type": "text", "text": chunk}
                        break
                    else:
                        break

            if data.get("done"):
                break
        except Exception as e:
            print(f"  [error] stream interrupted: {e}", flush=True)
            yield {"type": "error", "message": "Stream interrupted. Please retry."}
            return

    # Flush remaining buffer
    remainder = strip_thinking(buffer)
    if remainder.strip():
        yield {"type": "text", "text": remainder}

    print(f"\n── LLM RAW OUTPUT ──\n{full_text}\n── END ──\n", flush=True)
    _log_query_response(user_query, strip_thinking(full_text))
    yield {"type": "done", "raw": strip_thinking(full_text)}


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


# ── Ollama keepalive ping ──────────────────────────────────────────────────────
_KEEPALIVE_INTERVAL = 3 * 60  # seconds between pings

def _ollama_keepalive_loop() -> None:
    """Background thread: ping Ollama every 3 minutes to keep LLM loaded in VRAM."""
    import time as _time
    _time.sleep(60)  # wait for startup to settle
    while True:
        try:
            host = _next_ollama_host()
            payload = {
                "model":      LLM_MODEL,
                "messages":   [{"role": "user", "content": "hi"}],
                "stream":     False,
                "think":      False,
                "keep_alive": -1,
                "options":    {"num_predict": 1},
            }
            r = _ollama_session.post(host + "/api/chat", json=payload, timeout=(10, 30))
            r.raise_for_status()
            print(f"  [keepalive] LLM warm ({host})", flush=True)
        except Exception as e:
            print(f"  [keepalive] ping failed: {e}", flush=True)
        _time.sleep(_KEEPALIVE_INTERVAL)

_keepalive_thread = threading.Thread(target=_ollama_keepalive_loop, daemon=True)
_keepalive_thread.start()


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