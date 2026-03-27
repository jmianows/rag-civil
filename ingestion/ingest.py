from metadata import detect_section, detect_doc_page, propagate_metadata
import argparse
import json
import os
import re
import fitz
import lancedb
import ollama
from pathlib import Path
import pyarrow as pa
#DIRECTORY STRUCTURE FOR PROPER INGESTION!!!!!!!    _links.json pass document links onto the parser.
#docs/
#├── FEDERAL/
#│   └── USA/                     ← country code (future: EU, CAN, etc.)
#│       ├── OSHA/
#│       │   ├── 29CFR1926.pdf
#│       │   ├── 29CFR1910.pdf
#│       │   └── _links.json      ← sidecar: {"29CFR1926.pdf": "https://..."}
#│       ├── FHWA/
#│       │   ├── MUTCD-2025.pdf
#│       │   └── _links.json
#│       └── USACE/
#│           └── EM_1110-2-1902.pdf
#├── STATE/
#│   ├── WA/                      ← two-letter state code
#│   │   ├── WSDOT/
#│   │   │   ├── WSDOT Design Manual.pdf
#│   │   │   └── _links.json
#│   │   └── ECOLOGY/
#│   │       └── some_regulation.pdf
#│   └── OR/
#│       └── ODOT/
#│           └── oregon_highway_design.pdf
#└── LOCAL/
#    ├── WA/                      ← same state code as STATE/WA
#    │   ├── Seattle/
#    │   │   └── SDOT/
#    │   │       └── seattle_traffic.pdf
#    │   └── King County/
#    │       └── Roads/
#    │           └── king_county_roads.pdf
#    └── OR/
#        └── Portland/
#            └── PBOT/
#                └── portland_bike.pdf

_PROJECT_ROOT = Path(__file__).parent.parent
import sys as _sys; _sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from rag.env_config import VECTORDB_DIR
except ImportError:
    VECTORDB_DIR = _PROJECT_ROOT / "vectordb"
EMBED_MODEL = "mxbai-embed-large"
CHUNK_SIZE    = 220   # words — at 1.3 tokens/word ≈ 286 tokens, safe headroom
CHUNK_OVERLAP = 20    # words
MAX_CHARS     = 900   # characters — roughly 180 words, hard safety ceiling

FAILED_LOG = Path(__file__).parent / "failed_chunks.jsonl"

# Runtime-set by CLI args in __main__
DOCS_DIR: Path = _PROJECT_ROOT / "docs"
FORCE_RERUN: bool = False

#for looking at pages with images
def ocr_pdf_page(page) -> str:
    import pytesseract
    from PIL import Image
    import io

    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))

    # first try auto column detection
    text = pytesseract.image_to_string(img, config='--psm 1')

    if not text or len(text.strip()) < 50:
        # fallback to single block
        text = pytesseract.image_to_string(img, config='--psm 6')

    return text.strip()

##What this does:
##Opens each PDF and reads it page by page
##Skips pages with less than 50 characters — those are almost certainly images, drawings, or blank pages
##Prints a warning for skipped pages so you can see which files are image-heavy
##Returns a list of pages with the text and metadata attached
##is_scanned_pdf takes files that are just images and tries to turn them into text because some of these are images of text rather than cached text.
def is_scanned_pdf(doc, sample_size: int = 30, threshold: float = 0.9) -> bool:
    total_pages = len(doc)
    pages_to_check = min(sample_size, total_pages)
    low_text_count = 0

    for page_num in range(pages_to_check):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if len(text) < 50:
            low_text_count += 1

    ratio = low_text_count / pages_to_check
    return ratio >= threshold


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    pages = []
    doc = fitz.open(str(pdf_path))

    use_ocr = is_scanned_pdf(doc)
    if use_ocr:
        print(f"  [scanned document detected] using OCR for {pdf_path.name}, {len(doc)} pages long.")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if len(text) < 50:
            if use_ocr:
                print(f"  [OCR] page {page_num + 1}")
                try:
                    text = ocr_pdf_page(page)
                except Exception as e:
                    print(f"  [OCR error] page {page_num + 1}: {e}")
                    continue
            else:
                print(f"  [low text] page {page_num + 1} — skipping")
                continue

        if len(text.strip()) < 50:
            continue

        is_toc, dot_ratio = is_table_of_contents(text)
        if is_toc:
            print(f"  [TOC] page {page_num + 1} in {pdf_path.name} — skipping")
            import json
            with open(FAILED_LOG, "a") as f:
                json.dump({
                    "type":       "TOC_skipped",
                    "file":       pdf_path.name,
                    "page":       page_num + 1,
                    "dot_ratio":  dot_ratio,
                    "text":       text[:300],
                }, f)
                f.write("\n")
            continue

        pages.append({
            "text": text.strip(),
            "page": page_num + 1,
            "source_file": pdf_path.name,
        })

    doc.close()
    print(f"  Extracted {len(pages)} text pages from {pdf_path.name}")
    return pages

##a bunch of periods break things
def is_table_of_contents(text: str) -> tuple[bool, float]:
    dot_chars = text.count(".") + text.count("\u2026")
    total_chars = len(text)
    if total_chars == 0:
        return False, 0.0
    dot_ratio = dot_chars / total_chars
    return dot_ratio > 0.15, round(dot_ratio, 3)

##What this does:

##First tries to split text at section headings using pattern matching — this handles numbered sections like 4.3.2, Section 4, or 4 GENERAL REQUIREMENTS
##If no section headings are found it falls back to sliding window chunks of 512 tokens with 50 token overlap
##If a section is longer than 512 tokens it splits it further with the sliding window
##Anything under 100 characters gets discarded — too short to be useful
##Every chunk carries the page number and source file forward from Block 2
def chunk_text(page_data: dict) -> list[dict]:
    text = page_data["text"]
    chunks = []
    
    section_pattern = re.compile(
        r'(?=(\n|^)('
        r'\d+\.\d+[\.\d]*'        # matches 1.2, 1.2.3, 1.2.3.4
        r'|Section\s+\d+'          # matches Section 4
        r'|SECTION\s+\d+'          # matches SECTION 4
        r'|\d+\s+[A-Z][A-Z\s]{4,}' # matches 4 GENERAL REQUIREMENTS
        r'))'
    )
    
    splits = section_pattern.split(text)
    splits = [s.strip() for s in splits if s and len(s.strip()) > 100]
    
    if len(splits) <= 1:
        words = text.split()
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk = " ".join(chunk_words)
            if len(chunk.strip()) > 50:
                chunks.append({
                    **page_data,
                    "text": chunk,
                    "chunk_index": len(chunks),
                })
    else:
        for i, split in enumerate(splits):
            words = split.split()
            if len(words) > CHUNK_SIZE:
                for j in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk = " ".join(words[j:j + CHUNK_SIZE])
                    if len(chunk.strip()) > 100:
                        chunks.append({
                            **page_data,
                            "text": chunk,
                            "chunk_index": len(chunks),
                        })
            else:
                chunks.append({
                    **page_data,
                    "text": split,
                    "chunk_index": len(chunks),
                })
    
    return chunks
##What this does:

##Reads the folder path of each PDF to automatically determine jurisdiction (FEDERAL, STATE, LOCAL) and agency (FHWA, OSHA, USACE, ADA, EPA, WSDOT)
##Detects the section number from the start of the chunk text if present
##Carries all metadata forward so every chunk stored in the database knows exactly where it came from
##Folder structure drives all metadata — no hardcoded agency lookup dicts needed.
##  FEDERAL/{Country}/{Agency}/file.pdf  →  jurisdiction=FEDERAL, state=USA, agency=OSHA
##  STATE/{ST}/{Agency}/file.pdf         →  jurisdiction=STATE,   state=WA,  agency=WSDOT
##  LOCAL/{ST}/{Locality}/{Agency}/file  →  jurisdiction=LOCAL,   state=WA,  locality=Seattle, agency=SDOT

def _load_link(pdf_path: Path, root: Path) -> str:
    """Return the public URL for this PDF from a sidecar or root registry, or ''."""
    # Option A: per-agency sidecar — check _links.json, links.json, and *_links.json glob
    for name in ("_links.json", "links.json"):
        sidecar = pdf_path.parent / name
        if sidecar.exists():
            try:
                links = json.loads(sidecar.read_text(encoding="utf-8"))
                url = links.get(pdf_path.name, "")
                if url:
                    return url
            except Exception:
                pass
    for sidecar in pdf_path.parent.glob("*_links.json"):
        try:
            links = json.loads(sidecar.read_text(encoding="utf-8"))
            url = links.get(pdf_path.name, "")
            if url:
                return url
        except Exception:
            pass
    # Option B: root-level _registry.json keyed by relative path
    registry = root / "_registry.json"
    if registry.exists():
        try:
            reg = json.loads(registry.read_text(encoding="utf-8"))
            rel = pdf_path.relative_to(root).as_posix()
            url = reg.get(rel, "")
            if url:
                return url
        except Exception:
            pass
    return ""


def tag_metadata(chunk: dict, pdf_path: Path, root: Path) -> dict:
    try:
        rel_parts = pdf_path.relative_to(root).parts
    except ValueError:
        rel_parts = pdf_path.parts

    tier = rel_parts[0].upper() if rel_parts else "UNKNOWN"

    jurisdiction = "UNKNOWN"
    state        = ""
    locality     = ""
    agency       = "UNKNOWN"

    if tier == "FEDERAL" and len(rel_parts) >= 4:
        # FEDERAL / {Country} / {Agency} / file.pdf
        jurisdiction = "FEDERAL"
        state        = rel_parts[1].upper()   # country code stored in state field (e.g. "USA")
        agency       = rel_parts[2].upper()
    elif tier == "STATE" and len(rel_parts) >= 4:
        # STATE / {ST} / {Agency} / file.pdf
        jurisdiction = "STATE"
        state        = rel_parts[1].upper()
        agency       = rel_parts[2].upper()
    elif tier == "LOCAL" and len(rel_parts) >= 4:
        # LOCAL / {ST} / {Locality} / {Agency} / file.pdf  (>=5 parts)
        # LOCAL / {ST} / {Locality} / file.pdf             (4 parts — agency falls back to locality)
        jurisdiction = "LOCAL"
        state        = rel_parts[1].upper()
        locality     = rel_parts[2].title()   # "King County" style casing
        agency       = rel_parts[3].upper() if len(rel_parts) >= 5 else locality.upper()

    file_link = _load_link(pdf_path, root)

    section  = detect_section(chunk["text"])
    doc_page = "UNKNOWN"

    doc_page_match = re.search(
        r'(?<!\d)(\d{1,4}-\d{1,4})(?!\d)',
        chunk["text"][:300]
    )
    if doc_page_match:
        candidate = doc_page_match.group(1)
        if candidate != section:
            doc_page = candidate

    return {
        **chunk,
        "jurisdiction": jurisdiction,
        "agency":       agency,
        "state":        state,
        "locality":     locality,
        "source_file":  pdf_path.name,
        "file_link":    file_link,
        "section":      section,
        "doc_page":     doc_page,
        "page":         chunk.get("page", 0),
        "chunk_index":  chunk.get("chunk_index", 0),
    }
#What this does:
#get_embedding — sends each chunk of text to your local nomic-embed-text model via Ollama and gets back a vector of numbers representing the meaning of that text.
#init_chromadb — creates or opens your persistent ChromaDB database in the vectordb/ folder. Using cosine similarity which is the best metric for text retrieval. If you re-run ingestion it won't duplicate chunks — upsert updates existing ones.
#store_chunks — builds up batches of chunks with their embeddings and metadata and writes them to ChromaDB in one operation. Wraps each embedding in a try/except so one bad chunk doesn't crash the whole ingestion run.
#The chunk ID format filename__p142__c3 means page 142, chunk index 3 — unique and human readable so you can trace any retrieved chunk back to its exact location.
def _write_standards_json(table, out_path: Path) -> None:
    rows = table.search().select(
        ["source_file", "agency", "jurisdiction", "state", "locality", "file_link"]
    ).limit(999999).to_list()
    seen: dict[str, dict] = {}
    for r in rows:
        sf = r.get("source_file", "")
        if sf and sf not in seen:
            seen[sf] = {
                "source_file":  sf,
                "agency":       r.get("agency", ""),
                "jurisdiction": r.get("jurisdiction", ""),
                "state":        r.get("state", ""),
                "locality":     r.get("locality", "") or "",
                "file_link":    r.get("file_link", ""),
            }
    standards = sorted(seen.values(), key=lambda x: (x["agency"], x["source_file"]))
    out_path.write_text(json.dumps(standards, indent=2))
    print(f"  Wrote {len(standards)} entries → {out_path}")


def get_embedding(text: str) -> list[float]:
    words = text.split()
    if len(words) > 250:
        text = " ".join(words[:250])

    embed_text = re.sub(r'[\u2026\.]{3,}', ' ', text)
    embed_text = re.sub(r'[ \t]+', ' ', embed_text).strip()

    words = embed_text.split()
    if len(words) > 250:
        embed_text = " ".join(words[:250])

    try:
        response = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=embed_text
        )
        return response["embedding"]
    except Exception as e:
        if "context length" in str(e):
            embed_text = " ".join(embed_text.split()[:150])
            try:
                response = ollama.embeddings(
                    model=EMBED_MODEL,
                    prompt=embed_text
                )
                return response["embedding"]
            except Exception as e2:
                import json
                with open(FAILED_LOG, "a") as f:
                    json.dump({
                        "error":      str(e2),
                        "word_count": len(embed_text.split()),
                        "char_count": len(embed_text),
                        "text":       embed_text[:500],
                        "repr":       repr(embed_text[:200]),
                    }, f)
                    f.write("\n")
                raise
        raise


def init_db() -> lancedb.table.LanceTable:
    db = lancedb.connect(str(VECTORDB_DIR))
    
    schema = pa.schema([
        pa.field("id",           pa.string()),
        pa.field("text",         pa.string()),
        pa.field("vector",       pa.list_(pa.float32(), 1024)),
        pa.field("source_file",  pa.string()),
        pa.field("jurisdiction", pa.string()),
        pa.field("agency",       pa.string()),
        pa.field("state",        pa.string()),
        pa.field("locality",     pa.string()),
        pa.field("file_link",    pa.string()),
        pa.field("section",      pa.string()),
        pa.field("llm_corrected_section",  pa.bool_()),
        pa.field("llm_corrected_doc_page", pa.bool_()),
        pa.field("doc_page",     pa.string()),
        pa.field("page",         pa.int32()),
        pa.field("chunk_index",  pa.int32()),
    ])

    try:
        table = db.open_table("civil_engineering_codes")
        print(f"  LanceDB opened — {table.count_rows()} existing chunks")
    except Exception:
        table = db.create_table("civil_engineering_codes", schema=schema)
        print("  LanceDB created new table")

    return table


def store_chunks(table: lancedb.table.LanceTable, chunks: list[dict]) -> None:
    if not chunks:
        return

    rows = []
    for chunk in chunks:
        chunk_id = (
            f"{chunk['source_file']}"
            f"__p{chunk['page']}"
            f"__c{chunk['chunk_index']}"
        )

        try:
            embedding = get_embedding(chunk["text"])
        except Exception as e:
            print(f"  [embedding error] {chunk_id}: {e}")
            continue

        rows.append({
            "id":           chunk_id,
            "text":         chunk["text"],
            "vector":       embedding,
            "source_file":  chunk["source_file"],
            "jurisdiction": chunk["jurisdiction"],
            "agency":       chunk["agency"],
            "state":        chunk["state"],
            "locality":     chunk["locality"],
            "file_link":    chunk.get("file_link", ""),
            "section":      chunk["section"],
            "llm_corrected_section":  False,
            "llm_corrected_doc_page": False,
            "doc_page":     chunk.get("doc_page", "UNKNOWN"),
            "page":         int(chunk["page"]),
            "chunk_index":  int(chunk["chunk_index"]),
        })

    if rows:
        table.add(rows)
        print(f"  Stored {len(rows)} chunks")
#skip already added data if enabled
def already_ingested(pdf_path: Path, table: lancedb.table.LanceTable) -> bool:
    results = table.search() \
        .where(f"source_file = '{pdf_path.name}'") \
        .limit(1) \
        .to_list()
    return len(results) > 0
#copies metadata if it isn't found 
def propagate_missing_metadata(chunks: list[dict]) -> list[dict]:
    last_section = "UNKNOWN"
    last_doc_page = "UNKNOWN"

    for chunk in chunks:
        if chunk["section"] != "UNKNOWN":
            last_section = chunk["section"]
        else:
            chunk["section"] = last_section

        if chunk["doc_page"] != "UNKNOWN":
            last_doc_page = chunk["doc_page"]
        else:
            chunk["doc_page"] = last_doc_page

    return chunks

def clean_page_text(text: str) -> str:
    """Light whitespace cleanup applied to every page before chunking."""
    text = text.replace('\x0c', '')                       # fitz form-feed chars
    text = text.replace('\r\n', '\n').replace('\r', '\n') # normalize line endings
    text = re.sub(r'\n{3,}', '\n\n', text)               # collapse 3+ blank lines
    text = re.sub(r'[ \t]+\n', '\n', text)               # trailing spaces on lines
    return text.strip()


def detect_repeating_lines(pages: list[dict], zone: str = 'header', zone_size: int = 4) -> set[str]:
    """
    Identify lines that repeat in the header or footer zone across most pages.
    A line must appear on ≥50% of pages and on at least 3 pages.
    Returns a set of stripped line strings to remove.
    """
    section_pat = re.compile(r'^\s*\d[\d\.\-\(\)a-zA-Z]{1,20}\s')
    candidate_counts: dict[str, int] = {}
    n_pages = len(pages)

    for page in pages:
        lines = [l for l in page["text"].splitlines() if l.strip()]
        zone_lines = lines[:zone_size] if zone == 'header' else (
            lines[-zone_size:] if len(lines) >= zone_size else lines
        )
        seen_this_page: set[str] = set()
        for line in zone_lines:
            key = line.strip()
            if not key or len(key) < 5 or len(key) > 120:
                continue
            if section_pat.match(key):   # looks like a code section — keep it
                continue
            if key not in seen_this_page:
                candidate_counts[key] = candidate_counts.get(key, 0) + 1
                seen_this_page.add(key)

    min_pages = max(3, int(n_pages * 0.50))
    return {line for line, count in candidate_counts.items() if count >= min_pages}


def strip_repeating_lines(text: str, header_set: set[str], footer_set: set[str]) -> str:
    """
    Remove identified header/footer lines from the top/bottom of page text.
    Stops at the first line not in the removal set (does not skip over content).
    Reverts to the original if stripping would leave < 100 chars.
    """
    lines = text.splitlines()

    start = 0
    while start < len(lines) and lines[start].strip() in header_set:
        start += 1

    end = len(lines)
    while end > start and lines[end - 1].strip() in footer_set:
        end -= 1

    result = "\n".join(lines[start:end]).strip()
    if len(result) < 100 and len(text.strip()) >= 100:
        return text   # safety: reverted — stripping removed too much
    return result if result else text


def process_pdf(pdf_path: Path, table: lancedb.table.LanceTable, root: Path) -> None:
    if not FORCE_RERUN and already_ingested(pdf_path, table):
        print(f"  [skipped] already in database: {pdf_path.name}")
        return

    print(f"\nProcessing: {pdf_path.name}")

    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        print(f"  [skipped] no usable text found in {pdf_path.name}")
        return

    # Light whitespace cleanup on every page
    for p in pages:
        p["text"] = clean_page_text(p["text"])

    # Detect and strip repeating page headers/footers
    if len(pages) >= 3:
        header_set = detect_repeating_lines(pages, zone='header', zone_size=4)
        footer_set = detect_repeating_lines(pages, zone='footer', zone_size=3)
        if header_set or footer_set:
            print(f"  [clean] {len(header_set)} repeating header line(s), "
                  f"{len(footer_set)} footer line(s) — removing")
            for p in pages:
                p["text"] = strip_repeating_lines(p["text"], header_set, footer_set)

    all_chunks = []
    for page_data in pages:
        chunks = chunk_text(page_data)
        tagged = [tag_metadata(chunk, pdf_path, root) for chunk in chunks]
        all_chunks.extend(tagged)

    all_chunks = propagate_missing_metadata(all_chunks)

    print(f"  Generated {len(all_chunks)} chunks")
    store_chunks(table, all_chunks)

#delete old data when rerunning
def reset_db() -> lancedb.table.LanceTable:
    db = lancedb.connect(str(VECTORDB_DIR))
    
    try:
        db.drop_table("civil_engineering_codes")
        print("  Existing table deleted")
    except Exception:
        pass

    schema = pa.schema([
        pa.field("id",           pa.string()),
        pa.field("text",         pa.string()),
        pa.field("vector",       pa.list_(pa.float32(), 1024)),
        pa.field("source_file",  pa.string()),
        pa.field("jurisdiction", pa.string()),
        pa.field("agency",       pa.string()),
        pa.field("state",        pa.string()),
        pa.field("locality",     pa.string()),
        pa.field("file_link",    pa.string()),
        pa.field("section",      pa.string()),
        pa.field("llm_corrected_section",  pa.bool_()),
        pa.field("llm_corrected_doc_page", pa.bool_()),
        pa.field("doc_page",     pa.string()),
        pa.field("page",         pa.int32()),
        pa.field("chunk_index",  pa.int32()),
    ])

    table = db.create_table("civil_engineering_codes", schema=schema)
    print("  New table created")
    return table

def main(docs_dir: Path, force: bool, only: str | None) -> None:
    global DOCS_DIR, FORCE_RERUN
    DOCS_DIR    = docs_dir
    FORCE_RERUN = force

    print("=" * 60)
    print("Civil Engineering RAG — Ingestion Pipeline")
    print(f"Root : {DOCS_DIR}")
    print("=" * 60)

    if not DOCS_DIR.exists():
        print(f"[error] Root path does not exist: {DOCS_DIR}")
        return

    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

    if FORCE_RERUN:
        print("\n--force: delete and re-ingest affected files.")
        confirm = input("Type Y to confirm, anything else to cancel: ").strip()
        if confirm != "Y":
            print("Cancelled.")
            return
        print("Confirmed — resetting database")
        table = reset_db()
    else:
        table = init_db()

    pdf_files = sorted(DOCS_DIR.rglob("*.pdf"))

    # Filter to subtree if --only specified
    if only:
        only_root = DOCS_DIR / only
        pdf_files = [p for p in pdf_files if p.is_relative_to(only_root)]
        print(f"--only {only}: {len(pdf_files)} PDF(s) in subtree")

    print(f"\nFound {len(pdf_files)} PDF file(s) to process\n")

    _EXPECTED_ROOTS = {"FEDERAL", "STATE", "LOCAL"}
    bad_paths = [p for p in pdf_files if p.relative_to(DOCS_DIR).parts[0].upper() not in _EXPECTED_ROOTS]
    if bad_paths:
        print(f"[warn] {len(bad_paths)} PDF(s) outside expected FEDERAL/STATE/LOCAL structure:")
        for bp in bad_paths:
            print(f"  {bp.relative_to(DOCS_DIR)}")
        print()

    failed  = []
    success = []

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}]", end=" ")
        try:
            process_pdf(pdf_path, table, DOCS_DIR)
            success.append(pdf_path.name)
        except Exception as e:
            print(f"  [error] {pdf_path.name}: {e}")
            failed.append(pdf_path.name)

    print("\n" + "=" * 60)
    print("Ingestion complete")
    print(f"  Successful : {len(success)} files")
    print(f"  Failed     : {len(failed)} files")
    print(f"  Total chunks in database: {table.count_rows()}")
    _write_standards_json(table, Path(__file__).parent.parent / "frontend" / "standards.json")

    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  - {f}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Civil Engineering RAG — Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Folder structure (relative to root):
  FEDERAL/{Country}/{Agency}/file.pdf   e.g. FEDERAL/USA/OSHA/29CFR1926.pdf
  STATE/{ST}/{Agency}/file.pdf          e.g. STATE/WA/WSDOT/manual.pdf
  LOCAL/{ST}/{Locality}/{Agency}/file   e.g. LOCAL/WA/Seattle/SDOT/doc.pdf

PDF links (optional, no code changes needed):
  Place _links.json next to PDFs:  {"filename.pdf": "https://..."}
  Or place _registry.json at root: {"FEDERAL/USA/OSHA/file.pdf": "https://..."}
        """,
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=str(_PROJECT_ROOT / "docs"),
        help="Root folder containing FEDERAL/ STATE/ LOCAL/ subfolders (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest files already in DB (drops and recreates table)",
    )
    parser.add_argument(
        "--only",
        metavar="SUBPATH",
        default=None,
        help="Only ingest this relative subtree, e.g. FEDERAL/USA/OSHA",
    )
    args = parser.parse_args()
    main(Path(args.root), args.force, args.only)