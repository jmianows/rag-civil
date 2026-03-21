from metadata import detect_section, detect_doc_page, propagate_metadata
import os
import re
import fitz
import lancedb
import ollama
from pathlib import Path
import pyarrow as pa

DOCS_DIR = Path("/home/justin/rag-civil/docs")
VECTORDB_DIR = Path("/home/justin/rag-civil/vectordb")
EMBED_MODEL = "mxbai-embed-large"
CHUNK_SIZE    = 220   # words — at 1.3 tokens/word ≈ 286 tokens, safe headroom
CHUNK_OVERLAP = 20    # words
MAX_CHARS     = 900   # characters — roughly 180 words, hard safety ceiling

FORCE_RERUN = True

FAILED_LOG = Path("/home/justin/rag-civil/ingestion/failed_chunks.jsonl")

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
##Carries all metadata forward so every chunk stored in ChromaDB knows exactly where it came from
##Adding a new state later is as simple as adding a line to the state_agencies dictionary
def tag_metadata(chunk: dict, pdf_path: Path) -> dict:
    parts = pdf_path.parts
    
    jurisdiction = "UNKNOWN"
    agency = "UNKNOWN"
    state = None
    locality = None

    federal_agencies = {
        "fhwa": "FHWA",
        "osha": "OSHA",
        "usace": "USACE",
        "ada": "ADA",
        "env": "EPA",
    }

    state_agencies = {
        "WA": "WSDOT",
        # "OR": "ODOT",
        # "CA": "Caltrans",
        # "ID": "ITD",
    }

    local_agencies = {
        "WA": {
            "KING":     "King County",
            "PIERCE":   "Pierce County",
            "SNOHOMISH":"Snohomish County",
            "SEATTLE":  "City of Seattle",
            "TACOMA":   "City of Tacoma",
            "SPOKANE":  "City of Spokane",
            # add more WA localities here
        },
        # "OR": {
        #     "PORTLAND": "City of Portland",
        # },
    }

    if "FEDERAL" in parts:
        jurisdiction = "FEDERAL"
        for part in parts:
            if part.lower() in federal_agencies:
                agency = federal_agencies[part.lower()]
                break

    elif "STATE" in parts:
        jurisdiction = "STATE"
        state_idx = parts.index("STATE")
        if state_idx + 1 < len(parts):
            state = parts[state_idx + 1].upper()
            agency = state_agencies.get(state, "UNKNOWN")

    elif "LOCAL" in parts:
        jurisdiction = "LOCAL"
        local_idx = parts.index("LOCAL")
        if local_idx + 1 < len(parts):
            state = parts[local_idx + 1].upper()
        if local_idx + 2 < len(parts):
            locality = parts[local_idx + 2].upper()
            agency = local_agencies.get(state, {}).get(locality, "UNKNOWN")

# Dynamic section detection
    # Strategy: find the first line that looks like a section identifier
    # Section identifiers start with a digit and contain dots, dashes, 
    # parentheticals, or brackets before hitting whitespace or text
    section  = detect_section(chunk["text"])
    doc_page = "UNKNOWN"  # filled in by retag.py after ingestion

    # Doc page detection — dash format page numbers like 1320-5, 13-205
    # Search first 300 chars, prefer patterns that look like division-page
    # Exclude section numbers already captured and things like phone numbers
    doc_page_match = re.search(
        r'(?<!\d)'          # not preceded by digit (avoid matching mid-number)
        r'(\d{1,4}-\d{1,4})'
        r'(?!\d)',          # not followed by digit
        chunk["text"][:300]
    )
    if doc_page_match:
        candidate = doc_page_match.group(1)
        # Only accept if it doesn't match the section we already found
        if candidate != section:
            doc_page = candidate

    return {
        **chunk,
        "jurisdiction": jurisdiction,
        "agency":       agency,
        "state":        state or "",
        "locality":     locality or "",
        "source_file":  pdf_path.name,
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
        pa.field("section",      pa.string()),
        pa.field("llm_corrected_section", pa.bool_()),
        pa.field("doc_page",     pa.string()),
        pa.field("page",         pa.int32()),
        pa.field("chunk_index",  pa.int32()),
    ])

    if "civil_engineering_codes" in db.db.list_tables():
        table = db.open_table("civil_engineering_codes")
        print(f"  LanceDB opened — {table.count_rows()} existing chunks")
    else:
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
            "section":      chunk["section"],
            "llm_corrected_section": False,
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

def process_pdf(pdf_path: Path, table: lancedb.table.LanceTable) -> None:
    if not FORCE_RERUN and already_ingested(pdf_path, table):
        print(f"  [skipped] already in database: {pdf_path.name}")
        return

    print(f"\nProcessing: {pdf_path.name}")

    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        print(f"  [skipped] no usable text found in {pdf_path.name}")
        return

    all_chunks = []
    for page_data in pages:
        chunks = chunk_text(page_data)
        tagged = [tag_metadata(chunk, pdf_path) for chunk in chunks]
        all_chunks.extend(tagged)

    all_chunks = propagate_missing_metadata(all_chunks)

    print(f"  Generated {len(all_chunks)} chunks")
    store_chunks(table, all_chunks)

#delete old data when rerunning
def reset_db() -> lancedb.table.LanceTable:
    db = lancedb.connect(str(VECTORDB_DIR))
    
    if "civil_engineering_codes" in db.db.list_tables():
        db.drop_table("civil_engineering_codes")
        print("  Existing table deleted")

    schema = pa.schema([
        pa.field("id",           pa.string()),
        pa.field("text",         pa.string()),
        pa.field("vector",       pa.list_(pa.float32(), 1024)),
        pa.field("source_file",  pa.string()),
        pa.field("jurisdiction", pa.string()),
        pa.field("agency",       pa.string()),
        pa.field("state",        pa.string()),
        pa.field("locality",     pa.string()),
        pa.field("section",      pa.string()),
        pa.field("llm_corrected_section", pa.bool_()),
        pa.field("doc_page",     pa.string()),
        pa.field("page",         pa.int32()),
        pa.field("chunk_index",  pa.int32()),
    ])

    table = db.create_table("civil_engineering_codes", schema=schema)
    print("  New table created")
    return table

def main():
    print("=" * 60)
    print("Civil Engineering RAG — Ingestion Pipeline")
    print("=" * 60)

    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

    if FORCE_RERUN:
        print("\nFORCE_RERUN is enabled — this will delete and re-ingest all files.")
        confirm = input("Type Y to confirm, anything else to cancel: ").strip()
        if confirm != "Y":
            print("Cancelled.")
            return
        print("Confirmed — resetting database")
        table = reset_db()
    else:
        table = init_db()

    pdf_files = sorted(DOCS_DIR.rglob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDF files to process\n")

    skipped = []
    failed  = []
    success = []

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}]", end=" ")
        try:
            process_pdf(pdf_path, table)
            success.append(pdf_path.name)
        except Exception as e:
            print(f"  [error] {pdf_path.name}: {e}")
            failed.append(pdf_path.name)

    print("\n" + "=" * 60)
    print("Ingestion complete")
    print(f"  Successful : {len(success)} files")
    print(f"  Failed     : {len(failed)} files")
    print(f"  Total chunks in database: {table.count_rows()}")

    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  - {f}")

    print("=" * 60)

if __name__ == "__main__":
    main()