from metadata import detect_section, detect_doc_page, propagate_metadata, DocumentSectionTracker
try:
    from ingestion.section_parser import extract_section_from_heading
except ImportError:
    from section_parser import extract_section_from_heading
try:
    from ingestion.common import _load_link
except ImportError:
    from common import _load_link
import argparse
import json
import os
import re
import statistics
import fitz
import lancedb
import ollama
from dataclasses import dataclass
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
DOCS_DIR: Path    = _PROJECT_ROOT / "docs"
FORCE_RERUN: bool = False
DRY_RUN: bool     = False


# ── Structured extraction data model ──────────────────────────────────────────

@dataclass
class Block:
    text:      str
    kind:      str    # "heading" | "body" | "header" | "footer"
    page:      int    # physical PDF page number (1-indexed)
    doc_page:  str    # printed page number captured from header/footer, e.g. "1510-23"
    section:   str    # section number inherited from last heading seen
    bold:      bool
    font_size: float
    font_name: str    # dominant font name from PyMuPDF span data
    bbox:      tuple  # (x0, y0, x1, y1)

# Leading list markers that disqualify a block from being a heading even if bold/large.
# Matches bullet chars and en/em dash used as list markers (not ASCII hyphen in section numbers).
_LIST_PREFIX_RE   = re.compile(r'^[•–—▪◦]\s')
# Exhibit/Figure/Table labels are captions, not section headings.
_CAPTION_PREFIX_RE = re.compile(r'^(?:Exhibit|Figure|Table|Fig\.?)\s+\d', re.IGNORECASE)
# URL/domain patterns — reference list lines, not section headings.
_URL_RE           = re.compile(r'(?:www\.|https?://|\.gov|\.org|\.com)', re.IGNORECASE)
# Real section labels — numeric (1-2, 5-19.1, 1510.04) or keyword (Chapter 5, Section 3).
# Used to guard against text-fallback prose labels overwriting a good numeric section.
_REAL_SECTION_RE  = re.compile(r'^\d|^(?:Chapter|Section|Part|Article)\s+\d', re.IGNORECASE)


def _looks_like_real_section(s: str) -> bool:
    return bool(_REAL_SECTION_RE.match(s.strip()))


def _extract_doc_page(text: str) -> str:
    """Try to extract a printed page number from a header or footer line.

    Priority order:
      1. 'Page N' / 'Page xv' keyword — unambiguous, handles roman numerals and
         dash-suffixed numbers (Page 1510-23). Checked first so document version
         numbers like 'M 22-01.23' don't get captured by the dash pattern.
      2. Dash-format 'NNNN-NN' — chapter-page style (1510-23, 5-17) for docs
         without a 'Page' keyword in their headers/footers.
      3. Bare integer — entire block is a standalone page number (ADA-style).
         Rejected if it looks like a year (1900-2099).
    """
    text = text.strip()

    # Priority 1: "Page N", "Page xv", "Page 1510-23", "Page 1 of 4"
    page_kw = re.search(
        r'(?:Page|PAGE)\s+([ivxlcdm]+|\d{1,4}(?:-\d{1,4})?)',
        text, re.IGNORECASE,
    )
    if page_kw:
        return page_kw.group(1)

    # Priority 2: dash-format page numbers e.g. "1510-23", "5-17"
    dash = re.search(r'\b(\d{1,4}-\d{1,4})\b', text)
    if dash:
        parts = dash.group(1).split('-')
        if all(p.isdigit() and int(p) < 5000 for p in parts):
            return dash.group(1)

    # Priority 3: entire block is a bare integer page number (e.g. ADA CFR footers)
    bare = re.fullmatch(r'\d{1,4}', text)
    if bare:
        n = int(text)
        if not (1900 <= n <= 2099):
            return text

    return ""


def _block_text_and_style(fitz_block: dict) -> tuple[str, float, bool, str]:
    """Extract joined text, median font size, bold flag, and dominant font name."""
    texts      = []
    sizes      = []
    font_names = []
    is_bold    = False
    for line in fitz_block.get("lines", []):
        for span in line.get("spans", []):
            t = span.get("text", "").strip()
            if t:
                texts.append(t)
                font_names.append(span.get("font", ""))
            sz = span.get("size", 0.0)
            if sz > 0:
                sizes.append(sz)
            if span.get("flags", 0) & 16:   # bit 4 = bold
                is_bold = True
    text         = " ".join(texts)
    font_size    = statistics.median(sizes) if sizes else 0.0
    dominant_font = max(set(font_names), key=font_names.count) if font_names else ""
    return text, font_size, is_bold, dominant_font

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
    try:
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
    finally:
        doc.close()
    print(f"  Extracted {len(pages)} text pages from {pdf_path.name}")
    return pages

def extract_blocks(pdf_path: Path) -> list[Block]:
    """
    Structured extraction using PyMuPDF dict-mode.

    Classifies each text block as header/footer/heading/body using page position
    and font characteristics, then runs a forward-pass state machine to assign
    authoritative section and doc_page metadata to every body/heading block
    BEFORE any text is discarded.

    Falls back to flat OCR text for scanned documents (section/doc_page = UNKNOWN).
    """
    doc = fitz.open(str(pdf_path))
    raw_blocks: list[Block] = []

    try:
        use_ocr = is_scanned_pdf(doc)
        if use_ocr:
            print(f"  [scanned] using OCR fallback for {pdf_path.name}")
            for page_num in range(len(doc)):
                page = doc[page_num]
                try:
                    text = ocr_pdf_page(page)
                except Exception as e:
                    print(f"  [OCR error] page {page_num + 1}: {e}")
                    continue
                text = clean_page_text(text)
                if len(text.strip()) < 50:
                    continue
                raw_blocks.append(Block(
                    text=text, kind="body", page=page_num + 1,
                    doc_page="UNKNOWN", section="UNKNOWN",
                    bold=False, font_size=0.0, font_name="", bbox=(0, 0, 0, 0),
                ))
            return raw_blocks

        for page_num in range(len(doc)):
            page        = doc[page_num]
            page_height = page.rect.height
            page_width  = page.rect.width
            header_y    = page_height * 0.08   # top 8%
            footer_y    = page_height * 0.92   # bottom 8%

            page_dict   = page.get_text("dict")
            text_blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]
            if not text_blocks:
                continue

            # Median font size across all spans on this page — used to detect headings
            all_sizes = [
                span.get("size", 0.0)
                for b in text_blocks
                for line in b.get("lines", [])
                for span in line.get("spans", [])
                if span.get("size", 0.0) > 0
            ]
            body_font = statistics.median(all_sizes) if all_sizes else 12.0
            heading_threshold = body_font * 1.15

            for fitz_block in text_blocks:
                text, font_size, bold, font_name = _block_text_and_style(fitz_block)
                text = clean_page_text(text)
                if len(text.strip()) < 10:
                    continue

                bbox = fitz_block.get("bbox", (0, 0, 0, 0))
                y0, y1 = bbox[1], bbox[3]
                block_width = bbox[2] - bbox[0]

                # Filter: multi-sentence text is body prose, not a heading.
                multi_sentence = bool(re.search(r'\.\s+[A-Z]', text))
                # Filter: leading bullet/en-dash/em-dash marks a list item, not a heading.
                is_list_item = bool(_LIST_PREFIX_RE.match(text))
                # Filter: Exhibit/Figure/Table captions are not section headings.
                is_caption = bool(_CAPTION_PREFIX_RE.match(text))
                # Filter: URL/domain reference lines are not section headings.
                is_url_block = bool(_URL_RE.search(text))

                # Classify by position first, then font
                if y1 <= header_y:
                    kind = "header"
                elif y0 >= footer_y:
                    kind = "footer"
                elif (font_size >= heading_threshold
                      and not multi_sentence and not is_list_item
                      and not is_caption and not is_url_block):
                    kind = "heading"
                elif (bold and len(text) < 200
                      and not multi_sentence
                      and not is_list_item
                      and not is_caption
                      and not is_url_block
                      and page_width > 0 and block_width / page_width >= 0.4):
                    kind = "heading"
                else:
                    kind = "body"

                raw_blocks.append(Block(
                    text=text, kind=kind, page=page_num + 1,
                    doc_page="UNKNOWN", section="UNKNOWN",
                    bold=bold, font_size=font_size, font_name=font_name, bbox=bbox,
                ))

    finally:
        doc.close()

    if not raw_blocks:
        return []

    # ── Pass 2: font consistency filter ───────────────────────────────────────
    # Size-triggered headings are ground truth for what a heading font looks like.
    # Bold-only headings whose font doesn't match are likely callouts/table rows.
    size_heading_fonts = [
        b.font_name for b in raw_blocks
        if b.kind == "heading" and b.font_size >= (
            # recompute per-block: if font_size was the trigger it was >= threshold
            # we don't store the threshold, but size-triggered blocks have larger fonts
            # than bold-only blocks, so use the median of heading font sizes as a proxy
            statistics.median([
                x.font_size for x in raw_blocks if x.kind == "heading"
            ]) * 0.85  # slightly below median to capture all size-triggered ones
        )
        and b.font_name
    ]
    if size_heading_fonts:
        # Top 2 most common font names among size-triggered headings
        font_counter: dict[str, int] = {}
        for fn in size_heading_fonts:
            font_counter[fn] = font_counter.get(fn, 0) + 1
        heading_fonts = {fn for fn, _ in sorted(font_counter.items(), key=lambda x: -x[1])[:2]}

        for block in raw_blocks:
            if (block.kind == "heading"
                    and block.font_name
                    and block.font_name not in heading_fonts):
                block.kind = "body"

    # ── Forward-pass state machine ─────────────────────────────────────────────
    # Walk blocks in document order. Headers/footers are mined for doc_page then
    # discarded. Headings update current_section. All body/heading blocks receive
    # the current state values before being added to output.
    tracker          = DocumentSectionTracker()
    current_section  = "UNKNOWN"
    current_doc_page = "UNKNOWN"
    output: list[Block] = []

    for block in raw_blocks:
        if block.kind in ("header", "footer"):
            candidate = _extract_doc_page(block.text)
            if candidate:
                current_doc_page = candidate
            continue   # strip from output

        block.doc_page = current_doc_page

        if block.kind == "heading":
            # Skip boilerplate blank-page markers entirely.
            if "intentionally left blank" in block.text.lower():
                continue
            candidate = extract_section_from_heading(block.text)
            # Only accept a text-fallback (prose) label if we don't already have
            # a real numeric/keyword section. Prevents verbose heading titles from
            # overwriting good section numbers mid-document.
            if _looks_like_real_section(candidate) or not _looks_like_real_section(current_section):
                current_section = candidate
            block.section = current_section
            is_toc, _ = is_table_of_contents(block.text)
            if is_toc:
                continue
            output.append(block)

        else:  # body
            is_toc, _ = is_table_of_contents(block.text)
            if is_toc:
                continue
            block.section = current_section
            output.append(block)

    n_stripped = len(raw_blocks) - len(output)
    print(f"  Extracted {len(output)} blocks from {pdf_path.name} "
          f"({n_stripped} header/footer/TOC blocks stripped)")
    return output


def chunk_blocks(blocks: list[Block]) -> list[dict]:
    """
    Section-aware chunker operating on Block objects.

    Flushes the accumulator at every heading boundary so no chunk ever spans
    two sections. The heading text is prepended to the first chunk of its section
    to give the embedder richer context. doc_page and section come directly from
    the Block — never re-detected from text.
    """
    if not blocks:
        return []

    results:     list[dict] = []
    chunk_index: int        = 0
    acc_words:   list[str]  = []
    acc_section:  str = "UNKNOWN"
    acc_doc_page: str = "UNKNOWN"
    acc_page:     int = 0

    def _flush_remainder() -> None:
        nonlocal chunk_index
        if len(acc_words) < 30:
            return
        results.append({
            "text":        " ".join(acc_words),
            "page":        acc_page,
            "doc_page":    acc_doc_page,
            "section":     acc_section,
            "chunk_index": chunk_index,
        })
        chunk_index += 1

    def _flush_sliding() -> None:
        """Emit a full CHUNK_SIZE chunk and keep CHUNK_OVERLAP words for next."""
        nonlocal acc_words, chunk_index
        chunk_str = " ".join(acc_words[:CHUNK_SIZE])
        if len(chunk_str.strip()) > 50:
            results.append({
                "text":        chunk_str,
                "page":        acc_page,
                "doc_page":    acc_doc_page,
                "section":     acc_section,
                "chunk_index": chunk_index,
            })
            chunk_index += 1
        acc_words[:] = acc_words[CHUNK_SIZE - CHUNK_OVERLAP:]

    for block in blocks:
        words = block.text.split()
        if not words:
            continue

        if block.kind == "heading":
            _flush_remainder()
            acc_words    = words          # heading text starts the new chunk
            acc_section  = block.section
            acc_doc_page = block.doc_page
            acc_page     = block.page

        else:  # body
            # Safety flush if section changes mid-stream without a heading block
            if (block.section != acc_section
                    and acc_section  != "UNKNOWN"
                    and block.section != "UNKNOWN"):
                _flush_remainder()
                acc_words = []

            acc_section  = block.section
            acc_doc_page = block.doc_page
            acc_page     = block.page
            acc_words.extend(words)

            while len(acc_words) >= CHUNK_SIZE:
                _flush_sliding()

    _flush_remainder()
    return results


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

    # section and doc_page are set authoritatively by extract_blocks() / chunk_blocks().
    # Do NOT re-detect from text — that would overwrite the structural metadata.
    return {
        **chunk,
        "jurisdiction": jurisdiction,
        "agency":       agency,
        "state":        state,
        "locality":     locality,
        "source_file":  pdf_path.name,
        "file_link":    file_link,
        "section":      chunk.get("section", "UNKNOWN"),
        "doc_page":     chunk.get("doc_page", "UNKNOWN"),
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
    safe = pdf_path.name.replace("'", "''").replace("\\", "\\\\")
    results = table.search() \
        .where(f"source_file = '{safe}'") \
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
    if not DRY_RUN and already_ingested(pdf_path, table):
        if not FORCE_RERUN:
            print(f"  [skipped] already in database: {pdf_path.name}")
            return
        safe = pdf_path.name.replace("'", "''").replace("\\", "\\\\")
        table.delete(f"source_file = '{safe}'")
        print(f"  [force] deleted existing rows for {pdf_path.name}, re-ingesting")

    print(f"\nProcessing: {pdf_path.name}")

    # ── Structured extraction: blocks with authoritative section + doc_page ───
    blocks = extract_blocks(pdf_path)
    if not blocks:
        print(f"  [skipped] no usable blocks found in {pdf_path.name}")
        return

    # ── Dry-run diagnostic: print sample of blocks and stop ──────────────────
    if DRY_RUN:
        print(f"  [dry-run] first 15 blocks:")
        for b in blocks[:15]:
            print(f"    [{b.kind:7s}] p{b.page:>3}  sec={b.section:<15}  "
                  f"doc_page={b.doc_page:<10}  {b.text[:80]!r}")
        return

    # ── Chunk using block metadata — no re-detection needed ──────────────────
    raw_chunks = chunk_blocks(blocks)
    all_chunks = [tag_metadata(c, pdf_path, root) for c in raw_chunks]

    # Safety net: propagate any remaining UNKNOWN values forward
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

def main(docs_dir: Path, force: bool, only: str | None, dry_run: bool = False) -> None:
    global DOCS_DIR, FORCE_RERUN, DRY_RUN
    DOCS_DIR    = docs_dir
    FORCE_RERUN = force
    DRY_RUN     = dry_run

    print("=" * 60)
    print("Civil Engineering RAG — Ingestion Pipeline")
    print(f"Root : {DOCS_DIR}")
    if DRY_RUN:
        print("Mode : DRY RUN — no database writes, block diagnostics only")
    print("=" * 60)

    if not DOCS_DIR.exists():
        print(f"[error] Root path does not exist: {DOCS_DIR}")
        return

    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

    if FORCE_RERUN and not DRY_RUN:
        print("\n--force: existing rows for each processed file will be deleted and re-ingested.")
        confirm = input("Type Y to confirm, anything else to cancel: ").strip()
        if confirm != "Y":
            print("Cancelled.")
            return
        print("Confirmed.")

    table = init_db()

    pdf_files = sorted(DOCS_DIR.rglob("*.pdf"))

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

    if DRY_RUN:
        print("\n" + "=" * 60)
        print("Dry run complete — no data written.")
        print("=" * 60)
        return

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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print block diagnostics without writing to the database",
    )
    args = parser.parse_args()
    main(Path(args.root), args.force, args.only, dry_run=args.dry_run)