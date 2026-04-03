import re
import lancedb
from pathlib import Path
from collections import defaultdict
try:
    from ingestion.metadata import detect_doc_page
    from ingestion.section_parser import DocumentSectionTracker
except ImportError:
    from metadata import detect_doc_page
    from section_parser import DocumentSectionTracker, segments_to_string
try:
    from rag.query_engine import _ensure_fts_index
except ImportError:
    from query_engine import _ensure_fts_index

try:
    from rag.env_config import VECTORDB_DIR
except ImportError:
    VECTORDB_DIR = Path(__file__).parent.parent / "vectordb"

try:
    from ingestion.common import _load_link
except ImportError:
    from common import _load_link


def retag():
    db = lancedb.connect(str(VECTORDB_DIR))
    table = db.open_table("civil_engineering_codes")

    print("Loading all chunks...")
    all_rows = table.search().limit(999999).to_list()
    print(f"Loaded {len(all_rows)} chunks")

    by_file = defaultdict(list)
    for row in all_rows:
        by_file[row["source_file"]].append(row)

    # ── Metadata refresh from current docs tree ──────────────────────────────
    DOCS_ROOT = Path(__file__).parent.parent / "docs"
    file_map = {p.name: p for p in DOCS_ROOT.rglob("*.pdf")}
    print(f"Found {len(file_map)} PDFs in docs tree for metadata refresh")

    def _derive_metadata(pdf_path: Path) -> dict:
        rel   = pdf_path.relative_to(DOCS_ROOT)
        parts = rel.parts
        tier  = parts[0].upper() if parts else "UNKNOWN"
        jurisdiction = state = agency = "UNKNOWN"
        locality = ""
        if tier == "FEDERAL" and len(parts) >= 4:
            jurisdiction, state, agency = "FEDERAL", parts[1].upper(), parts[2].upper()
        elif tier == "STATE" and len(parts) >= 4:
            jurisdiction, state, agency = "STATE", parts[1].upper(), parts[2].upper()
        elif tier == "LOCAL" and len(parts) >= 4:
            jurisdiction = "LOCAL"
            state    = parts[1].upper()
            locality = parts[2].title()
            agency   = parts[3].upper() if len(parts) >= 5 else locality.upper()
        return {
            "jurisdiction": jurisdiction,
            "state":        state,
            "locality":     locality,
            "agency":       agency,
            "file_link":    _load_link(pdf_path, DOCS_ROOT),
        }

    refreshed = 0
    stale_files: set[str] = set()
    for row in all_rows:
        if row["source_file"] in file_map:
            row.update(_derive_metadata(file_map[row["source_file"]]))
            refreshed += 1
        else:
            stale_files.add(row["source_file"])
    print(f"Refreshed metadata for {refreshed} rows")
    if stale_files:
        print(f"[warn] {len(stale_files)} source file(s) not found in docs tree — metadata kept as-is:")
        for sf in sorted(stale_files):
            print(f"  {sf}")
    # ─────────────────────────────────────────────────────────────────────────

    updated_rows = []

    for filename, rows in by_file.items():
        rows.sort(key=lambda r: (r["page"], r["chunk_index"]))

        # page-level doc_page detection
        by_page = defaultdict(list)
        for row in rows:
            by_page[row["page"]].append(row)

        page_numbers = {}
        for page_num, page_chunks in by_page.items():
            full_page_text = "\n".join(c["text"] for c in page_chunks)
            page_numbers[page_num] = detect_doc_page(full_page_text, "")

        # section detection using state machine per document
        tracker = DocumentSectionTracker()

        for row in rows:
            row["section"]  = tracker.process_chunk(row["text"])
            row["doc_page"] = page_numbers.get(row["page"], "UNKNOWN")

        updated_rows.extend(rows)
        print(f"  Retagged {len(rows)} chunks from {filename}")

    # Ensure boolean columns exist so create_table schema is consistent
    for row in updated_rows:
        row.setdefault("llm_corrected_section", False)
        row.setdefault("llm_corrected_doc_page", False)

    # Atomic swap: build into a tmp table, then rename over the live table so a
    # crash between steps doesn't destroy the database.
    _TMP = "civil_engineering_codes_retag_tmp"
    if _TMP in db.table_names():
        db.drop_table(_TMP)

    print("Writing updated chunks to temporary table...")
    tmp_table = db.create_table(_TMP, data=updated_rows)
    print(f"  {tmp_table.count_rows()} chunks written")

    print("\nApplying corrections to temporary table...")
    apply_corrections(tmp_table)

    print("\nBuilding FTS index on temporary table...")
    _ensure_fts_index(tmp_table)
    print("FTS index ready.")

    print("\nSwapping tables...")
    db.drop_table("civil_engineering_codes")
    try:
        db.rename_table(_TMP, "civil_engineering_codes")
    except Exception as e:
        print(f"[CRITICAL] rename failed after drop: {e}", flush=True)
        print(f"[CRITICAL] Attempting recovery: renaming '{_TMP}' → 'civil_engineering_codes'", flush=True)
        try:
            db.rename_table(_TMP, "civil_engineering_codes")
            print("[CRITICAL] Recovery succeeded — service should be operational.", flush=True)
            # Recovery worked; don't re-raise the original error
        except Exception as e2:
            print(f"[CRITICAL] Recovery also failed: {e2}", flush=True)
            print(f"[CRITICAL] Manual fix: python -c \"import lancedb; db=lancedb.connect('vectordb'); db.rename_table('{_TMP}', 'civil_engineering_codes')\"", flush=True)
            raise RuntimeError(f"Table swap failed and recovery failed. DB is down. Original: {e}; Recovery: {e2}") from e2
    new_table = db.open_table("civil_engineering_codes")
    print(f"Done — {new_table.count_rows()} chunks live.")

    try:
        from rag.query_engine import invalidate_db_table
        invalidate_db_table()
        print("DB connection cache cleared.")
    except Exception:
        pass

CORRECTIONS_LOG = Path(__file__).parent / "corrections.jsonl"


def log_correction(
    source_file: str,
    page: int,
    chunk_index: int,
    old_value: str,
    correct_value: str,
    field: str = "section",
) -> None:
    import json
    from datetime import datetime
    with open(CORRECTIONS_LOG, "a") as f:
        json.dump({
            "source_file":   source_file,
            "page":          page,
            "chunk_index":   chunk_index,
            "old_value":     old_value,
            "correct_value": correct_value,
            "field":         field,
            "timestamp":     datetime.now().isoformat(),
        }, f)
        f.write("\n")
    print(f"  Logged {field} correction: {source_file} p{page} c{chunk_index} → {correct_value}")


def apply_corrections(table) -> int:
    import json
    if not CORRECTIONS_LOG.exists():
        print("  No corrections log found")
        return 0

    applied = 0
    with open(CORRECTIONS_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)

            # Support both old format (correct_section) and new format (correct_value + field)
            field         = c.get("field", "section")
            correct_value = c.get("correct_value") or c.get("correct_section", "")
            if not correct_value or correct_value == "UNKNOWN":
                continue

            chunk_id = (
                f"{c['source_file']}"
                f"__p{c['page']}"
                f"__c{c['chunk_index']}"
            )

            if field == "section":
                values = {"section": correct_value, "llm_corrected_section": True}
            elif field == "doc_page":
                values = {"doc_page": correct_value, "llm_corrected_doc_page": True}
            else:
                continue

            try:
                safe_id = chunk_id.replace("'", "''").replace("\\", "\\\\")
                table.update(where=f"id = '{safe_id}'", values=values)
                applied += 1
            except Exception as e:
                print(f"  Failed to apply {field} correction {chunk_id}: {e}")

    print(f"  Applied {applied} corrections")
    return applied

if __name__ == "__main__":
    retag()