import re
import lancedb
from pathlib import Path
from collections import defaultdict
try:
    from ingestion.metadata import detect_doc_page, propagate_metadata
    from ingestion.section_parser import (
        extract_section_candidate,
        segments_to_string,
        is_valid_advance,
    )
except ImportError:
    from metadata import detect_doc_page, propagate_metadata
    from section_parser import (
        extract_section_candidate,
        segments_to_string,
        is_valid_advance,
    )
try:
    from rag.query_engine import _ensure_fts_index
except ImportError:
    from query_engine import _ensure_fts_index

try:
    from rag.env_config import VECTORDB_DIR
except ImportError:
    VECTORDB_DIR = Path(__file__).parent.parent / "vectordb"


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
    def _load_link(pdf_path: Path, root: Path) -> str:
        import json as _json
        for name in ("_links.json", "links.json"):
            sidecar = pdf_path.parent / name
            if sidecar.exists():
                try:
                    links = _json.loads(sidecar.read_text(encoding="utf-8"))
                    url = links.get(pdf_path.name, "")
                    if url:
                        return url
                except Exception:
                    pass
        for sidecar in pdf_path.parent.glob("*_links.json"):
            try:
                links = _json.loads(sidecar.read_text(encoding="utf-8"))
                url = links.get(pdf_path.name, "")
                if url:
                    return url
            except Exception:
                pass
        registry = root / "_registry.json"
        if registry.exists():
            try:
                import json as _json2
                reg = _json2.loads(registry.read_text(encoding="utf-8"))
                url = reg.get(pdf_path.relative_to(root).as_posix(), "")
                if url:
                    return url
            except Exception:
                pass
        return ""

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
    for row in all_rows:
        if row["source_file"] in file_map:
            row.update(_derive_metadata(file_map[row["source_file"]]))
            refreshed += 1
    print(f"Refreshed metadata for {refreshed} rows")
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
        current_segments = None

        for row in rows:
            lines = row["text"].strip().splitlines()
            detected = None

            for line_num, line in enumerate(lines[:10]):
                candidate = extract_section_candidate(line)
                if candidate is None:
                    continue

                candidate_str = segments_to_string(candidate)

                if is_valid_advance(current_segments, candidate):
                    current_segments = candidate
                    detected = candidate_str
                    break

                if line_num < 2:
                    current_segments = candidate
                    detected = candidate_str
                    break

            row["section"]  = detected if detected else (
                segments_to_string(current_segments)
                if current_segments else "UNKNOWN"
            )
            row["doc_page"] = page_numbers.get(row["page"], "UNKNOWN")

        updated_rows.extend(rows)
        print(f"  Retagged {len(rows)} chunks from {filename}")

    print("Writing updated chunks back to database...")
    db.drop_table("civil_engineering_codes")
    new_table = db.create_table("civil_engineering_codes", data=updated_rows)
    print(f"Done — {new_table.count_rows()} chunks written")

    print("\nApplying corrections...")
    apply_corrections(new_table)

    print("\nBuilding FTS index...")
    _ensure_fts_index(new_table)
    print("FTS index ready.")

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
                table.update(where=f"id = '{chunk_id}'", values=values)
                applied += 1
            except Exception as e:
                print(f"  Failed to apply {field} correction {chunk_id}: {e}")

    print(f"  Applied {applied} corrections")
    return applied

if __name__ == "__main__":
    retag()