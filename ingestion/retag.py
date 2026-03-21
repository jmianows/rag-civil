import re
import lancedb
from pathlib import Path
from collections import defaultdict
from metadata import detect_doc_page, propagate_metadata
from section_parser import (
    extract_section_candidate,
    segments_to_string,
    is_valid_advance,
)

VECTORDB_DIR = Path("/home/justin/rag-civil/vectordb")


def retag():
    db = lancedb.connect(str(VECTORDB_DIR))
    table = db.open_table("civil_engineering_codes")

    print("Loading all chunks...")
    all_rows = table.search().limit(999999).to_list()
    print(f"Loaded {len(all_rows)} chunks")

    by_file = defaultdict(list)
    for row in all_rows:
        by_file[row["source_file"]].append(row)

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

CORRECTIONS_LOG = Path("/home/justin/rag-civil/ingestion/corrections.jsonl")


def log_correction(
    source_file: str,
    page: int,
    chunk_index: int,
    old_section: str,
    correct_section: str,
) -> None:
    import json
    from datetime import datetime
    with open(CORRECTIONS_LOG, "a") as f:
        json.dump({
            "source_file":     source_file,
            "page":            page,
            "chunk_index":     chunk_index,
            "old_section":     old_section,
            "correct_section": correct_section,
            "timestamp":       datetime.now().isoformat(),
        }, f)
        f.write("\n")
    print(f"  Logged correction: {source_file} p{page} c{chunk_index} -> {correct_section}")


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
            correction = json.loads(line)
            chunk_id = (
                f"{correction['source_file']}"
                f"__p{correction['page']}"
                f"__c{correction['chunk_index']}"
            )
            try:
                table.update(
                    where=f"id = '{chunk_id}'",
                    values={
                        "section":               correction["correct_section"],
                        "llm_corrected_section": True,
                    }
                )
                applied += 1
            except Exception as e:
                print(f"  Failed to apply correction {chunk_id}: {e}")

    print(f"  Applied {applied} corrections")
    return applied

if __name__ == "__main__":
    retag()