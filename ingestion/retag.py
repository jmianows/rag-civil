
from metadata import detect_section, detect_doc_page, propagate_metadata
import re
import json
import lancedb
from pathlib import Path

VECTORDB_DIR = Path("/home/justin/rag-civil/vectordb")

def detect_page_number_from_full_page(chunks_on_page: list[dict]) -> str:
    full_page_text = "\n".join(c["text"] for c in chunks_on_page)
    lines = full_page_text.strip().splitlines()

    # get last 5 non-empty lines
    bottom_lines = [l.strip() for l in lines if l.strip()][-5:]

    for line in reversed(bottom_lines):
        if not line:
            continue

        # check start of line
        start_match = re.match(
            r'^('
            r'(?:Page|PAGE)\s+\d+'      # "Page 42"
            r'|\d{1,4}-\d{1,4}'        # "1320-5"
            r'|\d{1,4}'                 # "42"
            r')\b',
            line
        )

        # check end of line
        end_match = re.search(
            r'\b('
            r'(?:Page|PAGE)\s+\d+'
            r'|\d{1,4}-\d{1,4}'
            r'|\d{1,4}'
            r')$',
            line
        )

        # prefer end match, fall back to start match
        match = end_match or start_match
        if match:
            candidate = match.group(1).strip()
            # reject years and anything over 4 digits
            if re.match(r'^\d{4}$', candidate):
                continue
            raw_num = candidate.split('-')[-1]
            if raw_num.isdigit() and int(raw_num) > 1999:
                continue
            return candidate

    # fallback — dash format in first 500 chars
    fallback = re.search(r'(?<!\d)(\d{1,4}-\d{1,4})(?!\d)', full_page_text[:500])
    if fallback:
        return fallback.group(1)

    return "UNKNOWN"


def retag():
    db = lancedb.connect(str(VECTORDB_DIR))
    table = db.open_table("civil_engineering_codes")

    print("Loading all chunks...")
    all_rows = table.search().limit(999999).to_list()
    print(f"Loaded {len(all_rows)} chunks")

    from collections import defaultdict

    # group by file
    by_file = defaultdict(list)
    for row in all_rows:
        by_file[row["source_file"]].append(row)

    updated_rows = []

    for filename, rows in by_file.items():
        # sort by page then chunk index
        rows.sort(key=lambda r: (r["page"], r["chunk_index"]))

        # group by page within this file
        by_page = defaultdict(list)
        for row in rows:
            by_page[row["page"]].append(row)

        # pass 1 — detect doc_page at page level using full page text
        page_numbers = {}
        for page_num, page_chunks in by_page.items():
            page_numbers[page_num] = detect_page_number_from_full_page(page_chunks)

        # pass 2 — detect section at chunk level
        for row in rows:
            row["section"]  = detect_section(row["text"])
            row["doc_page"] = page_numbers[row["page"]]

        # propagate section forward across chunks
        rows = propagate_metadata(rows)

        updated_rows.extend(rows)
        print(f"  Retagged {len(rows)} chunks from {filename}")

    print("Writing updated chunks back to database...")
    db.drop_table("civil_engineering_codes")
    new_table = db.create_table("civil_engineering_codes", data=updated_rows)
    print(f"Done — {new_table.count_rows()} chunks written")


if __name__ == "__main__":
    retag()