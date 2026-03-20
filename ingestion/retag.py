
from metadata import detect_section, detect_doc_page, propagate_metadata
import re
import json
import lancedb
from pathlib import Path

VECTORDB_DIR = Path("/home/justin/rag-civil/vectordb")

def retag():
    db = lancedb.connect(str(VECTORDB_DIR))
    table = db.open_table("civil_engineering_codes")

    print("Loading all chunks...")
    all_rows = table.search().limit(999999).to_list()
    print(f"Loaded {len(all_rows)} chunks")

    # group by source file to propagate metadata per document
    from collections import defaultdict
    by_file = defaultdict(list)
    for row in all_rows:
        by_file[row["source_file"]].append(row)

    # sort each file's chunks by page then chunk_index
    for filename in by_file:
        by_file[filename].sort(key=lambda r: (r["page"], r["chunk_index"]))

    print("Retagging...")
    updated = 0
    for filename, rows in by_file.items():
        for row in rows:
            row["section"]  = detect_section(row["text"])
            row["doc_page"] = detect_doc_page(row["text"], row["section"])

        rows = propagate_metadata(rows)

        for row in rows:
            table.update(
                where=f"id = '{row['id']}'",
                values={
                    "section":  row["section"],
                    "doc_page": row["doc_page"],
                }
            )
            updated += 1

        print(f"  Retagged {len(rows)} chunks from {filename}")

    print(f"\nDone — updated {updated} chunks")


if __name__ == "__main__":
    retag()