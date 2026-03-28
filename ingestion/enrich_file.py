"""
Batch LLM enrichment for UNKNOWN section/doc_page values in a specific source file.
Regex extraction and forward-fill propagation already ran at ingestion time.
This handles what they missed by batching chunks into single LLM calls, then
propagating the results to fill remaining gaps.

Usage:
    .venv/bin/python ingestion/enrich_file.py <source_file> [--apply] [--all]
                     [--section-only] [--page-only]

Options:
    --apply        Write changes to DB (default: dry run, print only)
    --all          Re-enrich all chunks, not just UNKNOWN ones
    --section-only Only fix section numbers
    --page-only    Only fix doc_page numbers
"""

import sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
from rag.query_engine import _ensure_fts_index
from ingestion.enrich_files import enrich_one_file

try:
    from rag.env_config import VECTORDB_DIR
except ImportError:
    VECTORDB_DIR = Path(__file__).parent.parent / "vectordb"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_file", help="Exact source_file value as stored in DB")
    parser.add_argument("--apply",        action="store_true", help="Write changes to DB")
    parser.add_argument("--all",          action="store_true", help="Re-enrich all chunks, not just UNKNOWN")
    parser.add_argument("--section-only", action="store_true", dest="section_only")
    parser.add_argument("--page-only",    action="store_true", dest="page_only")
    args = parser.parse_args()

    db    = lancedb.connect(str(VECTORDB_DIR))
    table = db.open_table("civil_engineering_codes")

    all_rows = table.search().limit(999999).to_list()
    rows = [r for r in all_rows if r["source_file"] == args.source_file]

    if not rows:
        print(f"No chunks found for '{args.source_file}'")
        print("\nAvailable files:")
        files = sorted({r["source_file"] for r in all_rows})
        for f in files:
            print(f"  {f}")
        sys.exit(1)

    rows.sort(key=lambda r: (r["page"], r["chunk_index"]))
    print(f"Found {len(rows)} chunks for {args.source_file}")

    unknown_sec  = sum(1 for r in rows if r.get("section",  "UNKNOWN") == "UNKNOWN")
    unknown_page = sum(1 for r in rows if r.get("doc_page", "UNKNOWN") == "UNKNOWN")
    print(f"  UNKNOWN sections:  {unknown_sec}/{len(rows)}")
    print(f"  UNKNOWN doc_pages: {unknown_page}/{len(rows)}")

    if not args.apply:
        print("\n[DRY RUN] Pass --apply to write changes.\n")

    fields = []
    if not args.page_only:    fields.append("section")
    if not args.section_only: fields.append("doc_page")

    total_written = enrich_one_file(table, rows, fields, args.apply, args.all)

    if args.apply and total_written:
        print(f"\nRebuilding FTS index...")
        _ensure_fts_index(table)
        print(f"Done. {total_written} chunks updated.")
    elif args.apply:
        print("\nNo changes written (all results were UNKNOWN).")
    else:
        print("\n[DRY RUN complete] Re-run with --apply to commit.")


if __name__ == "__main__":
    main()
