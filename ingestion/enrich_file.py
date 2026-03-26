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

import sys, argparse, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
from rag.query_engine import (
    _ollama_session, _next_ollama_host, strip_thinking,
    LLM_MODEL, OLLAMA_KEEP_ALIVE, _ensure_fts_index,
)
from ingestion.retag import log_correction

VECTORDB_DIR = Path(__file__).parent.parent / "vectordb"
BATCH_SIZE = 8


def batch_enrich(rows: list[dict], field: str) -> dict[str, str]:
    """Send up to BATCH_SIZE chunks to the LLM in one call.
    Returns {chunk_id: value} for all chunks where a value was identified."""
    if field == "section":
        ask = "the section number (e.g. '4D.03', '1926.502', '6-02.3')"
    else:
        ask = "the document page number or range (e.g. '142', '5-100')"

    source_file = rows[0]["source_file"]
    lines = "\n\n".join(f"[{i+1}] {r['text'][:300]}" for i, r in enumerate(rows))
    mapping = {
        str(i + 1): f"{r['source_file']}__p{r['page']}__c{r['chunk_index']}"
        for i, r in enumerate(rows)
    }

    prompt = (
        f"You are reading chunks from {source_file}.\n"
        f"For each numbered chunk, identify {ask}.\n"
        f'Return ONLY a JSON object like {{"1": "4D.03", "2": "UNKNOWN"}}.\n'
        f"Do not explain.\n\n{lines}"
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0, "num_predict": 200},
    }
    try:
        resp = _ollama_session.post(_next_ollama_host() + "/api/chat", json=payload)
        raw = strip_thinking(resp.json()["message"]["content"].strip())
        start, end = raw.find("{"), raw.rfind("}") + 1
        parsed = json.loads(raw[start:end]) if start >= 0 else {}
        return {mapping[k]: v for k, v in parsed.items() if k in mapping}
    except Exception as e:
        print(f"  [batch warn] {e}")
        return {}


def forward_propagate(rows: list[dict], field: str, updates: dict[str, str]) -> dict[str, str]:
    """Forward-fill field values using LLM results combined with existing known values.
    Returns {chunk_id: value} for chunks that were filled by propagation only."""
    last = "UNKNOWN"
    propagated = {}
    for r in rows:
        cid = f"{r['source_file']}__p{r['page']}__c{r['chunk_index']}"
        val = updates.get(cid) or r.get(field, "UNKNOWN")
        if val not in ("UNKNOWN", ""):
            last = val
        elif last != "UNKNOWN" and cid not in updates:
            propagated[cid] = last
    return propagated


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

    do_section = not args.page_only
    do_page    = not args.section_only
    total_written = 0

    fields = []
    if do_section: fields.append("section")
    if do_page:    fields.append("doc_page")

    for field in fields:
        targets = rows if args.all else [
            r for r in rows if r.get(field, "UNKNOWN") == "UNKNOWN"
        ]
        print(f"\n── {field} — {len(targets)} chunks to enrich ──")
        if not targets:
            print("  Nothing to do.")
            continue

        # Phase 1: batch LLM calls
        llm_updates: dict[str, str] = {}
        batches = [targets[i:i + BATCH_SIZE] for i in range(0, len(targets), BATCH_SIZE)]
        for i, batch in enumerate(batches):
            print(f"  batch {i+1}/{len(batches)} ({len(batch)} chunks)...", end=" ", flush=True)
            result = batch_enrich(batch, field)
            resolved = sum(1 for v in result.values() if v not in ("UNKNOWN", ""))
            print(f"{resolved} resolved")
            llm_updates.update({k: v for k, v in result.items()
                                 if v not in ("UNKNOWN", "")})

        # Phase 2: propagate LLM results to fill gaps
        prop_updates = forward_propagate(rows, field, llm_updates)
        print(f"  Propagation filled {len(prop_updates)} additional chunks")

        all_updates = {**llm_updates, **prop_updates}
        print(f"  Total changes: {len(all_updates)}")

        # Apply / print
        flag = "llm_corrected_section" if field == "section" else "llm_corrected_doc_page"
        for cid, new_val in sorted(all_updates.items()):
            row = next(
                (r for r in rows
                 if f"{r['source_file']}__p{r['page']}__c{r['chunk_index']}" == cid),
                None,
            )
            old_val = row.get(field, "UNKNOWN") if row else "?"
            src = "llm" if cid in llm_updates else "prop"
            print(f"  [{src}] {cid}  {old_val!r} → {new_val!r}")
            if args.apply and row:
                table.update(
                    where=f"id = '{cid}'",
                    values={field: new_val, flag: True},
                )
                log_correction(
                    row["source_file"], row["page"], row["chunk_index"],
                    old_val, new_val, field=field,
                )
                total_written += 1

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
