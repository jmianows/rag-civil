"""
Batch LLM enrichment for UNKNOWN/wrong section and doc_page values.

Single-file mode (same as enrich_file.py):
    .venv/bin/python ingestion/enrich_files.py <source_file> [--apply] [--all]
                     [--section-only] [--page-only]

Bulk scan mode — detect and enrich problem files automatically:
    .venv/bin/python ingestion/enrich_files.py --scan [--apply] [--section-only] [--page-only]
                     [--unknown-threshold N] [--majority-threshold N] [--yes]

Options:
    --apply               Write changes to DB (default: dry run, print only)
    --all                 Re-enrich all chunks, not just UNKNOWN ones (use for majority-same files)
    --section-only        Only fix section numbers
    --page-only           Only fix doc_page numbers
    --scan                Scan all files, flag problems, and optionally enrich them
    --unknown-threshold N Flag files with >N% UNKNOWN sections (default: 20)
    --majority-threshold N Flag files where one section value covers >N% of chunks (default: 60)
    --yes                 Skip confirmation prompt in scan mode
"""

import sys, argparse, json, re
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
from rag.query_engine import (
    _ollama_session, _next_ollama_host, strip_thinking,
    LLM_MODEL, OLLAMA_KEEP_ALIVE, _ensure_fts_index,
)
from ingestion.retag import log_correction

try:
    from rag.env_config import VECTORDB_DIR
except ImportError:
    VECTORDB_DIR = Path(__file__).parent.parent / "vectordb"

BATCH_SIZE = 8


# ── LLM batch call ─────────────────────────────────────────────────────────────

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
        resp.raise_for_status()
        raw = strip_thinking(resp.json()["message"]["content"].strip())
        start, end = raw.find("{"), raw.rfind("}") + 1
        parsed = json.loads(raw[start:end]) if start >= 0 else {}
        return {mapping[k]: v for k, v in parsed.items() if k in mapping}
    except Exception as e:
        print(f"  [batch warn] {e}")
        return {}


# ── Propagation ─────────────────────────────────────────────────────────────────

def forward_propagate(rows: list[dict], field: str, updates: dict[str, str]) -> dict[str, str]:
    """Forward-fill field values using LLM results combined with existing known values.
    Returns {chunk_id: value} for chunks filled by propagation only."""
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


def backward_propagate(rows: list[dict], field: str, all_updates: dict[str, str]) -> dict[str, str]:
    """Fill leading UNKNOWN chunks at the top of a document by propagating the
    first known value backward. Fills only the initial run of unknowns."""
    # Find first known value (from LLM results or DB)
    first_known = None
    for r in rows:
        cid = f"{r['source_file']}__p{r['page']}__c{r['chunk_index']}"
        val = all_updates.get(cid) or r.get(field, "UNKNOWN")
        if val not in ("UNKNOWN", ""):
            first_known = val
            break

    if first_known is None:
        return {}

    back_updates = {}
    for r in rows:
        cid = f"{r['source_file']}__p{r['page']}__c{r['chunk_index']}"
        val = all_updates.get(cid) or r.get(field, "UNKNOWN")
        if val not in ("UNKNOWN", ""):
            break  # reached the first chunk that has a value — stop
        if cid not in all_updates:
            back_updates[cid] = first_known
    return back_updates


# ── Core enrichment for one file ───────────────────────────────────────────────

def enrich_one_file(
    table,
    rows: list[dict],
    fields: list[str],
    apply: bool,
    use_all: bool,
) -> int:
    """Run LLM enrichment + propagation for one source file across the given fields.
    Returns number of chunks written."""
    source_file = rows[0]["source_file"]
    rows = sorted(rows, key=lambda r: (r["page"], r["chunk_index"]))
    total_written = 0

    for field in fields:
        targets = rows if use_all else [
            r for r in rows if r.get(field, "UNKNOWN") in ("UNKNOWN", "")
        ]
        print(f"\n  ── {field} — {len(targets)} chunks to enrich ──")
        if not targets:
            print("    Nothing to do.")
            continue

        # Phase 1: batch LLM calls
        llm_updates: dict[str, str] = {}
        batches = [targets[i:i + BATCH_SIZE] for i in range(0, len(targets), BATCH_SIZE)]
        for i, batch in enumerate(batches):
            print(f"    batch {i+1}/{len(batches)} ({len(batch)} chunks)...", end=" ", flush=True)
            result = batch_enrich(batch, field)
            resolved = sum(1 for v in result.values() if v not in ("UNKNOWN", ""))
            print(f"{resolved} resolved")
            llm_updates.update({k: v for k, v in result.items()
                                 if v not in ("UNKNOWN", "")})

        # Phase 2: forward propagation
        prop_fwd = forward_propagate(rows, field, llm_updates)
        all_updates = {**llm_updates, **prop_fwd}

        # Phase 3: backward propagation (fill leading unknowns)
        prop_bwd = backward_propagate(rows, field, all_updates)
        new_bwd = {k: v for k, v in prop_bwd.items() if k not in all_updates}
        all_updates.update(new_bwd)

        print(f"    LLM resolved:          {len(llm_updates)}")
        print(f"    Forward propagation:   {len(prop_fwd)}")
        print(f"    Backward propagation:  {len(new_bwd)}")
        print(f"    Total changes:         {len(all_updates)}")

        flag = "llm_corrected_section" if field == "section" else "llm_corrected_doc_page"
        for cid, new_val in sorted(all_updates.items()):
            row = next(
                (r for r in rows
                 if f"{r['source_file']}__p{r['page']}__c{r['chunk_index']}" == cid),
                None,
            )
            old_val = row.get(field, "UNKNOWN") if row else "?"
            src = "llm" if cid in llm_updates else ("bwd" if cid in new_bwd else "fwd")
            print(f"    [{src}] {cid}  {old_val!r} → {new_val!r}")
            if apply and row:
                table.update(
                    where=f"id = '{cid}'",
                    values={field: new_val, flag: True},
                )
                log_correction(
                    row["source_file"], row["page"], row["chunk_index"],
                    old_val, new_val, field=field,
                )
                total_written += 1

    return total_written


# ── Problem file detection ──────────────────────────────────────────────────────

_BOILERPLATE = {"table", "figure", "fig", "note", "section", "appendix", "exhibit",
                "chart", "diagram", "page", "header", "footer"}

def _looks_like_false_positive(val: str) -> bool:
    """Return True if a majority section value is likely a regex/parser artifact
    rather than a real section designator.

    Heuristics:
    - Purely numeric (digits, dots, dashes, spaces): "1.0", "3-2", "42"
    - Short (≤6 chars) with no letters beyond a trailing dot or dash
    - Known boilerplate heading words
    """
    v = val.strip()
    if not v or v == "UNKNOWN":
        return False
    # All-numeric (possibly with separators)
    if re.fullmatch(r"[\d][\d.\-\s]*", v):
        return True
    # Short value with only one word that is a boilerplate heading
    if v.lower() in _BOILERPLATE:
        return True
    # Very short (≤4 chars) — likely a page/figure number caught by mistake
    if len(v) <= 4:
        return True
    return False


def scan_problems(
    by_file: dict[str, list[dict]],
    unknown_threshold: int,
    majority_threshold: int,
) -> list[dict]:
    """Scan all files and return a list of problem file records (scans 'section' field).
    Each record: {source_file, chunks, unknown_pct, top_val, top_pct, reason, use_all}
    """
    field = "section"
    problems = []
    for source_file, rows in sorted(by_file.items()):
        total = len(rows)
        values = [r.get(field, "UNKNOWN") or "UNKNOWN" for r in rows]
        unknown_count = sum(1 for v in values if v == "UNKNOWN")
        unknown_pct = 100 * unknown_count // total if total else 0

        counter = Counter(v for v in values if v != "UNKNOWN")
        top_val, top_count = counter.most_common(1)[0] if counter else ("—", 0)
        top_pct = 100 * top_count // total if total else 0

        reasons = []
        use_all = False

        if unknown_pct > unknown_threshold:
            reasons.append(f"{unknown_pct}% UNKNOWN")

        if top_pct > majority_threshold and counter and _looks_like_false_positive(top_val):
            # How many distinct pages does the majority value span?
            top_pages = {r["page"] for r in rows if (r.get(field) or "UNKNOWN") == top_val}
            reasons.append(f"majority '{top_val}' on {top_pct}% of chunks ({len(top_pages)} pages) — looks like false positive")
            use_all = True  # existing values may be wrong — re-enrich all

        if reasons:
            problems.append({
                "source_file": source_file,
                "chunks":      total,
                "unknown_pct": unknown_pct,
                "top_val":     top_val,
                "top_pct":     top_pct,
                "reason":      "; ".join(reasons),
                "use_all":     use_all,
            })

    return problems


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source_file", nargs="?",
                        help="Exact source_file value as stored in DB (single-file mode)")
    parser.add_argument("--apply",        action="store_true", help="Write changes to DB")
    parser.add_argument("--all",          action="store_true", help="Re-enrich all chunks, not just UNKNOWN")
    parser.add_argument("--section-only", action="store_true", dest="section_only")
    parser.add_argument("--page-only",    action="store_true", dest="page_only")
    parser.add_argument("--scan",         action="store_true",
                        help="Scan all files for problems (bulk mode)")
    parser.add_argument("--unknown-threshold",  type=int, default=20, dest="unknown_threshold",
                        metavar="N", help="Flag files with >N%% UNKNOWN sections (default: 20)")
    parser.add_argument("--majority-threshold", type=int, default=60, dest="majority_threshold",
                        metavar="N", help="Flag files where top section >N%% of chunks (default: 60)")
    parser.add_argument("--yes", "-y",    action="store_true",
                        help="Skip confirmation prompt in scan mode")
    args = parser.parse_args()

    if not args.source_file and not args.scan:
        parser.error("Provide a source_file or use --scan for bulk mode")

    db    = lancedb.connect(str(VECTORDB_DIR))
    table = db.open_table("civil_engineering_codes")

    fields = []
    if not args.page_only:    fields.append("section")
    if not args.section_only: fields.append("doc_page")

    # ── Single-file mode ───────────────────────────────────────────────────────
    if args.source_file:
        rows = table.search().where(
            f"source_file = '{args.source_file.replace(chr(39), '')}'"
        ).limit(999999).to_list()

        if not rows:
            print(f"No chunks found for '{args.source_file}'")
            print("\nSearching for close matches...")
            all_files = sorted({r["source_file"]
                                 for r in table.search().select(["source_file"]).limit(999999).to_list()})
            term = args.source_file.lower()
            matches = [f for f in all_files if term in f.lower()]
            if matches:
                print("Possible matches:")
                for m in matches[:10]:
                    print(f"  {m}")
            else:
                print("No matches found. Available files:")
                for f in all_files[:30]:
                    print(f"  {f}")
            sys.exit(1)

        rows.sort(key=lambda r: (r["page"], r["chunk_index"]))
        print(f"Found {len(rows)} chunks for {args.source_file}")
        for field in fields:
            unknown = sum(1 for r in rows if r.get(field, "UNKNOWN") in ("UNKNOWN", ""))
            print(f"  UNKNOWN {field}: {unknown}/{len(rows)}")

        if not args.apply:
            print("\n[DRY RUN] Pass --apply to write changes.\n")

        n = enrich_one_file(table, rows, fields, args.apply, args.all)

        if args.apply and n:
            print(f"\nRebuilding FTS index...")
            _ensure_fts_index(table)
            print(f"Done. {n} chunks updated.")
        elif args.apply:
            print("\nNo changes written (all results were UNKNOWN).")
        else:
            print("\n[DRY RUN complete] Re-run with --apply to commit.")
        return

    # ── Bulk scan mode ─────────────────────────────────────────────────────────
    print("Loading all chunks for scan...")
    all_rows = table.search().limit(999999).to_list()
    print(f"Loaded {len(all_rows)} chunks across", end=" ")

    from collections import defaultdict
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_file[r["source_file"]].append(r)
    print(f"{len(by_file)} source files\n")

    problems = scan_problems(by_file, args.unknown_threshold, args.majority_threshold)

    if not problems:
        print(f"No problem files found "
              f"(unknown_threshold={args.unknown_threshold}%, "
              f"majority_threshold={args.majority_threshold}%)")
        return

    # Print problem table
    col = max(len(p["source_file"]) for p in problems)
    col = max(col, 20)
    print(f"{'source_file':<{col}}  {'chunks':>6}  {'unk%':>4}  {'top%':>4}  {'mode':<5}  reason")
    print("─" * (col + 40))
    for p in problems:
        mode = "--all" if p["use_all"] else "unk  "
        print(f"{p['source_file']:<{col}}  {p['chunks']:>6}  {p['unknown_pct']:>3}%  "
              f"{p['top_pct']:>3}%  {mode}  {p['reason']}")

    print(f"\n{len(problems)} problem file(s) flagged.")

    if not args.apply:
        print("\n[DRY RUN] Pass --apply to write changes, or add --yes to skip confirmation.")
        print("Preview of what would run:")
        for p in problems:
            mode = "--all" if p["use_all"] else ""
            print(f"  enrich_files.py '{p['source_file']}' --apply {mode}")
        return

    # Confirmation
    if not args.yes:
        confirm = input(f"\nEnrich {len(problems)} file(s)? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

    total_written = 0
    for i, p in enumerate(problems, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(problems)}] {p['source_file']}")
        print(f"  Reason: {p['reason']}")
        print(f"  Mode:   {'--all (re-enrich existing)' if p['use_all'] else 'UNKNOWN only'}")
        rows = by_file[p["source_file"]]
        n = enrich_one_file(table, rows, fields, apply=True, use_all=p["use_all"])
        total_written += n
        print(f"  Written: {n} chunks")

    if total_written:
        print(f"\n{'='*60}")
        print(f"Rebuilding FTS index...")
        _ensure_fts_index(table)
        print(f"Done. {total_written} total chunks updated across {len(problems)} files.")
    else:
        print(f"\nNo changes written (all LLM results were UNKNOWN).")


if __name__ == "__main__":
    main()
