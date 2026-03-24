"""
Prompt regression test suite for rag-civil.

Usage:
    # Full 100-prompt run, output to default paths
    .venv/bin/python tests/run_prompts.py

    # Run specific prompts by number
    .venv/bin/python tests/run_prompts.py --only 3 14 22 39

    # Custom output prefix
    .venv/bin/python tests/run_prompts.py --out /tmp/run9

Outputs:
    <prefix>.log  — full human-readable transcript
    <prefix>.json — structured summary (read this for quick analysis)

JSON schema per prompt:
    {
      "n": int,
      "prompt": str,
      "status": "OK" | "FAIL",
      "spurious_fail": bool,   # FAIL emitted despite [[SRC_N]] citations
      "time_s": float,
      "sources": [{"source_file", "agency", "section", "page"}, ...],
      "response_preview": str  # first 300 chars of response
    }
"""

import sys, time, json, re, argparse, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.query_engine import query

# ── Prompt definitions ────────────────────────────────────────────────────────

PROMPTS = {
    1:  "What is the minimum lane width for a rural arterial highway according to WSDOT?",
    2:  "What is the maximum cross-slope for a pedestrian access route per PROWAG?",
    3:  "What does OSHA require to prevent cave-ins in trenches?",
    4:  "What is the minimum concrete cover for reinforcing steel in a bridge pier column exposed to saltwater?",
    5:  "What is the stopping sight distance for a 45 mph design speed?",
    6:  "What are the required elements of a Stormwater Pollution Prevention Plan (SWPPP)?",
    7:  "What is the minimum headwall setback distance for a culvert outlet?",
    8:  "What does the MUTCD require for retroreflectivity of regulatory signs?",
    9:  "What are WSDOT requirements for painting structural steel on highway bridges?",
    10: "What compaction requirements does USACE specify for highway embankment fill?",
    11: "What is the minimum compressive strength of concrete for bridge deck construction?",
    12: "What is the maximum allowable slope ratio for a temporary cut in Type B soil?",
    13: "What foundation bearing capacity does AASHTO recommend for spread footings on sand?",
    14: "What are OSHA requirements for worker protection in excavations deeper than 5 feet?",
    15: "What does OSHA require for portable ladder angle and extension above landing?",
    16: "What PPE does OSHA require for workers in roadway construction zones?",
    17: "What are OSHA requirements for entering a permit-required confined space?",
    18: "What does the MUTCD specify for mounting height of road signs above pavement?",
    19: "What does the MUTCD require for speed limit sign placement near school zones?",
    20: "What freeboard does USACE recommend for a levee protecting an urban area?",
    21: "What minimum factor of safety for slope stability does USACE recommend for embankments?",
    22: "What are the general permit requirements for stormwater discharge from construction sites?",
    23: "What is the definition of an engineer of record in WSDOT contracts?",
    24: "What are the required elements of a traffic control plan for a lane closure?",
    25: "What is the seismic design category for a highway bridge in a high-seismic zone?",
    26: "What ASME code governs pressure vessel inspection intervals for industrial boilers?",
    27: "What does OSHA 29 CFR 1926.502 require for fall protection systems?",
    28: "What does the MUTCD require for Type III barricades at road closures?",
    29: "What are the standard steps for installing a precast concrete box culvert?",
    30: "What are WSDOT requirements for a Traffic Control Supervisor on construction projects?",
    31: "What are OSHA requirements for scaffold platform width and guardrails under 29 CFR 1926.451?",
    32: "What does OSHA require for concrete and masonry construction formwork under 29 CFR 1926.703?",
    33: "What are OSHA requirements for crane operator qualifications under 29 CFR 1926.1427?",
    34: "What does OSHA specify for electrical safety grounding on construction sites?",
    35: "What are OSHA requirements for stairways and ladders in construction under 29 CFR 1926.1052?",
    36: "What does OSHA require for personal protective equipment eye and face protection under 29 CFR 1926.102?",
    37: "What are OSHA requirements for demolition operations under 29 CFR 1926 Subpart T?",
    38: "What does OSHA specify for fire prevention plans on construction sites?",
    39: "What are OSHA requirements for steel erection bolted connections under 29 CFR 1926.755?",
    40: "What does OSHA 29 CFR 1910 require for lockout/tagout energy control procedures?",
    41: "What are OSHA requirements for walking-working surfaces and floor openings?",
    42: "What does OSHA specify for machine guarding requirements in general industry?",
    43: "What does WSDOT specify for horizontal curve design and superelevation on rural highways?",
    44: "What are WSDOT requirements for sidewalk width and design adjacent to curbs?",
    45: "What does WSDOT specify for the design of roundabouts?",
    46: "What are WSDOT requirements for shared use path width and design?",
    47: "What does WSDOT specify for guardrail end treatment and terminal design?",
    48: "What are WSDOT drainage ditch design criteria for rural highways?",
    49: "What does WSDOT require for accessible pedestrian facility design in new construction?",
    50: "What are WSDOT requirements for vertical clearance on highway structures?",
    51: "What does WSDOT specify for live load deflection limits for bridge girders?",
    52: "What are WSDOT requirements for bridge deck drainage scuppers and overhangs?",
    53: "What does WSDOT specify for minimum bearing seat width for bridges?",
    54: "What are WSDOT requirements for seismic design of bridge substructures?",
    55: "What does WSDOT specify for fatigue design of steel bridge details?",
    56: "What are WSDOT requirements for pile foundation design under bridges?",
    57: "What does WSDOT specify for concrete curing time and temperature requirements in the field?",
    58: "What are WSDOT requirements for hot mix asphalt compaction and density testing?",
    59: "What does WSDOT specify for aggregate base course material requirements?",
    60: "What are WSDOT standard specification requirements for temporary erosion and sediment control?",
    61: "What does WSDOT specify for structural steel welding inspection requirements?",
    62: "What does WSDOT specify for minimum embedment depth requirements for driven piles?",
    63: "What are WSDOT requirements for retaining wall drainage design?",
    64: "What does WSDOT specify for settlement limits for bridge approach embankments?",
    65: "What does MUTCD specify for the color and minimum size of diamond-shaped warning signs?",
    66: "What are MUTCD requirements for lane closure taper length on expressways?",
    67: "What does MUTCD specify for pedestrian signal walk interval timing?",
    68: "What are MUTCD requirements for pavement marking colors and centerline design?",
    69: "What does MUTCD specify for work zone speed limit reduction criteria?",
    70: "What are MUTCD requirements for flagger hand signal procedures?",
    71: "What does MUTCD specify for minimum green time requirements at signalized intersections?",
    72: "What are MUTCD requirements for school zone sign placement and operation?",
    73: "What are the turbidity and pH benchmark monitoring values in the EPA CGP?",
    74: "What does the EPA CGP require for inspection frequency on active construction sites?",
    75: "What are the EPA CGP requirements for stabilization of disturbed areas?",
    76: "What does 40 CFR Part 122 require for NPDES permit application content?",
    77: "What does the EPA CGP require for sites near impaired water bodies?",
    78: "What does USACE EM 1110-2-1913 specify for levee freeboard for agricultural levees?",
    79: "What are USACE requirements for riprap sizing on levee side slopes?",
    80: "What does USACE EM 1110-2-1902 specify for the ordinary method of slices for slope stability?",
    81: "What are USACE requirements for instrumentation in new embankment dams?",
    82: "What does PROWAG require for detectable warning surface size and color?",
    83: "What are ADA requirements for curb ramp flare slope and width?",
    84: "What does PROWAG specify for accessible pedestrian signal features?",
    85: "What are ADA requirements for accessible parking space dimensions and slope?",
    86: "What does the FHWA fp-24 specification require for structural concrete mix design?",
    87: "What are the fp-24 requirements for reinforcing steel bar testing and certification?",
    88: "What does fp-24 specify for asphalt binder performance grade selection criteria?",
    89: "What are the requirements for temporary traffic signals at work zones?",
    90: "What does WSDOT specify for noise analysis and mitigation in highway design?",
    91: "What are the requirements for dewatering discharge from construction excavations?",
    92: "What does OSHA require for hazard communication and safety data sheets on construction sites?",
    93: "What are the inspection requirements for temporary support systems during bridge construction?",
    94: "What does WSDOT specify for the design of wildlife crossings or animal passage structures?",
    95: "What are requirements for high-visibility safety apparel for workers near traffic?",
    96: "What does the International Building Code require for occupancy classification of warehouses?",
    97: "What are NFPA 13 requirements for fire sprinkler system design in industrial buildings?",
    98: "What does ASCE 7 specify for wind pressure design on highway sign structures?",
    99: "What are the International Residential Code requirements for residential foundation frost depth?",
    100: "What does the ASTM standard specify for the tensile strength testing of structural bolts?",
}

_SRC_RE = re.compile(r'\[\[SRC_\d+\]\]')


def run(prompt_nums: list[int], log_path: Path, json_path: Path):
    sep = "=" * 70
    results = []
    times = {}

    with log_path.open("w") as log:
        def emit(line=""):
            print(line, flush=True)
            log.write(line + "\n")

        run_meta = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompts_run": prompt_nums,
        }
        emit(f"Run started: {run_meta['timestamp']}  |  {len(prompt_nums)} prompts")

        for n in prompt_nums:
            p = PROMPTS[n]
            emit(f"\n{sep}\nPROMPT {n}: {p}\n{sep}")
            t0 = time.time()
            r = query(p)
            elapsed = round(time.time() - t0, 1)
            times[n] = elapsed

            resp = r["response"]
            has_fail = "[[FAIL]]" in resp
            has_src  = bool(_SRC_RE.search(resp))
            spurious = has_fail and has_src
            status   = "FAIL" if has_fail else "OK"

            emit("RESPONSE:\n" + resp)
            emit(f"\nSOURCES ({len(r['chunks'])}):")
            sources = []
            for j, c in enumerate(r["chunks"], 1):
                emit(f"  [{j}] {c['source_file']} | {c['agency']} | §{c['section']} | p{c['page']}")
                sources.append({
                    "source_file": c["source_file"],
                    "agency":      c["agency"],
                    "section":     c["section"],
                    "page":        c["page"],
                })
            emit(f"\n{status}{'(spurious)' if spurious else ''} | {elapsed}s")

            results.append({
                "n":               n,
                "prompt":          p,
                "status":          status,
                "spurious_fail":   spurious,
                "time_s":          elapsed,
                "sources":         sources,
                "response_preview": resp[:300].replace("\n", " "),
            })

        # ── Summary ──────────────────────────────────────────────────────────
        mean_t    = round(sum(times.values()) / len(times), 1)
        fails     = [r for r in results if r["status"] == "FAIL"]
        spurious  = [r for r in results if r["spurious_fail"]]
        slowest   = sorted(results, key=lambda x: x["time_s"], reverse=True)[:5]

        emit("\n\n=== TIMING SUMMARY ===")
        for res in sorted(results, key=lambda x: x["n"]):
            emit(f"  P{res['n']:>3}: {res['status']:4} | {res['time_s']:.1f}s")
        emit(f"  MEAN: {mean_t}s")
        emit(f"  FAILs ({len(fails)}): {[r['n'] for r in fails]}")
        emit(f"  Spurious FAILs: {[r['n'] for r in spurious]}")
        emit(f"  Slowest 5: {[(r['n'], r['time_s']) for r in slowest]}")

        summary = {
            **run_meta,
            "mean_time_s":   mean_t,
            "total_time_s":  round(sum(times.values()), 1),
            "fail_count":    len(fails),
            "fail_prompts":  [r["n"] for r in fails],
            "spurious_fails":[r["n"] for r in spurious],
            "slowest_5":     [(r["n"], r["time_s"]) for r in slowest],
            "results":       results,
        }
        json_path.write_text(json.dumps(summary, indent=2))
        emit(f"\nJSON summary → {json_path}")
        emit(f"Full log     → {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", type=int, default=None,
                        help="Run only these prompt numbers")
    parser.add_argument("--out", default="/tmp/rag_run",
                        help="Output path prefix (default: /tmp/rag_run)")
    args = parser.parse_args()

    nums = sorted(args.only if args.only else PROMPTS.keys())
    prefix = Path(args.out)
    run(nums, prefix.with_suffix(".log"), prefix.with_suffix(".json"))
