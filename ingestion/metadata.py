import re
from pathlib import Path
from section_parser import (
    extract_section_candidate,
    segments_to_string,
    is_valid_advance,
)


class DocumentSectionTracker:
    def __init__(self):
        self.current_segments = None
        self.current_string   = "UNKNOWN"

    def process_chunk(self, text: str) -> str:
        lines = text.strip().splitlines()

        for line_num, line in enumerate(lines[:10]):
            candidate = extract_section_candidate(line)
            if candidate is None:
                continue

            candidate_str = segments_to_string(candidate)

            if is_valid_advance(self.current_segments, candidate):
                self.current_segments = candidate
                self.current_string   = candidate_str
                return candidate_str

            # if no valid advance found from context
            # accept if it appears in first 2 lines (strong signal)
            if line_num < 2:
                self.current_segments = candidate
                self.current_string   = candidate_str
                return candidate_str

        return self.current_string


def detect_section(text: str) -> str:
    # stateless fallback for single chunk use
    lines = text.strip().splitlines()
    for line in lines[:10]:
        candidate = extract_section_candidate(line)
        if candidate:
            return segments_to_string(candidate)
    return "UNKNOWN"


def detect_doc_page(text: str, section: str) -> str:
    lines = text.strip().splitlines()
    for line in reversed([l.strip() for l in lines if l.strip()][-5:]):
        page_match = re.search(
            r'(?:'
            r'(?:Page|PAGE)\s+(\d+)'
            r'|(\d{1,4}-\d{1,4})'
            r'|(?<!\d)(\d{1,4})(?!\d)'
            r')$',
            line
        )
        if page_match:
            candidate = next(g for g in page_match.groups() if g is not None)
            if candidate != section:
                if not re.match(r'^\d{4}$', candidate):
                    raw_num = candidate.split('-')[-1]
                    if raw_num.isdigit() and int(raw_num) < 2000:
                        return candidate
        break

    fallback = re.search(r'(?<!\d)(\d{1,4}-\d{1,4})(?!\d)', text[:500])
    if fallback:
        return fallback.group(1)

    return "UNKNOWN"


def propagate_metadata(rows: list[dict]) -> list[dict]:
    last_section  = "UNKNOWN"
    last_doc_page = "UNKNOWN"

    for row in rows:
        if row["section"] != "UNKNOWN":
            last_section = row["section"]
        else:
            row["section"] = last_section

        if row["doc_page"] != "UNKNOWN":
            last_doc_page = row["doc_page"]
        else:
            row["doc_page"] = last_doc_page

    return rows