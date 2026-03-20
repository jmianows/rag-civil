import re


def detect_section(text: str) -> str:
    section_patterns = [
        # WSDOT style: 1310.02(13)(b) — must have 4 digit prefix
        r'^([1-9]\d{3}\.\d{1,3}(?:\.\d{1,3})*(?:\([^\)]+\))*)',
        # OSHA CFR style: 1926.502(d)(1)(ii) — must have parentheticals
        r'^([1-9]\d{1,3}\.\d{1,4}(?:\([a-z0-9]+\))+)',
        # MUTCD style: 2E.40, 6N.01 — digit + letters + dot + digits
        r'^([1-9][A-Z]{1,2}\.\d{2,3})',
        # CFR/ADA style: 122.42, 703.2.5 — 3 digit prefix minimum
        r'^([1-9]\d{2}\.\d{1,3}(?:\.\d{1,3})*)',
        # USACE/multi-part: 8.3.1 — must have at least 3 parts
        r'^([1-9]\d{0,2}(?:\.\d{1,3}){2,})',
    ]

    lines = text.strip().splitlines()

    for line_num, line in enumerate(lines[:10]):
        line = line.strip()
        if not line:
            continue

        for pattern in section_patterns:
            match = re.match(pattern, line)
            if match:
                candidate = match.group(1).strip()

                # reject standalone years
                if re.match(r'^\d{4}$', candidate):
                    continue

                # reject phone numbers — any segment over 4 digits
                segments = re.split(r'[.\-]', candidate.split('(')[0])
                if any(len(s) > 4 for s in segments):
                    continue

                # reject RCW statute style — 3 segments all 2+ digits
                # without a 4-digit prefix
                dot_segments = candidate.split('(')[0].split('.')
                if len(dot_segments) == 3:
                    if all(len(s) >= 2 for s in dot_segments):
                        if len(dot_segments[0]) < 4:
                            continue

                # must be followed by space, end of line, or parenthetical
                end_pos = match.end()
                if end_pos < len(line) and line[end_pos] not in (' ', '\t', '(', '['):
                    continue

                # remainder check — must not start with lowercase or math
                remainder = line[end_pos:].strip()
                if remainder:
                    if remainder[0].islower():
                        continue
                    if re.match(r'^[\d\+\-\=\<\>\/\*\%]', remainder):
                        continue

                # position check — after line 2 only accept strong patterns
                if line_num > 2:
                    if not re.match(
                        r'^[1-9]\d{3}\.|.*\([^\)]+\)',
                        candidate
                    ):
                        continue

                if len(candidate) >= 3:
                    return candidate

    return "UNKNOWN"


def detect_doc_page(text: str, section: str) -> str:
    # Strategy 1 — check last non-empty line for a trailing page number
    lines = text.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        # check if the last token on the line is a page number
        # handles formats like: "1320-5", "Page 42", "42", "1320-5"
        page_match = re.search(
            r'(?:'
            r'(?:Page|PAGE)\s+(\d+)'    # "Page 42"
            r'|(\d{1,4}-\d{1,4})'       # "1320-5" dash format
            r'|(?<!\d)(\d{1,4})(?!\d)'  # standalone number "42"
            r')$',
            line
        )
        if page_match:
            candidate = next(g for g in page_match.groups() if g is not None)
            if candidate != section:
                return candidate
        break

    # Strategy 2 — fallback, search first 300 chars for dash format
    doc_page_match = re.search(
        r'(?<!\d)(\d{1,4}-\d{1,4})(?!\d)',
        text[:300]
    )
    if doc_page_match:
        candidate = doc_page_match.group(1)
        if candidate != section:
            return candidate

    return "UNKNOWN"


def propagate_metadata(rows: list[dict]) -> list[dict]:
    last_section = "UNKNOWN"
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