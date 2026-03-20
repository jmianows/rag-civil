import re


def detect_section(text: str) -> str:
    section_patterns = [
        r'^(\d{2,4}\.\d+(?:\.\d+)*(?:\([^\)]+\))*)',
        r'^(\d+\.\d+(?:\([a-z0-9]+\))+)',
        r'^(\d+(?:\.\d+){2,})',
        r'^((?:Section|SECTION)\s+\d+(?:\.\d+)*)',
        r'^(\d+\.\d+)',
    ]
    lines = text.strip().splitlines()
    for line in lines[:10]:
        line = line.strip()
        if not line:
            continue
        for pattern in section_patterns:
            match = re.match(pattern, line)
            if match:
                candidate = match.group(1).strip()
                if len(candidate) >= 3 and not re.match(r'^\d{4}$', candidate):
                    return candidate
    return "UNKNOWN"


def detect_doc_page(text: str, section: str) -> str:
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