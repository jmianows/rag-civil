import re


def detect_doc_page(text: str, section: str) -> str:
    """Scan full page text for a printed page number.
    Used by retag.py which works on concatenated chunk text, not isolated header/footer blocks.
    """
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
        break  # only check the last matching line

    fallback = re.search(r'(?<!\d)(\d{1,4}-\d{1,4})(?!\d)', text[:500])
    if fallback:
        return fallback.group(1)

    return "UNKNOWN"
