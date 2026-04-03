import re
from typing import Optional
#this is for correctly tagging section numbers. It tries to find the top and proceed down and up the heirarchy as it proceeds through the text.

# Segment types in order of hierarchy
# Each represents one level of a section number
# e.g. 1310.02(13)(b) = [INT(1310), INT(02), PAREN_INT(13), PAREN_ALPHA(b)]

SEGMENT_PATTERNS = [
    # parenthetical roman numeral: (iv), (xiii)
    ("PAREN_ROMAN",  re.compile(r'^\(([ivxlcdm]+)\)', re.IGNORECASE)),
    # parenthetical alpha: (a), (b), (A), (B)
    ("PAREN_ALPHA",  re.compile(r'^\(([a-zA-Z])\)')),
    # parenthetical integer: (1), (13)
    ("PAREN_INT",    re.compile(r'^\((\d+)\)')),
    # bracket roman: [iv]
    ("BRACK_ROMAN",  re.compile(r'^\[([ivxlcdm]+)\]', re.IGNORECASE)),
    # bracket alpha: [a], [A]
    ("BRACK_ALPHA",  re.compile(r'^\[([a-zA-Z])\]')),
    # bracket integer: [1]
    ("BRACK_INT",    re.compile(r'^\[(\d+)\]')),
    # dot-separated integer: .02, .3
    ("DOT_INT",      re.compile(r'^\.(\d+)')),
    # dot-separated alpha: .A, .B
    ("DOT_ALPHA",    re.compile(r'^\.([a-zA-Z]+)')),
    # hyphen-separated integer: -02, -3
    ("HYPHEN_INT",   re.compile(r'^-(\d+)')),
    # leading integer only — no LEAD_ALPHA
    ("LEAD_INT",     re.compile(r'^(\d+)')),
]

ROMAN_VALUES = {
    'i': 1, 'v': 5, 'x': 10, 'l': 50,
    'c': 100, 'd': 500, 'm': 1000
}


def roman_to_int(s: str) -> int:
    s = s.lower()
    total = 0
    prev = 0
    for ch in reversed(s):
        val = ROMAN_VALUES.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total


def segment_to_int(seg_type: str, value: str) -> int:
    if 'ROMAN' in seg_type:
        return roman_to_int(value)
    if 'ALPHA' in seg_type:
        # convert single letter to int: a=1, b=2, A=1
        return ord(value.lower()[0]) - ord('a') + 1
    try:
        return int(value)
    except ValueError:
        return 0


def parse_section_number(text: str) -> Optional[list[dict]]:
    text = text.strip()
    if not text:
        return None

    segments = []
    pos = 0

    while pos < len(text):
        matched = False
        remaining = text[pos:]

        for seg_type, pattern in SEGMENT_PATTERNS:
            # skip LEAD patterns after first segment
            if seg_type.startswith('LEAD') and segments:
                continue

            match = pattern.match(remaining)
            if match:
                value = match.group(1)
                segments.append({
                    'type':     seg_type,
                    'value':    value,
                    'int_val':  segment_to_int(seg_type, value),
                    'raw':      match.group(0),
                })
                pos += len(match.group(0))
                matched = True
                break

        if not matched:
            break

    if not segments:
        return None

# reject phone numbers — any integer segment over 9999
    for seg in segments:
        if 'INT' in seg['type'] and seg['int_val'] > 9999:
            return None
    # must have consumed a meaningful portion of the token
    consumed = text[:pos]
    if len(consumed) < 2:
        return None
    return segments


def segments_to_string(segments: list[dict]) -> str:
    return ''.join(s['raw'] for s in segments)


def is_valid_advance(
    current: Optional[list[dict]],
    candidate: list[dict],
    max_jump: int = 20
) -> bool:
    if current is None:
        # no context yet — require stronger evidence for acceptance
        # two-segment numbers with small first segment need 3+ chars total
        candidate_str = segments_to_string(candidate)
        if len(candidate) == 2:
            first_val = candidate[0]['int_val']
            # small first segment (under 100) two-part numbers
            # require the second segment to have parentheticals
            # or be clearly a subsection (not a decimal)
            if first_val < 100:
                has_paren = any(
                    'PAREN' in s['type'] or 'BRACK' in s['type']
                    for s in candidate
                )
                has_hyphen = any(s['type'] == 'HYPHEN_INT' for s in candidate)
                if not has_paren and not has_hyphen:
                    return False
        return True

    cur_len = len(current)
    can_len = len(candidate)

    # same depth
    if can_len == cur_len:
        for i in range(cur_len - 1):
            if i >= len(candidate):
                return False
            if current[i]['value'].lower() != candidate[i]['value'].lower():
                if i == cur_len - 2:
                    cur_val = current[i]['int_val']
                    can_val = candidate[i]['int_val']
                    if 0 < can_val - cur_val <= max_jump:
                        return True
                return False
        cur_last = current[-1]['int_val']
        can_last = candidate[-1]['int_val']
        return 0 < can_last - cur_last <= max_jump

    # going deeper
    if can_len > cur_len:
        for i in range(cur_len):
            if current[i]['value'].lower() != candidate[i]['value'].lower():
                return False
        new_seg = candidate[cur_len]
        return new_seg['int_val'] <= 5

    # going shallower
    if can_len < cur_len:
        for i in range(can_len - 1):
            if i >= len(current):
                return False
            if current[i]['value'].lower() != candidate[i]['value'].lower():
                return False
        cur_at_level = current[can_len - 1]['int_val']
        can_last = candidate[-1]['int_val']
        return 0 < can_last - cur_at_level <= max_jump

    return False


def _validate_segments_common(segments: list[dict], line: str) -> bool:
    """Shared validation for both strict and relaxed section number paths.

    Checks applied by both extract_section_candidate and _extract_section_numeric_relaxed:
      - no leading zeros on first segment
      - year rejection (unless CFR-style with parentheticals/multiple dots/large dot)
      - first segment > 8000
      - any integer segment > 9999 (phone numbers)
      - any integer segment == 0 (not a real subsection)
      - USACE document number pattern (lead + hyphen + 4-digit hyphen)
      - remainder is not a math operator or bare number

    Returns True if segments pass all common checks, False if they should be rejected.
    """
    first_val = segments[0]['value']
    first_int = segments[0]['int_val']

    if first_val.startswith('0'):
        return False

    if re.match(r'^(19|20)\d{2}$', first_val):
        has_paren = any('PAREN' in s['type'] or 'BRACK' in s['type'] for s in segments)
        has_multiple_dot = sum(1 for s in segments if s['type'] == 'DOT_INT') > 1
        has_large_dot = any(s['type'] == 'DOT_INT' and s['int_val'] >= 100 for s in segments)
        if not has_paren and not has_multiple_dot and not has_large_dot:
            return False

    if first_int > 8000:
        return False

    for seg in segments:
        if 'INT' in seg['type'] and seg['int_val'] > 9999:
            return False

    for seg in segments:
        if 'INT' in seg['type'] and seg['int_val'] == 0:
            return False

    if (len(segments) == 3
            and segments[1]['type'] == 'HYPHEN_INT'
            and segments[2]['type'] == 'HYPHEN_INT'
            and len(segments[2]['value']) == 4):
        return False

    consumed = segments_to_string(segments)
    remainder = line[len(consumed):].strip()
    if remainder:
        if re.match(r'^[\+\=\<\>\/\*\%]', remainder):
            return False
        if re.match(r'^\d+\.?\d*$', remainder):
            return False

    return True


def extract_section_candidate(line: str) -> Optional[list[dict]]:
    line = line.strip()
    if not line:
        return None

    segments = parse_section_number(line)
    if not segments:
        return None

    # must have at least 2 segments
    if len(segments) < 2:
        return None

    # first segment must be LEAD_INT
    if segments[0]['type'] != 'LEAD_INT':
        return None

    if not _validate_segments_common(segments, line):
        return None

    first_val = segments[0]['value']
    first_int = segments[0]['int_val']

    # reject MUTCD-style page numbers: XX.XXX (2-digit lead, 3-digit dot second)
    if (len(segments) == 2
            and segments[1]['type'] == 'DOT_INT'
            and len(first_val) == 2
            and len(segments[1]['value']) == 3):
        return None

    # single digit lead requires 3+ segments (e.g. 4.3.1)
    if len(first_val) < 2 and len(segments) < 3:
        return None

    #MAYBE TOO SPECIFIC TO BE USEFUL LONG TERM BUT ITS HERE
    # reject two-segment numbers where second segment is suspiciously large
    # real subsection numbers at dot level rarely exceed 50
    # e.g. 126.75, 39.37 — likely coordinates or measurements
    # exception: CFR-style where first segment is 3+ digits (122.44 is borderline
    # but 1926.502 has parentheticals so won't reach here as 2-segment)
    if (len(segments) == 2
            and segments[1]['type'] == 'DOT_INT'
            and segments[1]['int_val'] > 50
            and first_int < 200):
        return None
    
    #MAYBE TOO SPECIFIC TO BE USEFUL LONG TERM BUT ITS HERE
    # reject two-segment dot numbers with 2-digit lead AND 2-digit second
    # and no parentheticals — e.g. 39.37, 18.05
    # legitimate 2-digit lead sections either have parentheticals (5.5(a))
    # or have a single-digit second segment (12.4, 17.3)
    if (len(segments) == 2
            and segments[1]['type'] == 'DOT_INT'
            and len(first_val) == 2
            and len(segments[1]['value']) >= 2
            and not any('PAREN' in s['type'] or 'BRACK' in s['type']
                       for s in segments)):
        return None

    consumed = segments_to_string(segments)
    remainder = line[len(consumed):].strip()
    if remainder and remainder[0].islower():
        return None

    return segments


# ── Heading-path section extraction ───────────────────────────────────────────

_CHAPTER_RE = re.compile(
    r'^(?:Chapter|CHAPTER|Section|SECTION|Part|PART|Article|ARTICLE)\s+(\d+\w*)',
    re.IGNORECASE
)


def _extract_section_numeric_relaxed(line: str) -> Optional[str]:
    """
    Attempt numeric section extraction with rules relaxed for confirmed heading blocks.

    Applies all common checks (see _validate_segments_common) but removes:
      - single-digit-lead 3-segment requirement
      - dot-second > 50 rejection
      - 2-digit lead + 2-digit dot rejection
      - MUTCD page number rejection
      - lowercase remainder rejection (title follows the section number)
    """
    line = line.strip()
    if not line:
        return None

    segments = parse_section_number(line)
    if not segments:
        return None

    if segments[0]['type'] != 'LEAD_INT':
        return None

    if not _validate_segments_common(segments, line):
        return None

    return segments_to_string(segments)


class DocumentSectionTracker:
    """Stateful section number tracker — advances through a document in order."""

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


def extract_section_from_heading(text: str) -> str:
    """
    Extract a section label from a confirmed heading block.

    Three-stage approach:
      1. Relaxed numeric extraction — accepts single-digit leads with 2 segments,
         X-Y hyphen style, and other patterns rejected in body text
      2. Chapter/Section/Part keyword pattern
      3. Heading text itself (truncated) — always returns something useful

    Never returns 'UNKNOWN'. Heading blocks are authoritative structural markers
    so even a text label is more useful than no label.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Stage 1 — relaxed numeric on each line
    for line in lines[:5]:
        result = _extract_section_numeric_relaxed(line)
        if result:
            return result

    # Stage 2 — Chapter/Section/Part keyword
    for line in lines[:5]:
        m = _CHAPTER_RE.match(line)
        if m:
            # Return the full matched prefix e.g. "Chapter 5"
            return line[:m.end()].strip()

    # Stage 3 — heading text fallback (strip trailing punctuation, cap at 50 chars)
    if lines:
        label = lines[0][:50].rstrip('.:;,- ')
        if label:
            return label

    return "UNKNOWN"