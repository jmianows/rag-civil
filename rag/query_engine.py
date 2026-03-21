import ollama
import lancedb
import re
from pathlib import Path
from dataclasses import dataclass

VECTORDB_DIR  = Path("/home/justin/rag-civil/vectordb")
EMBED_MODEL   = "mxbai-embed-large"
LLM_MODEL = "qwen3:4b-instruct"
N_RESULTS     = 5
CONTEXT_WINDOW_BEFORE = 1
CONTEXT_WINDOW_AFTER  = 2
MAX_CHUNK_CHARS = 1000 
#This sets up the configuration and a clean dataclass to carry retrieved chunk data through the pipeline.
@dataclass
class RetrievedChunk:
    text:                  str
    source_file:           str
    agency:                str
    jurisdiction:          str
    state:                 str
    section:               str
    doc_page:              str
    page:                  int
    chunk_index:           int
    distance:              float
    llm_corrected_section: bool = False


def get_db_table() -> lancedb.table.LanceTable:
    db = lancedb.connect(str(VECTORDB_DIR))
    return db.open_table("civil_engineering_codes")


def embed_query(query: str) -> list[float]:
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=f"Represent this sentence for searching relevant passages: {query}"
    )
    return response["embedding"]


def retrieve_chunks(
    query: str,
    table: lancedb.table.LanceTable,
    n_results: int = N_RESULTS,
    filter_agency: str = None,
    filter_jurisdiction: str = None,
    filter_state: str = None,
) -> list[RetrievedChunk]:

    embedding = embed_query(query)
    search = table.search(embedding)

    # build metadata filter if any provided
    filters = []
    if filter_agency:
        filters.append(f"agency = '{filter_agency}'")
    if filter_jurisdiction:
        filters.append(f"jurisdiction = '{filter_jurisdiction}'")
    if filter_state:
        filters.append(f"state = '{filter_state}'")

    if filters:
        search = search.where(" AND ".join(filters))

    results = search.limit(n_results).to_list()

    chunks = []
    for r in results:
        chunks.append(RetrievedChunk(
            text=r["text"],
            source_file=r["source_file"],
            agency=r["agency"],
            jurisdiction=r["jurisdiction"],
            state=r.get("state", ""),
            section=r["section"],
            llm_corrected_section=r.get("llm_corrected_section", False),
            doc_page=r.get("doc_page", "UNKNOWN"),
            page=r["page"],
            chunk_index=r["chunk_index"],
            distance=r.get("_distance", 0.0),
        ))

    return chunks

def expand_context(
    chunk: RetrievedChunk,
    table: lancedb.table.LanceTable,
    window_before: int = 1,
    window_after: int = 2,
) -> str:
    source_file  = chunk.source_file
    page         = chunk.page
    chunk_index  = chunk.chunk_index

    neighbors = []

    # fetch chunks before
    for offset in range(-window_before, 0):
        neighbor_id = f"{source_file}__p{page}__c{chunk_index + offset}"
        try:
            result = table.search() \
                .where(f"id = '{neighbor_id}'") \
                .limit(1) \
                .to_list()
            if result:
                neighbors.append((chunk_index + offset, result[0]["text"]))
        except Exception:
            continue

    # always include the chunk itself
    neighbors.append((chunk_index, chunk.text))

    # fetch chunks after — look across page boundaries too
    for offset in range(1, window_after + 1):
        neighbor_id = f"{source_file}__p{page}__c{chunk_index + offset}"
        try:
            result = table.search() \
                .where(f"id = '{neighbor_id}'") \
                .limit(1) \
                .to_list()
            if result:
                neighbors.append((chunk_index + offset, result[0]["text"]))
                continue
        except Exception:
            pass

        # if not found on same page try next page chunk 0
        if offset == 1:
            next_page_id = f"{source_file}__p{page + 1}__c0"
            try:
                result = table.search() \
                    .where(f"id = '{next_page_id}'") \
                    .limit(1) \
                    .to_list()
                if result:
                    neighbors.append((page + 1, result[0]["text"]))
            except Exception:
                pass

    neighbors.sort(key=lambda x: x[0])
    texts = [t for _, t in neighbors]

    if len(texts) == 1:
        return texts[0]

    combined = texts[0]
    for i in range(1, len(texts)):
        combined = remove_overlap(combined, texts[i])

    return combined

def remove_overlap(text_a: str, text_b: str, min_overlap: int = 20) -> str:
    max_check = min(len(text_a), len(text_b), 200)

    for length in range(max_check, min_overlap - 1, -1):
        suffix = text_a[-length:]
        if text_b.startswith(suffix):
            return text_a + text_b[length:]

    return text_a + " " + text_b

def group_chunks(
    chunks: list[RetrievedChunk],
    table: lancedb.table.LanceTable,
) -> list[dict]:
    # expand each chunk with neighboring context
    expanded = []
    for chunk in chunks:
        text = expand_context(chunk, table)
        expanded.append({
            "chunk":    chunk,
            "text":     text,
        })

    # group chunks that are from the same source file and close in section
    groups = []
    used = set()

    for i, item in enumerate(expanded):
        if i in used:
            continue

        group = [item]
        used.add(i)

        for j, other in enumerate(expanded):
            if j in used:
                continue

            same_file = (
                item["chunk"].source_file == other["chunk"].source_file
            )
            same_section = (
                item["chunk"].section == other["chunk"].section
                and item["chunk"].section != "UNKNOWN"
            )

            if same_section:
                group.append(other)
                used.add(j)

            close_pages = (
                abs(item["chunk"].page - other["chunk"].page) <= 2
            )

            if same_file and (same_section or close_pages):
                group.append(other)
                used.add(j)

        groups.append(group)

    return groups
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
### THE SYSTEM PROMPT IS HERE SUPER IMPORTANT!!!!!!!!!! 
SYSTEM_PROMPT = """You are a civil engineering code and standards lookup tool.

Your job is to read the retrieved code sections and present their requirements accurately.

Rules:
- Present only information from the retrieved sections. Never add outside knowledge.
- Copy requirement language verbatim. Do not paraphrase or reword. Truncate with "..." if too long.
- Preserve exact wording of shall, should, and may — these define mandatory vs guidance vs optional.
- Present only directly applicable requirements. Omit background, context, and procedural setup.
- Prioritize shall statements. Include should and may only if no shall statements exist or the query asks for guidance.
- Show most local jurisdiction first, expanding to federal below.
- Present each standard separately with its own citation block.
- Present ALL retrieved sections that contain relevant information, each with its own citation block.
- Do not stop after the first relevant section.
- If ALL retrieved sections contain no relevant information say exactly: "The provided standards do not address this query. Think I should? Request manuals to add using the bottom right!"
- Only use the "not addressed" message if NO retrieved sections are relevant. Do not append it after presenting relevant results.
- Never provide engineering advice, opinions, or recommendations.
- Even if two sections cover similar topics, present both separately if they contain distinct requirements.

Format:
- Bullet points per requirement.
- Citation block at the end of each passage:

  [SOURCE]
  Document: <source_file> <section>, page <doc_page>

- No preamble, summaries, or conclusions.
- Do not restate the question."""


def format_context(groups: list[dict]) -> str:
    context_blocks = []
    
    for group in groups:
        # combine all text in the group
        combined_texts = []
        for item in group:
            combined_texts.append(item["text"])

        # deduplicate overlapping text between group items
        combined = combined_texts[0]
        for i in range(1, len(combined_texts)):
            combined = remove_overlap(combined, combined_texts[i])

        # use metadata from first chunk in group
        chunk = group[0]["chunk"]

        section_display = chunk.section
        if chunk.llm_corrected_section:
            section_display = f"{chunk.section} [llm corrected]"

        if len(combined) > MAX_CHUNK_CHARS:
            combined = combined[:MAX_CHUNK_CHARS] + "..."

        block = f"""--- RETRIEVED SECTION ---
{combined}

[SOURCE]
Document: {chunk.source_file} {section_display}, page {chunk.doc_page}
PDF Page: {chunk.page}
---"""

        context_blocks.append(block)

    return "\n\n".join(context_blocks)

def strip_thinking(text: str) -> str:
    # remove any thinking blocks Qwen3 emits despite think=False
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

def generate_response(query: str, context: str) -> str:
    import requests
    import json

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nRetrieved sections:\n\n{context}",
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature":    0.1,
            "num_ctx":        4096,
            "num_predict":    1024,
            "repeat_penalty": 1.1,
            "num_gpu":        999,
        }
    }

    response = requests.post(
        "http://localhost:11434/api/chat",
        json=payload
    )

    data = response.json()
    content = data["message"]["content"].strip()
    return strip_thinking(content)
#if the parser finds a section it will go back, correct it, and then flag it did so for us.
def correct_section(
    source_file: str,
    page: int,
    chunk_index: int,
    new_section: str,
) -> bool:
    from ingestion.retag import log_correction
    table = get_db_table()
    chunk_id = f"{source_file}__p{page}__c{chunk_index}"

    try:
        result = table.search() \
            .where(f"id = '{chunk_id}'") \
            .limit(1) \
            .to_list()
        old_section = result[0]["section"] if result else "UNKNOWN"
    except Exception:
        old_section = "UNKNOWN"

    try:
        table.update(
            where=f"id = '{chunk_id}'",
            values={
                "section":               new_section,
                "llm_corrected_section": True,
            }
        )
    except Exception as e:
        print(f"  Failed to update {chunk_id}: {e}")
        return False

    log_correction(source_file, page, chunk_index, old_section, new_section)
    return True
def enrich_section(text: str, source_file: str) -> str:
    import requests

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"You are reading a chunk from {source_file}.\nIdentify the section number this chunk belongs to.\nReturn ONLY the section number (e.g. '1926.502', '1310.02(13)(b)', '6-02.3').\nIf you cannot determine a section number return UNKNOWN.\nDo not explain your answer.\n\nText:\n{text[:500]}",
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 20,
        }
    }

    response = requests.post(
        "http://localhost:11434/api/chat",
        json=payload
    )

    data = response.json()
    result = data["message"]["content"].strip()
    result = strip_thinking(result)

    import re
    if re.match(r'^[\d][\d\.\-\(\)\[\]a-zA-Z]*$', result) and len(result) >= 3:
        return result
    return "UNKNOWN"

def query(
    user_query: str,
    filter_agency: str = None,
    filter_jurisdiction: str = None,
    filter_state: str = None,
    enrich_unknown_sections: bool = True,
) -> dict:

    print("  [1/5] Connecting to database...")
    table = get_db_table()

    print("  [2/5] Embedding query...")
    chunks = retrieve_chunks(
        query=user_query,
        table=table,
        n_results=N_RESULTS,
        filter_agency=filter_agency,
        filter_jurisdiction=filter_jurisdiction,
        filter_state=filter_state,
    )
    print(f"  [3/5] Retrieved {len(chunks)} chunks")

    if not chunks:
        return {
            "query":    user_query,
            "response": "The provided standards do not address this query.",
            "chunks":   [],
        }

    if enrich_unknown_sections:
        unknown = [c for c in chunks if c.section == "UNKNOWN"]
        if unknown:
            print(f"  [4/5] Enriching {len(unknown)} unknown sections...")
            for chunk in unknown:
                chunk.section = enrich_section(chunk.text, chunk.source_file)
        else:
            print("  [4/5] All sections known — skipping enrichment")
    else:
        print("  [4/5] Section enrichment disabled")

    print("  [5/5] Generating response...")
    groups = group_chunks(chunks, table)
    context = format_context(groups)
    response = generate_response(user_query, context)

    return {
        "query":    user_query,
        "response": response,
        "chunks":   [
            {
                "source_file":           c.source_file,
                "agency":                c.agency,
                "jurisdiction":          c.jurisdiction,
                "section":               c.section,
                "doc_page":              c.doc_page,
                "page":                  c.page,
                "distance":              c.distance,
                "llm_corrected_section": c.llm_corrected_section,
            }
            for c in chunks
        ],
    }


if __name__ == "__main__":
    import sys
    import json

    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "stormwater requirements for construction sites"

    print(f"\nQuery: {user_query}\n")
    print("Retrieving and generating response...\n")

    result = query(user_query)

    print(result["response"])
    print("\n--- Retrieved chunks ---")
    for c in result["chunks"]:
        print(f"  {c['source_file']} | {c['agency']} | Section {c['section']} | Page {c['doc_page']}")