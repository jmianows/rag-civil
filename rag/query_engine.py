import ollama
import lancedb
from pathlib import Path
from dataclasses import dataclass

VECTORDB_DIR  = Path("/home/justin/rag-civil/vectordb")
EMBED_MODEL   = "mxbai-embed-large"
LLM_MODEL     = "qwen3:8b"
N_RESULTS     = 5
CONTEXT_WINDOW = 1
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
    window: int = CONTEXT_WINDOW,
) -> str:
    source_file = chunk.source_file
    page        = chunk.page
    chunk_index = chunk.chunk_index

    neighbors = []
    for offset in range(-window, window + 1):
        if offset == 0:
            neighbors.append((chunk_index, chunk.text))
            continue

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

Your job is to read the retrieved code sections provided and present their requirements clearly and accurately.

Rules you must follow without exception:
- Only use information from the retrieved sections provided. Do not add your own knowledge, but you are allowed to add vocabulary to smooth things out between passages.
- Do not change words like "shall" "should" or "may", because in the MUTCD they define how mandatory standards are.
- If the retrieved sections do not contain relevant information, say exactly: "The provided standards do not address this query. Think I should? Request manuals to add using the bottom right!"
- Never provide general engineering advice, opinions, or recommendations.
- Never fill gaps with assumed knowledge even if you are confident in the answer.
- If multiple standards address the same topic, present each one separately.
- Always include the full citation block at the end of each passage.

Response format:
- Write out each relevant passage cleanly and completely.
- At the end of each passage include a citation block in this exact format:

  [SOURCE]
  Document: <source_file>
  Section: <section>
  Document Page: <doc_page>
  PDF Page: <page>

- Group passages from the same section together under one citation block.
- Do not add preamble, summaries, or conclusions. Always include exceptions found.
- Do not restate the question. """


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

        block = f"""--- RETRIEVED SECTION ---
{combined}

[SOURCE]
Document: {chunk.source_file}
Agency: {chunk.agency}
Jurisdiction: {chunk.jurisdiction}
Section: {section_display}
Document Page: {chunk.doc_page}
PDF Page: {chunk.page}
---"""

        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def generate_response(
    query: str,
    context: str,
) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"Query: {query}\n\nRetrieved sections:\n\n{context}",
        }
    ]

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={
            "temperature":    0.1,
            "num_ctx":        4096,
            "num_predict":    2048,
            "repeat_penalty": 1.1,
        },
        think=False,
    )

    return response["message"]["content"].strip()
#this is the script to try and find sections when the automatic parser cannot. 
def enrich_section(text: str, source_file: str) -> str:
    prompt = f"""You are reading a chunk from {source_file}.
Identify the section number this chunk belongs to.
Return ONLY the section number (e.g. '1926.502', '1310.02(13)(b)', '6-02.3').
If you cannot determine a section number return UNKNOWN.
Do not explain your answer.

Text:
{text[:500]}"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0,
            "num_predict": 20,
        },
        think=False,
    )
    result = response["message"]["content"].strip()

    # validate result looks like a section number
    import re
    if re.match(r'^[\d][\d\.\-\(\)\[\]a-zA-Z]*$', result) and len(result) >= 3:
        return result
    return "UNKNOWN"
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

def query(
    user_query: str,
    filter_agency: str = None,
    filter_jurisdiction: str = None,
    filter_state: str = None,
    enrich_unknown_sections: bool = True,
) -> dict:

    table = get_db_table()

    # retrieve chunks
    chunks = retrieve_chunks(
        query=user_query,
        table=table,
        n_results=N_RESULTS,
        filter_agency=filter_agency,
        filter_jurisdiction=filter_jurisdiction,
        filter_state=filter_state,
    )

    if not chunks:
        return {
            "query":    user_query,
            "response": "The provided standards do not address this query.",
            "chunks":   [],
        }

    # enrich UNKNOWN sections with LLM if enabled
    if enrich_unknown_sections:
        for chunk in chunks:
            if chunk.section == "UNKNOWN":
                chunk.section = enrich_section(chunk.text, chunk.source_file)

    # group and expand context
    groups = group_chunks(chunks, table)

    # format context for LLM
    context = format_context(groups)

    # generate response
    response = generate_response(user_query, context)

    return {
        "query":    user_query,
        "response": response,
         "chunks": [
            {
                "source_file":           c.source_file,
                "agency":                c.agency,
                "jurisdiction":          c.jurisdiction,
                "section":               c.section,
                "llm_corrected_section": c.llm_corrected_section,
                "doc_page":              c.doc_page,
                "page":                  c.page,
                "distance":              c.distance,
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