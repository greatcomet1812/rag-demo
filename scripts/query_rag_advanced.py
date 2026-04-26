"""
query_rag.py
------------
Generic, scoped RAG:
1) Retrieve with cosine + MMR (via search_faiss.retrieve)
2) Dynamically scope context:
   - pick best section by query overlap
   - STITCH all chunks from that same section (to avoid partial lists)
   - keep lines that are bullets, colon-lines, course-code lines, or contain query terms
3) Constrained answer with citations
"""

from pathlib import Path
import sys, re, json
sys.path.append(str(Path(__file__).resolve().parent))

from search_faiss import retrieve
from ollama import chat

TOP_K = 4
MAX_CHARS_PER_CHUNK = 900
EMBED_PATH = Path("outputs/embedded_output.json")

SYSTEM_PROMPT = """You answer ONLY using the provided context.

                Output rules:
                Answer ONLY what the user asked; ignore unrelated categories.
                - Be concise: 3–8 bullet points.
                - Preserve logical operators EXACTLY as stated (AND, OR, "choose one from").
                - When a requirement has multiple parts, use nested bullets and write AND/OR explicitly.
                - Do not flatten AND into OR, and do not drop mandatory steps.
                - Cite sources at the end as [filename].
                - If the answer is not clearly in the context, reply: 'Not found in context'.
                - Do NOT include generic advice. Do NOT paste long passages.
                """

STOP = {"the","a","an","of","to","for","in","on","and","or","with","is","are","be","as","at","by","from"}

COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,4}(?:/[A-Z]{2,4})?\s?\d{3}\b")
COLON_LINE_RE  = re.compile(r"^[A-Z][A-Za-z/&\-\s]+:\s")  # e.g., "Physics: ", "Geological Sciences: "

def tokenize(s: str):
    return [t for t in re.sub(r"[^a-z0-9]+"," ", s.lower()).split() if t not in STOP and len(t) > 2]

def line_relevant(line: str, qterms: set) -> bool:
    """Keep: bullets/numbered, colon-lines (Discipline: ...), course-code lines, or any line with query terms."""
    raw = line.rstrip()
    l = raw.strip().lower()
    if not l:
        return False
    if l.startswith(("-", "*")):  # bullets
        return True
    if re.match(r"^\d+[\.\)]\s+", raw.strip()):  # numbered list
        return True
    if COLON_LINE_RE.match(raw):  # "Physics: ...", "Geological Sciences: ..."
        return True
    if COURSE_CODE_RE.search(raw):  # "PHYS 201", "GEOL/ERTH 202", "CH 221", "PSY 201", "BI 211"
        return True
    return any(w in l for w in qterms)

def score_chunk_for_query(chunk: dict, qterms: set) -> float:
    # Prefer section-title matches much more strongly than long-body overlap.
    path_parts = chunk.get("section_path", [])
    sec = " ".join(path_parts).lower()
    title = (path_parts[-1] if path_parts else chunk.get("section_title", "")).lower()
    txt = chunk["text"][:MAX_CHARS_PER_CHUNK].lower()

    title_hits = sum(1 for w in qterms if w in title)
    sec_hits = sum(1 for w in qterms if w in sec)
    txt_hits = sum(1 for w in qterms if w in txt)

    score = title_hits * 3.0 + sec_hits * 1.5 + txt_hits * 0.35

    # Give a further boost when the heading covers a large share of the query terms.
    if qterms:
        title_coverage = title_hits / len(qterms)
        sec_coverage = sec_hits / len(qterms)
        if title_coverage >= 0.45:
            score += 2.0
        if sec_coverage >= 0.60:
            score += 1.5

    return score

def load_full_section(source_file: str, section_path: list) -> list:
    """Load ALL chunks from the same file+section_path to avoid partial lists."""
    data = json.loads(EMBED_PATH.read_text(encoding="utf-8"))
    return [d for d in data
            if d.get("source_file") == source_file
            and d.get("section_path") == section_path]

def extract_relevant_lines_from_text(text: str, qterms: set, max_lines: int = 240):
    kept = []
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n and len(kept) < max_lines:
        ln = lines[i]
        # Start of a "Discipline: ..." block
        if COLON_LINE_RE.match(ln):
            block = [ln.rstrip()]
            j = i + 1
            while j < n:
                nxt = lines[j]
                if not nxt.strip():  # blank line ends block
                    break
                if COLON_LINE_RE.match(nxt):  # next discipline starts
                    break
                block.append(nxt.rstrip())
                j += 1
            kept.append(" ".join(block).strip())
            i = j
            continue
        # Otherwise, keep bullets, numbered, course-code, or qterm lines
        if line_relevant(ln, qterms):
            kept.append(ln.strip())
        i += 1
    return kept[:max_lines]

def build_scoped_context(query: str, retrieved: list):
    qterms = set(tokenize(query))
    # Re-score retrieved chunks locally so we can prefer the most on-topic section title.
    rescored = sorted(retrieved, key=lambda c: score_chunk_for_query(c, qterms), reverse=True)
    if not rescored:
        return "", set()

    # Use the best section as the anchor, then pull in all chunks from that same section.
    best = rescored[0]
    section_path = best.get("section_path", [])
    source_file = best.get("source_file", "")
    stitched_chunks = load_full_section(source_file, section_path) or [best]

    # Stitch the full section back together before filtering lines for the prompt.
    stitched_text = "\n".join((c.get("text") or "")[:MAX_CHARS_PER_CHUNK] for c in stitched_chunks)

    # Extract only relevant lines (bullets, colon-lines, course codes, query-term lines)
    lines = extract_relevant_lines_from_text(stitched_text, qterms)
    if not lines:
        # Fallback: use the best few rescored chunks (trimmed)
        parts = []
        for c in rescored[:max(3, TOP_K)]:
            section = " > ".join(c.get("section_path", []))
            hdr = f"[Source: {c['source_file']}" + (f" | Section: {section}]" if section else "]")
            parts.append(f"{hdr}\n{c['text'][:MAX_CHARS_PER_CHUNK]}")
        return "\n\n---\n\n".join(parts), {c["source_file"] for c in rescored[:max(3, TOP_K)]}

    # Build a compact prompt block with source attribution.
    section = " > ".join(section_path)
    header = f"[Source: {source_file} | Section: {section}]"
    context_text = f"{header}\n" + "\n".join(lines)
    return context_text, {source_file}

def generate_answer(
    query: str,
    retrieved_chunks: list,
    model_name: str = "mistral",
    debug_show_context: bool = False,
):
    """
    Build scoped context (stitch full section, keep relevant lines),
    show it (debug), then ask Mistral via Ollama.
    """
    # Build a single context string + set of source files
    context_text, srcs = build_scoped_context(query, retrieved_chunks)

    # Debug: see exactly what the model will read
    if debug_show_context:
        print("\n--- CONTEXT SENT TO MODEL ---\n")
        print(context_text)
        print("\n--- END CONTEXT ---\n")

    # Ask the model (Ollama)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_text}\n\nReturn: bullet points only."}
    ]
    resp = chat(model=model_name, messages=messages)
    answer = resp["message"]["content"].strip()

    # Citations
    citations = " ".join(f"[{s}]" for s in sorted(srcs)) if srcs else ""
    if citations:
        answer += f"\n\nSources: {citations}"
    return answer


if __name__ == "__main__":
    user_query = input("Enter your question: ").strip()

    print("\n🔍 Searching relevant chunks...")
    chunks = retrieve(user_query, top_k=TOP_K)
    for c in chunks:
        section = " > ".join(c.get("section_path", []))
        print(f"[Rank {c['rank']}] {c['source_file']} | Section: {section} (similarity={c['score']:.4f}, higher is better)")
        print(c["text"][:160], "...\n")

    print("💬 Generating answer with Mistral...\n")
    ans = generate_answer(user_query, chunks)
    print("=== ANSWER ===")
    print(ans)
