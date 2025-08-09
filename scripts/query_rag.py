"""
query_rag_min.py
----------------
Minimal RAG for the demo:
1) Retrieve with cosine+MMR (via search_faiss.retrieve)
2) Build a simple context (top-k trimmed chunks with file/section headers)
3) Ask Mistral with a tight prompt and show citations

Usage:
    python3 scripts/query_rag_min.py
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from search_faiss import retrieve
from ollama import chat

TOP_K = 4
MAX_CHARS_PER_CHUNK = 2000

SYSTEM_PROMPT = """You answer ONLY using the provided context.

Rules:
- Be concise: 3–6 bullet points.
- Answer only what was asked; ignore unrelated sections.
- Do not copy long passages; summarize what’s in context.
- If not clearly in context, say 'Not found in context'.
- Cite sources at the end as [filename].
"""

def build_context(chunks):
    parts = []
    for c in chunks:
        section = " > ".join(c.get("section_path", []))
        hdr = f"[Source: {c['source_file']}" + (f" | Section: {section}]" if section else "]")
        parts.append(f"{hdr}\n{c['text'][:MAX_CHARS_PER_CHUNK]}")
    return "\n\n---\n\n".join(parts), {c["source_file"] for c in chunks}

def ask_llm(query, context_text, srcs):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_text}\n\nReturn: bullet points only."}
    ]
    resp = chat(model="mistral", messages=messages)
    answer = resp["message"]["content"].strip()
    cites = " ".join(f"[{s}]" for s in sorted(srcs))
    return f"{answer}\n\nSources: {cites}"

if __name__ == "__main__":
    user_query = input("Enter your question: ").strip()

    print("\n🔍 Searching relevant chunks...")
    chunks = retrieve(user_query, top_k=TOP_K)
    for c in chunks:
        section = " > ".join(c.get("section_path", []))
        print(f"[Rank {c['rank']}] {c['source_file']} | Section: {section} (similarity={c['score']:.4f}, higher is better)")
        print(c["text"][:160], "...\n")

    print("💬 Generating answer with Mistral...\n")
    context_text, srcs = build_context(chunks)
    out = ask_llm(user_query, context_text, srcs)
    print("=== ANSWER ===")
    print(out)
