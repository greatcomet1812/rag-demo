"""
embed_documents.py
-------------------
Reads markdown files from `data/`, splits into section-aware chunks,
embeds with Ollama `nomic-embed-text`, and saves JSON to `outputs/embedded_output.json`.

Usage:
    python3 scripts/embed_documents.py
"""

import re
import json
from pathlib import Path
from typing import List, Dict
from ollama import embeddings

DATA_DIR = Path("data")
OUTPUT_PATH = Path("outputs/embedded_output.json")

# ---- Markdown parsing & chunking ----

HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)$')

def tokenize(s: str) -> List[str]:
    return [t for t in re.sub(r"[^a-z0-9]+", " ", s.lower()).split() if len(t) > 2]

def parse_markdown_sections(text: str) -> List[Dict]:
    """
    Parses a markdown into blocks with heading context.
    Returns list of sections: {level, title, start, end}.
    """
    lines = text.splitlines()
    sections = []
    stack = []  # (level, title, start_line_idx)
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line.strip())
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            # close sections deeper or equal to this level
            while stack and stack[-1][0] >= level:
                lvl, ttl, start = stack.pop()
                sections.append({"level": lvl, "title": ttl, "start": start, "end": i})
            stack.append((level, title, i))
    # close any remaining
    n = len(lines)
    while stack:
        lvl, ttl, start = stack.pop()
        sections.append({"level": lvl, "title": ttl, "start": start, "end": n})

    if not sections:
        # whole doc as one section
        sections = [{"level": 1, "title": "Document", "start": 0, "end": n}]
    return sorted(sections, key=lambda s: s["start"])

def section_path(sections: List[Dict], idx: int) -> List[str]:
    """
    Given a section index, reconstruct a simple heading path up to that section.
    (We approximate by collecting prior headings with lower level.)
    """
    path = [sections[idx]["title"]]
    my_level = sections[idx]["level"]
    # walk backward for parents
    for j in range(idx-1, -1, -1):
        if sections[j]["level"] < my_level:
            path.insert(0, sections[j]["title"])
            my_level = sections[j]["level"]
    return path

def chunk_text_by_sections(text: str, max_chars: int = 1200, overlap: int = 150) -> List[Dict]:
    lines = text.splitlines()
    sections = parse_markdown_sections(text)
    chunks = []
    for i, sec in enumerate(sections):
        sec_text = "\n".join(lines[sec["start"]:sec["end"]]).strip()
        path = section_path(sections, i)  # e.g., ["CS Core Requirements", "Mathematics (16 credits)"]
        # sliding window within section
        start = 0
        while start < len(sec_text):
            end = min(len(sec_text), start + max_chars)
            piece = sec_text[start:end]
            if piece.strip():
                chunks.append({
                    "section_path": path,
                    "section_title": path[-1] if path else "",
                    "text": piece
                })
            if end == len(sec_text):
                break
            start = max(0, end - overlap)
    return chunks

# ---- Embedding ----

def embed_batch(texts: List[str]) -> List[List[float]]:
    out = []
    for t in texts:
        res = embeddings(model="nomic-embed-text", prompt=t)
        out.append(res["embedding"])
    return out

def process_file(md_path: Path) -> List[Dict]:
    md = md_path.read_text(encoding="utf-8")
    chunks = chunk_text_by_sections(md)
    for c in chunks:
        c["source_file"] = md_path.name
    return chunks

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict] = []
    for fp in sorted(DATA_DIR.glob("*.md")):
        all_chunks.extend(process_file(fp))

    # Build embedding prompts that include a compact header => better retrieval
    embed_inputs = []
    for c in all_chunks:
        header = f"[Section: {' > '.join(c['section_path'])} | File: {c['source_file']}]"
        embed_inputs.append(f"{header}\n{c['text']}")

    vectors = embed_batch(embed_inputs)
    output = []
    for c, v in zip(all_chunks, vectors):
        output.append({
            "source_file": c["source_file"],
            "section_path": c["section_path"],
            "section_title": c["section_title"],
            "text": c["text"],
            "embedding": v
        })

    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Embeddings saved to {OUTPUT_PATH} (chunks={len(output)})")

if __name__ == "__main__":
    main()
