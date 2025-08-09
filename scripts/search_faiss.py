"""
search_faiss.py
---------------
Cosine-similarity search with:
- query normalization
- generic token-overlap boost (query terms ↔ chunk text)
- section-heading overlap boost (query terms ↔ section title/path)
- MMR reranking to avoid near-duplicates

Exports:
- retrieve(query: str, top_k: int) -> List[dict]
- CLI demo via search(...)
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from ollama import embeddings
import re

EMBED_PATH = Path("outputs/embedded_output.json")
INDEX_PATH = Path("outputs/faiss_index.bin")

CAND_MULT = 6
STOP = {"the","a","an","of","to","for","in","on","and","or","with","is","are","be","as","at","by","from"}

def _tokenize(s: str):
    return [t for t in re.sub(r"[^a-z0-9]+", " ", s.lower()).split() if t not in STOP and len(t) > 2]

def embed_text(text: str) -> np.ndarray:
    """Return (1, d) normalized embedding (cosine)."""
    res = embeddings(model="nomic-embed-text", prompt=text)
    q = np.array([res["embedding"]], dtype="float32")
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    return q

def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

def _build_candidates(scores: np.ndarray, indices: np.ndarray, data: list) -> List[Dict]:
    cand = []
    for pos, idx in enumerate(indices[0], start=1):
        if idx < 0 or idx >= len(data):
            continue
        item = data[idx]
        vec = _norm(np.array(item["embedding"], dtype="float32"))
        cand.append({
            "rank": pos,
            "score": float(scores[0][pos-1]),  # cosine similarity (higher=better)
            "source_file": item["source_file"],
            "section_path": item.get("section_path", []),
            "section_title": item.get("section_title", ""),
            "text": item["text"],
            "vec": vec
        })
    return cand

def _overlap_boost(r: Dict, qterms: set, per_hit: float = 0.02, section_bonus: float = 0.03):
    text = r["text"].lower()
    hits = sum(1 for w in qterms if w in text)
    r["score"] += per_hit * hits
    # section-aware, generic boost (works for ANY topic)
    sp = " ".join(r.get("section_path", [])).lower()
    hits_sec = sum(1 for w in qterms if w in sp)
    if hits_sec:
        r["score"] += section_bonus * hits_sec

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # vectors already normalized

def _mmr(results: List[Dict], top_k: int, lam: float = 0.7) -> List[Dict]:
    selected, candidates = [], results[:]
    while candidates and len(selected) < top_k:
        if not selected:
            best = max(candidates, key=lambda x: x["score"])
        else:
            def mmr_score(x):
                sim_to_sel = max(_cos(x["vec"], s["vec"]) for s in selected)
                return lam * x["score"] - (1 - lam) * sim_to_sel
            best = max(candidates, key=mmr_score)
        selected.append(best); candidates.remove(best)
    for i, r in enumerate(selected, start=1):
        r["rank"] = i
    return selected

def retrieve(query: str, top_k: int = 3) -> List[Dict]:
    with open(EMBED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    index = faiss.read_index(str(INDEX_PATH))
    qvec = embed_text(query)
    cand_k = min(len(data), max(top_k, top_k * CAND_MULT))
    sims, idxs = index.search(qvec, cand_k)

    results = _build_candidates(sims, idxs, data)

    # Generic, domain-agnostic boosts based on the user's query terms
    qterms = set(_tokenize(query))
    for r in results:
        _overlap_boost(r, qterms)

    # MMR diversity
    final = _mmr(results, top_k=top_k, lam=0.7)
    return final

def search(query: str, top_k: int = 4) -> None:
    res = retrieve(query, top_k=top_k)
    for r in res:
        sec = f" | Section: {' > '.join(r.get('section_path', []))}" if r.get("section_path") else ""
        print(f"[Rank {r['rank']}] {r['source_file']}{sec} (similarity={r['score']:.4f}, higher is better)")
        print(r["text"][:220], "\n")

if __name__ == "__main__":
    search("computer science degree requirements")
