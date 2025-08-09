"""
build_faiss.py
--------------
Load precomputed embeddings (JSON), build a cosine-similarity FAISS index,
and save it to disk.

Usage:
    python3 scripts/build_faiss.py
"""

import json
from pathlib import Path
import numpy as np
import faiss

EMBED_PATH = Path("outputs/embedded_output.json")
INDEX_PATH = Path("outputs/faiss_index.bin")  # keep consistent across files

def main():
    with open(EMBED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build matrix of embeddings
    emb = np.array([d["embedding"] for d in data], dtype="float32")

    # Normalize for cosine similarity = inner product on unit vectors
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms

    index = faiss.IndexFlatIP(emb.shape[1])  # inner product == cosine on normalized
    index.add(emb)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"✅ FAISS (cosine) index saved to {INDEX_PATH} with {index.ntotal} vectors.")

if __name__ == "__main__":
    main()
