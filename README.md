# Local RAG Assistant (Demo)

Minimal, fully-local **RAG** demo that answers questions about the UO CS program using local markdown docs.  
Built to show end-to-end understanding of the RAG pipeline.

## Why this exists
A small, time-boxed solo project to turn meeting-room knowledge into a tangible result:  
**documents → embeddings → vector search → grounded LLM answer with citations**.

## Stack
- **Embeddings:** Ollama `nomic-embed-text`  
- **Vector DB:** FAISS (cosine, with simple MMR)  
- **LLM:** Ollama `mistral`  
- **Data:** `data/*.md` (UO CS pages, section-aware chunking)

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull mistral
```

## Run (reproducible)
```bash
# 1) Build embeddings (reads data/*.md -> outputs/embedded_output.json)
python scripts/embed_documents.py

# 2) Build FAISS index (outputs/faiss_index.bin)
python scripts/build_faiss.py

# 3) Query end-to-end
python scripts/query_rag.py
# example prompts:
# - Explain the science requirement for the CS major, including eligible subjects.
# - List all concentrations available for the CS major.
# - What are the lower-division core courses for the CS major?
```

## What it shows
- Local embeddings → FAISS cosine retrieval (+ MMR)  
- Section-aware chunking for better grounding  
- Mistral answers **only from context**, with **file citations**

## Known limitations (v1)
- Answers are limited to the included markdown pages.  
- Some long, multi-part lines (e.g., Biology AND/OR) may be summarized imperfectly.  
- Occasionally a nearby requirement line (e.g., Writing) appears—kept to show raw grounding + citations.

## Next?
- Streamlit micro-UI (1 text box + results)  
- Deterministic list rendering for complex “AND/OR” requirement blocks  
- Expand/clean data sources
