# app.py
# Minimal Streamlit UI for your local RAG demo (Ollama + FAISS + Mistral)

import sys
from pathlib import Path
import streamlit as st

# make `scripts/` importable
sys.path.append(str(Path(__file__).resolve().parent / "scripts"))

from search_faiss import retrieve  # your cosine+MMR retriever
from ollama import chat

TOP_K_DEFAULT = 4
MAX_CHARS_PER_CHUNK = 900

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

def ask_llm(query, context_text, srcs, model_name="mistral"):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_text}\n\nReturn: bullet points only."},
    ]
    resp = chat(model=model_name, messages=messages)
    answer = resp["message"]["content"].strip()
    cites = " ".join(f"[{s}]" for s in sorted(srcs))
    return f"{answer}\n\nSources: {cites}"

st.set_page_config(page_title="Local RAG Assistant", page_icon="🧭", layout="wide")

st.title("Local RAG Assistant")
st.caption("Ollama + FAISS + Mistral · section-aware retrieval · citations")

with st.sidebar:
    st.subheader("Settings")
    model = st.text_input("Ollama model", value="mistral", help="Must be available via `ollama pull`")
    top_k = st.slider("Top‑K retrieved chunks", min_value=2, max_value=8, value=TOP_K_DEFAULT, step=1)
    show_chunks = st.checkbox("Show retrieved chunks", value=True)
    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown("- Ask specific questions (e.g., *science requirement*, *concentrations*, *lower-division core*).")
    st.markdown("- Answers are grounded in `data/*.md` and cite source files.")

query = st.text_input("Ask about the UO CS program…")
go = st.button("Ask")

if go and query.strip():
    with st.spinner("Retrieving…"):
        chunks = retrieve(query.strip(), top_k=top_k)

    if not chunks:
        st.warning("No results retrieved. Try a different query.")
    else:
        if show_chunks:
            st.subheader("Retrieved context")
            for c in chunks:
                section = " > ".join(c.get("section_path", []))
                with st.expander(f"[Rank {c['rank']}] {c['source_file']} | {section}  ·  similarity={c['score']:.4f}"):
                    st.code(c["text"][:1000])

        with st.spinner("Generating answer…"):
            context_text, srcs = build_context(chunks)
            answer = ask_llm(query.strip(), context_text, srcs, model_name=model)

        st.subheader("Answer")
        st.markdown(answer)
