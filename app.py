"""
app.py
------
Minimal Streamlit UI for local RAG demo:
1) Retrieve with cosine+MMR (via search_faiss.retrieve)
2) Build a section-aware context (top-k trimmed chunks with file/section headers)
3) Ask Mistral (via Ollama) with a concise prompt and show citations

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path
import streamlit as st

# Allow imports from scripts/
sys.path.append(str(Path(__file__).resolve().parent / "scripts"))

from search_faiss import retrieve   # cosine similarity + MMR retriever
from ollama import chat             # local Ollama API call wrapper
from query_rag_advanced import build_scoped_context

TOP_K_DEFAULT = 4
# Retrieve a wider candidate pool than the UI displays.
MIN_CANDIDATE_K = 6

SYSTEM_PROMPT = """You answer ONLY using the provided context.

                Rules:
                - Be concise: 3–6 bullet points.
                - Answer only what was asked; ignore unrelated sections.
                - Do not copy long passages; summarize what’s in context.
                - If not clearly in context, say 'Not found in context'.
                - Cite sources at the end as [filename].
                """

def ask_llm(query, context_text, srcs, model_name="mistral"):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_text}\n\nReturn: bullet points only."},
    ]
    resp = chat(model=model_name, messages=messages)
    answer = resp["message"]["content"].strip()
    cites = " ".join(f"[{s}]" for s in sorted(srcs))
    return f"{answer}\n\nSources: {cites}"

# Streamlit UI setup
st.set_page_config(page_title="Local RAG Assistant", page_icon="🧭", layout="wide")
st.title("Local RAG Assistant")
st.caption("Ollama + FAISS + Mistral · section-aware retrieval · citations")

# Sidebar settings
with st.sidebar:
    st.subheader("Settings")
    model = st.text_input("Ollama model", value="mistral", help="Must be available via `ollama pull`")
    top_k = st.slider("Top-K retrieved chunks", min_value=2, max_value=8, value=TOP_K_DEFAULT, step=1)
    show_chunks = st.checkbox("Show retrieved chunks", value=True)
    show_model_context = st.checkbox("Show final context sent to model", value=False)
    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown("- Ask specific questions (e.g., *science requirement*, *concentrations*, *lower-division core*).")
    st.markdown("- Answers are grounded in `data/*.md` and cite source files.")

# Main input area
query = st.text_input("Ask about the UO CS program…")
go = st.button("Ask")

if go and query.strip():
    with st.spinner("Retrieving…"):
        # Keep retrieval broad enough to find the right section even when the visible Top-K is small.
        retrieval_k = max(top_k, MIN_CANDIDATE_K)
        chunks = retrieve(query.strip(), top_k=retrieval_k)
        # Only show the number of chunks selected in the sidebar.
        visible_chunks = chunks[:top_k]

    if not chunks:
        st.warning("No results retrieved. Try a different query.")
    else:
        if show_chunks:
            st.subheader("Retrieved context")
            for c in visible_chunks:
                section = " > ".join(c.get("section_path", []))
                with st.expander(f"[Rank {c['rank']}] {c['source_file']} | {section}  ·  similarity={c['score']:.4f}"):
                    st.code(c["text"][:1000])

        with st.spinner("Generating answer…"):
            # Rebuild context around the single best-matching section before calling the LLM.
            context_text, srcs = build_scoped_context(query.strip(), chunks)
            answer = ask_llm(query.strip(), context_text, srcs, model_name=model)

        if show_model_context:
            st.subheader("Final context sent to model")
            st.code(context_text)

        st.subheader("Answer")
        st.markdown(answer)
