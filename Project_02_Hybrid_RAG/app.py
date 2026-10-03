"""Streamlit demo for the Hybrid RAG pipeline."""
import shutil
import time
from pathlib import Path

import streamlit as st

from src.ingestion import (
    Embedder,
    IngestionProcessor,
    QdrantIndexer,
    chunk_document,
)
from src.retrieval import HybridRetriever


# ---------- Config ----------

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Session state ----------

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# ---------- Helpers ----------

def ingest_files(uploaded_files) -> dict:
    """Save uploads to disk, then run the full ingestion pipeline."""
    stats = {"files": 0, "chunks": 0, "vectors": 0, "points": 0}

    # 1. Save uploaded files
    for uf in uploaded_files:
        dest = UPLOAD_DIR / uf.name
        dest.write_bytes(uf.getbuffer())
        stats["files"] += 1

    # 2. Extract
    processor = IngestionProcessor()
    docs = processor.process_directory(UPLOAD_DIR)
    if not docs:
        return stats

    # 3. Chunk
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    stats["chunks"] = len(chunks)

    # 4. Embed
    embedder = Embedder()
    vectors = embedder.embed([c["text"] for c in chunks])
    stats["vectors"] = len(vectors)

    # 5. Index
    indexer = QdrantIndexer()
    indexer.reset()
    stats["points"] = indexer.upsert(chunks, vectors)

    # 6. Save chunk store for BM25
    from chunk_store import save_chunks
    save_chunks(chunks)

    return stats


# ---------- UI ----------

st.set_page_config(page_title="Hybrid RAG", page_icon="🔍", layout="wide")
st.title("🔍 Hybrid RAG Demo")
st.caption("Dense + Sparse retrieval, fused with Reciprocal Rank Fusion")

tab_ingest, tab_query = st.tabs(["📥 Ingest", "💬 Query"])


# ============ TAB 1: INGEST ============

with tab_ingest:
    st.header("Upload your content")
    st.write(
        "Supported: `.txt` `.md` `.pdf` `.png` `.jpg` `.docx` `.html` — "
        "drop any mix and they'll be extracted, chunked, embedded, and indexed."
    )

    uploaded = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["txt", "md", "pdf", "png", "jpg", "jpeg", "docx", "html", "htm"],
    )

    if st.button("🚀 Ingest", type="primary", disabled=not uploaded):
        with st.spinner("Running ingestion pipeline ..."):
            start = time.time()
            stats = ingest_files(uploaded)
            elapsed = time.time() - start

        if stats["chunks"] == 0:
            st.error("No content could be extracted. Check your file formats.")
        else:
            st.success(f"✅ Ingestion complete in {elapsed:.2f}s")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Files", stats["files"])
            c2.metric("Chunks", stats["chunks"])
            c3.metric("Vectors", stats["vectors"])
            c4.metric("Points in Qdrant", stats["points"])

            # Force-retriever rebuild so the query tab sees fresh data
            st.session_state.retriever = None


# ============ TAB 2: QUERY ============

with tab_query:
    st.header("Ask a question")
    st.write("Uses hybrid retrieval (dense + sparse) fused with RRF, then Groq for the answer.")

    question = st.text_input(
        "Your question",
        placeholder="e.g. How do I reset my password?",
    )

    top_k = st.slider("Top-K chunks to retrieve", 1, 10, 3)
    show_context = st.checkbox("Show retrieved chunks", value=True)

    if st.button("💡 Ask", type="primary", disabled=not question):
        # Lazy-load retriever
        if st.session_state.retriever is None:
            with st.spinner("Loading retriever ..."):
                try:
                    st.session_state.retriever = HybridRetriever()
                except FileNotFoundError:
                    st.error("No chunks indexed. Go to the Ingest tab first.")
                    st.stop()

        with st.spinner("Retrieving ..."):
            chunks, sources = st.session_state.retriever.search(
                question,
                top_k=top_k,
                per_retriever_k=20,
                return_sources=True,
            )

        with st.spinner("Generating answer ..."):
            from query_hybrid import generate
            answer = generate(question, chunks)

        st.subheader("Answer")
        st.write(answer)

        if show_context:
            st.subheader("Retrieved context")
            for i, c in enumerate(chunks, 1):
                with st.expander(
                    f"[{i}] rrf={c.get('rrf_score', 0):.5f} — {c['source']}"
                ):
                    st.write(c["text"])

            with st.expander("🔬 Retrieval details"):
                st.write(f"**Dense candidates:** {len(sources['dense'])}")
                st.write(f"**Sparse candidates:** {len(sources['sparse'])}")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Dense top-5**")
                    for i, r in enumerate(sources["dense"][:5], 1):
                        st.write(f"{i}. score={r['score']:.4f} — {Path(r['source']).name}")
                with col2:
                    st.write("**Sparse top-5**")
                    for i, r in enumerate(sources["sparse"][:5], 1):
                        st.write(f"{i}. score={r['score']:.4f} — {Path(r['source']).name}")