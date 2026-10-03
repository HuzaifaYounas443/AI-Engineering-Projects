# Project 02: Hybrid RAG

A hybrid retrieval-augmented generation (RAG) application that combines
dense semantic search and sparse BM25 keyword search. Results from both
retrievers are merged with Reciprocal Rank Fusion (RRF), then passed to a
Groq-hosted, OpenAI-compatible language model for grounded answer generation.

## Features

- Ingests TXT, Markdown, PDF, DOCX, HTML, and common image files.
- Extracts and chunks source documents with configurable overlap.
- Creates local embeddings with `BAAI/bge-small-en-v1.5` through FastEmbed.
- Stores dense vectors in Qdrant.
- Persists chunks for sparse BM25 retrieval.
- Fuses dense and sparse rankings to improve semantic and exact-term recall.
- Provides a Streamlit interface and command-line ingestion/query scripts.
- Includes source attribution in generated answers.

## Architecture

See [architecture.md](architecture.md) for the component diagram and data flows.

## Prerequisites

- Python 3.10 or newer
- Docker Desktop (for Qdrant)
- A Groq API key for LLM answer generation

## Setup

From this directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install fastembed qdrant-client rank-bm25 python-dotenv openai streamlit
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env`. Keep `.env` local; it is intentionally excluded
from version control.

Start Qdrant:

```powershell
docker compose up -d
```

## Usage

### 1. Add source documents

Place supported documents in `data/`. Uploaded Streamlit files are stored in
`data/uploads/`.

### 2. Build the index

```powershell
python main.py
```

This extracts documents, creates chunks and embeddings, writes the local
`chunks_store.json` used by BM25, and upserts vectors into Qdrant.

### 3. Query from the command line

Hybrid retrieval only:

```powershell
python query_hybrid.py
```

Dense retrieval only:

```powershell
python query_retrieval.py
```

Hybrid retrieval with Groq generation:

```powershell
python query_with_llm.py
```

### 4. Run the Streamlit UI

```powershell
streamlit run app.py
```

Open the URL printed by Streamlit, upload documents, and ask questions after
the index has been built.

## Configuration

The ingestion defaults are defined in `src/ingestion/config.py`:

- `QDRANT_URL`: Qdrant HTTP endpoint, default `http://localhost:6335`.
- `COLLECTION_NAME`: Qdrant collection name, default `website_knowledge`.
- `EMBEDDING_MODEL`: FastEmbed model used for dense vectors.
- `CHUNK_SIZE` and `CHUNK_OVERLAP`: chunking parameters.

## Project layout

```text
.
├── app.py                  # Streamlit application
├── main.py                 # Document ingestion and indexing
├── query_hybrid.py         # Hybrid retrieval plus Groq generation
├── query_retrieval.py     # Dense retrieval example
├── query_with_llm.py      # Dense retrieval plus generation example
├── chunk_store.py          # JSON persistence for BM25 chunks
├── docker-compose.yml      # Local Qdrant service
├── data/                   # Example/source documents
└── src/
    ├── ingestion/          # Extraction, chunking, embedding, indexing
    └── retrieval/          # Dense, sparse, and fused retrieval
```

## Notes

- Run commands from the project root so relative paths resolve correctly.
- The Qdrant container exposes port `6335` for HTTP and `6336` for gRPC.
- Local vector data, generated chunk stores, uploads, virtual environments,
  and credentials are ignored by Git.
