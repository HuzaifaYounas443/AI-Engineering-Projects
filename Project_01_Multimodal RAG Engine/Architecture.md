# Architecture

## Overview

Hybrid RAG separates document ingestion, vector retrieval, and answer
generation. The ingestion path converts heterogeneous files into searchable
chunks. The query path embeds a question, performs cosine-similarity search in
Qdrant, and can pass the retrieved context to an LLM.

```text
Source files in data/
        |
        v
IngestionProcessor
  (extension routing)
        |
        v
File extractors
  txt/md | PDF | image/OCR | DOCX | HTML
        |
        v
ExtractedDocument
        |
        v
chunk_document
  500 chars / 100 char overlap
        |
        v
Embedder (FastEmbed)
  BAAI/bge-small-en-v1.5
        |
        v
QdrantIndexer
  cosine vectors + source payload
        |
        +----------------------+
        |                      |
        v                      v
query_retrieval.py       query_with_llm.py
semantic results          Groq answer generation
```

## Repository structure

```text
01-hybrid-rag/
|-- .env                         # Local secrets; do not commit
|-- requirements.txt             # Python dependencies
|-- Readme.md                    # Setup and execution instructions
|-- Architecture.md              # This architecture document
|-- main.py                      # Ingest data/ into Qdrant
|-- query_retrieval.py           # Semantic retrieval demonstration
|-- query_with_llm.py            # Retrieval plus Groq generation
|-- data/
|   `-- notes.txt                # Example knowledge-base document
|-- src/
|   `-- ingestion/
|       |-- __init__.py          # Public ingestion API
|       |-- config.py            # Qdrant, model, and chunk settings
|       |-- processor.py         # File validation and extractor routing
|       |-- chunker.py           # Overlapping text chunk creation
|       |-- embedder.py          # FastEmbed wrapper
|       |-- indexer.py            # Qdrant collection and upsert operations
|       `-- extractors/
|           |-- __init__.py      # Extractor registry
|           |-- base.py          # Extracted document abstraction
|           |-- text_extractor.py
|           |-- pdf_extractor.py
|           |-- image_extractor.py
|           |-- docx_extractor.py
|           `-- html_extractor.py
`-- venv/                        # Local virtual environment, not source
```

## Components

### Ingestion

`main.py` calls `IngestionProcessor.process_directory("data")`. The processor
checks file extensions against `SUPPORTED_EXTENSIONS` and delegates extraction
to the registered extractor:

- Text extractor: plain text, Markdown, and `.text`.
- PDF extractor: text extraction through `pdfplumber`.
- Image extractor: grayscale conversion and OCR through Pillow and Tesseract.
- DOCX extractor: paragraphs and tables through `python-docx`.
- HTML extractor: removes non-content tags and parses text with BeautifulSoup
  and `lxml`.

Each extractor returns an `ExtractedDocument` containing content, source path,
file type, and metadata.

### Chunking and embeddings

`chunk_document` calls `chunk_text` with a 500-character chunk size and a
100-character overlap. Each chunk keeps its source path, file type, and chunk
index. `Embedder` creates one 384-dimensional vector per chunk with
`BAAI/bge-small-en-v1.5`.

### Vector storage

`QdrantIndexer` connects to the Qdrant service at `http://localhost:6333` and
creates the `website_knowledge` collection when necessary. Vectors use cosine
distance. Each point stores the following payload:

```text
text, source, type, chunk_index
```

Point IDs are UUIDs, so rerunning ingestion adds new points rather than
replacing points from the same source.

### Retrieval

Both query scripts embed the question with the same FastEmbed model used during
ingestion. They call Qdrant `query_points` against `website_knowledge` and
request payloads with the result. `query_retrieval.py` returns five chunks by
default; `query_with_llm.py` returns three.

### Generation

`query_with_llm.py` formats retrieved chunks as source-tagged context and sends
them to a Groq endpoint through the OpenAI Python client:

```text
https://api.groq.com/openai/v1
```

The `GROQ_API_KEY` environment variable authenticates the request. The system
prompt restricts the answer to retrieved context and asks the model to cite the
source file. The configured model is `openai/gpt-oss-20b`.

## Runtime dependencies

```text
Python application
  |-- FastEmbed + ONNX Runtime: local embeddings
  |-- Qdrant client: vector database communication
  |-- extraction libraries: PDF, DOCX, HTML, and image handling
  |-- OpenAI client: Groq-compatible chat API
  `-- python-dotenv: root .env loading

External services
  |-- Qdrant at localhost:6333
  |-- Groq API for query_with_llm.py
  `-- Tesseract OCR executable for image files
```

## Data flow and operational notes

1. Put supported source files in `data/`.
2. Run `main.py` once to create or update the vector index.
3. Run either query script against the populated collection.
4. Keep the embedding model and vector dimension unchanged between indexing and
   querying.
5. Re-indexing is additive; clear the Qdrant collection manually when a clean
   rebuild is required.

The current implementation is a local command-line pipeline. It does not
provide an HTTP API, authentication layer, document deletion workflow, or
automatic collection cleanup.
