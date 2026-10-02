# Hybrid RAG

This project implements a small retrieval-augmented generation (RAG) pipeline for
website and policy content. It extracts text from supported files, splits the text
into overlapping chunks, creates vector embeddings with FastEmbed, stores the
vectors in Qdrant, and optionally uses Groq's OpenAI-compatible API to generate an
answer grounded in the retrieved context.

## Features

- Ingests `.txt`, `.md`, `.text`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.docx`, `.html`,
  and `.htm` files.
- Uses OCR through Tesseract for image files.
- Uses `BAAI/bge-small-en-v1.5` embeddings with 384-dimensional vectors.
- Stores vectors and source metadata in the `website_knowledge` Qdrant collection.
- Supports semantic search without an LLM and RAG answers with a Groq-hosted model.

## Requirements

- Python 3.10 or newer.
- A running Qdrant instance at `http://localhost:6333`.
- A Groq API key for `query_with_llm.py`.
- Tesseract OCR installed at
  `C:\Program Files\Tesseract-OCR\tesseract.exe` when image ingestion is needed.

## Installation on Windows

From the project directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell execution policy prevents activation, run the commands with the
interpreter directly instead:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start Qdrant with Docker:

```powershell
docker run --name hybrid-rag-qdrant -p 6333:6333 qdrant/qdrant
```

If the container already exists, start it with:

```powershell
docker start hybrid-rag-qdrant
```

## Configuration

Create a `.env` file in the project root for the LLM example:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

`src/ingestion/config.py` contains the current local configuration:

- Qdrant URL: `http://localhost:6333`
- Collection: `website_knowledge`
- Embedding model: `BAAI/bge-small-en-v1.5`
- Chunk size: 500 characters
- Chunk overlap: 100 characters

The first embedding run downloads the FastEmbed model.

## How to run

### 1. Index the documents

Place source files in `data/`, then run:

```powershell
python main.py
```

This extracts supported files, chunks their content, creates embeddings, and
upserts the chunks into Qdrant. The Qdrant dashboard is available at
<http://localhost:6333/dashboard>.

To index another directory:

```powershell
python -c "from main import main; main('path/to/data')"
```

### 2. Run semantic retrieval only

```powershell
python query_retrieval.py
```

This embeds the sample question, retrieves the five most similar chunks, and
prints their scores, sources, and text. Change the `question` value in the
`__main__` block to search for a different question.

### 3. Run retrieval with LLM generation

Make sure `.env` contains `GROQ_API_KEY`, then run:

```powershell
python query_with_llm.py
```

The script retrieves the three most relevant chunks and sends only that context
to the Groq OpenAI-compatible endpoint using the `openai/gpt-oss-20b` model. The
generated response is instructed to cite its source file. Change the `question`
value in the `__main__` block to ask a different question.

## Troubleshooting

- **Connection refused by Qdrant:** start Qdrant and verify
  <http://localhost:6333/dashboard> is reachable.
- **Missing `GROQ_API_KEY`:** add the key to the root `.env` file before running
  `query_with_llm.py`.
- **OCR errors:** install Tesseract OCR at the path configured in
  `src/ingestion/extractors/image_extractor.py`.
- **No supported files found:** verify that the input directory contains one of
  the extensions listed above.
- **Empty or stale results:** rerun `python main.py` after changing files. Existing
  points are not automatically removed from the Qdrant collection.

## License and data

Do not commit `.env` or API keys. The files in `data/` are treated as the
knowledge base and are embedded into the local Qdrant collection.
