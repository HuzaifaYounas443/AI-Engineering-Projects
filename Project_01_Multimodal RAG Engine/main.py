"""Ingest all files from a directory into Qdrant."""
from pathlib import Path

from src.ingestion import (
    Embedder,
    IngestionProcessor,
    QdrantIndexer,
    chunk_document,
)


def main(data_dir: str = "data") -> None:
    source = Path(data_dir)
    if not source.exists():
        raise SystemExit(f"Data directory not found: {source}")

    print(f"▶ Extracting from {source} ...")
    docs = IngestionProcessor().process_directory(source)
    print(f"  ✓ {len(docs)} documents extracted")

    if not docs:
        raise SystemExit("No supported files found.")

    print("\n▶ Chunking ...")
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    print(f"  ✓ {len(all_chunks)} chunks")

    print("\n▶ Embedding ...")
    embedder = Embedder()
    vectors = embedder.embed([c["text"] for c in all_chunks])
    print(f"  ✓ {len(vectors)} vectors (dim={len(vectors[0])})")

    print("\n▶ Indexing into Qdrant ...")
    indexer = QdrantIndexer()
    written = indexer.upsert(all_chunks, vectors)
    print(f"  ✓ {written} points written")

    print(f"\n✅ Done. {len(docs)} files → {written} chunks.")
    print("   Dashboard: http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()