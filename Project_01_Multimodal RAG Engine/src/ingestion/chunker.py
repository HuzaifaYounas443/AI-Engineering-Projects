"""Split text into overlapping chunks."""
from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .extractors.base import ExtractedDocument


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be < size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_document(doc: ExtractedDocument) -> list[dict]:
    pieces = chunk_text(doc.content)
    return [
        {
            "text": p,
            "source": doc.source_path,
            "type": doc.file_type,
            "chunk_index": i,
        }
        for i, p in enumerate(pieces)
    ]