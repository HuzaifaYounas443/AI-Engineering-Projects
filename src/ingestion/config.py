"""Central settings for the ingestion pipeline."""
from pathlib import Path

# Qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "website_knowledge"
VECTOR_DIM = 384

# Embeddings
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# File type routing
SUPPORTED_EXTENSIONS = {
    ".txt": "text", ".md": "text", ".text": "text",
    ".pdf": "pdf",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".docx": "docx",
    ".html": "html", ".htm": "html",
}