"""Persist all chunks to disk so BM25 can load them at query time."""
import json
from pathlib import Path


STORE_PATH = Path("chunks_store.json")


def save_chunks(chunks: list[dict]) -> None:
    """Write all chunks to a JSON file."""
    STORE_PATH.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  ✓ Saved {len(chunks)} chunks to {STORE_PATH}")


def load_chunks() -> list[dict]:
    """Read chunks from disk. Raises if the file is missing."""
    if not STORE_PATH.exists():
        raise FileNotFoundError(
            f"{STORE_PATH} not found. Run `python main.py` first to build the index."
        )
    return json.loads(STORE_PATH.read_text(encoding="utf-8"))