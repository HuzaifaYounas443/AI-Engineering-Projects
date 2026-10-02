"""Router — picks the extractor based on file extension."""
from pathlib import Path

from .config import SUPPORTED_EXTENSIONS
from .extractors import EXTRACTOR_REGISTRY
from .extractors.base import ExtractedDocument


class UnsupportedFileTypeError(Exception):
    pass


class IngestionProcessor:
    """Routes files to the correct extractor."""

    def process(self, file_path: str | Path) -> ExtractedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        file_type = SUPPORTED_EXTENSIONS.get(ext)
        if file_type is None:
            raise UnsupportedFileTypeError(
                f"Unsupported: {ext}. Supported: {list(SUPPORTED_EXTENSIONS.keys())}"
            )

        extractor = EXTRACTOR_REGISTRY[file_type]
        return extractor.extract(path)

    def process_directory(self, dir_path: str | Path) -> list[ExtractedDocument]:
        directory = Path(dir_path)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        results = []
        for f in sorted(directory.iterdir()):
            if not f.is_file() or f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                results.append(self.process(f))
            except Exception as e:
                print(f"  ⚠ Skipping {f.name}: {e}")
        return results