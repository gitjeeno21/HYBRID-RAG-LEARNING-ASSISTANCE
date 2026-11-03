"""Manage PDFs in the active knowledge base."""

from pathlib import Path
from typing import Protocol


class DeletableVectorStore(Protocol):
    """The Chroma operation needed to remove a paper from the index."""

    def delete(self, where: dict[str, str]) -> None: ...


def remove_pdf_from_library(
    filename: str,
    vector_store: DeletableVectorStore,
    pdf_dir: Path,
    removed_pdf_dir: Path,
    keep_copy: bool,
) -> Path | None:
    """Remove a PDF and its indexed chunks, optionally keeping an archive copy."""
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.lower().endswith(".pdf"):
        raise ValueError("Select a PDF with a simple filename ending in .pdf")

    source = pdf_dir / safe_name
    if not source.is_file():
        raise FileNotFoundError(f"{safe_name} is no longer in the active library")

    removed_pdf_dir.mkdir(parents=True, exist_ok=True)
    archived = removed_pdf_dir / safe_name
    version = 2
    while archived.exists():
        archived = removed_pdf_dir / f"{Path(safe_name).stem} ({version}).pdf"
        version += 1

    source.replace(archived)
    try:
        vector_store.delete(where={"filename": safe_name})
    except Exception as exc:
        try:
            archived.replace(source)
        except OSError as restore_error:
            raise RuntimeError(
                f"Could not remove {safe_name} from the index, and its PDF "
                f"could not be restored from {archived}: {restore_error}"
            ) from exc
        raise

    if keep_copy:
        return archived

    try:
        archived.unlink()
    except OSError as exc:
        raise OSError(
            f"Removed {safe_name} from the index, but could not permanently "
            f"delete its PDF. It remains at {archived}: {exc}"
        ) from exc
    return None
