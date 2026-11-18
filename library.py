"""Manage PDFs in the active knowledge base."""

import re
from pathlib import Path
from typing import Protocol

# Control characters, NUL and both path separators. WHY both separators:
# pathlib only treats "\\" as a separator on Windows, so "..\\x.pdf" was
# rejected on Windows but accepted as an ordinary name on Linux (found by CI).
_FORBIDDEN = re.compile(r"[\x00-\x1f\x7f/\\]")


def is_safe_pdf_name(name: object) -> bool:
    """True only for a plain PDF filename, judged the same on every OS.

    Rejects separators of either kind, NUL and control characters, names
    starting with a dot (hidden files, "..", "..pdf"), names over 255
    characters, and anything not ending in ".pdf".
    """
    return (isinstance(name, str) and 4 < len(name) <= 255
            and not name.startswith(".") and not name.endswith((" ", "."))
            and name.lower().endswith(".pdf") and not _FORBIDDEN.search(name))


class DeletableVectorStore(Protocol):
    """The Chroma operation needed to remove a paper from the index."""

    def delete(self, where: dict[str, str]) -> None: ...


def rename_pdf_in_library(filename: str, new_filename: str, pdf_dir: Path) -> Path:
    """Rename a PDF and return its new path."""
    old, new = filename, new_filename
    if not (is_safe_pdf_name(old) and is_safe_pdf_name(new)):
        raise ValueError("Use simple .pdf filenames")
    source = pdf_dir / old
    destination = pdf_dir / new
    if not source.is_file():
        raise FileNotFoundError(f"{old} was not found")
    if destination.exists():
        raise FileExistsError(f"{new} already exists")
    source.replace(destination)
    return destination


def remove_pdf_from_library(
    filename: str,
    vector_store: DeletableVectorStore,
    pdf_dir: Path,
    removed_pdf_dir: Path,
    keep_copy: bool,
) -> Path | None:
    """Remove a PDF and its indexed chunks, optionally keeping an archive copy."""
    safe_name = filename
    if not is_safe_pdf_name(safe_name):
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
