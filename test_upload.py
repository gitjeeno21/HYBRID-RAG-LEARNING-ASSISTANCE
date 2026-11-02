"""Offline tests for uploading and indexing PDFs through the app."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from upload import add_pdf_to_library


class FakeVectorStore:
    def __init__(self) -> None:
        self.added_documents: list[Document] = []
        self.deleted_filters: list[dict[str, str]] = []

    def add_documents(self, documents: list[Document]) -> None:
        self.added_documents.extend(documents)

    def delete(self, where: dict[str, str]) -> None:
        self.deleted_filters.append(where)


class PdfUploadTests(unittest.TestCase):
    def test_pdf_is_saved_and_indexed_with_final_citation_metadata(self) -> None:
        pdf_bytes = b"%PDF-1.7 mock pdf bytes"
        store = FakeVectorStore()
        page = Document(
            page_content="A paper page about attention.",
            metadata={"page": 0},
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_dir = Path(temporary_directory)
            with (
                patch("upload.PyPDFLoader") as loader,
                patch("upload.split_documents", return_value=[page]),
            ):
                loader.return_value.load.return_value = [page]
                chunks_added = add_pdf_to_library(
                    "new_paper.pdf",
                    pdf_bytes,
                    store,
                    pdf_dir,
                    chunk_size=1500,
                    chunk_overlap=50,
                )

            self.assertEqual(chunks_added, 1)
            self.assertEqual((pdf_dir / "new_paper.pdf").read_bytes(), pdf_bytes)
            self.assertEqual(len(store.added_documents), 1)
            self.assertEqual(
                store.added_documents[0].metadata["filename"],
                "new_paper.pdf",
            )
            self.assertEqual(
                store.added_documents[0].metadata["source"],
                str(pdf_dir / "new_paper.pdf"),
            )
            self.assertEqual(store.deleted_filters, [])

    def test_rejects_non_pdf_and_path_traversal_before_writing(self) -> None:
        store = FakeVectorStore()
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_dir = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "valid PDF header"):
                add_pdf_to_library(
                    "paper.pdf",
                    b"not a pdf",
                    store,
                    pdf_dir,
                    chunk_size=1500,
                    chunk_overlap=50,
                )
            with self.assertRaisesRegex(ValueError, "simple filename"):
                add_pdf_to_library(
                    "..\\outside.pdf",
                    b"%PDF-1.7",
                    store,
                    pdf_dir,
                    chunk_size=1500,
                    chunk_overlap=50,
                )
            self.assertEqual(list(pdf_dir.iterdir()), [])

    def test_refuses_to_overwrite_an_existing_pdf(self) -> None:
        store = FakeVectorStore()
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_dir = Path(temporary_directory)
            existing_pdf = pdf_dir / "paper.pdf"
            existing_pdf.write_bytes(b"%PDF-1.7 existing")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                add_pdf_to_library(
                    "paper.pdf",
                    b"%PDF-1.7 replacement",
                    store,
                    pdf_dir,
                    chunk_size=1500,
                    chunk_overlap=50,
                )
            self.assertEqual(existing_pdf.read_bytes(), b"%PDF-1.7 existing")


if __name__ == "__main__":
    unittest.main()
