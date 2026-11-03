"""Offline tests for removing PDFs from the active knowledge base."""

import tempfile
import unittest
from pathlib import Path

from library import remove_pdf_from_library


class FakeVectorStore:
    def __init__(self, fail_delete: bool = False) -> None:
        self.deleted_filters: list[dict[str, str]] = []
        self.fail_delete = fail_delete

    def delete(self, where: dict[str, str]) -> None:
        self.deleted_filters.append(where)
        if self.fail_delete:
            raise RuntimeError("index unavailable")


class PdfRemovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.pdf_dir = self.root / "data"
        self.removed_pdf_dir = self.root / "removed_pdfs"
        self.pdf_dir.mkdir()
        (self.pdf_dir / "paper.pdf").write_bytes(b"%PDF-1.7 paper")
        self.store = FakeVectorStore()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_archives_pdf_and_removes_its_indexed_chunks(self) -> None:
        archived = remove_pdf_from_library(
            "paper.pdf",
            self.store,
            self.pdf_dir,
            self.removed_pdf_dir,
            keep_copy=True,
        )

        self.assertEqual(archived, self.removed_pdf_dir / "paper.pdf")
        self.assertEqual(archived.read_bytes(), b"%PDF-1.7 paper")
        self.assertFalse((self.pdf_dir / "paper.pdf").exists())
        self.assertEqual(self.store.deleted_filters, [{"filename": "paper.pdf"}])

    def test_permanently_removes_pdf_and_its_indexed_chunks(self) -> None:
        archived = remove_pdf_from_library(
            "paper.pdf",
            self.store,
            self.pdf_dir,
            self.removed_pdf_dir,
            keep_copy=False,
        )

        self.assertIsNone(archived)
        self.assertFalse((self.pdf_dir / "paper.pdf").exists())
        self.assertFalse((self.removed_pdf_dir / "paper.pdf").exists())
        self.assertEqual(self.store.deleted_filters, [{"filename": "paper.pdf"}])

    def test_restores_pdf_when_index_removal_fails(self) -> None:
        self.store.fail_delete = True

        with self.assertRaisesRegex(RuntimeError, "index unavailable"):
            remove_pdf_from_library(
                "paper.pdf",
                self.store,
                self.pdf_dir,
                self.removed_pdf_dir,
                keep_copy=True,
            )

        self.assertTrue((self.pdf_dir / "paper.pdf").exists())
        self.assertFalse((self.removed_pdf_dir / "paper.pdf").exists())

    def test_archived_name_collision_keeps_both_pdf_copies(self) -> None:
        self.removed_pdf_dir.mkdir()
        (self.removed_pdf_dir / "paper.pdf").write_bytes(b"%PDF-1.7 older")

        archived = remove_pdf_from_library(
            "paper.pdf",
            self.store,
            self.pdf_dir,
            self.removed_pdf_dir,
            keep_copy=True,
        )

        self.assertEqual(archived, self.removed_pdf_dir / "paper (2).pdf")
        self.assertEqual(archived.read_bytes(), b"%PDF-1.7 paper")
        self.assertEqual(
            (self.removed_pdf_dir / "paper.pdf").read_bytes(),
            b"%PDF-1.7 older",
        )

    def test_rejects_paths_and_non_pdf_names(self) -> None:
        for filename in ("..\\outside.pdf", "paper.txt"):
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    remove_pdf_from_library(
                        filename,
                        self.store,
                        self.pdf_dir,
                        self.removed_pdf_dir,
                        keep_copy=True,
                    )


if __name__ == "__main__":
    unittest.main()
