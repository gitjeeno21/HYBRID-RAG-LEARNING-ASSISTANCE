"""Cold-start fixes: the on-disk parse cache and the startup warm-up."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

import workspace


def fake_loader(text):
    class Loader:
        calls = 0

        def __init__(self, path):
            pass

        def load(self):
            Loader.calls += 1
            return [Document(page_content=text, metadata={"page": 0, "page_label": "1"})]
    return Loader


class ParseCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.pdf = root / "pdfs" / "paper.pdf"
        self.pdf.parent.mkdir()
        self.pdf.write_bytes(b"%PDF-1.4 version one")
        self.cache = root / "cache"
        workspace._parse_pdf.cache_clear()
        self.addCleanup(workspace._parse_pdf.cache_clear)

    def test_a_restart_reuses_the_parsed_text_from_disk(self):
        loader = fake_loader("hello")
        with patch("workspace.PyPDFLoader", loader):
            first = workspace.load_pdf_pages(self.pdf, self.cache)
            workspace._parse_pdf.cache_clear()              # simulate a new process
            second = workspace.load_pdf_pages(self.pdf, self.cache)
        self.assertEqual(loader.calls, 1)
        self.assertEqual([p.page_content for p in second], ["hello"])
        self.assertEqual(second[0].metadata["filename"], "paper.pdf")
        self.assertEqual(first[0].metadata, second[0].metadata)

    def test_an_edited_pdf_is_parsed_again(self):
        with patch("workspace.PyPDFLoader", fake_loader("old")):
            workspace.load_pdf_pages(self.pdf, self.cache)
        time.sleep(0.01)
        self.pdf.write_bytes(b"%PDF-1.4 version two, longer")
        workspace._parse_pdf.cache_clear()
        with patch("workspace.PyPDFLoader", fake_loader("new")):
            pages = workspace.load_pdf_pages(self.pdf, self.cache)
        self.assertEqual(pages[0].page_content, "new")

    def test_a_corrupt_cache_file_is_ignored(self):
        self.cache.mkdir()
        (self.cache / "paper.pdf.json").write_text("{not json", encoding="utf-8")
        with patch("workspace.PyPDFLoader", fake_loader("fresh")):
            self.assertEqual(workspace.load_pdf_pages(self.pdf, self.cache)[0].page_content, "fresh")

    def test_drop_cache_forgets_a_removed_paper(self):
        ws = workspace.Workspace(1)
        ws.cache_dir = self.cache
        self.cache.mkdir()
        (self.cache / "paper.pdf.json").write_text("{}", encoding="utf-8")
        ws.drop_cache("paper.pdf")
        ws.drop_cache("never-existed.pdf")                 # no error
        self.assertFalse((self.cache / "paper.pdf.json").exists())


class WarmUpTests(unittest.TestCase):
    def test_preload_can_be_disabled(self):
        import main
        from fastapi.testclient import TestClient

        with patch.dict(os.environ, {"APP_PRELOAD": "0"}), patch("workspace.warm_up") as warm:
            with TestClient(main.app):                      # runs the lifespan
                pass
        warm.assert_not_called()

    def test_startup_runs_the_warm_up_in_the_background(self):
        import main
        from fastapi.testclient import TestClient

        with patch.dict(os.environ, {"APP_PRELOAD": "1"}), patch("workspace.warm_up") as warm:
            with TestClient(main.app):
                for _ in range(50):
                    if warm.called:
                        break
                    time.sleep(0.02)
        warm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
