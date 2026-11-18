"""PDF filename validation must give the same answer on every operating system.

Found by CI: pathlib treats a backslash as a separator only on Windows, so
"..\\outside.pdf" was rejected on Windows but accepted on Linux.
"""

import unittest

from library import is_safe_pdf_name

BACKSLASH = "\\"


class SafePdfNameTests(unittest.TestCase):
    def test_plain_names_are_accepted(self):
        for name in ("paper.pdf", "My Paper (v2).PDF", "90-Day ML Engineer Roadmap.pdf", "name .pdf"):
            with self.subTest(name=name):
                self.assertTrue(is_safe_pdf_name(name))

    def test_traversal_with_either_separator_is_rejected(self):
        names = ["../outside.pdf", "a/b.pdf", "/etc/a.pdf",
                 ".." + BACKSLASH + "outside.pdf", "a" + BACKSLASH + "b.pdf",
                 "C:" + BACKSLASH + "x" + BACKSLASH + "a.pdf"]
        for name in names:
            with self.subTest(name=name):
                self.assertFalse(is_safe_pdf_name(name))

    def test_hidden_control_and_odd_names_are_rejected(self):
        for name in (".hidden.pdf", "..pdf", ".pdf", "a" + chr(0) + ".pdf", "a" + chr(10) + ".pdf",
                     "x.txt", "trailing.pdf ", "x" * 252 + ".pdf", "", None, 7):
            with self.subTest(name=repr(name)):
                self.assertFalse(is_safe_pdf_name(name))


if __name__ == "__main__":
    unittest.main()
