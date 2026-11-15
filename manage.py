"""Administration commands for the multi-user app.

    python manage.py users                                   # list accounts and library sizes
    python manage.py import-pdfs --email you@example.com data/*.pdf
    python manage.py reindex --email you@example.com         # rebuild one user's index
    python manage.py reindex --all                           # rebuild every user's index

Accounts are created in the app (Create account). import-pdfs copies PDFs into
an existing account's private library and indexes them, exactly like uploading
them through the app; files already in that library are skipped.

Run these commands while the server is STOPPED (or restart it afterwards):
Chroma caches its vector index in memory per process, so a running server
may not see chunks another process wrote.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import auth  # noqa: E402
import db  # noqa: E402
from ingest import best_chunk_config  # noqa: E402
from upload import add_pdf_to_library  # noqa: E402
from workspace import Workspace  # noqa: E402


def _user(email: str) -> auth.User:
    user = auth.get_user_by_email(email)
    if user is None:
        raise SystemExit(f"No account for {email}. Create it in the app first (Create account).")
    return user


def cmd_users(_: argparse.Namespace) -> None:
    with db.connect() as conn:
        rows = conn.execute("SELECT id, email, name, created_at FROM users ORDER BY id").fetchall()
    if not rows:
        print("No accounts yet. Create one in the app.")
    for row in rows:
        papers = Workspace(row["id"]).pdf_names()
        print(f"{row['id']:>4}  {row['email']:<32} {len(papers):>3} PDF(s)  created {row['created_at']}")


def cmd_import(args: argparse.Namespace) -> None:
    user = _user(args.email)
    ws = Workspace(user.id)
    paths = [Path(p) for pattern in args.paths for p in (glob.glob(pattern) or [pattern])]
    chunk_size, chunk_overlap = best_chunk_config()
    added = skipped = 0
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".pdf":
            print(f"skip  {path} (not a PDF file)")
            skipped += 1
            continue
        if ws.pdf_path(path.name):
            print(f"skip  {path.name} (already in the library)")
            skipped += 1
            continue
        try:
            count = add_pdf_to_library(path.name, path.read_bytes(), ws.store, ws.pdf_dir,
                                       chunk_size, chunk_overlap)
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"FAIL  {path.name}: {exc}")
            skipped += 1
            continue
        print(f"added {path.name} ({count} chunks)")
        added += 1
    ws.invalidate()
    chunks, _ = ws.resources()   # verifies the dense and BM25 indexes match
    print(f"\n{added} added, {skipped} skipped. {user.email} now has "
          f"{len(ws.pdf_names())} PDF(s), {len(chunks)} chunks, index verified.")


def cmd_reindex(args: argparse.Namespace) -> None:
    if args.all:
        with db.connect() as conn:
            ids = [row["id"] for row in conn.execute("SELECT id FROM users ORDER BY id")]
    else:
        ids = [_user(args.email).id]
    for user_id in ids:
        ws = Workspace(user_id)
        count = ws.reindex()
        ws.resources()
        print(f"user {user_id}: {len(ws.pdf_names())} PDF(s), {count} chunks, index verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("users", help="list accounts").set_defaults(run=cmd_users)
    imp = sub.add_parser("import-pdfs", help="copy PDFs into an account's library and index them")
    imp.add_argument("--email", required=True)
    imp.add_argument("paths", nargs="+", help="PDF files or glob patterns")
    imp.set_defaults(run=cmd_import)
    rei = sub.add_parser("reindex", help="rebuild the dense index from the files")
    target = rei.add_mutually_exclusive_group(required=True)
    target.add_argument("--email")
    target.add_argument("--all", action="store_true")
    rei.set_defaults(run=cmd_reindex)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
