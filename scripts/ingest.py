"""Ingest files/folders and URLs through the same pipeline as POST /api/v1/documents/upload.

    python scripts/ingest.py                       # everything in data/documents
    python scripts/ingest.py data/sample           # a folder
    python scripts/ingest.py report.pdf --url https://example.com/post
"""
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.rag import ingest  # noqa: E402
from app.rag.loaders import file_type  # noqa: E402
from app.services.conversation_service import USER_ID  # noqa: E402


def collect(paths: list[str]) -> list[Path]:
    allowed = get_settings().allowed_upload_types
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        candidates = sorted(p.rglob("*")) if p.is_dir() else [p]
        files += [f for f in candidates if f.is_file() and file_type(f.name) in allowed]
    return files


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help="files or folders (default: data/documents)")
    p.add_argument("--url", action="append", default=[], help="URL to ingest (repeatable)")
    args = p.parse_args()

    files = collect(args.paths or ([] if args.url else [str(ROOT / "data" / "documents")]))
    if not files and not args.url:
        print("nothing to ingest")
        return

    limit = get_settings().max_upload_mb * 1024 * 1024
    failed = 0
    for item in [*files, *args.url]:
        name = item.name if isinstance(item, Path) else item[:200]
        if isinstance(item, Path) and item.stat().st_size > limit:
            print(f"SKIP   {name}: larger than {get_settings().max_upload_mb} MB")
            continue
        try:
            if isinstance(item, Path):
                doc = await ingest(user_id=USER_ID, filename=name, data=item.read_bytes())
            else:
                doc = await ingest(user_id=USER_ID, filename=name, url=item)
            print(f"{doc.status.upper():7}{name}  id={doc.id} chunks={doc.metadata.get('chunks')}")
        except Exception as e:  # keep going; report every failure
            failed += 1
            print(f"FAILED {name}: {getattr(e, 'message', e)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
