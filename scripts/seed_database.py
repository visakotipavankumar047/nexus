"""Apply app/database/migrations/*.sql (in name order) to DATABASE_URL.

    python scripts/seed_database.py [--dry-run]

Migrations use IF NOT EXISTS, so re-running is safe.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import inspect  # noqa: E402

from app.database.models import Base  # noqa: E402
from app.database.supabase import get_engine  # noqa: E402

MIGRATIONS = ROOT / "app" / "database" / "migrations"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="list migrations without running them")
    args = p.parse_args()

    files = sorted(MIGRATIONS.glob("*.sql"))
    for f in files:
        print(("would apply " if args.dry_run else "applying ") + f.name)
        if not args.dry_run:
            with get_engine().begin() as conn:  # one transaction per file
                conn.exec_driver_sql(f.read_text(encoding="utf-8"))

    if not args.dry_run:
        missing = set(Base.metadata.tables) - set(inspect(get_engine()).get_table_names())
        print("schema OK: all tables present" if not missing else f"MISSING tables: {sorted(missing)}")
        sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
