from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src.core.config import get_settings
from src.database import close_database, init_database
from src.models.material import Material, replace_material_pages


async def migrate(*, dry_run: bool, limit: int | None) -> int:
    settings = get_settings()
    await init_database(settings)
    migrated = 0
    try:
        query = Material.find({"pages.0": {"$exists": True}})
        if limit:
            query = query.limit(limit)
        materials = await query.to_list()
        for material in materials:
            if not material.pages:
                continue
            migrated += 1
            print(f"{'Would migrate' if dry_run else 'Migrating'} {material.id} ({len(material.pages)} pages)")
            if dry_run:
                continue
            await replace_material_pages(material, material.pages)
            material.pages = []
            await material.save()
    finally:
        await close_database()
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Move embedded Material.pages into the material_pages collection.")
    parser.add_argument("--dry-run", action="store_true", help="Print affected materials without writing changes.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of materials to migrate.")
    args = parser.parse_args()
    migrated = asyncio.run(migrate(dry_run=args.dry_run, limit=args.limit))
    print(f"Done. Materials matched: {migrated}")


if __name__ == "__main__":
    main()
