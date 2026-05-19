#!/usr/bin/env python3
"""Download Spire Codex cards and write sectioned files under data/cards/."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sts2_agent.codex_cards import CARD_SECTIONS, CARDS_ROOT, import_codex_cards

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Spire Codex cards by color section")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if index.json is less than 24h old",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CARDS_ROOT,
        help=f"Output root (default: {CARDS_ROOT})",
    )
    args = parser.parse_args()

    try:
        report = import_codex_cards(force=args.force, root=args.out)
    except Exception as exc:
        logger.error("Import failed: %s", exc)
        return 1

    print()
    print("=" * 60)
    print("SPIRE CODEX CARD IMPORT")
    print("=" * 60)
    print(f"Fetched at:   {report.fetched_at}")
    print(f"Total cards:  {report.total_cards}")
    print(f"Index:        {report.index_path}")
    print()
    print("Sections:")
    for key, label in CARD_SECTIONS:
        count = report.section_counts.get(key, 0)
        print(f"  {label:14} ({key:12})  {count:4} cards  ->  by_color/{key}.json")
    if report.unmapped:
        print()
        print(f"Unmapped: {len(report.unmapped)}")
        for line in report.unmapped[:10]:
            print(f"  - {line}")
        if len(report.unmapped) > 10:
            print(f"  ... and {len(report.unmapped) - 10} more (see index.json)")
    elif not report.wrote_paths:
        print()
        print("(Used existing cache — pass --force to re-download)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
