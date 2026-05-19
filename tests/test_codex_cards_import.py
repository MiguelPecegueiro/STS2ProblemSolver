"""Spire Codex card import and section classification."""

from __future__ import annotations

import json
from pathlib import Path

from sts2_agent.codex_cards import (
    CARD_SECTIONS,
    classify_card,
    partition_cards,
    write_card_database,
)


def test_classify_ironclad_and_curse() -> None:
    assert classify_card({"name": "Strike", "color": "ironclad"}) == "ironclad"
    assert classify_card({"name": "Bane", "color": "curse", "id": "ASCENDERS_BANE"}) == "curse"
    assert classify_card({"name": "Alchemize", "color": "colorless"}) == "colorless"
    assert classify_card({"name": "Disintegration", "color": "token"}) == "token"


def test_partition_writes_all_sections(tmp_path: Path) -> None:
    sample = [
        {"id": "A", "name": "Zeta", "color": "ironclad", "type_key": "Attack"},
        {"id": "B", "name": "Alpha", "color": "silent", "type_key": "Skill"},
        {"id": "C", "name": "Curse", "color": "curse", "type_key": "Curse"},
    ]
    report = write_card_database(sample, fetched_at="2026-01-01T00:00:00+00:00", root=tmp_path)
    assert report.total_cards == 3
    assert report.section_counts["ironclad"] == 1
    assert report.section_counts["silent"] == 1
    assert report.section_counts["curse"] == 1
    assert (tmp_path / "index.json").exists()
    iron = json.loads((tmp_path / "by_color" / "ironclad.json").read_text(encoding="utf-8"))
    assert iron["label"] == "Ironclad"
    assert iron["cards"][0]["name"] == "Zeta"
    assert iron["cards"][0]["section"] == "ironclad"


def test_section_files_exist_for_empty_buckets(tmp_path: Path) -> None:
    write_card_database([], root=tmp_path)
    for key, _ in CARD_SECTIONS:
        assert (tmp_path / "by_color" / f"{key}.json").exists()
