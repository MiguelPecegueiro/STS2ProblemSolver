"""Spire Codex monster import and section classification."""

from __future__ import annotations

import json
from pathlib import Path

from sts2_agent.codex_monsters import (
    MONSTER_SECTIONS,
    classify_monster,
    partition_monsters,
    primary_act,
    write_monster_database,
)


def test_classify_by_type() -> None:
    assert classify_monster({"name": "Jaw Worm", "type": "Normal"}) == "normal"
    assert classify_monster({"name": "Skulking Colony", "type": "Elite"}) == "elite"
    assert classify_monster({"name": "Hexaghost", "type": "Boss"}) == "boss"


def test_primary_act_from_encounters() -> None:
    m = {
        "encounters": [
            {"act": "Act 1 - Underdocks", "room_type": "Elite"},
        ]
    }
    assert primary_act(m) == "Act 1 - Underdocks"


def test_partition_writes_all_sections(tmp_path: Path) -> None:
    sample = [
        {
            "id": "A",
            "name": "Zombie",
            "type": "Normal",
            "min_hp": 10,
            "encounters": [{"act": "Act 1 - Overgrowth"}],
        },
        {
            "id": "B",
            "name": "Elite X",
            "type": "Elite",
            "min_hp": 50,
        },
        {
            "id": "C",
            "name": "Boss Y",
            "type": "Boss",
            "min_hp": 200,
        },
    ]
    report = write_monster_database(
        sample, fetched_at="2026-01-01T00:00:00+00:00", root=tmp_path
    )
    assert report.total_monsters == 3
    assert report.section_counts["normal"] == 1
    assert report.section_counts["elite"] == 1
    assert report.section_counts["boss"] == 1
    assert (tmp_path / "index.json").exists()
    elite = json.loads((tmp_path / "by_type" / "elite.json").read_text(encoding="utf-8"))
    assert elite["label"] == "Elite"
    assert elite["monsters"][0]["section"] == "elite"
    assert elite["monsters"][0]["primary_act"] is None


def test_section_files_exist_for_empty_buckets(tmp_path: Path) -> None:
    write_monster_database([], root=tmp_path)
    for key, _ in MONSTER_SECTIONS:
        assert (tmp_path / "by_type" / f"{key}.json").exists()
