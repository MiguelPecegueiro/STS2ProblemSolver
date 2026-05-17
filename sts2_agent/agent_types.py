"""Shared types for agent decisions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Decision:
    action: dict | None
    reasons: list[str] = field(default_factory=list)

    @property
    def has_action(self) -> bool:
        return self.action is not None
