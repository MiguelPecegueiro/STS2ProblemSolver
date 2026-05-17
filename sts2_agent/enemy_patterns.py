"""Enemy patterns - learned compendium only (see enemy_compendium.py)."""

from sts2_agent.enemy_compendium import (  # noqa: F401
    LearnedCompendiumKB,
    ResolvedIntent,
    assess_combat_debuff_pressure,
    begin_combat_observation,
    clear_combat_history,
    enrich_incoming_damage,
    finalize_combat_observation,
    get_compendium_kb,
    get_enemy_pattern_kb,
    group_compendium_by_encounter,
    last_resolved_intents,
    move_incoming_damage,
    player_damage_taken_multiplier,
    predict_enemy_move,
    record_enemy_intents_from_state,
    reload_compendium,
    resolve_enemy_intent,
)
