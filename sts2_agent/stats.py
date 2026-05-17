"""Read training run summaries and print aggregate stats."""



from __future__ import annotations



import json

from collections import Counter, defaultdict

from pathlib import Path



PACKAGE_DIR = Path(__file__).resolve().parent

RUNS_PATH = PACKAGE_DIR.parent / "data" / "runs.jsonl"





def load_runs() -> list[dict]:

    if not RUNS_PATH.exists():

        return []

    runs: list[dict] = []

    try:

        with RUNS_PATH.open(encoding="utf-8") as fh:

            for line in fh:

                line = line.strip()

                if not line:

                    continue

                try:

                    runs.append(json.loads(line))

                except json.JSONDecodeError:

                    continue

    except OSError:

        return []

    return runs





def _run_source(r: dict) -> str:

    return str(r.get("source") or "agent").lower()





DEFAULT_AGENT_VERSION = "rules_v1"


def _agent_version(r: dict) -> str:
    return str(r.get("agent_version") or DEFAULT_AGENT_VERSION)





def _run_score(r: dict) -> float:

    if r.get("run_score") is not None:

        try:

            return float(r["run_score"])

        except (TypeError, ValueError):

            pass

    return float(r.get("reward") or 0.0)





def _hp_conservation_pct(r: dict) -> float:

    val = r.get("avg_hp_pct_after_combat")

    if val is not None:

        try:

            return float(val) * 100.0

        except (TypeError, ValueError):

            pass

    return 0.0





def _score_trend_last_n(runs: list[dict], n: int = 10) -> str:

    if not runs:

        return "-"

    recent = runs[-n:]

    scores = [_run_score(r) for r in recent]

    if len(scores) < 2:

        return f"{scores[0]:.0f}" if scores else "-"

    first_half = scores[: len(scores) // 2]

    second_half = scores[len(scores) // 2 :]

    avg_first = sum(first_half) / len(first_half)

    avg_second = sum(second_half) / len(second_half)

    delta = avg_second - avg_first

    arrow = "up" if delta > 5 else "down" if delta < -5 else "flat"

    return f"{avg_second:.0f} avg ({arrow} vs prior {len(first_half)})"





def _summarize_group(runs: list[dict], label: str) -> None:

    if not runs:

        print(f"\n{label}: no runs")

        return



    total = len(runs)

    wins = sum(1 for r in runs if r.get("won"))

    floors = [int(r.get("floors_reached") or 0) for r in runs]

    scores = [_run_score(r) for r in runs]

    hp_conservation = [_hp_conservation_pct(r) for r in runs]



    win_rate = (wins / total * 100) if total else 0

    avg_floor = sum(floors) / total if total else 0

    avg_score = sum(scores) / total if total else 0

    avg_hp = sum(hp_conservation) / total if total else 0



    print(f"\n{label} ({total} runs)")

    print(f"  Win rate:            {win_rate:.1f}% ({wins}/{total})")

    print(f"  Avg floor reached:   {avg_floor:.1f}")

    print(f"  Avg run score:       {avg_score:.1f}")

    print(f"  Avg HP conservation: {avg_hp:.1f}% after combat")

    if total >= 10:

        print(f"  Score trend (last 10): {_score_trend_last_n(runs, 10)}")





def _group_agent_by_version(agent_runs: list[dict]) -> dict[str, list[dict]]:

    by_ver: dict[str, list[dict]] = defaultdict(list)

    for r in agent_runs:

        by_ver[_agent_version(r)].append(r)

    return dict(sorted(by_ver.items()))





def _print_version_comparison_table(by_version: dict[str, list[dict]]) -> None:

    if len(by_version) < 2:

        return



    print("\nAgent version comparison")

    width = max(20, max(len(ver) for ver in by_version) + 2)
    print(f"  {'Version':<{width}} {'Runs':>6} {'Win%':>8} {'Avg floor':>10} {'Avg score':>10}")
    print("  " + "-" * (width + 38))

    for ver, subset in by_version.items():

        total = len(subset)

        wins = sum(1 for r in subset if r.get("won"))

        wr = (wins / total * 100) if total else 0

        avg_floor = sum(int(r.get("floors_reached") or 0) for r in subset) / total if total else 0

        avg_score = sum(_run_score(r) for r in subset) / total if total else 0

        print(f"  {ver:<{width}} {total:>6} {wr:>7.1f}% {avg_floor:>10.1f} {avg_score:>10.1f}")





def _print_version_death_causes(by_version: dict[str, list[dict]]) -> None:

    for ver, subset in by_version.items():

        losses = [r for r in subset if not r.get("won")]

        if not losses:

            continue

        causes = Counter(str(r.get("cause_of_death") or "unknown") for r in losses)

        print(f"\n  Top deaths ({ver}):")

        for cause, count in causes.most_common(3):

            print(f"    ({count}x) {cause[:72]}")





def print_summary() -> None:

    runs = load_runs()

    if not runs:

        print(f"No runs found at {RUNS_PATH}")

        return



    human_runs = [r for r in runs if _run_source(r) == "human"]

    agent_runs = [r for r in runs if _run_source(r) != "human"]

    by_version = _group_agent_by_version(agent_runs)



    print("STS2 - Training Run Summary")

    print("=" * 40)

    print(f"Total runs in file: {len(runs)}")

    print(f"  Human imports:      {len(human_runs)}")

    print(f"  Agent runs:         {len(agent_runs)}")

    if by_version:

        for ver, subset in by_version.items():

            print(f"    {ver:<22} {len(subset)}")



    _summarize_group(human_runs, "Human gameplay")

    _summarize_group(agent_runs, "Agent (all)")



    for ver, subset in by_version.items():

        _summarize_group(subset, f"Agent - {ver}")



    _print_version_comparison_table(by_version)

    _print_version_death_causes(by_version)



    if human_runs and agent_runs:

        h_wr = sum(1 for r in human_runs if r.get("won")) / len(human_runs) * 100

        a_wr = sum(1 for r in agent_runs if r.get("won")) / len(agent_runs) * 100

        h_floor = sum(int(r.get("floors_reached") or 0) for r in human_runs) / len(human_runs)

        a_floor = sum(int(r.get("floors_reached") or 0) for r in agent_runs) / len(agent_runs)

        print("\nHuman vs agent (all versions)")

        print(f"  Win rate:      {h_wr:.1f}% human  |  {a_wr:.1f}% agent")

        print(f"  Avg floor:     {h_floor:.1f} human  |  {a_floor:.1f} agent")



    all_for_causes = agent_runs if agent_runs else runs

    causes = Counter(

        str(r.get("cause_of_death") or "unknown")

        for r in all_for_causes

        if not r.get("won")

    )

    if causes:

        print("\nMost common causes of death (all agent runs):")

        for cause, count in causes.most_common(5):

            print(f"  ({count}x) {cause[:80]}")



    pool = agent_runs or runs

    best_score = max(pool, key=_run_score)

    best_score_val = _run_score(best_score)

    print("\nBest run by score (agent, any version):")

    print(f"  Run ID:        {best_score.get('run_id')}")

    print(f"  Agent version: {_agent_version(best_score)}")

    print(f"  Character:     {best_score.get('character')}")

    print(f"  Run score:     {best_score_val:.0f}")

    print(f"  Floors:        {best_score.get('floors_reached')}")

    print(f"  Won:           {best_score.get('won')}")



    for ver, subset in by_version.items():

        if not subset:

            continue

        best_ver = max(subset, key=_run_score)

        print(f"\nBest run by score (agent - {ver}):")

        print(f"  Run ID:      {best_ver.get('run_id')}")

        print(f"  Character:   {best_ver.get('character')}")

        print(f"  Run score:   {_run_score(best_ver):.0f}")

        print(f"  Floors:      {best_ver.get('floors_reached')}")

        print(f"  Won:         {best_ver.get('won')}")



    if human_runs:

        best_human = max(human_runs, key=_run_score)

        print("\nBest run by score (human):")

        print(f"  Run ID:      {best_human.get('run_id')}")

        print(f"  Character:   {best_human.get('character')}")

        print(f"  Run score:   {_run_score(best_human):.0f}")

        print(f"  Floors:      {best_human.get('floors_reached')}")

        print(f"  Won:         {best_human.get('won')}")





def main() -> int:

    print_summary()

    return 0





if __name__ == "__main__":

    raise SystemExit(main())


