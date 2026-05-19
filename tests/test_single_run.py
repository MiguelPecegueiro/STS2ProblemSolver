"""--single-run exit after one completed run."""

from __future__ import annotations

from unittest.mock import MagicMock

from sts2_agent.graceful_shutdown import SingleRunController
from sts2_agent.menu import MenuFlow


def test_single_run_waits_until_run_started() -> None:
    ctrl = SingleRunController()
    menu = MenuFlow()
    assert not ctrl.should_exit(
        {"state_type": "menu"},
        run_in_progress=False,
        menu_flow=menu,
    )


def test_single_run_exits_after_game_over_and_pipeline_idle() -> None:
    ctrl = SingleRunController()
    menu = MenuFlow()

    ctrl.observe({"state_type": "map"}, run_in_progress=True)
    ctrl.observe({"state_type": "game_over"}, run_in_progress=False)

    pipeline = MagicMock()
    pipeline._run_active = False

    import sts2_agent.data_pipeline as dp

    original = dp.get_pipeline
    dp.get_pipeline = lambda: pipeline  # type: ignore[method-assign]
    try:
        assert ctrl.should_exit(
            {"state_type": "game_over"},
            run_in_progress=False,
            menu_flow=menu,
        )
        assert menu._block_restart is True
    finally:
        dp.get_pipeline = original  # type: ignore[method-assign]
