"""Menu flow - abandon stale continue/abandon main menu."""

from sts2_agent.menu import MenuFlow


def test_abandon_stale_main_menu():
    flow = MenuFlow(abandon_stale_runs=True)
    state = {
        "state_type": "menu",
        "menu_screen": "main",
        "options": ["continue", "abandon_run", "multiplayer", "quit"],
    }
    action, reasons = flow.decide(state)
    assert action == {"action": "menu_select", "option": "abandon_run"}
    assert flow._awaiting_abandon_confirm
    assert any("abandon stale" in r for r in reasons)


def test_abandon_confirm_yes_then_singleplayer():
    flow = MenuFlow(abandon_stale_runs=True)
    flow._awaiting_abandon_confirm = True
    popup = {
        "state_type": "menu",
        "menu_screen": "popup",
        "options": ["yes", "no"],
    }
    action, reasons = flow.decide(popup)
    assert action == {"action": "menu_select", "option": "yes"}
    assert not flow._awaiting_abandon_confirm
    assert any("confirm abandon" in r for r in reasons)

    main = {
        "state_type": "menu",
        "menu_screen": "main",
        "options": ["singleplayer", "multiplayer", "quit"],
    }
    action, _ = flow.decide(main)
    assert action == {"action": "menu_select", "option": "singleplayer"}


def test_no_abandon_loop_while_awaiting_confirm():
    flow = MenuFlow(abandon_stale_runs=True)
    flow._awaiting_abandon_confirm = True
    state = {
        "state_type": "menu",
        "menu_screen": "main",
        "options": ["continue", "abandon_run", "multiplayer", "quit"],
    }
    action, reasons = flow.decide(state)
    assert action is None
    assert any("waiting for abandon confirm" in r for r in reasons)


def test_no_abandon_when_singleplayer_available():
    flow = MenuFlow(abandon_stale_runs=True)
    state = {
        "state_type": "menu",
        "menu_screen": "main",
        "options": ["singleplayer", "multiplayer", "quit"],
    }
    action, _ = flow.decide(state)
    assert action == {"action": "menu_select", "option": "singleplayer"}


def test_stuck_without_flag():
    flow = MenuFlow(abandon_stale_runs=False)
    state = {
        "state_type": "menu",
        "menu_screen": "main",
        "options": ["continue", "abandon_run", "quit"],
    }
    action, reasons = flow.decide(state)
    assert action is None
    assert any("unhandled" in r or "waiting" in r for r in reasons)
