import pytest

from lucidgym import register_lucidgym_components
from lucidgym.environments.textarena_env import TextArenaEnv


pytest.importorskip("textarena")


def test_lucidgym_countdown_board_is_tuple_based() -> None:
    """TextArenaEnv should see tuple observations without the progress suffix."""
    register_lucidgym_components()
    env = TextArenaEnv(env_id="LucidCountdown-v0-raw", num_players=1)
    observation, _ = env.reset()

    raw_messages = observation["raw_observation"]
    assert isinstance(raw_messages, list) and raw_messages, "Expected tuple observations from TextArena."
    assert all(isinstance(entry, tuple) and len(entry) == 3 for entry in raw_messages)

    board_text = [msg["text"] for msg in observation["messages"] if msg["type"] == "GAME_BOARD"]
    assert board_text, "Countdown should emit at least one GAME_BOARD message on reset."
    assert all("Current progress score" not in text for text in board_text)
    env.close()
