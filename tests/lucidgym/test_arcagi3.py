from __future__ import annotations

import json
from pathlib import Path

import pytest

from lucidgym.agents.arcagi3_agent import ArcAgi3Agent
from lucidgym.environments.arcagi3.arcagi3_env import ArcAgi3Env
from lucidgym.environments.arcagi3.client import ArcAgi3Client
from lucidgym.environments.arcagi3.mocks import StaticArcTransport
from lucidgym.environments.arcagi3.structs import GameAction

ROOT = Path(__file__).resolve().parents[2]
SESSION_PATH = ROOT / "examples" / "arcagi3" / "mock_session.json"


def _load_session() -> dict:
    return json.loads(SESSION_PATH.read_text())


def test_client_cycles_through_mock_frames() -> None:
    session_data = _load_session()
    client = ArcAgi3Client(
        root_url="http://mock.arc.local",
        default_card_id="ARC-demo-card",
        default_game_id="ARC-demo-game",
        transport=StaticArcTransport(session_data),
    )

    frame = client.reset()
    assert frame.score == 0
    assert frame.available_actions[0] is GameAction.ACTION1

    next_frame = client.step(GameAction.ACTION1)
    assert next_frame.score == 1

    winning_frame = client.step(GameAction.ACTION6, payload={"x": 1, "y": 1})
    assert winning_frame.state.name == "WIN"

    scorecard = client.scorecard()
    assert "ARC-demo-game" in scorecard.cards


def test_env_episode_with_mock_transport() -> None:
    session_data = _load_session()
    transport = StaticArcTransport(session_data)
    env = ArcAgi3Env(
        card_id="ARC-demo-card",
        game_id="ARC-demo-game",
        root_url="http://mock.arc.local",
        transport=transport,
    )

    obs, info = env.reset()
    assert obs["score"] == 0
    assert info["arc"]["state"] == "NOT_FINISHED"

    obs, reward, done, info = env.step({"action": "ACTION1"})
    assert reward == pytest.approx(1.0)
    assert not done

    obs, reward, done, info = env.step({"action": "ACTION6", "x": 1, "y": 1})
    assert reward == pytest.approx(2.0)
    assert done
    assert info["arc"]["state"] == "WIN"
    assert "replay_url" in info["arc"]

    env.close()


def test_arcagi3_env_from_dict_loads_mock_path(tmp_path: Path) -> None:
    session_copy = tmp_path / "session.json"
    session_copy.write_text(SESSION_PATH.read_text())

    env = ArcAgi3Env.from_dict(
        {
            "card_id": "ARC-demo-card",
            "game_id": "ARC-demo-game",
            "root_url": "http://mock.arc.local",
            "mock_session_path": str(session_copy),
        }
    )

    obs, _ = env.reset()
    assert obs["game_id"] == "ARC-demo-game"
    env.close()


def test_agent_parses_llm_json() -> None:
    agent = ArcAgi3Agent()
    observation = {
        "card_id": "ARC-demo-card",
        "game_id": "ARC-demo-game",
        "step": 0,
        "score": 0,
        "state": "NOT_FINISHED",
        "grid_ascii": "..\\n##",
        "available_actions": [{"name": "ACTION6", "id": 6, "requires_coordinates": True}],
    }
    agent.update_from_env(observation, reward=0.0, done=False, info={})

    action = agent.update_from_model('{"action": "ACTION6", "x": 1, "y": 2, "reasoning": {"note": "test"}}')
    assert action.action["action"] == "ACTION6"
    assert action.action["x"] == 1
    assert action.action["y"] == 2
    assert "reasoning" in action.action

    with pytest.raises(ValueError):
        agent.update_from_model("{}")
