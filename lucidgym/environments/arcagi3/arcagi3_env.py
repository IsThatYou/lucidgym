"""
ARC-AGI-3 environment wrapper using the arc_agi toolkit.

This adapter provides a Gym-compatible interface for ARC-AGI-3 games
using the official arc_agi toolkit for all backend communication.
"""
from __future__ import annotations

from typing import Any, Mapping

import arc_agi
from arc_agi import OperationMode
from arcengine import FrameDataRaw, GameAction, GameState

from rllm.environments.base.base_env import BaseEnv


class ArcAgi3Env(BaseEnv):
    """
    Wraps the ARC-AGI toolkit with Gym-compatible reset/step mechanics.
    """

    def __init__(
        self,
        game_id: str,
        max_actions: int = 80,
        reward_mode: str = "binary",
        reward_scale: float = 1.0,
        # Arcade pass-through args
        arc_api_key: str = "",
        arc_base_url: str = "https://three.arcprize.org",
        operation_mode: OperationMode = OperationMode.NORMAL,
    ) -> None:
        self.game_id = game_id
        self.max_actions = max_actions
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self._arc = arc_agi.Arcade(
            arc_api_key=arc_api_key,
            arc_base_url=arc_base_url,
            operation_mode=operation_mode,
        )
        self._env = None
        self._actions_taken = 0
        self._last_obs: FrameDataRaw | None = None

    def reset(self, task: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and return the initial observation."""
        """This is different from the GameAction Reset method. We open a scorecard here and pass it to the environment."""
        game_id = task.get("game_id", self.game_id)
        tags = task.get("tags", [])
        self._scorecard_id = self.open_scorecard(tags=tags)
        self._env = self._arc.make(game_id, scorecard_id=self._scorecard_id)
        obs = self._env.reset()
        self._last_obs = obs
        observation = self._format_observation(obs)
        return observation

    def step(self, action_payload: Any) -> tuple[dict, float, bool]:
        """Take a step in the environment."""
        if self._env is None or self._last_obs is None:
            raise RuntimeError("ArcAgi3Env.step called before reset.")

        action, payload, reasoning = self._coerce_action(action_payload)
        obs = self._env.step(action, data=payload, reasoning=reasoning[:10000])
        self._actions_taken += 1
        self._last_obs = obs
        reward = self._compute_reward(obs)
        done = obs.state in (GameState.WIN, GameState.GAME_OVER) or self._actions_taken >= self.max_actions
        observation = self._format_observation(obs)
        return observation, reward, done

    def close(self) -> None:
        """Close the environment and scorecard."""
        self.close_scorecard(self._scorecard_id)
        self._env = None
        self._last_obs = None
        self._actions_taken = 0

    def open_scorecard(self, tags: list[str] | None = None) -> str:
        """Open a new scorecard."""
        return self._arc.open_scorecard(tags=tags)

    def close_scorecard(self, card_id: str | None = None):
        """Close a scorecard by ID."""
        return self._arc.close_scorecard(card_id)

    def get_scorecard(self) -> str:
        """Get the scorecard ID."""
        return self._arc.get_scorecard(self._scorecard_id)

    def _format_observation(self, obs: FrameDataRaw) -> dict[str, Any]:
        """Format FrameDataRaw into observation dict."""
        return {
            "game_id": obs.game_id,
            "state": obs.state.name,
            "score": obs.levels_completed,
            "frame": [layer.tolist() if hasattr(layer, "tolist") else layer for layer in obs.frame],
            "available_actions": obs.available_actions,
        }

    def _coerce_action(self, action_payload: Any) -> tuple[GameAction, dict[str, Any], Any | None]:
        """Convert action payload into GameAction, data dict, and reasoning."""
        if isinstance(action_payload, Mapping):
            action = action_payload.get("action")
            reasoning = action_payload.get("reasoning")
            payload = {k: v for k, v in action_payload.items() if k not in {"action", "reasoning"}}
            return action, payload, reasoning
        raise TypeError(f"Unsupported action payload type: {type(action_payload)}")

    def _compute_reward(self, obs: FrameDataRaw) -> float:
        """Compute reward from observation."""
        if self.reward_mode == "score":
            base = obs.levels_completed
        elif self.reward_mode == "binary":
            base = 1.0 if obs.state == GameState.WIN else 0.0
        else:
            base = 0.0
        return float(base) * float(self.reward_scale)

    @staticmethod
    def from_dict(info: dict) -> "ArcAgi3Env":
        """Create an ArcAgi3Env instance from a dictionary.

        Args:
            info: A dictionary containing environment configuration.
                Required keys:
                    - game_id: The game identifier (e.g., "ls20")
                Optional keys:
                    - max_actions: Maximum actions per episode (default: 80)
                    - reward_mode: "binary" or "score" (default: "binary")
                    - reward_scale: Reward multiplier (default: 1.0)
                    - arc_api_key: API key for arc_agi (default: "")
                    - arc_base_url: Base URL for arc_agi (default: "https://three.arcprize.org")
                    - operation_mode: OperationMode enum value (default: OperationMode.NORMAL)

        Returns:
            An initialized ArcAgi3Env instance.
        """
        game_id = info.get("game_id")
        if not game_id:
            raise ValueError("'game_id' is required in info dict")

        # Handle operation_mode conversion from string if needed
        operation_mode = info.get("operation_mode", OperationMode.NORMAL)
        if isinstance(operation_mode, str):
            operation_mode = OperationMode[operation_mode.upper()]

        return ArcAgi3Env(
            game_id=game_id,
            max_actions=info.get("max_actions", 80),
            reward_mode=info.get("reward_mode", "binary"),
            reward_scale=info.get("reward_scale", 1.0),
            arc_api_key=info.get("arc_api_key", ""),
            arc_base_url=info.get("arc_base_url", "https://three.arcprize.org"),
            operation_mode=operation_mode,
        )
