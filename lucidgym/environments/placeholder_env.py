"""
Placeholder environment demonstrating the BaseEnv contract.

Real LucidGym environments (Gymnasium, TextArena, ARC, ...) will replace
this class. The placeholder allows early wiring/testing of registries and
training configs without touching upstream modules.
"""
from __future__ import annotations

from typing import Any, Tuple

from rllm.environments.base.base_env import BaseEnv


class LucidGymPlaceholderEnv(BaseEnv):
    """No-op environment that immediately terminates when stepped."""

    def __init__(self, task: dict | None = None) -> None:
        self.task = task or {}
        self._last_action = None
        self._done = False

    def reset(self, task: dict | None = None) -> Tuple[dict, dict]:
        if task is not None:
            self.task = task
        self._done = False
        self._last_action = None
        info = {"status": "placeholder_env_reset"}
        return self.task, info

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        self._last_action = action
        self._done = True
        next_obs: dict = {"note": "Placeholder env terminates immediately."}
        info = {"status": "placeholder_env_step", "action": action}
        return next_obs, 0.0, self._done, info

    def close(self) -> None:
        self._last_action = None

    @staticmethod
    def from_dict(info: dict) -> "LucidGymPlaceholderEnv":
        return LucidGymPlaceholderEnv(task=info.get("task"))
