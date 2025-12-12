"""
Placeholder agent showing how LucidGym subclasses rllm's BaseAgent.

Concrete agents will override ``chat_completions`` construction plus the
``update_from_model`` logic to convert model responses into environment
actions. This stub keeps the code importable so registry wiring can be
validated during Phase 1.
"""
from __future__ import annotations

from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class LucidGymPlaceholderAgent(BaseAgent):
    """Minimal BaseAgent implementation used for wiring tests."""

    def __init__(self, name: str = "lucidgym_agent") -> None:
        self._name = name
        self._trajectory = Trajectory(name=name)

    def reset(self) -> None:
        self._trajectory = Trajectory(name=self._name)

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        # Replace with task-specific prompts when building real agents.
        return [{"role": "system", "content": "LucidGym placeholder agent awaiting implementation."}]

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **_: Any) -> None:
        step = Step(observation=observation, reward=reward, done=done, info=info)
        self._trajectory.steps.append(step)

    def update_from_model(self, response: str, **_: Any) -> Action:
        raise NotImplementedError("Replace LucidGymPlaceholderAgent with a concrete implementation before use.")
