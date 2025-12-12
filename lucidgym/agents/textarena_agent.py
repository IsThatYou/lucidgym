"""
Baseline TextArena agent that converts serialized observations into chat prompts.
"""
from __future__ import annotations

from typing import Any, Iterable

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class TextArenaAgent(BaseAgent):
    """
    Simple agent scaffold that treats TextArena messages as alternating user prompts.

    Real agents should subclass this and override prompt construction/decoding logic,
    but this baseline already works with the TextArenaEnv wrapper.
    """

    def __init__(self, system_prompt: str | None = None, name: str = "textarena_agent") -> None:
        self.system_prompt = system_prompt
        self.name = name
        self._chat_history: list[dict[str, str]] = []
        self._trajectory = Trajectory(name=name)

    def reset(self) -> None:
        self._chat_history = []
        self._trajectory = Trajectory(name=self.name)

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}]
        else:
            messages = []
        messages.extend(self._chat_history)
        return messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **_: Any) -> None:
        if observation:
            user_message = self._format_observation(observation)
            if user_message:
                self._chat_history.append({"role": "user", "content": user_message})

        step = Step(observation=observation, reward=reward, done=done, info=info, chat_completions=self.chat_completions.copy())
        self._trajectory.steps.append(step)

    def update_from_model(self, response: str, **_: Any) -> Action:
        response = (response or "").strip()
        action = Action(action=response)

        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = response
            self._trajectory.steps[-1].action = action.action

        self._chat_history.append({"role": "assistant", "content": response})
        return action

    # ------------------------------------------------------------------ #
    def _format_observation(self, observation: dict) -> str:
        messages = observation.get("messages", [])
        player_id = observation.get("current_player_id")
        env_id = observation.get("env_id", "unknown-env")
        lines = [f"[{env_id}] Player {player_id} turn"] if player_id is not None else [f"[{env_id}] Observation"]

        for msg in messages:
            from_role = msg.get("from_role", msg.get("from_id", "unknown"))
            msg_type = msg.get("type", "MESSAGE")
            text = msg.get("text", "")
            lines.append(f"{from_role} ({msg_type}): {text}")

        return "\n".join(lines)
