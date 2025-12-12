"""
ARC-AGI-3 agent scaffold that converts environment observations into prompts.

The agent works in two modes:

1. ``llm`` (default) – format observations into chat prompts and parse the LLM's
   response into a structured ARC ``GameAction`` dict.
2. ``passthrough`` – delegate action selection to a caller-supplied function so
   existing ARC agents can be reused without rewriting them for LucidGym.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable
import textwrap

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

from lucidgym.environments.arcagi3.structs import GameAction

DEFAULT_SYSTEM_PROMPT = (
    "You control an ARC-AGI-3 agent. Observe the grid, reason about the task, "
    "then emit a tool call with a 'name' field matching ACTION1-7 plus "
    "optional coordinates (x,y) when the action requires them. Include reasoning summarizing why the action was chosen."
)

PassthroughFn = Callable[[dict[str, Any], Trajectory], dict[str, Any]]


class ArcAgi3Agent(BaseAgent):
    """
    Minimal agent facade for ARC-AGI-3 environments.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        name: str = "arcagi3_agent",
        mode: str = "llm",
        passthrough_fn: PassthroughFn | None = None,
    ) -> None:
        if mode not in {"llm", "passthrough"}:
            raise ValueError("mode must be either 'llm' or 'passthrough'.")
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.name = name
        self.mode = mode
        self._passthrough_fn = passthrough_fn
        self._latest_tool_call_id = "call_12345"
        self.reset()

    def reset(self) -> None:
        self._chat_history: list[dict[str, str]] = []
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._chat_history)
        return messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    # ------------------------------------------------------------------ #
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **_: Any) -> None:
        # Update internal state
        self._last_observation = observation
        if len(observation) != 0:
            user_msg = self._format_observation(self._last_observation)
            if user_msg:
                self._chat_history.append({"role": "tool", "tool_call_id": self._latest_tool_call_id, "content": user_msg})

            step = Step(observation=observation, reward=reward, done=done, info=info, chat_completions=self.chat_completions.copy())
            self._trajectory.steps.append(step)
        else:
            # no action, no observation, ask llm to do it again.
            self._chat_history.append({"role": "tool", "tool_call_id": self._latest_tool_call_id, "content": "Unable to parse action response. Make sure your response calls a valid tool call."})
            step = Step(observation=observation, reward=reward, done=done, info=info, chat_completions=self.chat_completions.copy())
            self._trajectory.steps.append(step)


    def update_from_model(self, response: str, **_: Any) -> Action:
        normalized_response = (response or "").strip()
        sentinel_requested = normalized_response.upper() == "__PASSTHROUGH__"
        use_passthrough = self.mode == "passthrough" or (sentinel_requested and self._passthrough_fn is not None)
        if use_passthrough:
            payload = self._call_passthrough()
            assistant_msg = f"[passthrough] {payload}"
        else:
            action_payload = self._parse_action_response(response)
            assistant_msg = response
            
        # action = Action(action=action_payload)
        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = response
            self._trajectory.steps[-1].action = action_payload

        self._chat_history.append({"role": "assistant", "content": assistant_msg})
        return action_payload

    # ------------------------------------------------------------------ #
    def _format_observation(self, observation: dict[str, Any]) -> str:
        if "grid_ascii" in observation:
            frame = observation.get("grid_ascii")
        elif "grid_flat" in observation:
            frame = observation.get("grid_flat")
        elif "frame" in observation:
            frame = observation.get("frame")
            # frame = "\n".join(",".join(str(x) for x in row) for row in frame[0])
            frame = self.pretty_print_3d(frame)
        available_actions = observation.get("available_actions") or []
        if available_actions:
            formatted = ", ".join(
                f"{a['name']}({'xy' if a.get('requires_coordinates') else 'ok'})" for a in available_actions
            )
        return textwrap.dedent(
            """
            # State:
            {state}
            Step {step}

            # Score:
            {score}

            # Frame:
            {frame}

            # Output next action:
            Reply with a several sentences/ paragraphs of plain-text strategy observation about the frame to inform your next action. The output should be a tool call indicating the action to take.
            """
        ).format(
            state=observation.get("state", "UNKNOWN"),
            step=observation.get("step", 0),
            # card=observation.get("card_id", "?"),
            # game=observation.get("game_id", "?"),
            score=observation.get("score", 0),
            frame=frame if frame else "N/A",
        )

    def _parse_action_response(self, response: str) -> dict[str, Any]:
        payload = self._parse_tool_call(response)
        if not payload:
            return {'action': None, "reasoning": response}
        action_name = str(payload.get("name", "")).strip().upper()
        action = GameAction.from_name(action_name)
        normalized: dict[str, Any] = {"action": action, "action_name": action.name,}

        if action.requires_coordinates():
            try:
                normalized["x"] = int(payload["arguments"]["x"])
                normalized["y"] = int(payload["arguments"]["y"])
            except (TypeError, ValueError):
                raise ValueError(f"{action.name} requires integer x/y fields.") from None

        normalized["reasoning"] = response

        return normalized

    def _parse_tool_call(self,response: str) -> Optional[dict[str, Any]]:
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            return None  # No tool call found
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None

    def _call_passthrough(self) -> dict[str, Any]:
        if not self._passthrough_fn:
            raise ValueError("passthrough_fn must be provided when mode='passthrough'.")
        observation = self._last_observation or {}
        return self._passthrough_fn(observation, self._trajectory)

    def build_functions(self) -> list[dict[str, Any]]:
        """Build JSON function description of game actions for LLM."""
        empty_params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        functions: list[dict[str, Any]] = [
            {
                "name": GameAction.RESET.name,
                "description": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION1.name,
                "description": "Send this simple input action (1, W, Up).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION2.name,
                "description": "Send this simple input action (2, S, Down).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION3.name,
                "description": "Send this simple input action (3, A, Left).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION4.name,
                "description": "Send this simple input action (4, D, Right).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION5.name,
                "description": "Send this simple input action (5, Enter, Spacebar, Delete).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION6.name,
                "description": "Send this complex input action (6, Click, Point).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "string",
                            "description": "Coordinate X which must be Int<0,63>",
                        },
                        "y": {
                            "type": "string",
                            "description": "Coordinate Y which must be Int<0,63>",
                        },
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
            },
        ]
        return functions

    def build_tools(self) -> list[dict[str, Any]]:
        """Support models that expect tool_call format."""
        functions = self.build_functions()
        tools: list[dict[str, Any]] = []
        for f in functions:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f["name"],
                        "description": f["description"],
                        "parameters": f.get("parameters", {}),
                        "strict": True,
                    },
                }
            )
        return tools
        
    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
            lines.append("")
        return "\n".join(lines)
