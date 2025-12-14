"""
16x16 text-based guided agents with multi-modal support.
Implements observation-action loop with downsampled grid representations.
"""
from __future__ import annotations
from typing import Any, Optional, List, Dict
import json
import logging
import base64
from openai import OpenAI

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

from lucidgym.environments.arcagi3.structs import GameAction, GameState
from lucidgym.utils.grid_processing import downsample_4x4, generate_numeric_grid_image_bytes
from lucidgym.prompts.text_prompts import (
    build_observation_system_text,
    build_observation_user_text,
    build_action_system_text,
    build_action_user_text,
)

log = logging.getLogger(__name__)


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        if isinstance(v, bool):
            return default
        return max(0, int(float(v)))
    except Exception:
        return default


def _build_tools() -> list[dict]:
    """Build the tool/function definitions for the ARC-AGI-3 API."""
    return [
        {
            "type": "function",
            "function": {
                "name": "RESET",
                "description": "Reset the game to start a new attempt",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ACTION1",
                "description": "Move Up",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ACTION2",
                "description": "Move Down",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ACTION3",
                "description": "Move Left",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ACTION4",
                "description": "Move Right",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ACTION5",
                "description": "Spacebar / Enter / No-op",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ACTION6",
                "description": "Click at coordinates (x, y)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "X coordinate (0-15 for 16x16 cell, or absolute)"},
                        "y": {"type": "integer", "description": "Y coordinate (0-15 for 16x16 cell, or absolute)"},
                    },
                    "required": ["x", "y"],
                },
            },
        },
    ]


class AS66GuidedAgent(BaseAgent):
    """
    16×16 downsample agent using rllm BaseAgent interface.
    Supports observation + single tool call per turn with multimodal input modes.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        name: str = "as66_guided_agent",
        input_mode: str = "text_only",
        model: str = "gpt-4o",
        reasoning_effort: str = "low",
        game_id: str | None = None,
    ) -> None:
        """
        Initialize the agent.

        Args:
            system_prompt: Override system prompt (optional)
            name: Agent name
            input_mode: 'text_only', 'image_only', or 'text_and_image'
            model: OpenAI model to use
            reasoning_effort: Reasoning effort level
            game_id: Game ID for prompt selection
        """
        self.name = name
        self.input_mode = input_mode
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.game_id = game_id
        self._system_prompt_override = system_prompt

        if self.input_mode not in ["text_only", "image_only", "text_and_image"]:
            log.warning(f"Invalid input_mode '{self.input_mode}', defaulting to 'text_only'.")
            self.input_mode = "text_only"

        self._client = OpenAI()
        self._latest_tool_call_id = "call_12345"
        self.reset()

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self._chat_history: list[dict] = []
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None
        self._token_total: int = 0
        self._action_counter: int = 0

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return message history formatted for chat API."""
        system_msg = self._system_prompt_override or build_observation_system_text(self.game_id)
        messages: list[dict] = [{"role": "system", "content": system_msg}]
        messages.extend(self._chat_history)
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Return the trajectory tracking object."""
        return self._trajectory

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **_: Any) -> None:
        """Process environment observation and update state."""
        self._last_observation = observation

        # Store in trajectory
        step = Step(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            chat_completions=self.chat_completions.copy()
        )
        self._trajectory.steps.append(step)

        # Add tool response to chat history
        if self._chat_history and self._chat_history[-1].get("role") == "assistant":
            # Format observation as tool response
            state = observation.get("state", "NOT_PLAYED")
            score = observation.get("score", 0)
            tool_content = f"State: {state} | Score: {score}"

            self._chat_history.append({
                "role": "tool",
                "tool_call_id": self._latest_tool_call_id,
                "content": tool_content
            })

    def update_from_model(self, response: str, **_: Any) -> Action:
        """Convert model response to Action."""
        # This agent actually calls the model internally, so we just trigger the action selection
        if not self._last_observation:
            return Action(action={"name": "RESET", "data": {}})

        obs = self._last_observation
        state = obs.get("state", "NOT_PLAYED")

        # Handle RESET needed states
        if state in ("NOT_PLAYED", "GAME_OVER"):
            action_dict = {"name": "RESET", "data": {}}
            return Action(action=action_dict)

        # Extract frame and downsample
        frame_3d = obs.get("frame", [])
        if not frame_3d:
            action_dict = {"name": "ACTION5", "data": {}}
            return Action(action=action_dict)

        ds16 = downsample_4x4(frame_3d, take_last_grid=True, round_to_int=True)
        score = obs.get("score", 0)

        # Step 1: Observation phase
        obs_text = self._call_observation_model(ds16, score)

        # Step 2: Action selection phase
        action_dict = self._call_action_model(ds16, obs_text)

        # Attach observation text as reasoning for ARC API replay logs
        action_dict["reasoning"] = obs_text

        # Update trajectory with model response
        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = obs_text + " | " + str(action_dict)
            self._trajectory.steps[-1].action = action_dict

        self._action_counter += 1
        return Action(action=action_dict)

    def _build_user_content(self, ds16: List[List[int]], user_prompt_text: str) -> List[Dict[str, Any]]:
        """Build the 'content' array for the API call based on input_mode."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt_text}]

        if self.input_mode in ["image_only", "text_and_image"]:
            try:
                png_bytes = generate_numeric_grid_image_bytes(ds16)
                b64_image = base64.b64encode(png_bytes).decode('utf-8')
                data_url = f"data:image/png;base64,{b64_image}"
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": "low"
                    }
                })
            except Exception as e:
                log.error(f"Failed to generate numeric grid image: {e}")

        return content

    def _call_observation_model(self, ds16: List[List[int]], score: int) -> str:
        """Call the model for observation/reasoning phase."""
        sys_msg = build_observation_system_text(self.game_id)

        include_text = self.input_mode in ["text_only", "text_and_image"]
        format_clarification = ""
        if self.input_mode == "image_only":
            format_clarification = "The board state is provided as an attached image of the 16x16 grid."
        elif self.input_mode == "text_and_image":
            format_clarification = "The board state is provided as both a textual matrix and an attached image."

        user_msg_text = build_observation_user_text(
            ds16, score, self._action_counter, self.game_id,
            format_clarification=format_clarification,
            include_text_matrix=include_text
        )

        user_content = self._build_user_content(ds16, user_msg_text)

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content}
        ]

        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
        }

        # Add reasoning_effort for GPT-5 series models
        if self.model.startswith("gpt-5"):
            api_params["reasoning_effort"] = self.reasoning_effort

        resp = self._client.chat.completions.create(**api_params)

        self._token_total += getattr(resp.usage, "total_tokens", 0) or 0
        text = (resp.choices[0].message.content or "").strip()

        # Suppress tool calls in observation phase
        if resp.choices[0].message.tool_calls:
            text = "(observation only; tool call suppressed)"

        return text

    def _call_action_model(self, ds16: List[List[int]], last_obs: str) -> dict:
        """Call the model for action selection phase."""
        sys_msg = build_action_system_text(self.game_id)

        include_text = self.input_mode in ["text_only", "text_and_image"]
        format_clarification = ""
        if self.input_mode == "image_only":
            format_clarification = "The board state is provided as an attached image."
        elif self.input_mode == "text_and_image":
            format_clarification = "The board state is provided as both text and image."

        user_msg_text = build_action_user_text(
            ds16, last_obs, self.game_id,
            format_clarification=format_clarification,
            include_text_matrix=include_text
        )

        user_content = self._build_user_content(ds16, user_msg_text)
        tools = _build_tools()

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content}
        ]

        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
        }

        # Add reasoning_effort for GPT-5 series models
        if self.model.startswith("gpt-5"):
            api_params["reasoning_effort"] = self.reasoning_effort

        resp = self._client.chat.completions.create(**api_params)

        self._token_total += getattr(resp.usage, "total_tokens", 0) or 0

        m = resp.choices[0].message

        # Note: GPT-5 models perform reasoning internally but don't expose reasoning content in API responses.
        # The reasoning_effort parameter controls the amount of thinking, but the actual reasoning
        # text is not returned. We can only see reasoning_tokens in usage stats.

        if not m.tool_calls:
            return {"name": "ACTION5", "data": {}}

        tc = m.tool_calls[0]
        self._latest_tool_call_id = tc.id
        name = tc.function.name

        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {}

        # Add to chat history
        self._chat_history.append({
            "role": "assistant",
            "tool_calls": [{
                "id": tc.id,
                "type": "function",
                "function": {"name": name, "arguments": tc.function.arguments}
            }]
        })

        # Handle ACTION6 coordinate mapping if needed
        if name == "ACTION6":
            x_16 = args.get("x", 0)
            y_16 = args.get("y", 0)
            # Scale 16x16 coordinates to 64x64 game space (4x upscaling)
            x_64 = x_16 * 4
            y_64 = y_16 * 4
            return {"name": name, "data": {"x": x_64, "y": y_64}}

        return {"name": name, "data": args}
