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

from rllm.agents.agent import Action, BaseAgent,Step, Trajectory
from lucidgym.agents.arcagi3_agent import ArcAgi3Agent

from lucidgym.environments.arcagi3.structs import GameAction, GameState
from lucidgym.utils.grid_processing import frame_to_grid_text, downsample_4x4, generate_numeric_grid_image_bytes


log = logging.getLogger(__name__)


def build_observation_system_text():
    return (
        "You are observing a 16x16 grid representation of a game state. "
        "Each cell contains an ASCII character representing different game elements. "
        "Your task is to analyze this grid and determine the best action to take. "
        "The grid shows the current game state with various symbols representing different game objects."
    )

def build_action_system_text():
    return (
        "You are selecting the best action based on your observation of the game state. "
        "Choose one of the available actions: ACTION1 (Up), ACTION2 (Down), ACTION3 (Left), ACTION4 (Right), ACTION5 (Enter), ACTION6 (Click)"
    )

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


class BasicObsActionAgent(ArcAgi3Agent):
    """
    Basic agent that generates an observation, and based on that generates an action.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        name: str = "basic_obs_action_agent",
        input_mode: str = "text_only",
        model: str = "gpt-5-nano",
        reasoning_effort: str = "low",
        downsample = True,
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
        super().__init__(system_prompt=system_prompt, name=name)
        self.input_mode = input_mode
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.game_id = game_id
        self._system_prompt_override = system_prompt
        self.downsample = downsample

        if self.input_mode not in ["text_only", "image_only", "text_and_image"]:
            log.warning(f"Invalid input_mode '{self.input_mode}', defaulting to 'text_only'.")
            self.input_mode = "text_only"

        self._client = OpenAI()
        self._latest_tool_call_id = "call_12345"
        self.reset()

    def reset(self) -> None:
        """Reset agent state for new episode."""
        super().reset()  # Initialize parent class attributes including _steps_this_episode
        self._chat_history: list[dict] = []
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None
        self._token_total: int = 0
        self._action_counter: int = 0
        self._pending_action: dict[str, Any] | None = None

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return message history formatted for chat API."""
        system_msg = self._system_prompt_override or build_observation_system_text()
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
            tool_content = self._format_observation(observation)

            self._chat_history.append({
                "role": "tool",
                "tool_call_id": self._latest_tool_call_id,
                "content": tool_content
            })

    def update_from_model(self, action_payload: dict | None = None, **_: Any) -> Action:
        """Convert model response to Action and record the trajectory step."""
        action_dict = action_payload or self._pending_action
        obs_text = action_dict["obs_text"]
        action_text = action_dict["action_text"]
        response_text = f"Observation: {obs_text}\nAction Text: {action_text}\nAction: {action_dict['name']}"

        if action_dict is None:
            # Fallback to internal generation to preserve legacy behavior
            response_text, action_dict = self.call_llm()

        if not response_text:
            response_text = str(action_dict)

        # print(f"Observation: {obs_text}\nAction: {action_text}\nResponse: {response_text}")

        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = response_text
            self._trajectory.steps[-1].action = action_dict

        self._action_counter += 1
        self._pending_action = None
        action = GameAction.from_name(action_dict["name"])
        action_dict2 = {"action": action, "reasoning": response_text}
        if action.requires_coordinates():
            action_dict2["x"] = action_dict["data"]["x"]
            action_dict2["y"] = action_dict["data"]["y"]
        return action_dict2

    async def call_llm(self, rollout_engine=None) -> tuple[str, dict]:
        """Run the two-phase observation/action LLM calls and return text + action dict."""
        obs = self._last_observation or {}
        state = obs.get("state", "NOT_PLAYED")

        # Handle RESET needed states
        # print(f"[DEBUG]:[guided]: obs={obs}")
        if state in ("NOT_PLAYED", "GAME_OVER"):
            action_dict = {"name": "RESET", "data": {}, "obs_text": "Game Over, starting new game.", "action_text": ""}
            self._pending_action = action_dict
            return action_dict

        # Extract frame and downsample
        frame_3d = obs.get("frame", [])

        if self.downsample:
            grid = frame_to_grid_text([downsample_4x4(frame_3d)])
        else:
            grid = frame_to_grid_text([frame_3d])
        score = obs.get("score", 0)

        # DEBUG alternate between ACTION1 and ACTION2
        # action_name = "ACTION1" if self._action_counter % 2 == 0 else "ACTION2"
        # action_dict = {"name": action_name, "data": {}, "obs_text": "", "action_text": ""}
            
        # # Step 1: Observation phase
        obs_text = await self._call_observation_model(grid, score, rollout_engine=rollout_engine)

        # # Step 2: Action selection phase
        action_dict = await self._call_action_model(grid, obs_text, rollout_engine=rollout_engine)
        action_dict["obs_text"] = obs_text

        # Stash for update_from_model to record
        self._pending_action = action_dict
        return action_dict

    def _build_user_content(self, grid: List[List[int]], user_prompt_text: str) -> List[Dict[str, Any]]:
        """Build the 'content' array for the API call based on input_mode."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt_text}]

        if self.input_mode in ["image_only", "text_and_image"]:
            try:
                png_bytes = generate_numeric_grid_image_bytes(grid)
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

    async def rollout(self, rollout_engine: OpenAIEngine, messages: List[Dict[str, Any]], tools=None):
        return await rollout_engine.get_model_response(messages, tools=tools)

    async def _call_observation_model(self, grid: List[List[int]], score: int, rollout_engine=None) -> str:
        """Call the model for observation/reasoning phase."""
        sys_msg = build_observation_system_text()

        include_text = self.input_mode in ["text_only", "text_and_image"]
        grid_text = "16x16" if self.downsample else "64x64"
        format_clarification = ""
        if self.input_mode == "image_only":
            format_clarification = f"The board state is provided as an attached image of the {grid_text} grid."
        elif self.input_mode == "text_and_image":
            format_clarification = "The board state is provided as both a textual matrix and an attached image."

        user_msg_text = (
            f"Score: {score}\n"
            f"Step: {self._action_counter}\n"
            f"Matrix {grid_text} (ASCII characters):\n{grid}\n\n"
            "Rationale:\n"
            "  • Identify the movable ASCII character(s) and relevant structures.\n"
            "  • Conclude which direction is best and why. Do not output an action here.\n"
            "  • Focus on the strategic importance of each character and how it relates to the goal."
        )

        user_content = self._build_user_content(grid, user_msg_text)

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content}
        ]

        model_output = await self.rollout(rollout_engine, messages)
        text = (getattr(model_output, "content", None) or getattr(model_output, "text", "") or "").strip()
        return text

    async def _call_action_model(self, grid: List[List[int]], last_obs: str, rollout_engine=None) -> dict:
        """Call the model for action selection phase."""
        sys_msg = build_action_system_text()

        include_text = self.input_mode in ["text_only", "text_and_image"]
        format_clarification = ""
        if self.input_mode == "image_only":
            format_clarification = "The board state is provided as an attached image."
        elif self.input_mode == "text_and_image":
            format_clarification = "The board state is provided as both text and image."

        user_msg_text = (
            "Choose the best single move as a function call.\n"
            f"{grid}"
            "Previous observation summary:\n"
            f"{last_obs}\n"
        )

        user_content = self._build_user_content(grid, user_msg_text)
        tools = _build_tools()

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content}
        ]

        model_output = await self.rollout(rollout_engine, messages, tools)

        m = model_output.tool_calls[0] if getattr(model_output, "tool_calls", None) else None

        # Note: GPT-5 models perform reasoning internally but don't expose reasoning content in API responses.
        # The reasoning_effort parameter controls the amount of thinking, but the actual reasoning
        # text is not returned. We can only see reasoning_tokens in usage stats.

        print(f"[DEBUG]:guided_text_16:model_output={model_output}")
        print(f"[DEBUG]:guided_text_16:rollout_engine._use_chat_completions={getattr(rollout_engine, '_use_chat_completions', 'N/A')}")
        # return {"name": "ACTION6", "data": {"x": 7, "y": 7}, "action_text": "ACTION6"}
        if m is None:
            return {"name": "ACTION5", "data": {}, "action_text": "ACTION5"}

        tc = m
        tc_id = tc.id
        self._latest_tool_call_id = tc_id
        name = tc.function.name
        arguments = getattr(tc, "function", {}).get("arguments") if isinstance(tc, dict) else tc.function.arguments

        try:
            args = json.loads(arguments or "{}")
        except Exception:
            args = {}

        # Add to chat history
        self._chat_history.append({
            "role": "assistant",
            "tool_calls": [{
                "id": tc_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments}
            }]
        })

        # Handle ACTION6 coordinate mapping if needed
        if name == "ACTION6":
            x_16 = args.get("x", 0)
            y_16 = args.get("y", 0)
            # Scale 16x16 coordinates to 64x64 game space (4x upscaling)
            x_64 = x_16 * 4
            y_64 = y_16 * 4
            return {"name": name, "data": {"x": x_64, "y": y_64}, "action_text": model_output.content}

        return {"name": name, "data": args, "action_text": model_output.content}
