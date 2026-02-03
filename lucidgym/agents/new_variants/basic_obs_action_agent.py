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


from rllm.tools.tool_base import ToolCall
from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from lucidgym.agents.arcagi3_agent import ArcAgi3Agent


from arcengine import GameAction, GameState
from lucidgym.utils.grid_processing import frame_to_grid_text, downsample_4x4, generate_numeric_grid_image_bytes, format_grid
from lucidgym.utils.representation import RepresentationConfig, GridFormat


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
        downsample: bool = True,
        game_id: str | None = None,
        representation: RepresentationConfig | None = None,
    ) -> None:
        """
        Initialize the agent.

        Args:
            system_prompt: Override system prompt (optional)
            name: Agent name
            input_mode: 'text_only', 'image_only', or 'text_and_image'
            model: OpenAI model to use
            reasoning_effort: Reasoning effort level
            downsample: Whether to downsample 64x64 to 16x16
            game_id: Game ID for prompt selection
            representation: RepresentationConfig for grid formatting
        """
        # Create representation config before calling super().__init__
        self.representation = representation or RepresentationConfig(
            downsample=downsample,
        )
        super().__init__(system_prompt=system_prompt, name=name, downsample=downsample, representation=self.representation)
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

    def update_from_env(self, observation: Any, reward: float, done: bool, **_: Any) -> None:
        """Process environment observation and update state."""
        self._last_observation = observation

        # Store in trajectory
        step = Step(
            observation=observation,
            reward=reward,
            done=done,
            info={},
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
        if action.is_complex():
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

        # Extract frame and format using representation config
        frame_3d = obs.get("frame", [])
        grid_text = self._format_grid(frame_3d)
        score = obs.get("score", 0)

        # DEBUG alternate between ACTION1 and ACTION2
        # action_name = "ACTION1" if self._action_counter % 2 == 0 else "ACTION2"
        # action_dict = {"name": action_name, "data": {}, "obs_text": "", "action_text": ""}
            
        # Step 1: Observation phase
        obs_text = await self._call_observation_model(grid_text, score, rollout_engine=rollout_engine)

        # Step 2: Action selection phase
        action_dict = await self._call_action_model(grid_text, obs_text, rollout_engine=rollout_engine)
        action_dict["obs_text"] = obs_text

        # Stash for update_from_model to record
        self._pending_action = action_dict
        return action_dict

    def _format_grid(self, frame_3d: List[List[List[int]]]) -> str:
        """Format the grid using the representation config."""
        if self.representation:
            if self.representation.downsample:
                grid_2d = downsample_4x4(frame_3d) if frame_3d else []
            else:
                # Get raw 2D grid from 3D frame
                grid_2d = frame_3d[-1] if frame_3d else []
            return format_grid(grid_2d, self.representation) if grid_2d else "No frame data"
        elif self.downsample:
            return frame_to_grid_text([downsample_4x4(frame_3d)])
        else:
            return frame_to_grid_text([frame_3d])

    def _build_user_content(self, grid_text: str, user_prompt_text: str) -> List[Dict[str, Any]]:
        """Build the 'content' array for the API call based on input_mode.
        
        Args:
            grid_text: Pre-formatted grid text string
            user_prompt_text: The user prompt text to include
        """
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt_text}]

        if self.input_mode in ["image_only", "text_and_image"]:
            try:
                # Note: image generation still needs the raw grid, not the formatted text
                # This would need the original grid passed separately if image support is needed
                log.warning("Image mode not fully supported with representation config yet")
            except Exception as e:
                log.error(f"Failed to generate numeric grid image: {e}")

        return content

    async def rollout(self, rollout_engine: OpenAIEngine, messages: List[Dict[str, Any]], tools=None):
        return await rollout_engine.get_model_response(messages, tools=tools)

    async def _call_observation_model(self, grid_text: str, score: int, rollout_engine=None) -> str:
        """Call the model for observation/reasoning phase.
        
        Args:
            grid_text: Pre-formatted grid text string
            score: Current game score
            rollout_engine: The rollout engine for LLM calls
        """
        sys_msg = build_observation_system_text()

        grid_size = "16x16" if self.downsample else "64x64"
        format_desc = self.representation.get_format_description() if self.representation else "ASCII characters"
        
        format_clarification = ""
        if self.input_mode == "image_only":
            format_clarification = f"The board state is provided as an attached image of the {grid_size} grid."
        elif self.input_mode == "text_and_image":
            format_clarification = "The board state is provided as both a textual matrix and an attached image."

        user_msg_text = (
            f"Score: {score}\n"
            f"Step: {self._action_counter}\n"
            f"Matrix {grid_size} ({format_desc}):\n{grid_text}\n\n"
            "Rationale:\n"
            "  • Identify the movable character(s) and relevant structures.\n"
            "  • Conclude which direction is best and why. Do not output an action here.\n"
            "  • Focus on the strategic importance of each character and how it relates to the goal."
        )

        user_content = self._build_user_content(grid_text, user_msg_text)
        user_content = user_content[0]['text']

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content}
        ]

        self._chat_history.append({"role": "user", "content": user_msg_text})

        model_output = await self.rollout(rollout_engine, messages)
        text = (getattr(model_output, "content", None) or getattr(model_output, "text", "") or "").strip()

        if text:
            self._chat_history.append({"role": "assistant", "content": text})
        return text

    async def _call_action_model(self, grid_text: str, last_obs: str, rollout_engine=None) -> dict:
        """Call the model for action selection phase.
        
        Args:
            grid_text: Pre-formatted grid text string
            last_obs: The observation text from the previous phase
            rollout_engine: The rollout engine for LLM calls
        """
        sys_msg = build_action_system_text()

        format_clarification = ""
        if self.input_mode == "image_only":
            format_clarification = "The board state is provided as an attached image."
        elif self.input_mode == "text_and_image":
            format_clarification = "The board state is provided as both text and image."

        user_msg_text = (
            "Choose the best single move as a function call.\n"
            f"{grid_text}\n"
            "Previous observation summary:\n"
            f"{last_obs}\n"
        )

        user_content = self._build_user_content(grid_text, user_msg_text)
        user_content = user_content[0]['text']
        tools = _build_tools()

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content}
        ]

        self._chat_history.append({"role": "user", "content": user_msg_text})

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
        if isinstance(tc, ToolCall):
            self._latest_tool_call_id = f"call_{tc.name}"
            name = tc.name
            args = tc.arguments
        else:
            self._latest_tool_call_id = tc.id
            name = tc.function.name
            arguments = tc.function.arguments
            args = json.loads(arguments or "{}")

        # Add to chat history
        self._chat_history.append({
            "role": "assistant",
            "tool_calls": [{
                "id": self._latest_tool_call_id,
                "type": "function",
                "function": {"name": name, "arguments": args}
            }]
        })

        # Handle ACTION6 coordinate mapping if needed
        if name == "ACTION6":
            # Ensure coordinates are integers (JSON may return strings)
            x_raw = _coerce_int(args.get("x", 0))
            y_raw = _coerce_int(args.get("y", 0))
            # Scale coordinates to 64x64 game space only if using downsampled 16x16 grid
            if self.downsample:
                # Clamp to valid 16x16 range before scaling
                x_raw = max(0, min(15, x_raw))
                y_raw = max(0, min(15, y_raw))
                x_64 = x_raw * 4
                y_64 = y_raw * 4
            else:
                # Clamp to valid 64x64 range
                x_64 = max(0, min(63, x_raw))
                y_64 = max(0, min(63, y_raw))
            return {"name": name, "data": {"x": x_64, "y": y_64}, "action_text": model_output.content}

        return {"name": name, "data": args, "action_text": model_output.content}
