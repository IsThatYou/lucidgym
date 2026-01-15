"""
16x16 text-based guided agents with rolling context.
Maintains history of last k steps to provide context in prompts.
"""
from __future__ import annotations
from typing import Any, Optional, List, Dict
from collections import deque
import json
import logging
import base64
from openai import OpenAI

from lucidgym.utils.openai_client import get_openai_client
from rllm.agents.agent import Action, BaseAgent,Step, Trajectory
from lucidgym.agents.arcagi3_agent import ArcAgi3Agent

from lucidgym.environments.arcagi3.structs import GameAction, GameState
from lucidgym.utils.grid_processing import frame_to_grid_text, downsample_4x4, generate_numeric_grid_image_bytes

# Optional Weave integration for LLM tracing
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    weave = None

log = logging.getLogger(__name__)

# Conditional weave decorator
def weave_op(func):
    """Apply @weave.op decorator if weave is available, otherwise no-op."""
    if WEAVE_AVAILABLE and weave:
        return weave.op(func)
    return func


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


class BasicObsActionAgentRollingContext(ArcAgi3Agent):
    """
    Basic agent with rolling context of last k steps.
    Includes step history in prompts to provide context.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        name: str = "basic_obs_action_agent_rolling_context",
        input_mode: str = "text_only",
        model: str = "gpt-5-nano",
        reasoning_effort: str = "low",
        downsample = True,
        game_id: str | None = None,
        context_window_size: int = 5,  # Keep last k steps in context
        crop_border: int = 0,  # Remove outer N pixels (e.g., 2 for scoring numbers)
    ) -> None:
        """
        Initialize the agent with rolling context.

        Args:
            system_prompt: Override system prompt (optional)
            name: Agent name
            input_mode: 'text_only', 'image_only', or 'text_and_image'
            model: OpenAI model to use
            reasoning_effort: Reasoning effort level
            downsample: Whether to downsample the grid
            game_id: Game ID for prompt selection
            context_window_size: Number of recent steps to keep in context
            crop_border: Remove outer N pixels from grid (e.g., 2 to remove scoring numbers)
        """
        # Set context_window_size before calling super().__init__ because parent calls reset()
        self.context_window_size = context_window_size
        self.crop_border = crop_border

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

        self._client = get_openai_client(model=model)
        self._latest_tool_call_id = "call_12345"

    def reset(self) -> None:
        """Reset agent state for new episode."""
        super().reset()
        self._chat_history: list[dict] = []
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None
        self._token_total: int = 0
        self._action_counter: int = 0
        self._pending_action: dict[str, Any] | None = None
        # Rolling context: track last k steps
        self._step_history: deque = deque(maxlen=self.context_window_size)
        # Store actual prompts/responses for logging
        self._last_observation_prompt: str = ""
        self._last_observation_response: str = ""
        self._last_action_prompt: str = ""
        self._last_action_response: str = ""

    def _format_step_history(self) -> str:
        """Format step history as context string."""
        if not self._step_history:
            return ""

        history_lines = ["**Recent History:**\n"]
        for step_info in self._step_history:
            history_lines.append(
                f"Step {step_info['step']}: Action={step_info['action']}, Score={step_info['score']}, State={step_info['state']}\n"
                f"Board:\n{step_info['grid']}\n"
            )
        return "\n".join(history_lines) + "\n"

    def _crop_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """Remove outer N pixels from grid."""
        if self.crop_border <= 0:
            return grid
        c = self.crop_border
        return [row[c:-c] for row in grid[c:-c]]

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

        # Build full prompt/response log
        full_prompts = []
        if self._last_observation_prompt:
            full_prompts.append({"role": "observation_phase", "content": self._last_observation_prompt})
        if self._last_observation_response:
            full_prompts.append({"role": "observation_response", "content": self._last_observation_response})
        if self._last_action_prompt:
            full_prompts.append({"role": "action_phase", "content": self._last_action_prompt})
        if self._last_action_response:
            full_prompts.append({"role": "action_response", "content": self._last_action_response})

        # Store in trajectory
        step = Step(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            chat_completions=full_prompts
        )
        self._trajectory.steps.append(step)

        # Add tool response to chat history
        if self._chat_history and self._chat_history[-1].get("role") == "assistant":
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
            response_text, action_dict = self.call_llm()

        if not response_text:
            response_text = str(action_dict)

        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = response_text
            self._trajectory.steps[-1].action = action_dict

        # Add to step history with full board state
        obs = self._last_observation or {}
        frame_3d = obs.get("frame", [])

        # Process grid same way as in call_llm
        if self.downsample and len(frame_3d) > 0:
            downsampled = downsample_4x4(frame_3d)
            cropped = self._crop_grid(downsampled)
            grid_str = frame_to_grid_text([cropped])
        elif len(frame_3d) > 0:
            grid_str = frame_to_grid_text([frame_3d])
        else:
            grid_str = ""

        self._step_history.append({
            "step": self._action_counter,
            "action": action_dict["name"],
            "score": obs.get("score", 0),
            "state": obs.get("state", "UNKNOWN"),
            "grid": grid_str
        })

        self._action_counter += 1
        self._pending_action = None
        action = GameAction.from_name(action_dict["name"])
        action_dict2 = {"action": action, "reasoning": response_text}
        if action.requires_coordinates():
            action_dict2["x"] = action_dict["data"]["x"]
            action_dict2["y"] = action_dict["data"]["y"]
        return action_dict2

    @weave_op
    async def call_llm(self, rollout_engine=None) -> tuple[str, dict]:
        """Run the two-phase observation/action LLM calls and return text + action dict."""
        obs = self._last_observation or {}
        state = obs.get("state", "NOT_PLAYED")

        if state in ("NOT_PLAYED", "GAME_OVER"):
            action_dict = {"name": "RESET", "data": {}, "obs_text": "Game Over, starting new game.", "action_text": ""}
            self._pending_action = action_dict
            return action_dict

        # Extract frame and downsample
        frame_3d = obs.get("frame", [])

        if self.downsample:
            downsampled = downsample_4x4(frame_3d)
            cropped = self._crop_grid(downsampled)
            grid = frame_to_grid_text([cropped])
        else:
            grid = frame_to_grid_text([frame_3d])
        score = obs.get("score", 0)

        # Step 1: Observation phase
        obs_text = await self._call_observation_model(grid, score, rollout_engine=rollout_engine)

        # Step 2: Action selection phase
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

    @weave_op
    async def _call_observation_model(self, grid: List[List[int]], score: int, rollout_engine=None) -> str:
        """Call the model for observation/reasoning phase."""
        sys_msg = build_observation_system_text()

        if self.downsample:
            base_size = 16
            actual_size = base_size - (2 * self.crop_border)
            grid_text = f"{actual_size}x{actual_size}" if self.crop_border > 0 else "16x16"
        else:
            grid_text = "64x64"
        history_context = self._format_step_history()

        user_msg_text = (
            f"{history_context}"
            f"**Current State:**\n"
            f"Score: {score}\n"
            f"Step: {self._action_counter}\n\n"
            f"**Current Matrix** {grid_text} (ASCII characters):\n{grid}\n\n"
            "Rationale:\n"
            "  • Identify the movable ASCII character(s) and relevant structures.\n"
            "  • Conclude which direction is best and why. Do not output an action here.\n"
            "  • Focus on the strategic importance of each character and how it relates to the goal."
        )

        # Store prompt for logging
        self._last_observation_prompt = f"SYSTEM: {sys_msg}\n\nUSER: {user_msg_text}"

        # Use text-only content for rollout engine (multimodal not supported)
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg_text}
        ]

        model_output = await self.rollout(rollout_engine, messages)
        text = (getattr(model_output, "content", None) or getattr(model_output, "text", "") or "").strip()

        # Store response for logging
        self._last_observation_response = text

        return text

    @weave_op
    async def _call_action_model(self, grid: List[List[int]], last_obs: str, rollout_engine=None) -> dict:
        """Call the model for action selection phase."""
        sys_msg = build_action_system_text()

        history_context = self._format_step_history()

        user_msg_text = (
            f"{history_context}"
            "Choose the best single move as a function call.\n"
            f"{grid}"
            "Previous observation summary:\n"
            f"{last_obs}\n"
        )

        # Store prompt for logging
        self._last_action_prompt = f"SYSTEM: {sys_msg}\n\nUSER: {user_msg_text}"

        tools = _build_tools()

        # Use text-only content for rollout engine (multimodal not supported)
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg_text}
        ]

        model_output = await self.rollout(rollout_engine, messages, tools)

        m = model_output.tool_calls[0] if getattr(model_output, "tool_calls", None) else None

        print(f"[DEBUG]:guided_text_16:model_output={model_output}")
        print(f"[DEBUG]:guided_text_16:rollout_engine._use_chat_completions={getattr(rollout_engine, '_use_chat_completions', 'N/A')}")

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

        # Store response for logging
        action_content = model_output.content or ""
        self._last_action_response = f"Tool Call: {name}({json.dumps(args)})\nContent: {action_content}"

        # Handle ACTION6 coordinate mapping if needed
        if name == "ACTION6":
            x_raw = args.get("x", 0)
            y_raw = args.get("y", 0)
            # Scale coordinates to 64x64 game space only if using downsampled 16x16 grid
            if self.downsample:
                x_64 = x_raw * 4
                y_64 = y_raw * 4
            else:
                x_64 = x_raw
                y_64 = y_raw
            return {"name": name, "data": {"x": x_64, "y": y_64}, "action_text": model_output.content}

        return {"name": name, "data": args, "action_text": model_output.content}
