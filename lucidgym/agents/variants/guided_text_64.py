"""
64x64 full-resolution text-based guided agent.
Identical to guided_text_16 but operates on full 64x64 grids without downsampling.
"""
from __future__ import annotations
from typing import Any, List, Dict
import json
import logging
import base64
from openai import OpenAI

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from lucidgym.agents.arcagi3_agent import ArcAgi3Agent

from lucidgym.environments.arcagi3.structs import GameAction, GameState
from lucidgym.utils.grid_processing import frame_to_grid_text, generate_numeric_grid_image_bytes
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
                        "x": {"type": "integer", "description": "X coordinate (0-63 for 64x64 grid)"},
                        "y": {"type": "integer", "description": "Y coordinate (0-63 for 64x64 grid)"},
                    },
                    "required": ["x", "y"],
                },
            },
        },
    ]


class AS66GuidedAgent64(ArcAgi3Agent):
    """
    64×64 full-resolution agent using rllm BaseAgent interface.
    No downsampling - operates on full game grid.
    Supports observation + single tool call per turn with multimodal input modes.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        name: str = "as66_guided_agent_64",
        input_mode: str = "text_only",
        model: str = "gpt-4o",
        reasoning_effort: str = "none",
        game_id: str | None = None,
        representation: "RepresentationConfig | None" = None,
        use_general: bool = False,
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
            representation: Grid representation configuration
            use_general: If True, use general learning prompts; if False, use game-specific prompts
        """
        from lucidgym.utils.representation import RepresentationConfig
        self.representation = representation or RepresentationConfig(downsample=False)
        super().__init__(system_prompt=system_prompt, name=name, representation=self.representation)
        self.input_mode = input_mode
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.game_id = game_id
        self.use_general = use_general
        self._system_prompt_override = system_prompt

        if self.input_mode not in ["text_only", "image_only", "text_and_image"]:
            log.warning(f"Invalid input_mode '{self.input_mode}', defaulting to 'text_only'.")
            self.input_mode = "text_only"

        self._client = OpenAI()
        self._latest_tool_call_id = "call_12345"
        self.reset()

    def reset(self) -> None:
        """Reset agent state for new episode."""
        super().reset()  # Initialize _steps_this_episode, _chat_history, _trajectory, _last_observation
        self._token_total: int = 0
        self._action_counter: int = 0
        self._pending_action: dict[str, Any] | None = None

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return message history formatted for chat API."""
        system_msg = self._system_prompt_override or build_observation_system_text(self.game_id, use_general=self.use_general)
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
        obs_text = action_dict.get("obs_text", "")
        action_text = action_dict.get("action_text", "")
        response_text = f"Observation: {obs_text}\nAction Text: {action_text}\nAction: {action_dict['name']}"

        if action_dict is None:
            # Fallback to internal generation to preserve legacy behavior
            response_text, action_dict = self.call_llm()

        if not response_text:
            response_text = str(action_dict)

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
        if state in ("NOT_PLAYED", "GAME_OVER"):
            action_dict = {"name": "RESET", "data": {}, "obs_text": "Game Over, starting new game.", "action_text": ""}
            self._pending_action = action_dict
            return action_dict

        # Extract frame - use full 64x64 grid (no downsampling)
        frame_3d = obs.get("frame", [])
        # Pass raw integer grid, not ASCII-converted - prompts describe integer codes
        grid_64 = frame_3d[-1] if frame_3d else []
        score = obs.get("score", 0)

        # Step 1: Observation phase
        obs_text = await self._call_observation_model(grid_64, score, rollout_engine=rollout_engine)

        # Step 2: Action selection phase
        action_dict = await self._call_action_model(grid_64, obs_text, rollout_engine=rollout_engine)
        action_dict["obs_text"] = obs_text

        # Stash for update_from_model to record
        self._pending_action = action_dict
        return action_dict

    def _build_user_content(self, grid_64: List[List[int]], user_prompt_text: str) -> List[Dict[str, Any]]:
        """Build the 'content' array for the API call based on input_mode."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt_text}]

        if self.input_mode in ["image_only", "text_and_image"]:
            try:
                png_bytes = generate_numeric_grid_image_bytes(grid_64)
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

    async def rollout(self, rollout_engine, messages: List[Dict[str, Any]], tools=None):
        return await rollout_engine.get_model_response(messages, tools=tools)

    async def _call_observation_model(self, grid_64: List[List[int]], score: int, rollout_engine=None) -> str:
        """Call the model for observation/reasoning phase."""
        sys_msg = build_observation_system_text(self.game_id, use_general=self.use_general, grid_size="64x64")

        include_text = self.input_mode in ["text_only", "text_and_image"]
        format_clarification = ""
        if self.input_mode == "image_only":
            format_clarification = "The board state is provided as an attached image of the 64x64 grid."
        elif self.input_mode == "text_and_image":
            format_clarification = "The board state is provided as both a textual matrix and an attached image."

        user_msg_text = build_observation_user_text(
            grid_64, score, self._action_counter, self.game_id,
            format_clarification=format_clarification,
            include_text_matrix=include_text,
            use_general=self.use_general,
            grid_size="64x64"
        )

        user_content = self._build_user_content(grid_64, user_msg_text)

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content}
        ]

        model_output = await self.rollout(rollout_engine, messages)
        text = (getattr(model_output, "content", None) or getattr(model_output, "text", "") or "").strip()
        return text

    async def _call_action_model(self, grid_64: List[List[int]], last_obs: str, rollout_engine=None) -> dict:
        """Call the model for action selection phase."""
        sys_msg = build_action_system_text(self.game_id, use_general=self.use_general, grid_size="64x64")

        include_text = self.input_mode in ["text_only", "text_and_image"]
        format_clarification = ""
        if self.input_mode == "image_only":
            format_clarification = "The board state is provided as an attached image."
        elif self.input_mode == "text_and_image":
            format_clarification = "The board state is provided as both text and image."

        user_msg_text = build_action_user_text(
            grid_64, last_obs, self.game_id,
            format_clarification=format_clarification,
            include_text_matrix=include_text,
            use_general=self.use_general,
            grid_size="64x64"
        )

        user_content = self._build_user_content(grid_64, user_msg_text)
        tools = _build_tools()

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content}
        ]

        model_output = await self.rollout(rollout_engine, messages, tools)

        m = model_output.tool_calls[0] if getattr(model_output, "tool_calls", None) else None

        print(f"[DEBUG]:guided_text_64:model_output={model_output}")
        print(f"[DEBUG]:guided_text_64:rollout_engine._use_chat_completions={getattr(rollout_engine, '_use_chat_completions', 'N/A')}")

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

        # Handle ACTION6 coordinates - no scaling needed for 64x64
        if name == "ACTION6":
            x_64 = args.get("x", 0)
            y_64 = args.get("y", 0)
            return {"name": name, "data": {"x": x_64, "y": y_64}, "action_text": model_output.content}

        return {"name": name, "data": args, "action_text": model_output.content}
