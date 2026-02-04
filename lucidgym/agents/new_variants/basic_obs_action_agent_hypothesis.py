"""
Basic observation-action agent with hypothesis tracking.
Updates hypotheses every N turns instead of every turn.
Combines rolling context with periodic hypothesis updates.
"""
from __future__ import annotations
from typing import Any, Optional, List, Dict
from collections import deque
import json
import logging
import base64

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from lucidgym.agents.arcagi3_agent import ArcAgi3Agent

from arcengine import GameAction, GameState
from lucidgym.utils.grid_processing import downsample_grid, generate_numeric_grid_image_bytes, format_grid
from lucidgym.utils.representation import RepresentationConfig, GridFormat

log = logging.getLogger(__name__)


def build_initial_hypotheses_prompt(click_only: bool = False) -> str:
    """System prompt for generating initial hypotheses."""
    action_desc = "clicking cells at (x, y) coordinates" if click_only else "directional movement (Up/Down/Left/Right) or clicking cells"

    return (
        "You are analyzing a grid-based puzzle game to understand its rules.\n\n"
        f"The game uses {action_desc}.\n"
        "The grid uses ASCII characters to represent different elements.\n\n"
        "Generate 3-5 hypotheses about:\n"
        "1. What different characters represent (player, walls, goals, collectibles, etc.)\n"
        "2. How the game mechanics work (movement, interactions, win conditions)\n"
        "3. What actions are likely to cause progress\n\n"
        "For each hypothesis, include:\n"
        "- A clear description of the rule\n"
        "- How to test it\n"
        "- Confidence level (low/medium/high)\n\n"
        "Be concise but specific."
    )


def build_update_hypotheses_prompt() -> str:
    """System prompt for updating hypotheses."""
    return (
        "Review the recent game history and update your hypotheses.\n\n"
        "For each hypothesis:\n"
        "- If evidence supports it, increase confidence\n"
        "- If evidence contradicts it, revise or discard it\n"
        "- If no relevant evidence, keep it unchanged\n\n"
        "Pay attention to:\n"
        "- Actions that caused state changes vs no changes\n"
        "- Patterns in successful vs unsuccessful moves\n"
        "- Level ups or game overs and what preceded them\n\n"
        "Output updated hypotheses with confidence levels."
    )


def build_observation_prompt(click_only: bool = False) -> str:
    """System prompt for observation phase."""
    if click_only:
        return (
            "Analyze the grid and your hypotheses to decide which cell to click.\n\n"
            "Consider:\n"
            "1. Your current hypotheses about the game\n"
            "2. Recent history - avoid repeating failed actions\n"
            "3. Patterns in the grid that match your hypotheses\n\n"
            "Recommend specific coordinates (x=row, y=column) to click.\n"
            "Explain your reasoning based on your hypotheses."
        )
    else:
        return (
            "Analyze the grid and your hypotheses to decide the best action.\n\n"
            "Consider:\n"
            "1. Your current hypotheses about the game\n"
            "2. Recent history - avoid repeating failed actions\n"
            "3. Patterns in the grid that match your hypotheses\n\n"
            "Recommend a specific action and explain your reasoning."
        )


def build_action_prompt(click_only: bool = False) -> str:
    """System prompt for action phase."""
    if click_only:
        return (
            "Execute the click recommended in your analysis.\n"
            "Call ACTION6 with the EXACT coordinates from your reasoning.\n\n"
            "Available: ACTION6(x, y) - click cell at row x, column y"
        )
    else:
        return (
            "Execute the action recommended in your analysis.\n\n"
            "Available actions:\n"
            "- ACTION1: Up\n"
            "- ACTION2: Down\n"
            "- ACTION3: Left\n"
            "- ACTION4: Right\n"
            "- ACTION5: Confirm/Enter\n"
            "- ACTION6(x, y): Click cell"
        )


def _build_tools(click_only: bool = False) -> list[dict]:
    """Build tool definitions."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "RESET",
                "description": "Reset the game",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    if not click_only:
        tools.extend([
            {"type": "function", "function": {"name": "ACTION1", "description": "Move Up", "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {"name": "ACTION2", "description": "Move Down", "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {"name": "ACTION3", "description": "Move Left", "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {"name": "ACTION4", "description": "Move Right", "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {"name": "ACTION5", "description": "Confirm/Enter", "parameters": {"type": "object", "properties": {}, "required": []}}},
        ])

    tools.append({
        "type": "function",
        "function": {
            "name": "ACTION6",
            "description": "Click at (x, y)",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Row (0-indexed)"},
                    "y": {"type": "integer", "description": "Column (0-indexed)"},
                },
                "required": ["x", "y"],
            },
        },
    })

    return tools


class BasicObsActionAgentHypothesis(ArcAgi3Agent):
    """
    Agent with rolling context and periodic hypothesis updates.
    Updates hypotheses every N turns instead of every turn.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        name: str = "basic_obs_action_agent_hypothesis",
        input_mode: str = "text_only",
        model: str = "gpt-5-nano",
        reasoning_effort: str = "low",
        downsample: bool = True,
        game_id: str | None = None,
        context_window_size: int = 5,
        crop_border: int = 0,
        representation: RepresentationConfig | None = None,
        downsample_block_size: int = 4,
        use_mode: bool = True,
        click_only: bool = False,
        hypothesis_update_interval: int = 5,  # Update hypotheses every N turns
    ) -> None:
        """
        Initialize the agent.

        Args:
            hypothesis_update_interval: Update hypotheses every N turns (default 5)
        """
        self.context_window_size = context_window_size
        self.crop_border = crop_border
        self.downsample_block_size = downsample_block_size
        self.use_mode = use_mode
        self.click_only = click_only
        self.hypothesis_update_interval = hypothesis_update_interval

        _representation = representation or RepresentationConfig(format=GridFormat.ASCII)

        super().__init__(system_prompt=system_prompt, name=name)

        self.representation = _representation
        self.input_mode = input_mode
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.game_id = game_id
        self._system_prompt_override = system_prompt
        self.downsample = downsample

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self._chat_history: list[dict] = []
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None
        self._token_total: int = 0
        self._action_counter: int = 0
        self._pending_action: dict[str, Any] | None = None
        self._step_history: deque = deque(maxlen=self.context_window_size)
        self._last_executed_action: str | None = None

        # Hypothesis tracking
        self._hypotheses: str = ""
        self._hypotheses_initialized: bool = False
        self._turns_since_hypothesis_update: int = 0

        # Prompt logging
        self._last_observation_prompt: str = ""
        self._last_observation_response: str = ""
        self._last_action_prompt: str = ""
        self._last_action_response: str = ""
        self._latest_tool_call_id: str = "call_12345"

    def _format_step_history(self) -> str:
        """Format step history."""
        if not self._step_history:
            return ""

        lines = ["**Recent History:**\n"]
        for step in self._step_history:
            marker = " ⚠️ NO STATE CHANGE" if step.get('no_state_change') else ""
            lines.append(f"Step {step['step']}: {step['action']}, Score={step['score']}{marker}\n")
        return "\n".join(lines) + "\n"

    def _crop_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """Crop border from grid."""
        if self.crop_border <= 0:
            return grid
        c = self.crop_border
        return [row[c:-c] for row in grid[c:-c]]

    def _format_grid(self, grid: List[List[int]]) -> str:
        """Format grid."""
        return format_grid(grid, self.representation)

    @property
    def chat_completions(self) -> list[dict]:
        return self._chat_history

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict = None, **_: Any) -> None:
        """Process environment observation."""
        self._last_observation = observation

        # Build log
        prompts = []
        if self._last_observation_prompt:
            prompts.append({"role": "observation_phase", "content": self._last_observation_prompt})
        if self._last_observation_response:
            prompts.append({"role": "observation_response", "content": self._last_observation_response})
        if self._last_action_prompt:
            prompts.append({"role": "action_phase", "content": self._last_action_prompt})
        if self._last_action_response:
            prompts.append({"role": "action_response", "content": self._last_action_response})

        step = Step(observation=observation, reward=reward, done=done, info=info, chat_completions=prompts)
        self._trajectory.steps.append(step)

    def update_from_model(self, action_payload: dict | None = None, **_: Any) -> Action:
        """Convert model response to Action."""
        action_dict = action_payload or self._pending_action

        obs = self._last_observation or {}
        frame_3d = obs.get("frame", [])

        # Get grid for history
        if self.downsample and len(frame_3d) > 0:
            downsampled = downsample_grid(frame_3d, block_size=self.downsample_block_size, use_mode=self.use_mode)
            cropped = self._crop_grid(downsampled)
            grid_str = self._format_grid(cropped)
        elif len(frame_3d) > 0:
            grid_str = self._format_grid(frame_3d[-1])
        else:
            grid_str = ""

        # Detect no state change
        no_state_change = False
        if self._step_history:
            if self._step_history[-1].get("grid", "") == grid_str:
                no_state_change = True

        # Format action display
        action_name = action_dict["name"]
        if action_name == "ACTION6":
            data = action_dict.get("data", {})
            action_display = f"ACTION6(x={data.get('x', 0)}, y={data.get('y', 0)})"
        else:
            action_display = action_name

        self._step_history.append({
            "step": self._action_counter,
            "action": action_display,
            "score": obs.get("score", 0),
            "state": obs.get("state", "UNKNOWN"),
            "grid": grid_str,
            "no_state_change": no_state_change
        })

        self._action_counter += 1
        self._turns_since_hypothesis_update += 1
        self._pending_action = None
        self._last_executed_action = action_dict["name"]

        action = GameAction.from_name(action_dict["name"])
        result = {"action": action, "reasoning": action_dict.get("obs_text", "")}
        if action == GameAction.ACTION6:
            result["x"] = action_dict["data"]["x"]
            result["y"] = action_dict["data"]["y"]
        return result

    async def call_llm(self, rollout_engine=None) -> dict:
        """Main LLM call with hypothesis management."""
        obs = self._last_observation or {}
        state = obs.get("state", "NOT_PLAYED")

        # Auto-reset
        if state in ("NOT_PLAYED", "GAME_OVER") and self._last_executed_action != "RESET":
            action_dict = {"name": "RESET", "data": {}, "obs_text": "Resetting game."}
            self._pending_action = action_dict
            return action_dict

        # Get grid
        frame_3d = obs.get("frame", [])
        if self.downsample:
            downsampled = downsample_grid(frame_3d, block_size=self.downsample_block_size, use_mode=self.use_mode)
            cropped = self._crop_grid(downsampled)
            grid = self._format_grid(cropped)
        else:
            grid = self._format_grid(frame_3d[-1] if frame_3d else [])

        score = obs.get("score", 0)

        # Calculate grid size
        if self.downsample:
            base_size = 64 // self.downsample_block_size
            grid_size = f"{base_size}x{base_size}"
        else:
            grid_size = "64x64"

        # Initialize or update hypotheses
        if not self._hypotheses_initialized:
            self._hypotheses = await self._generate_initial_hypotheses(grid, rollout_engine)
            self._hypotheses_initialized = True
            self._turns_since_hypothesis_update = 0
        elif self._turns_since_hypothesis_update >= self.hypothesis_update_interval:
            self._hypotheses = await self._update_hypotheses(rollout_engine)
            self._turns_since_hypothesis_update = 0

        # Observation phase
        obs_text = await self._call_observation(grid, grid_size, score, rollout_engine)

        # Action phase
        action_dict = await self._call_action(obs_text, rollout_engine)
        action_dict["obs_text"] = obs_text

        self._pending_action = action_dict
        return action_dict

    async def _generate_initial_hypotheses(self, grid: str, rollout_engine) -> str:
        """Generate initial hypotheses."""
        sys_msg = build_initial_hypotheses_prompt(self.click_only)
        user_msg = f"**Initial Grid:**\n{grid}\n\nGenerate hypotheses about this game."

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        response = await rollout_engine.get_model_response(messages)
        return (getattr(response, "content", None) or "").strip()

    async def _update_hypotheses(self, rollout_engine) -> str:
        """Update hypotheses based on recent history."""
        sys_msg = build_update_hypotheses_prompt()

        history = self._format_step_history()
        user_msg = (
            f"**Current Hypotheses:**\n{self._hypotheses}\n\n"
            f"{history}\n"
            "Update hypotheses based on this evidence."
        )

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        response = await rollout_engine.get_model_response(messages)
        return (getattr(response, "content", None) or self._hypotheses).strip()

    async def _call_observation(self, grid: str, grid_size: str, score: int, rollout_engine) -> str:
        """Observation phase."""
        sys_msg = build_observation_prompt(self.click_only)

        history = self._format_step_history()
        user_msg = (
            f"**Hypotheses:**\n{self._hypotheses}\n\n"
            f"{history}"
            f"**Current State:** Score={score}, Step={self._action_counter}\n\n"
            f"**Grid ({grid_size}):**\n{grid}\n\n"
            "Analyze and recommend an action."
        )

        self._last_observation_prompt = f"SYSTEM: {sys_msg}\n\nUSER: {user_msg}"

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        response = await rollout_engine.get_model_response(messages)
        text = (getattr(response, "content", None) or "").strip()
        self._last_observation_response = text
        return text

    async def _call_action(self, obs_text: str, rollout_engine) -> dict:
        """Action phase."""
        sys_msg = build_action_prompt(self.click_only)

        if self.click_only:
            user_msg = f"**Analysis:**\n{obs_text}\n\nExecute ACTION6 with the recommended coordinates."
        else:
            user_msg = f"**Analysis:**\n{obs_text}\n\nExecute the recommended action."

        self._last_action_prompt = f"SYSTEM: {sys_msg}\n\nUSER: {user_msg}"

        tools = _build_tools(self.click_only)
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        response = await rollout_engine.get_model_response(messages, tools=tools)

        tc = response.tool_calls[0] if getattr(response, "tool_calls", None) else None

        if tc is None:
            self._last_action_response = "No tool call, defaulting to ACTION5"
            return {"name": "ACTION5", "data": {}}

        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except:
            args = {}

        self._last_action_response = f"Tool: {name}({json.dumps(args)})"

        # Handle ACTION6 coordinate scaling
        if name == "ACTION6":
            x_raw = args.get("x", 0)
            y_raw = args.get("y", 0)
            if self.downsample:
                x_64 = x_raw * self.downsample_block_size
                y_64 = y_raw * self.downsample_block_size
            else:
                x_64, y_64 = x_raw, y_raw
            return {"name": name, "data": {"x": x_64, "y": y_64}}

        return {"name": name, "data": args}
