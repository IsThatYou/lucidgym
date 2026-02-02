"""
16x16 text-based agent with cycle detection.
Uses a simple directed graph to detect when the agent is revisiting states.
"""
from __future__ import annotations

from typing import Any, Optional, List, Dict, Set
from collections import deque
import json
import logging
import hashlib

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from lucidgym.agents.arcagi3_agent import ArcAgi3Agent
from lucidgym.environments.arcagi3.structs import GameAction, GameState
from lucidgym.utils.grid_processing import frame_to_grid_text, downsample_4x4

try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    weave = None

log = logging.getLogger(__name__)


def weave_op(func):
    if WEAVE_AVAILABLE and weave:
        return weave.op(func)
    return func


def build_observation_system_text():
    return (
        "You are playing a game represented by a 16x16 grid.\n"
        "Your task is to observe the position and analyze potential moves.\n\n"
        "Movement model:\n"
        "- There is one main movable piece. It may be a unique integer or small block.\n"
        "- When you choose a direction (Up, Down, Left, Right), the piece slides until blocked.\n"
        "- Sliding can wrap across board edges if unobstructed.\n"
        "- If no obstacles in a direction, the piece returns to start (no movement).\n\n"
        "For observation, analyze:\n"
        "1. **FIRST: Check KNOWN CYCLES** - These are action sequences that return to this exact state\n"
        "2. Locate the movable piece(s) and key structures\n"
        "3. For each direction, simulate where the piece would land\n"
        "4. Determine which direction best progresses toward the goal\n\n"
        "DO NOT call an action tool here - only provide analysis.\n\n"
        "CRITICAL: If KNOWN CYCLES are listed, you MUST avoid starting any of those action sequences!"
    )


def build_action_system_text():
    return (
        "Select exactly one move by calling a single tool. Do not include prose.\n"
        "Available tools:\n"
        "- ACTION1 = Up\n"
        "- ACTION2 = Down\n"
        "- ACTION3 = Left\n"
        "- ACTION4 = Right\n\n"
        "IMPORTANT: If KNOWN CYCLES are listed, do NOT choose an action that starts any of those sequences!"
    )


def _build_tools() -> list[dict]:
    return [
        {"type": "function", "function": {"name": "RESET", "description": "Reset the game", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "ACTION1", "description": "Move Up", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "ACTION2", "description": "Move Down", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "ACTION3", "description": "Move Left", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "ACTION4", "description": "Move Right", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "ACTION5", "description": "Spacebar / Enter / No-op", "parameters": {"type": "object", "properties": {}, "required": []}}},
    ]


class CycleDetector:
    """Directed graph for cycle detection - finds all cycles from current state."""

    def __init__(self, max_cycle_length: int = 8):
        self.max_cycle_length = max_cycle_length
        self.reset()

    def reset(self):
        # Map: state_hash -> {action -> resulting_state_hash}
        self._transitions: Dict[str, Dict[str, str]] = {}
        # Track visit counts per state
        self._visit_counts: Dict[str, int] = {}

    def _hash_state(self, grid_text: str) -> str:
        return hashlib.md5(grid_text.encode()).hexdigest()[:12]

    def record_transition(self, from_grid: str, action: str, to_grid: str) -> None:
        """Record a state transition."""
        from_hash = self._hash_state(from_grid)
        to_hash = self._hash_state(to_grid)

        if from_hash not in self._transitions:
            self._transitions[from_hash] = {}
        self._transitions[from_hash][action] = to_hash

        # Track visits
        self._visit_counts[to_hash] = self._visit_counts.get(to_hash, 0) + 1

    def get_tried_actions(self, grid_text: str) -> Dict[str, str]:
        """Get actions already tried from this state and their results."""
        state_hash = self._hash_state(grid_text)
        return self._transitions.get(state_hash, {})

    def find_all_cycles(self, start_grid: str) -> List[List[str]]:
        """
        Find all cycles that start and end at the given state.
        Returns list of action trajectories (each trajectory is a list of action names).
        Uses DFS with path tracking.
        """
        start_hash = self._hash_state(start_grid)
        cycles: List[List[str]] = []

        def dfs(current_hash: str, path: List[str], visited: Set[str]):
            # Don't exceed max cycle length
            if len(path) > self.max_cycle_length:
                return

            # Get all outgoing transitions from current state
            transitions = self._transitions.get(current_hash, {})

            for action, next_hash in transitions.items():
                new_path = path + [action]

                # Found a cycle back to start!
                if next_hash == start_hash and len(new_path) > 0:
                    cycles.append(new_path)
                    continue

                # Continue exploring if we haven't visited this state in current path
                if next_hash not in visited:
                    new_visited = visited | {next_hash}
                    dfs(next_hash, new_path, new_visited)

        # Start DFS from the start state
        dfs(start_hash, [], {start_hash})

        # Sort by cycle length
        cycles.sort(key=len)
        return cycles

    def get_untried_actions(self, grid_text: str, all_actions: List[str]) -> List[str]:
        """Get actions that haven't been tried from this state."""
        tried = set(self.get_tried_actions(grid_text).keys())
        return [a for a in all_actions if a not in tried]

    def build_context(self, current_grid: str) -> str:
        """Build context string showing all known cycles from this state."""
        lines = []
        current_hash = self._hash_state(current_grid)

        # Find all cycles from this state
        cycles = self.find_all_cycles(current_grid)

        if cycles:
            lines.append("=" * 60)
            lines.append("KNOWN CYCLES FROM THIS STATE (these action sequences return here):")
            lines.append("=" * 60)
            for i, cycle in enumerate(cycles, 1):
                trajectory = " -> ".join(cycle)
                lines.append(f"  Cycle {i}: {trajectory}")
            lines.append("")
            lines.append("AVOID these action sequences - they lead back to this same state!")
            lines.append("=" * 60)
            lines.append("")

        # Show tried actions from this state
        tried = self.get_tried_actions(current_grid)
        if tried:
            lines.append("**Actions tried from this state:**")
            for action, result_hash in tried.items():
                same_state = (result_hash == current_hash)
                if same_state:
                    lines.append(f"  - {action}: NO EFFECT (same state)")
                else:
                    lines.append(f"  - {action}: -> different state")
            lines.append("")

        # Suggest untried actions
        all_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        untried = self.get_untried_actions(current_grid, all_actions)
        if untried:
            lines.append(f"**Untried actions from this state:** {', '.join(untried)}")
            lines.append("")

        if lines:
            return "\n".join(lines) + "\n"
        return ""


class BasicObsActionAgentCycleDetection(ArcAgi3Agent):
    """Agent with cycle detection to avoid getting stuck in loops."""

    def __init__(
        self,
        system_prompt: str | None = None,
        name: str = "basic_obs_action_agent_cycle_detection",
        model: str = "gpt-5-nano",
        reasoning_effort: str = "low",
        downsample=True,
        game_id: str | None = None,
        context_window_size: int = 5,
        crop_border: int = 0,
        use_as66_prompts: bool = True,
        **kwargs,  # Accept extra kwargs for harness compatibility
    ) -> None:
        self.crop_border = crop_border
        self.context_window_size = context_window_size

        super().__init__(system_prompt=system_prompt, name=name)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.game_id = game_id
        self._system_prompt_override = system_prompt
        self.downsample = downsample
        self._latest_tool_call_id = "call_12345"

    def reset(self) -> None:
        super().reset()
        self._chat_history: list[dict] = []
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None
        self._action_counter: int = 0
        self._pending_action: dict[str, Any] | None = None
        self._cycle_detector = CycleDetector()
        self._last_grid_text: str = ""
        self._last_observation_prompt: str = ""
        self._last_observation_response: str = ""
        self._last_action_prompt: str = ""
        self._last_action_response: str = ""
        self._last_executed_action: str | None = None

    def _crop_grid(self, grid: List[List[int]]) -> List[List[int]]:
        if self.crop_border <= 0:
            return grid
        c = self.crop_border
        return [row[c:-c] for row in grid[c:-c]]

    def _get_grid_text(self, obs: dict) -> str:
        frame_3d = obs.get("frame", [])
        if not frame_3d:
            return ""
        if self.downsample:
            downsampled = downsample_4x4(frame_3d)
            cropped = self._crop_grid(downsampled)
            return frame_to_grid_text([cropped])
        return frame_to_grid_text([frame_3d])

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **_: Any) -> None:
        # Record transition in cycle detector
        new_grid_text = self._get_grid_text(observation)
        if self._last_grid_text and self._last_executed_action and new_grid_text:
            self._cycle_detector.record_transition(
                self._last_grid_text,
                self._last_executed_action,
                new_grid_text
            )

        self._last_observation = observation
        self._last_grid_text = new_grid_text

        # Build prompts log
        full_prompts = []
        if self._last_observation_prompt:
            full_prompts.append({"role": "observation_phase", "content": self._last_observation_prompt})
        if self._last_observation_response:
            full_prompts.append({"role": "observation_response", "content": self._last_observation_response})
        if self._last_action_prompt:
            full_prompts.append({"role": "action_phase", "content": self._last_action_prompt})
        if self._last_action_response:
            full_prompts.append({"role": "action_response", "content": self._last_action_response})

        step = Step(observation=observation, reward=reward, done=done, info=info, chat_completions=full_prompts)
        self._trajectory.steps.append(step)

    def update_from_model(self, action_payload: dict | None = None, **_: Any) -> Action:
        action_dict = action_payload or self._pending_action
        if action_dict is None:
            action_dict = {"name": "ACTION1", "data": {}, "obs_text": "", "action_text": ""}

        obs_text = action_dict.get("obs_text", "")
        action_text = action_dict.get("action_text", "")
        response_text = f"Observation: {obs_text}\nAction: {action_dict['name']}"

        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = response_text
            self._trajectory.steps[-1].action = action_dict

        self._action_counter += 1
        self._pending_action = None
        self._last_executed_action = action_dict["name"]

        action = GameAction.from_name(action_dict["name"])
        return {"action": action, "reasoning": response_text}

    @weave_op
    async def call_llm(self, rollout_engine=None) -> dict:
        obs = self._last_observation or {}
        state = obs.get("state", "NOT_PLAYED")

        if state in ("NOT_PLAYED", "GAME_OVER") and self._last_executed_action != "RESET":
            action_dict = {"name": "RESET", "data": {}, "obs_text": "Game Over", "action_text": ""}
            self._pending_action = action_dict
            return action_dict

        grid_text = self._get_grid_text(obs)
        score = obs.get("score", 0)

        # Get cycle detection context
        cycle_context = self._cycle_detector.build_context(grid_text)

        # Observation phase
        obs_text = await self._call_observation_model(grid_text, score, cycle_context, rollout_engine)

        # Action phase
        action_dict = await self._call_action_model(grid_text, obs_text, cycle_context, rollout_engine)
        action_dict["obs_text"] = obs_text

        self._pending_action = action_dict
        return action_dict

    async def _call_observation_model(self, grid: str, score: int, cycle_context: str, rollout_engine) -> str:
        sys_msg = build_observation_system_text()

        if self.downsample:
            actual_size = 16 - (2 * self.crop_border)
            grid_size = f"{actual_size}x{actual_size}"
        else:
            grid_size = "64x64"

        user_msg = (
            f"{cycle_context}"
            f"**Current State:**\n"
            f"Score: {score}\n"
            f"Step: {self._action_counter}\n\n"
            f"**Current Matrix** {grid_size}:\n{grid}\n\n"
            "Analyze the board and determine the best move."
        )

        self._last_observation_prompt = f"SYSTEM: {sys_msg}\n\nUSER: {user_msg}"

        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
        model_output = await rollout_engine.get_model_response(messages)
        text = (getattr(model_output, "content", None) or getattr(model_output, "text", "") or "").strip()

        self._last_observation_response = text
        return text

    async def _call_action_model(self, grid: str, obs_text: str, cycle_context: str, rollout_engine) -> dict:
        sys_msg = build_action_system_text()

        user_msg = (
            f"{cycle_context}"
            f"Choose the best move based on your analysis.\n"
            f"{grid}\n\n"
            f"Analysis: {obs_text}\n"
        )

        self._last_action_prompt = f"SYSTEM: {sys_msg}\n\nUSER: {user_msg}"

        tools = _build_tools()
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
        model_output = await rollout_engine.get_model_response(messages, tools=tools)

        tc = model_output.tool_calls[0] if getattr(model_output, "tool_calls", None) else None

        if tc is None:
            return {"name": "ACTION5", "data": {}, "action_text": ""}

        self._latest_tool_call_id = tc.id
        name = tc.function.name

        try:
            args = json.loads(tc.function.arguments or "{}")
        except:
            args = {}

        self._last_action_response = f"Tool Call: {name}({json.dumps(args)})"

        return {"name": name, "data": args, "action_text": model_output.content or ""}
