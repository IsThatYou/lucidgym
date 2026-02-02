"""
Graph Explorer Observation-Action Agent.

Combines:
1. Two-phase observation/action LLM calls (like basic_obs_action agents)
2. GraphExplorer systematic exploration memory

The agent:
- Phase 1 (Observation): Analyzes the board state, identifies pieces, obstacles, goals
- Phase 2 (Action): Chooses action informed by BOTH observation AND graph exploration memory
- GraphExplorer tracks which actions have been tried and guides systematic exploration
"""
from __future__ import annotations
from typing import Any, Optional, List, Dict, Hashable, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
import base64
import hashlib
import numpy as np

from lucidgym.utils.openai_client import get_openai_client
from lucidgym.utils.representation import RepresentationConfig
from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from lucidgym.agents.arcagi3_agent import ArcAgi3Agent

from arcengine import GameAction, GameState
from lucidgym.utils.grid_processing import downsample_4x4, generate_numeric_grid_image_bytes

# Consonants palette for grid display (tokenizes well, no confusing symbols)
CONSONANTS = "BCDFGHJKLMNPQRSTVWXZ"


def grid_to_ascii(grid: list[list[int]]) -> str:
    """Convert grid to ASCII using consonants palette."""
    max_val = 16
    lines = []
    for row in grid:
        chars = []
        for v in row:
            idx = int(v) % len(CONSONANTS)
            chars.append(CONSONANTS[idx])
        lines.append("".join(chars))
    return "\n".join(lines)

# Import GraphExplorer from the other module
from lucidgym.agents.new_variants.graph_explorer_agent import GraphExplorer

# Optional Weave integration
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    weave = None

log = logging.getLogger(__name__)


def weave_op(func):
    """Apply @weave.op decorator if weave is available, otherwise no-op."""
    if WEAVE_AVAILABLE and weave:
        return weave.op(func)
    return func


def build_observation_system_text(use_as66_prompts: bool = False):
    if use_as66_prompts:
        return (
            "You are playing a game represented by a 16×16 grid.\n"
            "Your task is to observe the position and analyze potential moves.\n\n"
            "Movement model:\n"
            "- There is one main movable piece. It may be a unique integer or small block.\n"
            "- When you choose a direction (Up, Down, Left, Right), the piece slides until blocked.\n"
            "- Sliding can wrap across board edges if unobstructed.\n"
            "- If no obstacles in a direction, the piece returns to start (no movement).\n\n"
            "Obstacles and structures:\n"
            "- Walls block movement (you stop adjacent to them).\n"
            "- Target region forms a U-shape (2x3 with center removed). Fill it to win.\n"
            "- Background cells are the playable area.\n"
            "- Boundaries delimit the play field.\n"
            "- Some levels have enemies (large blocks) - collision means game over.\n\n"
            "For observation, analyze:\n"
            "1. Locate the movable piece(s) and key structures\n"
            "2. For each direction, simulate where the piece would land\n"
            "3. Consider enemy movement if present\n"
            "4. Determine which direction best progresses toward the goal\n\n"
            "DO NOT call an action tool here - only provide analysis."
        )
    else:
        return (
            "You are observing a 16x16 grid representation of a game state. "
            "Each cell contains an ASCII character representing different game elements. "
            "Your task is to analyze this grid and determine the best action to take. "
            "DO NOT call an action tool here - only provide analysis."
        )


def build_action_system_text(use_as66_prompts: bool = False):
    if use_as66_prompts:
        return (
            "Select exactly one move by calling a single tool.\n"
            "Available tools:\n"
            "- ACTION1 = Up\n"
            "- ACTION2 = Down\n"
            "- ACTION3 = Left\n"
            "- ACTION4 = Right\n\n"
            "You have access to a Graph Explorer Memory that tracks:\n"
            "- Which actions you've already tried from this state\n"
            "- Which actions are still untested\n"
            "- Whether previous actions succeeded (moved to new state) or failed (blocked)\n\n"
            "Use BOTH your board analysis AND the graph memory to choose wisely:\n"
            "- Prefer untested actions to explore new possibilities\n"
            "- Avoid repeating actions that failed (blocked) at this state\n"
            "- If an action previously led to a new state, consider if that path is promising"
        )
    else:
        return (
            "Choose one of the available actions: ACTION1 (Up), ACTION2 (Down), "
            "ACTION3 (Left), ACTION4 (Right), ACTION5 (Enter), ACTION6 (Click). "
            "Use both your board analysis and the graph exploration memory to make your choice."
        )


def _build_tools() -> list[dict]:
    """Build the tool/function definitions."""
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
                        "x": {"type": "integer", "description": "X coordinate"},
                        "y": {"type": "integer", "description": "Y coordinate"},
                    },
                    "required": ["x", "y"],
                },
            },
        },
    ]


class GraphExplorerObsActionAgent(ArcAgi3Agent):
    """
    Agent combining two-phase observation/action with GraphExplorer memory.

    Flow:
    1. Observation phase: LLM analyzes board state (pieces, obstacles, goals)
    2. Action phase: LLM chooses action using BOTH observation AND graph memory
    3. GraphExplorer records transition and updates exploration state
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        name: str = "graph_explorer_obs_action_agent",
        input_mode: str = "text_only",
        model: str = "gpt-5-nano",
        reasoning_effort: str = "low",
        downsample: bool = True,
        game_id: str | None = None,
        crop_border: int = 0,
        use_as66_prompts: bool = False,
        n_groups: int = 1,
        verbose_level: int = 0,
        # Harness compatibility params
        include_text_diff: bool = True,
        context_length_limit: int = -1,
        representation: RepresentationConfig | None = None,
        use_general: bool = False,
    ) -> None:
        """
        Initialize the agent.

        Args:
            system_prompt: Override system prompt
            name: Agent name
            input_mode: 'text_only', 'image_only', or 'text_and_image'
            model: OpenAI model to use
            reasoning_effort: Reasoning effort level
            downsample: Whether to downsample grids
            game_id: Game ID
            crop_border: Border pixels to crop for state hashing
            use_as66_prompts: Use AS66-specific game prompts
            n_groups: Number of priority groups for exploration
            verbose_level: GraphExplorer verbosity (0-2)
        """
        self.crop_border = crop_border
        self.use_as66_prompts = use_as66_prompts

        super().__init__(system_prompt=system_prompt, name=name)

        self.input_mode = input_mode
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.downsample = downsample
        self.game_id = game_id
        self.n_groups = n_groups
        self.verbose_level = verbose_level
        self._system_prompt_override = system_prompt

        if self.input_mode not in ["text_only", "image_only", "text_and_image"]:
            log.warning(f"Invalid input_mode '{self.input_mode}', defaulting to 'text_only'.")
            self.input_mode = "text_only"

        self._client = get_openai_client(model=model)
        self._latest_tool_call_id = "call_12345"

        # GraphExplorer persists across resets
        self.graph_explorer = GraphExplorer(n_groups=n_groups, verbose_level=verbose_level)

    def reset(self) -> None:
        """Reset agent state for new episode (but keep GraphExplorer)."""
        super().reset()
        self._chat_history: list[dict] = []
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None
        self._token_total: int = 0
        self._action_counter: int = 0
        self._pending_action: dict[str, Any] | None = None

        # Track current state for graph
        self._current_node_id: Optional[str] = None
        self._last_executed_action: Optional[str] = None

        # Prompts for logging
        self._last_observation_prompt: str = ""
        self._last_observation_response: str = ""
        self._last_action_prompt: str = ""
        self._last_action_response: str = ""

        # Note: graph_explorer persists!

    def _compute_state_hash(self, grid: List[List[int]]) -> str:
        """Compute hash of grid state (with border cropping) using consonants representation."""
        if self.crop_border > 0 and len(grid) > 2 * self.crop_border:
            c = self.crop_border
            grid = [row[c:-c] for row in grid[c:-c]]

        # Use consonants representation for hashing (same as what LLM sees)
        ascii_repr = grid_to_ascii(grid)
        return hashlib.md5(ascii_repr.encode()).hexdigest()

    def _get_grid_from_obs(self, obs: dict) -> List[List[int]]:
        """Extract and process grid from observation."""
        frame_3d = obs.get("frame", [])
        if not frame_3d:
            return []

        if self.downsample:
            return downsample_4x4(frame_3d)
        return frame_3d

    def _crop_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """Remove outer N pixels from grid for display."""
        if self.crop_border <= 0:
            return grid
        c = self.crop_border
        if len(grid) <= 2 * c:
            return grid
        return [row[c:-c] for row in grid[c:-c]]

    def _format_graph_context(self, state_hash: str) -> str:
        """Format GraphExplorer context for LLM."""
        if self.graph_explorer.empty or state_hash not in self.graph_explorer._nodes:
            return "**Graph Explorer Memory:** No exploration data yet.\n"

        node_data = self.graph_explorer._nodes[state_hash]
        stats = self.graph_explorer.get_frontier_stats()

        lines = ["**Graph Explorer Memory:**\n"]

        # Overall stats
        lines.append(f"States discovered: {stats['total_nodes']} | ")
        lines.append(f"Actions tested: {stats['tested_edges']}/{stats['total_edges']}\n\n")

        # Current state action status
        lines.append("**Actions at this state:**\n")

        for edge_idx in range(node_data.num_candidates):
            action_name = f"ACTION{edge_idx + 1}"
            result = node_data.edge_data["result"][edge_idx]

            if result == -1:
                status = "UNTESTED"
            elif result == 0:
                status = "FAILED (blocked)"
            elif result == 1:
                status = "SUCCESS (moved to new state)"
            else:
                status = "UNKNOWN"

            # Check if untested
            is_untested = edge_idx in node_data.group2remaining_candidate_ids.get(0, set())
            marker = "→" if is_untested else " "

            lines.append(f"  {marker} {action_name}: {status}\n")

        # Summary
        untested_count = sum(
            len(node_data.group2remaining_candidate_ids[g])
            for g in range(self.graph_explorer.active_group + 1)
        )
        if untested_count > 0:
            untested_names = [
                f"ACTION{i+1}"
                for i in sorted(node_data.group2remaining_candidate_ids.get(0, set()))
            ]
            lines.append(f"\n**Recommendation:** Try untested actions: {', '.join(untested_names)}\n")
        else:
            lines.append(f"\n**Note:** All actions tested at this state.\n")

        return "".join(lines)

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return message history formatted for chat API."""
        system_msg = self._system_prompt_override or build_observation_system_text(self.use_as66_prompts)
        messages: list[dict] = [{"role": "system", "content": system_msg}]
        messages.extend(self._chat_history)
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Return the trajectory tracking object."""
        return self._trajectory

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict = None, **_: Any) -> None:
        """Process environment observation and update graph."""
        self._last_observation = observation

        obs = observation or {}
        grid = self._get_grid_from_obs(obs)

        if len(grid) > 0 and self._last_executed_action:
            # Record transition in graph
            state_hash = self._compute_state_hash(grid)
            prev_state = self._current_node_id
            game_state = obs.get("state", "UNKNOWN")

            # Determine success code
            if game_state == "GAME_OVER":
                success = -1
            elif prev_state and state_hash != prev_state:
                success = 1  # New state
            else:
                success = 0  # No movement (blocked)

            # Record in graph (skip RESET)
            if not self.graph_explorer.empty and prev_state is not None and self._last_executed_action != "RESET":
                action_map = {
                    "ACTION1": 0, "ACTION2": 1, "ACTION3": 2,
                    "ACTION4": 3, "ACTION5": 4, "ACTION6": 5
                }
                edge_idx = action_map.get(self._last_executed_action, 0)

                try:
                    self.graph_explorer.record_test(
                        node=prev_state,
                        edge_idx=edge_idx,
                        success=success,
                        target_node=state_hash if success == 1 else None,
                        target_num_candidates=6 if success == 1 else None,
                    )
                except Exception as e:
                    log.warning(f"Failed to record transition: {e}")

            # Update current node
            if success == 1:
                self._current_node_id = state_hash

        # Store in trajectory
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

        # Add tool response to chat history
        if self._chat_history and self._chat_history[-1].get("role") == "assistant":
            tool_content = self._format_observation(observation)
            self._chat_history.append({
                "role": "tool",
                "tool_call_id": self._latest_tool_call_id,
                "content": tool_content
            })

    def update_from_model(self, action_payload: dict | None = None, **_: Any) -> Action:
        """Convert model response to Action."""
        action_dict = action_payload or self._pending_action

        if action_dict is None:
            action_dict = {"name": "ACTION1", "data": {}, "obs_text": "", "action_text": ""}

        obs_text = action_dict.get("obs_text", "")
        action_text = action_dict.get("action_text", "")
        response_text = f"Observation Analysis:\n{obs_text}\n\nAction Reasoning:\n{action_text}\n\nChose: {action_dict['name']}"

        if not response_text:
            response_text = str(action_dict)

        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = response_text
            self._trajectory.steps[-1].action = action_dict

        self._action_counter += 1
        self._pending_action = None
        self._last_executed_action = action_dict["name"]

        action = GameAction.from_name(action_dict["name"])
        action_dict2 = {"action": action, "reasoning": response_text}
        if action == GameAction.ACTION6:
            action_dict2["x"] = action_dict["data"]["x"]
            action_dict2["y"] = action_dict["data"]["y"]
        return action_dict2

    @weave_op
    async def call_llm(self, rollout_engine=None) -> tuple[str, dict]:
        """Run two-phase observation/action LLM calls with graph memory."""
        obs = self._last_observation or {}
        state = obs.get("state", "NOT_PLAYED")

        # Auto-RESET if needed
        if state in ("NOT_PLAYED", "GAME_OVER") and self._last_executed_action != "RESET":
            action_dict = {"name": "RESET", "data": {}, "obs_text": "Game requires reset.", "action_text": ""}
            self._pending_action = action_dict
            return action_dict

        # Get current grid
        grid = self._get_grid_from_obs(obs)
        if not len(grid):
            action_dict = {"name": "ACTION1", "data": {}, "obs_text": "", "action_text": ""}
            self._pending_action = action_dict
            return action_dict

        state_hash = self._compute_state_hash(grid)

        # Initialize graph if needed
        if self.graph_explorer.empty:
            self.graph_explorer.initialize(start_node=state_hash, num_candidates=6)
            self._current_node_id = state_hash

        # Ensure current node is in graph
        if state_hash not in self.graph_explorer._nodes:
            self.graph_explorer._add_new_node(state_hash, num_candidates=6)

        self._current_node_id = state_hash

        # Prepare grid for display (consonants palette)
        display_grid = self._crop_grid(grid)
        grid_text = grid_to_ascii(display_grid)
        score = obs.get("score", 0)

        # Phase 1: Observation
        obs_text = await self._call_observation_model(grid_text, score, rollout_engine)

        # Phase 2: Action (with graph memory)
        graph_context = self._format_graph_context(state_hash)
        action_dict = await self._call_action_model(grid_text, obs_text, graph_context, rollout_engine)
        action_dict["obs_text"] = obs_text

        self._pending_action = action_dict
        return action_dict

    @weave_op
    async def _call_observation_model(self, grid_text: str, score: int, rollout_engine=None) -> str:
        """Phase 1: Analyze the board state."""
        sys_msg = build_observation_system_text(self.use_as66_prompts)

        if self.downsample:
            base_size = 16
            actual_size = base_size - (2 * self.crop_border)
            grid_size = f"{actual_size}x{actual_size}" if self.crop_border > 0 else "16x16"
        else:
            grid_size = "64x64"

        user_msg = (
            f"**Current State:**\n"
            f"Score: {score}\n"
            f"Step: {self._action_counter}\n\n"
            f"**Board ({grid_size}):**\n{grid_text}\n\n"
            "Analyze the board:\n"
            "1. Identify the movable piece(s) and their current position\n"
            "2. Identify obstacles, walls, boundaries\n"
            "3. Identify the goal/target area\n"
            "4. For each direction (Up/Down/Left/Right), predict where the piece would move\n"
            "5. Note any dangers (enemies, game-over conditions)\n\n"
            "Provide your analysis. Do NOT choose an action yet."
        )

        self._last_observation_prompt = f"SYSTEM: {sys_msg}\n\nUSER: {user_msg}"

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        model_output = await rollout_engine.get_model_response(messages)

        # Extract reasoning
        obs_reasoning = getattr(model_output, "reasoning", None)
        if not obs_reasoning:
            obs_reasoning = getattr(model_output, "content", "") or ""
        if not obs_reasoning and hasattr(model_output, "text"):
            obs_reasoning = getattr(model_output, "text", "")

        obs_reasoning = (obs_reasoning or "").strip()
        self._last_observation_response = obs_reasoning

        return obs_reasoning

    @weave_op
    async def _call_action_model(self, grid_text: str, obs_analysis: str, graph_context: str, rollout_engine=None) -> dict:
        """Phase 2: Choose action using observation + graph memory."""
        sys_msg = build_action_system_text(self.use_as66_prompts)

        user_msg = (
            f"**Your Board Analysis:**\n{obs_analysis}\n\n"
            f"{graph_context}\n"
            f"**Board:**\n{grid_text}\n\n"
            "Based on your analysis AND the graph exploration memory:\n"
            "- Choose an action that makes progress toward the goal\n"
            "- Prefer UNTESTED actions to explore new possibilities\n"
            "- Avoid actions that previously FAILED (blocked) at this state\n"
            "- If an action succeeded before, consider if that direction is promising\n\n"
            "Call ONE action tool now."
        )

        self._last_action_prompt = f"SYSTEM: {sys_msg}\n\nUSER: {user_msg}"

        tools = _build_tools()
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        model_output = await rollout_engine.get_model_response(messages, tools=tools)

        # Extract reasoning
        action_reasoning = getattr(model_output, "reasoning", None)
        if not action_reasoning:
            action_reasoning = getattr(model_output, "content", "") or ""
        if not action_reasoning and hasattr(model_output, "text"):
            action_reasoning = getattr(model_output, "text", "")

        action_reasoning = (action_reasoning or "").strip()

        # Parse tool call
        m = model_output.tool_calls[0] if getattr(model_output, "tool_calls", None) else None

        if m is None:
            action_text = f"LLM Reasoning: {action_reasoning}\nNo tool call, defaulting to ACTION1"
            return {"name": "ACTION1", "data": {}, "action_text": action_text}

        tc = m
        tc_id = tc.id
        self._latest_tool_call_id = tc_id
        name = tc.function.name
        arguments = tc.function.arguments if hasattr(tc.function, 'arguments') else "{}"

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

        action_text = f"LLM Reasoning: {action_reasoning}\nChose: {name}"
        self._last_action_response = action_text

        # Handle ACTION6 coordinate mapping
        if name == "ACTION6":
            x_raw = args.get("x", 0)
            y_raw = args.get("y", 0)
            if self.downsample:
                x_64 = x_raw * 4
                y_64 = y_raw * 4
            else:
                x_64 = x_raw
                y_64 = y_raw
            return {"name": name, "data": {"x": x_64, "y": y_64}, "action_text": action_text}

        return {"name": name, "data": args, "action_text": action_text}

    async def rollout(self, rollout_engine, messages: List[Dict[str, Any]], tools=None):
        """Call rollout engine for model response."""
        return await rollout_engine.get_model_response(messages, tools=tools)
