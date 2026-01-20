"""
16x16 text-based guided agents with directed graph memory.
Maintains a graph of state transitions to track visited states and avoid cycles.
"""
from __future__ import annotations
from typing import Any, Optional, List, Dict, Set, Tuple
import json
import logging
import base64
import hashlib
from openai import OpenAI

from lucidgym.utils.openai_client import get_openai_client
from lucidgym.utils.representation import RepresentationConfig
from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
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


def build_observation_system_text(use_as66_prompts: bool = False):
    if use_as66_prompts:
        # AS66-specific rules (generic, no hardcoded integer values)
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
            "IMPORTANT: You have access to a state graph showing previously visited states.\n"
            "- Avoid actions that lead to states you've already visited\n"
            "- Pay special attention to cycles (repeated action sequences)\n"
            "- If a state has been visited multiple times without progress, try a different approach\n\n"
            "DO NOT call an action tool here - only provide analysis."
        )
    else:
        # Generic prompts
        return (
            "You are observing a 16x16 grid representation of a game state. "
            "Each cell contains an ASCII character representing different game elements. "
            "Your task is to analyze this grid and determine the best action to take. "
            "The grid shows the current game state with various symbols representing different game objects. "
            "You have access to a state graph that tracks previously visited states - use this to avoid repeating unsuccessful actions."
        )

def build_action_system_text(use_as66_prompts: bool = False):
    if use_as66_prompts:
        # AS66-specific action prompt
        return (
            "Select exactly one move by calling a single tool. Do not include prose.\n"
            "Available tools:\n"
            "- ACTION1 = Up\n"
            "- ACTION2 = Down\n"
            "- ACTION3 = Left\n"
            "- ACTION4 = Right\n\n"
            "Consider the state graph information to avoid cycles and repeated failures."
        )
    else:
        # Generic action prompt
        return (
            "You are selecting the best action based on your observation of the game state. "
            "Choose one of the available actions: ACTION1 (Up), ACTION2 (Down), ACTION3 (Left), ACTION4 (Right), ACTION5 (Enter), ACTION6 (Click). "
            "Use the state graph to avoid repeating actions that have led to dead-ends or cycles."
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


class StateNode:
    """Represents a state in the directed graph."""

    def __init__(self, state_hash: str):
        self.state_hash = state_hash
        self.visit_count = 0
        self.score = 0
        self.step_numbers: List[int] = []  # Steps when this state was visited
        self.outgoing_edges: List[StateEdge] = []  # Actions taken from this state
        self.incoming_edges: List[StateEdge] = []  # Actions that led to this state


class StateEdge:
    """Represents an action transition between states."""

    def __init__(self, from_node: StateNode, to_node: StateNode, action: str, step: int, score_delta: float):
        self.from_node = from_node
        self.to_node = to_node
        self.action = action
        self.step = step
        self.score_delta = score_delta


class StateGraph:
    """Directed graph tracking state transitions."""

    def __init__(self, crop_border: int = 0):
        self.nodes: Dict[str, StateNode] = {}  # state_hash -> StateNode
        self.current_state_hash: Optional[str] = None
        self.previous_state_hash: Optional[str] = None
        self.total_steps = 0
        self.crop_border = crop_border

    def _crop_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """Remove outer N pixels from grid before hashing."""
        if self.crop_border <= 0 or len(grid) == 0:
            return grid
        c = self.crop_border
        # Ensure we don't crop more than the grid size
        if len(grid) <= 2 * c or len(grid[0]) <= 2 * c:
            return grid
        return [row[c:-c] for row in grid[c:-c]]

    def _compute_state_hash(self, grid: List[List[int]], score: int) -> str:
        """Compute hash of grid state (after cropping border).

        The border pixels (typically 2 pixels on each side) contain score counters
        that change independently of actual game state. By cropping these before
        hashing, we can correctly identify when the board position is identical
        even if the score display has changed.
        """
        # Crop border pixels (score counters) before hashing
        cropped_grid = self._crop_grid(grid)
        # Flatten grid and create hash
        grid_str = json.dumps(cropped_grid)
        # Don't include score in hash - we want to identify same board state regardless of score
        # This helps detect when actions lead to the same position (cycles)
        state_str = grid_str
        return hashlib.md5(state_str.encode()).hexdigest()

    def add_or_update_state(self, grid: List[List[int]], score: int, step: int) -> StateNode:
        """Add new state or update existing one."""
        state_hash = self._compute_state_hash(grid, score)

        if state_hash not in self.nodes:
            self.nodes[state_hash] = StateNode(state_hash)

        node = self.nodes[state_hash]
        node.visit_count += 1
        node.score = score
        node.step_numbers.append(step)

        # Update previous and current state tracking
        self.previous_state_hash = self.current_state_hash
        self.current_state_hash = state_hash
        self.total_steps = step

        return node

    def add_transition(self, action: str, new_grid: List[List[int]], new_score: int, step: int) -> StateEdge:
        """Add transition from current state to new state."""
        if self.current_state_hash is None:
            # First state, just record it
            self.add_or_update_state(new_grid, new_score, step)
            return None

        # Get or create new state
        new_node = self.add_or_update_state(new_grid, new_score, step)

        # Get previous state
        prev_node = self.nodes.get(self.previous_state_hash)
        if prev_node is None:
            return None

        # Calculate score delta
        score_delta = new_score - prev_node.score

        # Create edge
        edge = StateEdge(prev_node, new_node, action, step, score_delta)
        prev_node.outgoing_edges.append(edge)
        new_node.incoming_edges.append(edge)

        return edge

    def detect_cycles(self, max_lookback: int = 10) -> List[Tuple[str, int]]:
        """Detect recent cycles (repeated states)."""
        if self.current_state_hash is None:
            return []

        current_node = self.nodes.get(self.current_state_hash)
        if current_node is None or current_node.visit_count < 2:
            return []

        # Find cycles: states visited multiple times recently
        cycles = []
        for state_hash, node in self.nodes.items():
            if node.visit_count >= 2:
                # Check if visits are recent
                recent_visits = [s for s in node.step_numbers if s >= self.total_steps - max_lookback]
                if len(recent_visits) >= 2:
                    cycles.append((state_hash, len(recent_visits)))

        return cycles

    def get_action_from_state(self, state_hash: str, action: str) -> Optional[StateEdge]:
        """Get the edge representing an action from a given state."""
        node = self.nodes.get(state_hash)
        if node is None:
            return None

        for edge in node.outgoing_edges:
            if edge.action == action:
                return edge
        return None

    def get_failed_actions(self, max_lookback: int = 10) -> Set[str]:
        """Get actions that recently led to no progress or negative outcomes."""
        if self.current_state_hash is None:
            return set()

        failed_actions = set()
        recent_step_threshold = self.total_steps - max_lookback

        for node in self.nodes.values():
            for edge in node.outgoing_edges:
                # Consider failed if: score didn't increase and led back to same state
                if edge.step >= recent_step_threshold:
                    if edge.score_delta <= 0 and edge.to_node.state_hash == edge.from_node.state_hash:
                        failed_actions.add(edge.action)

        return failed_actions


class BasicObsActionAgentGraphMemory(ArcAgi3Agent):
    """
    Basic agent with directed graph memory.
    Tracks state transitions in a graph to avoid cycles and repeated failures.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        name: str = "basic_obs_action_agent_graph_memory",
        input_mode: str = "text_only",
        model: str = "gpt-5-nano",
        reasoning_effort: str = "low",
        downsample = True,
        game_id: str | None = None,
        max_context_states: int = 10,  # Max states to include in context
        crop_border: int = 0,  # Remove outer N pixels (set to 2 for AS66 to ignore score display)
        use_as66_prompts: bool = False,  # Use AS66-specific game rules
        include_text_diff: bool = True,  # For harness compatibility (unused)
        context_length_limit: int = -1,  # For harness compatibility (unused)
        representation: RepresentationConfig | None = None,  # For harness compatibility (unused)
        use_general: bool = False,  # For harness compatibility (unused)
    ) -> None:
        """
        Initialize the agent with graph memory.

        Args:
            system_prompt: Override system prompt (optional)
            name: Agent name
            input_mode: 'text_only', 'image_only', or 'text_and_image'
            model: OpenAI model to use
            reasoning_effort: Reasoning effort level
            downsample: Whether to downsample the grid
            game_id: Game ID for prompt selection
            max_context_states: Maximum number of states to include in LLM context
            crop_border: Remove outer N pixels from grid for hashing (e.g., 2 to ignore AS66 score display).
                        This ensures state hashing ignores the score counter border pixels.
            use_as66_prompts: Use AS66-specific game rules in prompts
            include_text_diff: Compatibility parameter for harness (unused in this agent)
            context_length_limit: Compatibility parameter for harness (unused in this agent)
            representation: Compatibility parameter for harness (unused in this agent)
            use_general: Compatibility parameter for harness (unused in this agent)
        """
        # Set parameters before calling super().__init__ because parent calls reset()
        self.max_context_states = max_context_states
        self.crop_border = crop_border
        self.use_as66_prompts = use_as66_prompts

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

        # Graph persists across episode resets
        # Pass crop_border so state hashing ignores score counter pixels
        self._state_graph = StateGraph(crop_border=self.crop_border)

    def reset(self) -> None:
        """Reset agent state for new episode (but keep graph)."""
        super().reset()
        self._chat_history: list[dict] = []
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None
        self._token_total: int = 0
        self._action_counter: int = 0
        self._pending_action: dict[str, Any] | None = None

        # Store actual prompts/responses for logging
        self._last_observation_prompt: str = ""
        self._last_observation_response: str = ""
        self._last_action_prompt: str = ""
        self._last_action_response: str = ""

        # Track last executed action to prevent double RESETs
        self._last_executed_action: str | None = None

        # Track last state for transition recording
        self._last_grid: Optional[List[List[int]]] = None
        self._last_score: int = 0

        # NOTE: _state_graph persists across resets

    def _format_graph_context(self) -> str:
        """Format graph information for LLM context."""
        if not self._state_graph.nodes:
            return ""

        lines = ["**State Graph Memory:**\n"]

        # Current state info
        if self._state_graph.current_state_hash:
            current_node = self._state_graph.nodes.get(self._state_graph.current_state_hash)
            if current_node:
                lines.append(f"Current state (visited {current_node.visit_count} time(s)):\n")

        # Detect and report cycles
        cycles = self._state_graph.detect_cycles(max_lookback=15)
        if cycles:
            lines.append("\n**⚠️ CYCLES DETECTED:**\n")
            for state_hash, visit_count in cycles[:5]:  # Show top 5 cycles
                node = self._state_graph.nodes[state_hash]
                lines.append(f"  - State visited {visit_count} times recently (score: {node.score})\n")
                # Show what actions led there
                if node.incoming_edges:
                    recent_edges = [e for e in node.incoming_edges if e.step >= self._state_graph.total_steps - 15]
                    if recent_edges:
                        actions = [e.action for e in recent_edges]
                        lines.append(f"    Actions leading here: {', '.join(actions)}\n")

        # Failed actions
        failed_actions = self._state_graph.get_failed_actions(max_lookback=15)
        if failed_actions:
            lines.append(f"\n**Actions with no progress:** {', '.join(failed_actions)}\n")
            lines.append("  ➜ Consider trying different actions\n")

        # Recent successful transitions (positive score deltas)
        successful_edges = []
        for node in self._state_graph.nodes.values():
            for edge in node.outgoing_edges:
                if edge.score_delta > 0 and edge.step >= self._state_graph.total_steps - 15:
                    successful_edges.append(edge)

        if successful_edges:
            successful_edges.sort(key=lambda e: e.score_delta, reverse=True)
            lines.append(f"\n**Recent successful actions (score increased):**\n")
            for edge in successful_edges[:5]:  # Show top 5
                lines.append(f"  - {edge.action}: +{edge.score_delta} points\n")

        # Graph statistics
        lines.append(f"\n**Graph Stats:**\n")
        lines.append(f"  - Total unique states: {len(self._state_graph.nodes)}\n")
        lines.append(f"  - Total steps: {self._state_graph.total_steps}\n")

        most_visited = max(self._state_graph.nodes.values(), key=lambda n: n.visit_count)
        lines.append(f"  - Most visited state: {most_visited.visit_count} visits\n")

        return "".join(lines)

    def _crop_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """Remove outer N pixels from grid."""
        if self.crop_border <= 0:
            return grid
        c = self.crop_border
        return [row[c:-c] for row in grid[c:-c]]

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

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **_: Any) -> None:
        """Process environment observation and update graph."""
        self._last_observation = observation

        # Update graph with new state
        obs = observation or {}
        frame_3d = obs.get("frame", [])

        if len(frame_3d) > 0:
            # Get downsampled grid (but don't crop yet - StateGraph will crop for hashing)
            if self.downsample:
                grid_for_graph = downsample_4x4(frame_3d)
            else:
                grid_for_graph = frame_3d

            score = obs.get("score", 0)

            # Add transition if we have a previous action
            # StateGraph will crop border pixels before hashing
            if self._last_executed_action:
                self._state_graph.add_transition(
                    self._last_executed_action,
                    grid_for_graph,
                    score,
                    self._action_counter
                )
            else:
                # First state
                self._state_graph.add_or_update_state(grid_for_graph, score, self._action_counter)

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

        if action_dict is None:
            # Fallback to default action if no action provided
            action_dict = {"name": "ACTION1", "data": {}, "obs_text": "", "action_text": ""}

        obs_text = action_dict.get("obs_text", "")
        action_text = action_dict.get("action_text", "")
        response_text = f"Observation: {obs_text}\nAction Text: {action_text}\nAction: {action_dict['name']}"

        if not response_text:
            response_text = str(action_dict)

        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = response_text
            self._trajectory.steps[-1].action = action_dict

        self._action_counter += 1
        self._pending_action = None

        # Track last executed action
        self._last_executed_action = action_dict["name"]

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

        # Only auto-RESET if state requires it AND we didn't just execute RESET
        if state in ("NOT_PLAYED", "GAME_OVER") and self._last_executed_action != "RESET":
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
        sys_msg = build_observation_system_text(self.use_as66_prompts)

        if self.downsample:
            base_size = 16
            actual_size = base_size - (2 * self.crop_border)
            grid_text = f"{actual_size}x{actual_size}" if self.crop_border > 0 else "16x16"
        else:
            grid_text = "64x64"

        # Get graph context instead of rolling history
        graph_context = self._format_graph_context()

        user_msg_text = (
            f"{graph_context}"
            f"**Current State:**\n"
            f"Score: {score}\n"
            f"Step: {self._action_counter}\n\n"
            f"**Current Matrix** {grid_text} (ASCII characters):\n{grid}\n\n"
            "Rationale:\n"
            "  • Identify the movable ASCII character(s) and relevant structures.\n"
            "  • Review the graph memory to see what states/actions have been tried.\n"
            "  • Avoid cycles and actions that have failed recently.\n"
            "  • Conclude which direction is best and why. Do not output an action here.\n"
            "  • Focus on making progress toward states with higher scores."
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
        sys_msg = build_action_system_text(self.use_as66_prompts)

        graph_context = self._format_graph_context()

        user_msg_text = (
            f"{graph_context}"
            "Choose the best single move as a function call.\n"
            f"{grid}\n"
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

        print(f"[DEBUG]:graph_memory:model_output={model_output}")
        print(f"[DEBUG]:graph_memory:rollout_engine._use_chat_completions={getattr(rollout_engine, '_use_chat_completions', 'N/A')}")

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
