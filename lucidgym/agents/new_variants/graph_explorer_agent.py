"""
Graph Explorer Agent - systematic exploration using frontier-based memory.

This agent uses a GraphExplorer memory module that:
- Tracks which actions have been tried from each state
- Maintains a frontier of partially-explored states
- Uses priority groups for systematic exploration
- Can navigate back to frontier states to continue exploration

The LLM is used as a policy head to choose which untested action to try.
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
from lucidgym.utils.grid_processing import frame_to_grid_text, downsample_4x4, generate_numeric_grid_image_bytes

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


# ============================================================================
# GraphExplorer - The Memory Module
# ============================================================================

@dataclass
class NodeData:
    """Data for a single node in the exploration graph."""
    node_id: Hashable
    num_candidates: int  # Number of possible actions from this state

    # Edge data arrays (indexed by action/edge index)
    edge_data: Dict[str, np.ndarray] = field(default_factory=dict)

    # Priority groups: group_id -> set of remaining candidate edge indices
    group2remaining_candidate_ids: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))

    # Distance from this node to nearest frontier (in active group)
    distance: int = -1

    def __post_init__(self):
        if not self.edge_data:
            # Initialize edge arrays
            self.edge_data = {
                "result": np.full(self.num_candidates, -1, dtype=np.int8),  # -1=untested, 0=fail, 1=success
                "target": np.full(self.num_candidates, None, dtype=object),  # target node (if success)
                "group": np.zeros(self.num_candidates, dtype=np.int8),  # priority group
                "distance": np.full(self.num_candidates, -1, dtype=np.int32),  # distance through this edge
            }

    def has_open_group(self, max_group: int) -> bool:
        """Check if node has untested edges in groups <= max_group."""
        for g in range(max_group + 1):
            if len(self.group2remaining_candidate_ids[g]) > 0:
                return True
        return False


class GraphExplorer:
    """
    Memory module for systematic graph exploration.

    Core concepts:
    - Nodes = states
    - Edges = actions (indexed 0 to num_actions-1)
    - Priority groups: systematically test edges group by group
    - Frontier = nodes with untested edges in active group
    - Can compute shortest paths back to frontier
    """

    def __init__(self, n_groups: int = 3, verbose_level: int = 0):
        """
        Initialize GraphExplorer.

        Args:
            n_groups: Number of priority groups (0=highest, n_groups-1=lowest)
            verbose_level: Logging verbosity (0=quiet, 1=basic, 2=detailed)
        """
        self.n_groups = n_groups
        self.verbose = verbose_level

        self._nodes: Dict[Hashable, NodeData] = {}
        self._next: Dict[Hashable, Tuple[int, Hashable]] = {}  # node -> (edge_idx, target) for shortest path

        self.active_group: int = 0  # Current priority group
        self.empty: bool = True

    def initialize(self, start_node: Hashable, num_candidates: int) -> None:
        """Initialize graph with starting node."""
        if not self.empty:
            log.warning("GraphExplorer already initialized, resetting")

        self._nodes.clear()
        self._next.clear()
        self.active_group = 0

        self._add_new_node(start_node, num_candidates)
        self.empty = False

        if self.verbose >= 1:
            log.info(f"GraphExplorer initialized with start_node={start_node}, num_candidates={num_candidates}")

    def _add_new_node(self, node_id: Hashable, num_candidates: int) -> NodeData:
        """Add a new node to the graph."""
        node = NodeData(node_id=node_id, num_candidates=num_candidates)

        # All edges start in group 0 (highest priority)
        for edge_idx in range(num_candidates):
            node.group2remaining_candidate_ids[0].add(edge_idx)

        self._nodes[node_id] = node

        if self.verbose >= 2:
            log.debug(f"Added new node: {node_id} with {num_candidates} candidates")

        return node

    def record_test(
        self,
        node: Hashable,
        edge_idx: int,
        success: int,
        target_node: Optional[Hashable] = None,
        target_num_candidates: Optional[int] = None,
    ) -> None:
        """
        Record the result of testing an edge.

        Args:
            node: Source node ID
            edge_idx: Edge/action index that was tested
            success: 1=success (new state), 0=fail/blocked, -1=error/reset
            target_node: Target node ID (required if success=1)
            target_num_candidates: Number of actions from target (required if success=1)
        """
        if node not in self._nodes:
            raise ValueError(f"Node {node} not in graph")

        node_data = self._nodes[node]

        # Record result
        node_data.edge_data["result"][edge_idx] = success

        if success == 1:
            # Success: record target and add target node if new
            if target_node is None or target_num_candidates is None:
                raise ValueError("target_node and target_num_candidates required for success=1")

            node_data.edge_data["target"][edge_idx] = target_node

            if target_node not in self._nodes:
                self._add_new_node(target_node, target_num_candidates)

        # Remove from remaining candidates
        current_group = int(node_data.edge_data["group"][edge_idx])
        if edge_idx in node_data.group2remaining_candidate_ids[current_group]:
            node_data.group2remaining_candidate_ids[current_group].remove(edge_idx)

        # Rebuild distances to frontier
        self._rebuild_distances()

        if self.verbose >= 2:
            log.debug(f"Recorded test: node={node}, edge={edge_idx}, success={success}, target={target_node}")

    def _rebuild_distances(self) -> None:
        """
        Rebuild BFS distances from all nodes to nearest frontier.
        Frontier = nodes with untested edges in active group.
        """
        # Reset distances
        for node_data in self._nodes.values():
            node_data.distance = -1
            for i in range(node_data.num_candidates):
                node_data.edge_data["distance"][i] = -1

        self._next.clear()

        # Find frontier nodes
        frontier: Set[Hashable] = set()
        for node_id, node_data in self._nodes.items():
            if node_data.has_open_group(self.active_group):
                frontier.add(node_id)
                node_data.distance = 0

        if not frontier:
            # No frontier: maybe need to advance to next group
            if self.verbose >= 1:
                log.info(f"No frontier in group {self.active_group}")
            return

        # BFS from frontier backwards through successful edges
        queue: deque[Hashable] = deque(frontier)

        while queue:
            current = queue.popleft()
            current_data = self._nodes[current]
            current_dist = current_data.distance

            # Look at all nodes that can reach current in one step
            for node_id, node_data in self._nodes.items():
                if node_data.distance != -1:
                    continue  # Already visited

                # Check if any edge from node_id leads to current
                for edge_idx in range(node_data.num_candidates):
                    if (node_data.edge_data["result"][edge_idx] == 1 and
                        node_data.edge_data["target"][edge_idx] == current):
                        # Found edge: node_id --[edge_idx]--> current
                        node_data.distance = current_dist + 1
                        node_data.edge_data["distance"][edge_idx] = current_dist + 1
                        self._next[node_id] = (edge_idx, current)
                        queue.append(node_id)
                        break  # Only need one path

    def get_next_hop(self, node: Hashable) -> Optional[Hashable]:
        """
        Get next node on shortest path to frontier.

        Returns:
            Next node on path, or None if no path exists
        """
        if node not in self._next:
            return None

        _, target = self._next[node]
        return target

    def get_frontier_stats(self) -> Dict[str, Any]:
        """Get statistics about current exploration state."""
        total_nodes = len(self._nodes)
        frontier_nodes = sum(1 for n in self._nodes.values() if n.has_open_group(self.active_group))

        total_edges = sum(n.num_candidates for n in self._nodes.values())
        tested_edges = sum(
            np.sum(n.edge_data["result"] != -1) for n in self._nodes.values()
        )

        return {
            "total_nodes": total_nodes,
            "frontier_nodes": frontier_nodes,
            "active_group": self.active_group,
            "total_edges": total_edges,
            "tested_edges": tested_edges,
            "untested_edges": total_edges - tested_edges,
        }

    def format_context(self, current_node: Hashable) -> str:
        """Format graph exploration context for LLM."""
        if current_node not in self._nodes:
            return ""

        node_data = self._nodes[current_node]
        stats = self.get_frontier_stats()

        lines = ["**Graph Explorer Memory:**\n"]

        # Overall stats
        lines.append(f"**Exploration Progress:**\n")
        lines.append(f"  - Total states discovered: {stats['total_nodes']}\n")
        lines.append(f"  - Frontier states (with untested actions): {stats['frontier_nodes']}\n")
        lines.append(f"  - Active priority group: {stats['active_group']}\n")
        lines.append(f"  - Actions tested: {stats['tested_edges']}/{stats['total_edges']}\n")

        # Current node status
        lines.append(f"\n**Current State Status:**\n")

        # Count untested actions at current node
        untested_here = sum(
            len(node_data.group2remaining_candidate_ids[g])
            for g in range(self.active_group + 1)
        )

        if untested_here > 0:
            lines.append(f"  - Untested actions available: {untested_here}\n")
            lines.append(f"  - This is a FRONTIER state - explore new actions!\n")

            # Show which actions are untested
            for g in range(self.active_group + 1):
                remaining = node_data.group2remaining_candidate_ids[g]
                if remaining:
                    action_names = [f"ACTION{i+1}" for i in sorted(remaining)]
                    lines.append(f"  - Group {g} untested: {', '.join(action_names)}\n")
        else:
            lines.append(f"  - All actions tested at this state\n")
            if node_data.distance > 0:
                lines.append(f"  - Distance to nearest frontier: {node_data.distance} steps\n")
                lines.append(f"  - Should navigate back to frontier\n")
            else:
                lines.append(f"  - No reachable frontier in current group\n")

        # Show successful transitions discovered
        success_count = np.sum(node_data.edge_data["result"] == 1)
        if success_count > 0:
            lines.append(f"\n**Discovered transitions from here:** {success_count}\n")
            for edge_idx in range(node_data.num_candidates):
                if node_data.edge_data["result"][edge_idx] == 1:
                    target = node_data.edge_data["target"][edge_idx]
                    lines.append(f"  - ACTION{edge_idx+1} → new state\n")

        return "".join(lines)


# ============================================================================
# Agent Wrapper
# ============================================================================

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
                        "x": {"type": "integer", "description": "X coordinate"},
                        "y": {"type": "integer", "description": "Y coordinate"},
                    },
                    "required": ["x", "y"],
                },
            },
        },
    ]


class GraphExplorerAgent(ArcAgi3Agent):
    """
    Agent that uses GraphExplorer for systematic exploration.

    The agent:
    - Uses GraphExplorer to track which actions have been tried
    - Uses LLM to choose which untested action to try at frontier states
    - Navigates back to frontier when current state is exhausted
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        name: str = "graph_explorer_agent",
        model: str = "gpt-5-nano",
        reasoning_effort: str = "low",
        downsample: bool = True,
        game_id: str | None = None,
        n_groups: int = 1,
        crop_border: int = 0,
        verbose_level: int = 0,
        # Harness compatibility params
        input_mode: str = "text_only",
        use_as66_prompts: bool = False,
        include_text_diff: bool = True,
        context_length_limit: int = -1,
        representation: RepresentationConfig | None = None,
        use_general: bool = False,
    ) -> None:
        """
        Initialize GraphExplorerAgent.

        Args:
            system_prompt: Override system prompt
            name: Agent name
            model: OpenAI model to use
            reasoning_effort: Reasoning effort level
            downsample: Whether to downsample grids
            game_id: Game ID
            n_groups: Number of priority groups for exploration
            crop_border: Border pixels to crop for state hashing
            verbose_level: GraphExplorer verbosity (0-2)
            (Other params for harness compatibility)
        """
        self.crop_border = crop_border

        super().__init__(system_prompt=system_prompt, name=name)

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.downsample = downsample
        self.game_id = game_id
        self.n_groups = n_groups
        self.verbose_level = verbose_level

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
        self._action_counter: int = 0
        self._pending_action: dict[str, Any] | None = None

        # Track current state
        self._current_node_id: Optional[str] = None
        self._last_executed_action: Optional[str] = None

        # Prompts for logging
        self._last_observation_prompt: str = ""
        self._last_observation_response: str = ""
        self._last_action_prompt: str = ""
        self._last_action_response: str = ""

        # Note: graph_explorer persists!

    def _compute_state_hash(self, grid: List[List[int]]) -> str:
        """Compute hash of grid state (with border cropping)."""
        if self.crop_border > 0 and len(grid) > 2 * self.crop_border:
            c = self.crop_border
            grid = [row[c:-c] for row in grid[c:-c]]

        grid_str = json.dumps(grid)
        return hashlib.md5(grid_str.encode()).hexdigest()

    def _get_grid_from_obs(self, obs: dict) -> List[List[int]]:
        """Extract and process grid from observation."""
        frame_3d = obs.get("frame", [])
        if not frame_3d:
            return []

        if self.downsample:
            return downsample_4x4(frame_3d)
        return frame_3d

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return message history formatted for chat API."""
        system_msg = (
            "You are an agent systematically exploring a game state space.\n"
            "You have a graph memory that tracks:\n"
            "  - Which states you've discovered\n"
            "  - Which actions you've tried from each state\n"
            "  - Which actions are still untested\n\n"
            "Your goal: systematically test all available actions.\n"
            "When at a frontier state (with untested actions), choose one to try.\n"
            "When all actions are tested, you'll navigate back to a frontier state.\n\n"
            "Be methodical and thorough in your exploration."
        )
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

            # Determine success code
            prev_state = self._current_node_id
            game_state = obs.get("state", "UNKNOWN")

            if game_state == "GAME_OVER":
                success = -1  # Error/reset
            elif state_hash != prev_state:
                success = 1  # New state
            else:
                success = 0  # No movement (blocked)

            # Record in graph if initialized (skip RESET actions)
            if not self.graph_explorer.empty and prev_state is not None and self._last_executed_action != "RESET":
                # Map action name to index (0-indexed, so ACTION1=0, ACTION2=1, etc.)
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
            # Fallback to default action if no action provided
            action_dict = {"name": "ACTION1", "data": {}, "obs_text": "", "action_text": ""}

        obs_text = action_dict.get("obs_text", "")
        action_text = action_dict.get("action_text", "")
        response_text = f"Observation: {obs_text}\n\n{action_text}\n\nChose Action: {action_dict['name']}"

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
        """Call LLM to select action using GraphExplorer context."""
        obs = self._last_observation or {}
        state = obs.get("state", "NOT_PLAYED")

        # Auto-RESET if needed
        if state in ("NOT_PLAYED", "GAME_OVER") and self._last_executed_action != "RESET":
            action_dict = {"name": "RESET", "data": {}, "obs_text": "Starting new game", "action_text": ""}
            self._pending_action = action_dict
            return action_dict

        # Get current state
        grid = self._get_grid_from_obs(obs)
        if not len(grid):
            # No grid, default action
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

        # Get exploration context
        graph_context = self.graph_explorer.format_context(state_hash)

        # Format grid for display
        if self.crop_border > 0 and len(grid) > 2 * self.crop_border:
            c = self.crop_border
            display_grid = [row[c:-c] for row in grid[c:-c]]
        else:
            display_grid = grid

        grid_text = frame_to_grid_text([display_grid])

        # Build prompt
        user_msg = (
            f"{graph_context}\n"
            f"**Current Board ({len(display_grid)}x{len(display_grid[0] if display_grid else 0)}):**\n"
            f"{grid_text}\n\n"
            "Select ONE action to try. Use the graph memory to guide your choice:\n"
            "- If at a FRONTIER state, choose an untested action\n"
            "- Prioritize systematic exploration over random choices\n"
            "- Call the appropriate ACTION tool (ACTION1-6)"
        )

        self._last_action_prompt = user_msg

        tools = _build_tools()
        messages = [
            {"role": "system", "content": self.chat_completions[0]["content"]},
            {"role": "user", "content": user_msg}
        ]

        # Get model response
        model_output = await self.rollout(rollout_engine, messages, tools)

        # Extract reasoning from the model (prioritize reasoning field for o1-like models)
        llm_reasoning = getattr(model_output, "reasoning", None)
        if not llm_reasoning:
            llm_reasoning = getattr(model_output, "content", "") or ""
        if not llm_reasoning and hasattr(model_output, "text"):
            llm_reasoning = getattr(model_output, "text", "")

        llm_reasoning = (llm_reasoning or "").strip()

        # Parse tool call
        m = model_output.tool_calls[0] if getattr(model_output, "tool_calls", None) else None

        if m is None:
            action_text = f"LLM Response: {llm_reasoning}\nDefaulting to ACTION1"
            action_dict = {"name": "ACTION1", "data": {}, "obs_text": graph_context, "action_text": action_text}
            self._pending_action = action_dict
            return action_dict

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

        # Build action text with LLM reasoning
        action_text = f"LLM Reasoning: {llm_reasoning}\nChose: {name}"
        self._last_action_response = f"Tool Call: {name}({json.dumps(args)})\nReasoning: {llm_reasoning}"

        # Handle coordinate mapping for ACTION6
        if name == "ACTION6":
            x_raw = args.get("x", 0)
            y_raw = args.get("y", 0)
            if self.downsample:
                x_64 = x_raw * 4
                y_64 = y_raw * 4
            else:
                x_64 = x_raw
                y_64 = y_raw
            action_dict = {"name": name, "data": {"x": x_64, "y": y_64}, "obs_text": graph_context, "action_text": action_text}
            self._pending_action = action_dict
            return action_dict

        action_dict = {"name": name, "data": args, "obs_text": graph_context, "action_text": action_text}
        self._pending_action = action_dict
        return action_dict

    async def rollout(self, rollout_engine, messages: List[Dict[str, Any]], tools=None):
        """Call rollout engine for model response."""
        return await rollout_engine.get_model_response(messages, tools=tools)
