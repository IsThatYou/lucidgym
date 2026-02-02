"""
16x16 text-based guided agents with directed graph memory.
Tracks visited states as nodes and actions/transitions as directed edges.
"""
from __future__ import annotations

from typing import Any, Optional, List, Dict
from collections import deque, defaultdict
import json
import logging
import base64
import hashlib

from openai import OpenAI

from lucidgym.utils.openai_client import get_openai_client
from lucidgym.utils.representation import RepresentationConfig
from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from lucidgym.agents.arcagi3_agent import ArcAgi3Agent

from arcengine import GameAction, GameState
from lucidgym.utils.grid_processing import frame_to_grid_text, downsample_4x4, generate_numeric_grid_image_bytes

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
            "DO NOT call an action tool here - only provide analysis.\n"
            "THE MOST IMPORTANT THING TO KEEP IN MIND IS THE RESULTS OF YOUR PAST ACTIONS. DO NOT REPEAT ACTIONS THAT CHANGED NOTHING."
        )
    return (
        "You are observing a 16x16 grid representation of a game state. "
        "Each cell contains an ASCII character representing different game elements. "
        "Your task is to analyze this grid and determine the best action to take. "
        "The grid shows the current game state with various symbols representing different game objects."
    )


def build_action_system_text(use_as66_prompts: bool = False):
    if use_as66_prompts:
        return (
            "Select exactly one move by calling a single tool. Do not include prose.\n"
            "Available tools:\n"
            "- ACTION1 = Up\n"
            "- ACTION2 = Down\n"
            "- ACTION3 = Left\n"
            "- ACTION4 = Right"
        )
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


class DirectedGraphMemory:
    def __init__(self, max_recent_transitions: int = 5) -> None:
        self.max_recent_transitions = max_recent_transitions
        self.reset()

    def reset(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._out_edges: dict[str, list[int]] = defaultdict(list)
        self._recent_edge_indices: deque[int] = deque(maxlen=max(0, self.max_recent_transitions))

    def _state_id(self, grid_text: str) -> str:
        return hashlib.md5(grid_text.encode("utf-8")).hexdigest()

    def upsert_node(self, *, grid_text: str, score: int, state: str, step: int) -> str:
        node_id = self._state_id(grid_text)
        existing = self._nodes.get(node_id)
        if existing is None:
            self._nodes[node_id] = {
                "id": node_id,
                "first_seen_step": step,
                "last_seen_step": step,
                "score": score,
                "state": state,
                "visit_count": 1,
            }
        else:
            existing["last_seen_step"] = step
            existing["score"] = score
            existing["state"] = state
            existing["visit_count"] = int(existing.get("visit_count", 0)) + 1
        return node_id

    def add_transition(
        self,
        *,
        from_grid_text: str,
        from_score: int,
        from_state: str,
        action_name: str,
        to_grid_text: str,
        to_score: int,
        to_state: str,
        step: int,
    ) -> None:
        from_id = self.upsert_node(grid_text=from_grid_text, score=from_score, state=from_state, step=step)
        to_id = self.upsert_node(grid_text=to_grid_text, score=to_score, state=to_state, step=step)

        changed = (from_grid_text != to_grid_text) or (from_score != to_score) or (from_state != to_state)
        edge = {
            "step": step,
            "from": from_id,
            "to": to_id,
            "action": action_name,
            "from_score": from_score,
            "to_score": to_score,
            "from_state": from_state,
            "to_state": to_state,
            "changed": changed,
        }
        idx = len(self._edges)
        self._edges.append(edge)
        self._out_edges[from_id].append(idx)
        if self.max_recent_transitions > 0:
            self._recent_edge_indices.append(idx)

    def build_context(self, *, current_grid_text: str) -> str:
        if not self._edges:
            return ""

        cur_id = self._state_id(current_grid_text)
        lines: list[str] = ["**Directed Graph Memory:**\n"]

        if self._recent_edge_indices:
            lines.append("**Recent Transitions:**\n")
            for idx in self._recent_edge_indices:
                e = self._edges[idx]
                delta = int(e.get("to_score", 0)) - int(e.get("from_score", 0))
                lines.append(
                    f"Step {e['step']}: {e['from_state']} --{e['action']}--> {e['to_state']} (Δscore={delta}, changed={e['changed']})"
                )
            lines.append("")

        outgoing = self._out_edges.get(cur_id, [])
        if outgoing:
            lines.append("**From This State, Tried Actions:**\n")
            per_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for idx in outgoing:
                per_action[self._edges[idx]["action"]].append(self._edges[idx])

            for action_name, edges in sorted(per_action.items(), key=lambda kv: kv[0]):
                last = edges[-1]
                loops_back = last["to"] == cur_id
                delta = int(last.get("to_score", 0)) - int(last.get("from_score", 0))
                lines.append(
                    f"- {action_name}: last_result={last['to_state']}, Δscore={delta}, changed={last['changed']}, returns_to_same_state={loops_back}"
                )
            lines.append("")

        return "\n".join(lines) + "\n"


class BasicObsActionAgentDirectedGraphMemory(ArcAgi3Agent):
    def __init__(
        self,
        system_prompt: str | None = None,
        name: str = "basic_obs_action_agent_directed_graph_memory",
        input_mode: str = "text_only",
        model: str = "gpt-5-nano",
        reasoning_effort: str = "low",
        downsample=True,
        game_id: str | None = None,
        context_window_size: int = 5,
        crop_border: int = 0,
        use_as66_prompts: bool = False,
        include_text_diff: bool = True,  # For harness compatibility (unused)
        context_length_limit: int = -1,  # For harness compatibility (unused)
        representation: RepresentationConfig | None = None,  # For harness compatibility (unused)
        use_general: bool = False,  # For harness compatibility (unused)
    ) -> None:
        self.context_window_size = context_window_size
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

    def reset(self) -> None:
        super().reset()
        self._chat_history: list[dict] = []
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None
        self._token_total: int = 0
        self._action_counter: int = 0
        self._pending_action: dict[str, Any] | None = None
        self._graph_memory = DirectedGraphMemory(max_recent_transitions=self.context_window_size)
        self._pending_transition: dict[str, Any] | None = None
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

    def _grid_text_from_observation(self, obs: dict[str, Any]) -> str:
        frame_3d = obs.get("frame", [])
        if not frame_3d:
            return ""

        if self.downsample:
            downsampled = downsample_4x4(frame_3d)
            cropped = self._crop_grid(downsampled)
            return frame_to_grid_text([cropped])
        return frame_to_grid_text([frame_3d])

    def _format_graph_context(self, current_grid_text: str) -> str:
        if self.context_window_size <= 0:
            return ""
        return self._graph_memory.build_context(current_grid_text=current_grid_text)

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        system_msg = self._system_prompt_override or build_observation_system_text(self.use_as66_prompts)
        messages: list[dict] = [{"role": "system", "content": system_msg}]
        messages.extend(self._chat_history)
        return messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict = None, **_: Any) -> None:
        self._last_observation = observation

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

        if self._chat_history and self._chat_history[-1].get("role") == "assistant":
            tool_content = self._format_observation(observation)
            self._chat_history.append({
                "role": "tool",
                "tool_call_id": self._latest_tool_call_id,
                "content": tool_content,
            })

        if self._pending_transition is not None:
            to_grid_text = self._grid_text_from_observation(observation)
            to_score = observation.get("score", 0)
            to_state = observation.get("state", "UNKNOWN")

            if to_grid_text:
                self._graph_memory.add_transition(
                    from_grid_text=self._pending_transition["grid"],
                    from_score=self._pending_transition["score"],
                    from_state=self._pending_transition["state"],
                    action_name=self._pending_transition["action"],
                    to_grid_text=to_grid_text,
                    to_score=to_score,
                    to_state=to_state,
                    step=self._pending_transition["step"],
                )
            self._pending_transition = None

    def update_from_model(self, action_payload: dict | None = None, **_: Any) -> Action:
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

        obs = self._last_observation or {}
        grid_text = self._grid_text_from_observation(obs)
        score = obs.get("score", 0)
        state = obs.get("state", "UNKNOWN")

        if grid_text:
            self._graph_memory.upsert_node(grid_text=grid_text, score=score, state=state, step=self._action_counter)

        self._pending_transition = {
            "step": self._action_counter,
            "action": action_dict["name"],
            "score": score,
            "state": state,
            "grid": grid_text,
        }

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
        obs = self._last_observation or {}
        state = obs.get("state", "NOT_PLAYED")

        if state in ("NOT_PLAYED", "GAME_OVER") and self._last_executed_action != "RESET":
            action_dict = {"name": "RESET", "data": {}, "obs_text": "Game Over, starting new game.", "action_text": ""}
            self._pending_action = action_dict
            return action_dict

        frame_3d = obs.get("frame", [])

        if self.downsample:
            downsampled = downsample_4x4(frame_3d)
            cropped = self._crop_grid(downsampled)
            grid_text = frame_to_grid_text([cropped])
        else:
            grid_text = frame_to_grid_text([frame_3d])

        score = obs.get("score", 0)

        if grid_text:
            self._graph_memory.upsert_node(
                grid_text=grid_text,
                score=score,
                state=state,
                step=self._action_counter,
            )

        obs_text = await self._call_observation_model(grid_text, score, rollout_engine=rollout_engine)
        action_dict = await self._call_action_model(grid_text, obs_text, rollout_engine=rollout_engine)
        action_dict["obs_text"] = obs_text

        self._pending_action = action_dict
        return action_dict

    def _build_user_content(self, grid: List[List[int]], user_prompt_text: str) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt_text}]

        if self.input_mode in ["image_only", "text_and_image"]:
            try:
                png_bytes = generate_numeric_grid_image_bytes(grid)
                b64_image = base64.b64encode(png_bytes).decode("utf-8")
                data_url = f"data:image/png;base64,{b64_image}"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "low"},
                })
            except Exception as e:
                log.error(f"Failed to generate numeric grid image: {e}")

        return content

    async def rollout(self, rollout_engine: OpenAIEngine, messages: List[Dict[str, Any]], tools=None):
        return await rollout_engine.get_model_response(messages, tools=tools)

    @weave_op
    async def _call_observation_model(self, grid: Any, score: int, rollout_engine=None) -> str:
        sys_msg = build_observation_system_text(self.use_as66_prompts)

        if self.downsample:
            base_size = 16
            actual_size = base_size - (2 * self.crop_border)
            grid_size_text = f"{actual_size}x{actual_size}" if self.crop_border > 0 else "16x16"
        else:
            grid_size_text = "64x64"

        history_context = self._format_graph_context(str(grid))

        user_msg_text = (
            f"{history_context}"
            f"**Current State:**\n"
            f"Score: {score}\n"
            f"Step: {self._action_counter}\n\n"
            f"**Current Matrix** {grid_size_text} (ASCII characters):\n{grid}\n\n"
            "Rationale:\n"
            "  • Identify the movable ASCII character(s) and relevant structures.\n"
            "  • Conclude which direction is best and why. Do not output an action here.\n"
            "  • Focus on the strategic importance of each character and how it relates to the goal."
        )

        self._last_observation_prompt = f"SYSTEM: {sys_msg}\n\nUSER: {user_msg_text}"

        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg_text}]
        model_output = await self.rollout(rollout_engine, messages)
        text = (getattr(model_output, "content", None) or getattr(model_output, "text", "") or "").strip()

        self._last_observation_response = text
        return text

    @weave_op
    async def _call_action_model(self, grid: Any, last_obs: str, rollout_engine=None) -> dict:
        sys_msg = build_action_system_text(self.use_as66_prompts)

        history_context = self._format_graph_context(str(grid))

        user_msg_text = (
            f"{history_context}"
            "Choose the best single move as a function call.\n"
            f"{grid}"
            "Previous observation summary:\n"
            f"{last_obs}\n"
        )

        self._last_action_prompt = f"SYSTEM: {sys_msg}\n\nUSER: {user_msg_text}"

        tools = _build_tools()
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg_text}]
        model_output = await self.rollout(rollout_engine, messages, tools)

        m = model_output.tool_calls[0] if getattr(model_output, "tool_calls", None) else None

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

        self._chat_history.append({
            "role": "assistant",
            "tool_calls": [{
                "id": tc_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }],
        })

        action_content = model_output.content or ""
        self._last_action_response = f"Tool Call: {name}({json.dumps(args)})\nContent: {action_content}"

        if name == "ACTION6":
            x_raw = args.get("x", 0)
            y_raw = args.get("y", 0)
            if self.downsample:
                x_64 = x_raw * 4
                y_64 = y_raw * 4
            else:
                x_64 = x_raw
                y_64 = y_raw
            return {"name": name, "data": {"x": x_64, "y": y_64}, "action_text": model_output.content}

        return {"name": name, "data": args, "action_text": model_output.content}
