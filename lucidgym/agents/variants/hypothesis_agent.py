"""
Memory-based agent with hypothesis tracking for AS66 game.
Uses external memory file to track move history and dynamic hypotheses.
"""
from __future__ import annotations
from typing import Any, List, Dict, Optional
import json
import logging
import uuid
from pathlib import Path
import os
import hashlib
from openai import OpenAI

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

from lucidgym.environments.arcagi3.structs import GameAction, GameState
from lucidgym.utils.grid_processing import downsample_4x4, matrix16_to_lines
from lucidgym.prompts.memory_prompts import (
    build_initial_hypotheses_system_prompt,
    build_initial_hypotheses_user_prompt,
    build_update_hypotheses_system_prompt,
    build_update_hypotheses_user_prompt,
    build_observation_system_prompt,
    build_observation_user_prompt,
    build_action_selection_system_prompt,
    build_action_selection_user_prompt,
)

log = logging.getLogger(__name__)


class AS66MemoryAgent(BaseAgent):
    """
    Agent using external memory to manage context, including move history and
    dynamic hypotheses about game mechanics. Uses a three-call process:
    Hypothesis Update, Observation, and Action Selection.
    """

    MEMORY_DIR = Path("memory")

    def __init__(
        self,
        name: str = "as66_memory_agent",
        model: str = "gpt-4o",
        reasoning_effort: str = "low",
        game_id: str | None = None,
        downsample: bool = True,
        include_text_diff: bool = True,
        context_length_limit: int = -1,
    ) -> None:
        """
        Initialize the memory agent.

        Args:
            name: Agent name
            model: OpenAI model to use
            reasoning_effort: Reasoning effort level
            game_id: Game identifier for memory file naming
            downsample: Whether to downsample grids to 16x16
            include_text_diff: Whether to include text diffs in memory
            context_length_limit: Max context tokens (-1 for unlimited)
        """
        self.name = name
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.game_id = game_id or f"game_{uuid.uuid4().hex[:8]}"
        self.downsample = downsample
        self.include_text_diff = include_text_diff
        self.context_length_limit = context_length_limit

        self._client = OpenAI()

        # Memory setup
        self.MEMORY_DIR.mkdir(exist_ok=True)
        self.memory_file_path = self.MEMORY_DIR / f"{self.game_id}_{uuid.uuid4()}.md"

        # State tracking
        self.seen_state_actions: set = set()
        self.move_history_content = "## Move History\n\n(No moves recorded yet.)"
        self.hypotheses_content = "## Hypotheses\n\n(No hypotheses generated yet.)"
        self._is_initialized = False
        self._token_total = 0

        # Latest tool call ID for chat history
        self._latest_tool_call_id = "call_12345"

        self.reset()

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self._chat_history: list[dict] = []
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None
        self._action_counter: int = 0

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return message history formatted for chat API."""
        messages: list[dict] = []
        messages.extend(self._chat_history)
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Return the trajectory tracking object."""
        return self._trajectory

    def _get_token_count(self, text: str) -> int:
        """Lightweight proxy for token counting (4 chars/token)."""
        return len(text) // 4

    def _get_truncated_history(self) -> str:
        """
        Apply context length limit with sliding window that preserves
        high-information (level up / game over) entries.
        """
        if self.context_length_limit == -1:
            return self.move_history_content

        try:
            header, all_entries_str = self.move_history_content.split("\n\n", 1)
            entries = all_entries_str.split("\n---\n\n")
        except ValueError:
            return self.move_history_content

        high_info_entries = []
        regular_entries = []

        for i, entry in enumerate(entries):
            if "LEVEL UP!" in entry or "GAME OVER!" in entry:
                high_info_entries.append((i, entry, self._get_token_count(entry)))
            else:
                regular_entries.append((i, entry, self._get_token_count(entry)))

        high_info_tokens = sum(tokens for _, _, tokens in high_info_entries)
        remaining_budget = self.context_length_limit - high_info_tokens

        kept_entries = [(i, entry) for i, entry, _ in high_info_entries]

        if remaining_budget > 0:
            kept_regular_tokens = 0
            for i, entry, tokens in reversed(regular_entries):
                if (kept_regular_tokens + tokens) <= remaining_budget:
                    kept_entries.append((i, entry))
                    kept_regular_tokens += tokens
                else:
                    break

        kept_entries.sort(key=lambda x: x[0])
        final_entries_str = "\n---\n\n".join([entry for _, entry in kept_entries])
        return f"{header}\n\n{final_entries_str}"

    def _read_memory(self) -> str:
        """Read current memory content."""
        move_history = self._get_truncated_history()
        return f"{move_history}\n\n{self.hypotheses_content}"

    def _write_memory(self) -> None:
        """Write memory to file."""
        content = self._read_memory()
        self.memory_file_path.write_text(content, encoding="utf-8")

    def _get_state_hash(self, grid: List[List[int]]) -> str:
        """Generate hash for state deduplication."""
        return hashlib.md5(json.dumps(grid, sort_keys=True).encode()).hexdigest()

    def _calculate_diff(self, grid1: List[List[int]], grid2: List[List[int]]) -> str:
        """Calculate textual diff between two grids."""
        diffs = []
        h = len(grid1)
        w = len(grid1[0]) if h > 0 else 0
        if len(grid2) != h or (h > 0 and len(grid2[0]) != w):
            return "Error: Grids have different dimensions."

        for r in range(h):
            for c in range(w):
                if grid1[r][c] != grid2[r][c]:
                    diffs.append(f"- Cell ({r}, {c}): {grid1[r][c]} -> {grid2[r][c]}")

        if not diffs:
            return "No change in board state."
        return "\n".join(diffs)

    def _initialize_memory(self, observation: dict) -> None:
        """Generate initial set of hypotheses."""
        log.info(f"[{self.game_id}] Initializing memory and hypotheses...")

        grid_3d = observation.get("frame", [])
        grid = downsample_4x4(grid_3d, take_last_grid=True, round_to_int=True) if self.downsample else (grid_3d[-1] if grid_3d else [])

        sys_prompt = build_initial_hypotheses_system_prompt()
        user_prompt = build_initial_hypotheses_user_prompt(grid)

        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        }

        # Add reasoning_effort for GPT-5 series models
        if self.model.startswith("gpt-5"):
            api_params["reasoning_effort"] = self.reasoning_effort

        resp = self._client.chat.completions.create(**api_params)

        self.hypotheses_content = f"## Hypotheses\n\n{(resp.choices[0].message.content or '').strip()}"
        self._token_total += getattr(resp.usage, "total_tokens", 0) or 0
        self._write_memory()
        self._is_initialized = True
        log.info(f"[{self.game_id}] Initial hypotheses generated and saved to {self.memory_file_path}")

    def _update_memory_from_action(
        self,
        prev_observation: dict,
        action_dict: dict,
        new_observation: dict
    ) -> bool:
        """
        Update memory with latest move and revised hypotheses.
        Returns True if the state-action pair was a repeat, False otherwise.
        """
        prev_grid_3d = prev_observation.get("frame", [])
        new_grid_3d = new_observation.get("frame", [])

        prev_grid = downsample_4x4(prev_grid_3d, take_last_grid=True, round_to_int=True) if self.downsample else (prev_grid_3d[-1] if prev_grid_3d else [])
        new_grid = downsample_4x4(new_grid_3d, take_last_grid=True, round_to_int=True) if self.downsample else (new_grid_3d[-1] if new_grid_3d else [])

        state_hash = self._get_state_hash(prev_grid)
        action_name = action_dict.get("name", "UNKNOWN")
        action_identifier = action_name

        state_action_tuple = (state_hash, action_identifier)

        if state_action_tuple in self.seen_state_actions:
            log.warning(f"[{self.game_id}] Repeated state-action pair detected: {action_identifier}. Applying penalty.")
            return True

        self.seen_state_actions.add(state_action_tuple)

        prev_score = prev_observation.get("score", 0)
        new_score = new_observation.get("score", 0)
        is_level_up = new_score > prev_score
        diff = self._calculate_diff(prev_grid, new_grid)

        entry_header = f"### Turn {len(self.seen_state_actions)}\n\n"

        entry_parts_list = [
            f"**Action:** `{action_identifier}`\n\n",
            "**State Before:**\n",
            "```\n",
            f"{matrix16_to_lines(prev_grid)}\n",
            "```\n\n",
            "**Resulting State:**\n",
            "```\n",
            f"{matrix16_to_lines(new_grid)}\n",
            "```\n"
        ]

        if self.include_text_diff:
            entry_parts_list.extend([
                "\n**Resulting Diff:**\n",
                "```\n",
                f"{diff}\n",
                "```\n"
            ])
        else:
            entry_parts_list.append("\n")

        entry_body = "".join(entry_parts_list)

        if is_level_up:
            history_entry = f"> **LEVEL UP! A new level begins below.**\n>\n{'> '.join((entry_header + entry_body).splitlines(True))}\n---\n\n"
        elif new_observation.get("state") == "GAME_OVER":
            history_entry = f"> **GAME OVER!**\n>\n{'> '.join((entry_header + entry_body).splitlines(True))}\n---\n\n"
        else:
            history_entry = f"{entry_header}{entry_body}\n---\n\n"

        if "(No moves recorded yet.)" in self.move_history_content:
            self.move_history_content = "## Move History\n\n" + history_entry
        else:
            self.move_history_content += history_entry

        # Update hypotheses
        log.info(f"[{self.game_id}] Updating hypotheses based on new move...")
        sys_prompt = build_update_hypotheses_system_prompt()
        user_prompt = build_update_hypotheses_user_prompt(self._read_memory())

        if is_level_up:
            special_instruction = (
                "**IMPORTANT CONTEXT: A LEVEL UP just occurred.** The game board has changed for the new level. "
                "Your primary task now is to re-validate your current hypotheses against this new environment. "
                "Generate a new set of five hypotheses that reflect your understanding of this new level.\n\n"
            )
            user_prompt = special_instruction + user_prompt

        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        }

        # Add reasoning_effort for GPT-5 series models
        if self.model.startswith("gpt-5"):
            api_params["reasoning_effort"] = self.reasoning_effort

        resp = self._client.chat.completions.create(**api_params)
        self._token_total += getattr(resp.usage, "total_tokens", 0) or 0

        new_hypotheses = (resp.choices[0].message.content or "").strip()
        self.hypotheses_content = "## Hypotheses\n\n" + (new_hypotheses if new_hypotheses else self.hypotheses_content.split("\n\n", 1)[1])

        self._write_memory()
        log.info(f"[{self.game_id}] Memory file updated.")
        return False

    def _get_observation_text(self, memory_content: str, grid: List[List[int]], score: int, step: int) -> str:
        """Call the LLM to get text-based observation and rationale."""
        sys_prompt = build_observation_system_prompt()
        user_prompt = build_observation_user_prompt(memory_content, grid, score, step)

        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        }

        # Add reasoning_effort for GPT-5 series models
        if self.model.startswith("gpt-5"):
            api_params["reasoning_effort"] = self.reasoning_effort

        resp = self._client.chat.completions.create(**api_params)
        self._token_total += getattr(resp.usage, "total_tokens", 0) or 0

        observation = (resp.choices[0].message.content or "No observation generated.").strip()
        log.info(f"[{self.game_id} | Step {step}] Observation Rationale: {observation}")
        return observation

    def _select_action(self, observation_text: str) -> dict:
        """Takes observation text and returns action via tool call."""
        sys_prompt = build_action_selection_system_prompt()
        user_prompt = build_action_selection_user_prompt(observation_text)

        tools = self._build_tools()

        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            "tools": tools,
            "tool_choice": "required",
        }

        # Add reasoning_effort for GPT-5 series models
        if self.model.startswith("gpt-5"):
            api_params["reasoning_effort"] = self.reasoning_effort

        resp = self._client.chat.completions.create(**api_params
        )
        self._token_total += getattr(resp.usage, "total_tokens", 0) or 0

        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            log.error("Action selection model failed to call a tool. Defaulting to ACTION1.")
            return {"name": "ACTION1", "data": {}}

        tool_call = tool_calls[0]
        action_name = tool_call.function.name
        self._latest_tool_call_id = tool_call.id

        try:
            arguments = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
        except json.JSONDecodeError:
            log.error(f"Failed to parse arguments for {action_name}. Defaulting to no arguments.")
            arguments = {}

        # Add to chat history
        self._chat_history.append({
            "role": "assistant",
            "tool_calls": [{
                "id": tool_call.id,
                "type": "function",
                "function": {"name": action_name, "arguments": tool_call.function.arguments}
            }]
        })

        log.info(f"[{self.game_id}] Selected Action: {action_name}")
        return {"name": action_name, "data": arguments}

    def _build_tools(self) -> list[dict]:
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
        ]

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **_: Any) -> None:
        """Process environment observation and update state."""
        prev_observation = self._last_observation
        self._last_observation = observation

        step = Step(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            chat_completions=self.chat_completions.copy()
        )
        self._trajectory.steps.append(step)

        # Add tool response to chat history if there was a previous action
        if self._chat_history and self._chat_history[-1].get("role") == "assistant":
            state = observation.get("state", "NOT_PLAYED")
            score = observation.get("score", 0)
            tool_content = f"State: {state} | Score: {score}"

            self._chat_history.append({
                "role": "tool",
                "tool_call_id": self._latest_tool_call_id,
                "content": tool_content
            })

        # Update memory if this is a post-action observation
        if prev_observation is not None and hasattr(self, '_last_action_dict'):
            self._update_memory_from_action(prev_observation, self._last_action_dict, observation)

    def update_from_model(self, response: str, **_: Any) -> Action:
        """Convert model response to Action."""
        if not self._last_observation:
            return Action(action={"name": "RESET", "data": {}})

        obs = self._last_observation
        state = obs.get("state", "NOT_PLAYED")

        # Initialize memory on first observation
        if not self._is_initialized:
            self._initialize_memory(obs)

        # Handle game states
        if state in ("NOT_PLAYED", "GAME_OVER"):
            return Action(action={"name": "RESET", "data": {}})

        frame_3d = obs.get("frame", [])
        if not frame_3d:
            return Action(action={"name": "ACTION1", "data": {}})

        grid = downsample_4x4(frame_3d, take_last_grid=True, round_to_int=True) if self.downsample else (frame_3d[-1] if frame_3d else [])
        score = obs.get("score", 0)

        # Get memory and observation
        memory_content = self._read_memory()
        observation_text = self._get_observation_text(memory_content, grid, score, self._action_counter)

        # Select action
        action_dict = self._select_action(observation_text)

        # Attach observation text (which includes hypotheses + reasoning) for ARC API replay logs
        action_dict["reasoning"] = observation_text

        # Store for memory update on next observation
        self._last_action_dict = action_dict

        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = observation_text + " | " + str(action_dict)
            self._trajectory.steps[-1].action = action_dict

        self._action_counter += 1
        return Action(action=action_dict)
