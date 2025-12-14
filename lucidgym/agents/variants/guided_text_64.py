"""
64x64 full-resolution text-based guided agent.
Identical to guided_text_16 but operates on full 64x64 grids without downsampling.
"""
from __future__ import annotations
from typing import Any, List, Dict
import json
import logging
from openai import OpenAI

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

from lucidgym.environments.arcagi3.structs import GameAction, GameState

log = logging.getLogger(__name__)


def _build_observation_system_text_64() -> str:
    return (
        "You are playing a game which is represented by a 64×64 matrix of integer codes. "
        "This matrix is the current state after a reset or after your most recent move. "
        "Your task is to observe the position and analyze the potential benefits and drawbacks of each possible move.\n\n"
        "Movement model:\n"
        "• There is one main movable square of integers. Its specific value may vary across levels, but it will be unique among the surrounding integers (it could even be a square of 2/6/8/9 among other ints). "
        "  There may also be multiple movable integers; they remain distinct from the rest of the board and move together under the same command.\n"
        "• When you choose a direction (Up, Down, Left, Right), each movable integer slides as far as possible in that direction. "
        "  Sliding wraps across the board edges when unobstructed (after passing beyond one edge, it reappears from the opposite edge and continues). "
        "  Sliding stops as soon as an obstacle blocks further motion.\n\n"
        "If you move in a direction with no obstacles ever, you move back to where you started. For example, if one moves up and the entire wraparound has no obstacles, the game state does not change due to your action and you remain stationary.\n\n"
        "Obstacles and board semantics:\n"
        "• 4 are walls. Sliding stops adjacent to a 4 and cannot overlap 4s.\n"
        "• 15 are background cells that constitute the playable area (free to traverse and occupy).\n"
        "• The board is typically delimited by boundaries such as 1 or 14. You can generally localize the playable field by the region filled with 15s.\n"
        "• 0 are target cells. They form a U shape that can be viewed as a larger 8×12 rectangle with a center block removed. "
        "  Your objective is to navigate the movable integer(s) into this space to complete the U by filling its cavity. "
        "  The 0 region also interrupts sliding (you stop upon reaching it when the motion would otherwise continue).\n"
        "• You may observe a perimeter/track behavior using 8 and 9; this indicates the consumption of available moves, the more 9s you see, the less moves you have\n\n"
        "Hostile entity (avoid at all costs but note frequently there is not one. They are large 12 by 12 and easy to spot):\n"
        "• Larger composite blocks consisting of 8 and 9 indicate an enemy. If any movable integer collides with this enemy, it is game over.\n"
        "• The position of 9 within that 8/9 block indicates the direction in which this enemy will step per your move (one tile, row, or column, potentially diagonally depending on level behavior). "
        "  If the 9 is centered and the entity is stationary, it remains stationary. Use history to infer whether it moves.\n\n"
        "Multiple movers:\n"
        "• If multiple movable integers exist, they all move together in the same chosen direction. Avoid any collision with the hostile entity while advancing the objective of completing the U.\n\n"
        "Target matching with multiple movers:\n"
        "• On later levels, there may be multiple target U regions that expect specific movers. The 0 U may include a single block of the intended mover's code to indicate which mover should fill that particular U.\n\n"
        "What to produce during observation (rationale only):\n"
        "• Identify the locations of the movable integer(s) and all relevant structures (0 region, 4 walls, 15 background, 1/14 boundaries, any 8/9 enemy mass). "
        "• For each direction (Up, Down, Left, Right), reason carefully about full wrap-around sliding: what blocking elements will be met, what will be the final resting locations, and how these outcomes change proximity/alignment to the U cavity. "
        "• Consider the enemy's response (8/9), including whether a move would cause immediate collision or a forced collision on the subsequent step. "
        "• Conclude which direction best progresses toward completing the 8×12 cavity in the 0 region while avoiding risk. "
        "This is a text-only analysis turn; do not name or call an action tool here. "
        "THE MOST IMPORTANT THING TO KEEP IN MIND IS THE RESULTS OF YOUR PAST ACTIONS AND PREVIOUSLY WHAT STATE CHANGE CAME FROM THEM. DO NOT REPEAT ACTIONS THAT CHANGED NOTHING."
    )


def _build_action_system_text_64() -> str:
    return (
        "Select exactly one move by calling a single tool. Do not include prose.\n"
        "Available tools:\n"
        "• ACTION1 = Up\n"
        "• ACTION2 = Down\n"
        "• ACTION3 = Left\n"
        "• ACTION4 = Right"
    )


def _matrix64_to_lines(mat: List[List[int]]) -> str:
    """Convert 64x64 matrix to text representation."""
    if not mat:
        return "(empty)"
    return "\n".join(" ".join(str(v) for v in row) for row in mat)


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
    ]


class AS66GuidedAgent64(BaseAgent):
    """
    64×64 full-resolution agent using rllm BaseAgent interface.
    No downsampling - operates on full game grid.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        name: str = "as66_guided_agent_64",
        model: str = "gpt-4o",
        reasoning_effort: str = "low",
    ) -> None:
        """
        Initialize the agent.

        Args:
            system_prompt: Override system prompt (optional)
            name: Agent name
            model: OpenAI model to use
            reasoning_effort: Reasoning effort level
        """
        self.name = name
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._system_prompt_override = system_prompt

        self._client = OpenAI()
        self._latest_tool_call_id = "call_12345"
        self.reset()

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self._chat_history: list[dict] = []
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None
        self._token_total: int = 0
        self._action_counter: int = 0

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return message history formatted for chat API."""
        system_msg = self._system_prompt_override or _build_observation_system_text_64()
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

        step = Step(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            chat_completions=self.chat_completions.copy()
        )
        self._trajectory.steps.append(step)

        if self._chat_history and self._chat_history[-1].get("role") == "assistant":
            state = observation.get("state", "NOT_PLAYED")
            score = observation.get("score", 0)
            tool_content = f"State: {state} | Score: {score}"

            self._chat_history.append({
                "role": "tool",
                "tool_call_id": self._latest_tool_call_id,
                "content": tool_content
            })

    def update_from_model(self, response: str, **_: Any) -> Action:
        """Convert model response to Action."""
        if not self._last_observation:
            return Action(action={"name": "RESET", "data": {}})

        obs = self._last_observation
        state = obs.get("state", "NOT_PLAYED")

        if state in ("NOT_PLAYED", "GAME_OVER"):
            return Action(action={"name": "RESET", "data": {}})

        frame_3d = obs.get("frame", [])
        if not frame_3d:
            return Action(action={"name": "ACTION5", "data": {}})

        # Use full 64x64 grid (last layer)
        grid_64 = frame_3d[-1] if frame_3d else []
        score = obs.get("score", 0)

        # Step 1: Observation phase
        obs_text = self._call_observation_model(grid_64, score)

        # Step 2: Action selection phase
        action_dict = self._call_action_model(grid_64, obs_text)

        # Attach observation text as reasoning for ARC API replay logs
        action_dict["reasoning"] = obs_text

        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = obs_text + " | " + str(action_dict)
            self._trajectory.steps[-1].action = action_dict

        self._action_counter += 1
        return Action(action=action_dict)

    def _call_observation_model(self, grid_64: List[List[int]], score: int) -> str:
        """Call the model for observation/reasoning phase."""
        sys_msg = _build_observation_system_text_64()

        matrix_text = _matrix64_to_lines(grid_64)
        user_msg = (
            f"Score: {score}\n"
            f"Step: {self._action_counter}\n"
            f"Matrix 64x64 (integer codes):\n{matrix_text}\n\n"
            "Rationale:\n"
            "  • Identify the movable integer(s) and relevant structures.\n"
            "  • For Up, Down, Left, Right: fully simulate wrap-around sliding, state blockers, and final landing positions.\n"
            "  • Explain how each landing affects progress toward completing the U cavity and whether the enemy's response threatens collision.\n"
            "  • Conclude which direction is best and why. Do not output an action here."
        )

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
        }

        # Add reasoning_effort for GPT-5 series models
        if self.model.startswith("gpt-5"):
            api_params["reasoning_effort"] = self.reasoning_effort

        resp = self._client.chat.completions.create(**api_params)

        self._token_total += getattr(resp.usage, "total_tokens", 0) or 0
        text = (resp.choices[0].message.content or "").strip()

        if resp.choices[0].message.tool_calls:
            text = "(observation only; tool call suppressed)"

        return text

    def _call_action_model(self, grid_64: List[List[int]], last_obs: str) -> dict:
        """Call the model for action selection phase."""
        sys_msg = _build_action_system_text_64()

        matrix_text = _matrix64_to_lines(grid_64)
        user_msg = (
            f"Choose the best single move as a function call.\n"
            f"Matrix 64x64 (integer codes):\n{matrix_text}\n\n"
            f"Previous observation summary:\n{last_obs}\n"
        )

        tools = _build_tools()

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
        }

        # Add reasoning_effort for GPT-5 series models
        if self.model.startswith("gpt-5"):
            api_params["reasoning_effort"] = self.reasoning_effort

        resp = self._client.chat.completions.create(**api_params)

        self._token_total += getattr(resp.usage, "total_tokens", 0) or 0

        m = resp.choices[0].message
        if not m.tool_calls:
            return {"name": "ACTION1", "data": {}}

        tc = m.tool_calls[0]
        self._latest_tool_call_id = tc.id
        name = tc.function.name

        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {}

        self._chat_history.append({
            "role": "assistant",
            "tool_calls": [{
                "id": tc.id,
                "type": "function",
                "function": {"name": name, "arguments": tc.function.arguments}
            }]
        })

        return {"name": name, "data": args}
