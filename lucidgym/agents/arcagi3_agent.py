"""
ARC-AGI-3 agent scaffold that converts environment observations into prompts.

The agent works in two modes:

1. ``llm`` (default) – format observations into chat prompts and parse the LLM's
   response into a structured ARC ``GameAction`` dict.
2. ``passthrough`` – delegate action selection to a caller-supplied function so
   existing ARC agents can be reused without rewriting them for LucidGym.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable
import textwrap
from openai.types.chat import ChatCompletionMessageFunctionToolCall

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput

from arcengine import GameAction
from lucidgym.utils.grid_processing import flatten_frame, downsample_4x4, frame_to_grid_text, format_grid
from lucidgym.utils.representation import RepresentationConfig, GridFormat



def build_initial_usr_prompt(grid):
    return (
        "Analyze the following game state grid and determine the best action to take.\n\n"
        f"{grid}"
    )


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert ARC-AGI-3 agent that plays a grid-based reasoning game. Observe the grid, reason about the task, "
    "then emit a tool call with a 'name' field matching ACTION1-6 plus "
    "optional coordinates (x,y) when the action requires them. Include reasoning summarizing why the action was chosen.\n\n"
    "Each time you act, the game state updates automatically and you will receive a new observation. "
)



PassthroughFn = Callable[[dict[str, Any], Trajectory], dict[str, Any]]


class ArcAgi3Agent(BaseAgent):
    """
    Minimal agent facade for ARC-AGI-3 environments.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        name: str = "arcagi3_agent",
        mode: str = "llm",
        downsample: bool = True,
        grid: bool = True,
        representation: RepresentationConfig | None = None,
    ) -> None:
        if mode not in {"llm", "passthrough"}:
            raise ValueError("mode must be either 'llm' or 'passthrough'.")
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.name = name
        self.mode = mode
        self._latest_tool_call_id = "call_12345"
        self.downsample = downsample
        self.grid = grid
        # Representation config for grid formatting
        self.representation = representation or RepresentationConfig(
            downsample=downsample,
        )
        self.reset()

    def reset(self) -> None:
        self._steps_this_episode = 0
        self._chat_history: list[dict[str, str]] = []
        self._trajectory = Trajectory(name=self.name)
        self._last_observation: dict[str, Any] | None = None

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._chat_history)
        return messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    # ------------------------------------------------------------------ #
    def update_from_env(self, observation: Any, reward: float, done: bool, **_: Any) -> None:
        # Update internal state
        self._last_observation = observation

        # Handle first observation (initial state)
        if len(self._chat_history) == 0:
            grid_text = self._format_observation(self._last_observation)
            initial_prompt = build_initial_usr_prompt(grid_text)
            self._chat_history.append({"role": "user", "content": initial_prompt})
            return


        if "tool_calls" not in self._chat_history[-1]:
            # no tool calls in the last message, ask llm to do it again.
            self._chat_history.append({"role": "user", "content": "No tool calls found in the response. Please make sure to call a valid tool."})
            step = Step(observation=observation, reward=reward, done=done, info={}, chat_completions=self.chat_completions.copy())
            self._trajectory.steps.append(step)
        else:
            user_msg = self._format_observation(self._last_observation)
            if user_msg:
                self._chat_history.append({"role": "tool", "tool_call_id": self._latest_tool_call_id, "content": user_msg})

            step = Step(observation=observation, reward=reward, done=done, info={}, chat_completions=self.chat_completions.copy())
            self._trajectory.steps.append(step)


    def update_from_model(self, response: dict | ModelOutput) -> dict:
        # Handle RESET needed states (with null safety)
        state = self._last_observation.get("state", "NOT_PLAYED") if self._last_observation else "NOT_PLAYED"
        if state in ("NOT_PLAYED", "GAME_OVER"):
            response = ModelOutput(text="Game Over, starting new game.", content="", reasoning="", tool_calls=["RESET"])
        action_payload = {}
        args = {}
        text = response.text
        content = response.content
        reasoning = getattr(response, 'reasoning', None)
        tool_calls = getattr(response, 'tool_calls', [])

        if len(tool_calls) == 0:
            name = "ACTION5"
        else:
            if isinstance(tool_calls[0], str):
                name = "RESET"
                self._latest_tool_call_id = "call_reset_1234"
                tool_calls = [ChatCompletionMessageFunctionToolCall(id="call_reset_1234", function={"name": "RESET", "arguments": "{}"}, type="function")]
            else:
                tc = response.tool_calls[0]
                self._latest_tool_call_id = tc.id
                name = tc.function.name
                arguments = tc.function.arguments
                args = json.loads(arguments or "{}")
                
                if name == "ACTION6":
                    x_raw = int(args.get("x", 0))
                    y_raw = int(args.get("y", 0))
                    if self.downsample:
                        # Clamp to 16x16 range before scaling
                        x_raw = max(0, min(15, x_raw))
                        y_raw = max(0, min(15, y_raw))
                        x_pos = x_raw * 4
                        y_pos = y_raw * 4
                    else:
                        # Clamp to 64x64 range
                        x_pos = max(0, min(63, x_raw))
                        y_pos = max(0, min(63, y_raw))
                    action_payload["x"] = x_pos
                    action_payload["y"] = y_pos
        action = GameAction.from_name(name)
        action_payload["action"] = action
        action_payload["reasoning"] = f"{text}\n{name} {args}"
        self._steps_this_episode +=1
            
        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = response
            self._trajectory.steps[-1].action = action_payload


        if tool_calls and len(tool_calls) > 0:
            tool_call_dicts = [x.to_dict() for x in tool_calls]
            message = {"role": "assistant", "content": text, "tool_calls": tool_call_dicts}
        else:
            message = {"role": "assistant", "content": text}

        print(f"[DEBUG]:arcagi3_agent:message={message}\nresponse={response}")
        self._chat_history.append(message)
        return action_payload

    # ------------------------------------------------------------------ #
    async def call_llm(self, rollout_engine=None) -> tuple[str, dict]:
        """Run the two-phase observation/action LLM calls and return text + action dict."""
        state = self._last_observation.get("state", "NOT_PLAYED") if self._last_observation else "NOT_PLAYED"

        # Handle RESET needed states
        # print(f"[DEBUG]:[guided]: obs={obs}")
        if state in ("NOT_PLAYED", "GAME_OVER"):
            model_output = ModelOutput(text="Game Over, starting new game.", content="", reasoning="", tool_calls=["RESET"])
            return model_output

        chat_so_far = self.chat_completions
        tools = self.build_tools()
        model_output = await self.rollout(rollout_engine, chat_so_far, tools)
        # print(f"[DEBUG]:arcagi3_agent: model_output={model_output}")
        
        return model_output
    async def rollout(self, rollout_engine: OpenAIEngine, messages: List[Dict[str, Any]], tools=None):
        return await rollout_engine.get_model_response(messages, tools=tools)
    
    def _format_observation(self, observation: dict[str, Any]) -> str:
        frame = observation.get("frame", [])

        # Use representation config if available
        if self.representation:
            if self.representation.downsample:
                grid_2d = downsample_4x4(frame) if frame else []
            else:
                # Get raw 2D grid from 3D frame
                grid_2d = frame[-1] if frame else []
            frame_text = format_grid(grid_2d, self.representation) if grid_2d else "No frame data"
            print(frame_text)
        elif self.downsample:
            frame = [downsample_4x4(frame)] if frame else []
            if self.grid:
                frame_text = frame_to_grid_text(frame)
            else:
                frame_text = self.pretty_print_3d(frame)
        else:
            if self.grid:
                frame_text = frame_to_grid_text(frame)
            else:
                frame_text = self.pretty_print_3d(frame)

        available_actions = observation.get("available_actions") or []

        return textwrap.dedent(
            """
            # State:
            {state}
            Step {step}

            # Score:
            {score}

            # Frame:
            {frame}
            """
        ).format(
            state=observation.get("state", "UNKNOWN"),
            step=self._steps_this_episode,
            score=observation.get("score", 0),
            frame=frame_text if frame_text else "N/A",
        )
        # Output next action:
        # Reply with a several sentences/ paragraphs of plain-text strategy observation about the frame to inform your next action. The output should be a tool call indicating the action to take.



    def build_functions(self) -> list[dict[str, Any]]:
        """Build JSON function description of game actions for LLM."""
        empty_params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        functions: list[dict[str, Any]] = [
            {
                "name": GameAction.RESET.name,
                "description": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION1.name,
                "description": "Send this simple input action (1, W, Up).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION2.name,
                "description": "Send this simple input action (2, S, Down).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION3.name,
                "description": "Send this simple input action (3, A, Left).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION4.name,
                "description": "Send this simple input action (4, D, Right).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION5.name,
                "description": "Send this simple input action (5, Enter, Spacebar, Delete).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION6.name,
                "description": "Send this complex input action (6, Click, Point).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "string",
                            "description": "Coordinate X which must be Int<0,63>",
                        },
                        "y": {
                            "type": "string",
                            "description": "Coordinate Y which must be Int<0,63>",
                        },
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
            },
        ]
        return functions

    def build_tools(self) -> list[dict[str, Any]]:
        """Support models that expect tool_call format."""
        functions = self.build_functions()
        tools: list[dict[str, Any]] = []
        for f in functions:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f["name"],
                        "description": f["description"],
                        "parameters": f.get("parameters", {}),
                        "strict": True,
                    },
                }
            )
        return tools
        
    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(" ".join(str(x) for x in row))
            lines.append("")
        return "\n".join(lines)
