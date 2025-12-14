"""
Wrapper to make rllm BaseAgent compatible with the legacy evaluation harness.
This allows the harness to continue using take_action(), choose_action(), etc.
while the agents use the new BaseAgent interface.
"""
from __future__ import annotations
from typing import Any, List
import requests

from rllm.agents.agent import BaseAgent, Action

from lucidgym.environments.arcagi3.structs import ActionInput, FrameData, GameAction, GameState

import logging
log = logging.getLogger(__name__)


class BaseAgentWrapper:
    """
    Wraps a BaseAgent to provide the legacy Agent interface expected by harness.py.

    The evaluation harness expects:
    - agent.take_action(action) -> FrameData
    - agent.choose_action(frames, latest_frame) -> GameAction
    - agent.append_frame(frame)
    - agent.agent_name (property)
    - agent.frames (list)
    - agent.action_counter (int)

    This wrapper translates between the harness's expectations and BaseAgent's interface.
    """

    def __init__(
        self,
        base_agent: BaseAgent,
        game_id: str,
        card_id: str,
        root_url: str,
        session: requests.Session
    ):
        """
        Initialize the wrapper.

        Args:
            base_agent: The BaseAgent instance to wrap
            game_id: Game ID for API calls
            card_id: Scorecard ID for API calls
            root_url: ARC API root URL
            session: requests.Session with auth headers/cookies
        """
        self.agent = base_agent
        self.game_id = game_id
        self.card_id = card_id
        self.root_url = root_url
        self.session = session

        # Legacy interface state
        self.frames: List[FrameData] = []
        self.action_counter: int = 0
        self.guid: str | None = None
        self._last_reasoning: str | None = None

    @property
    def agent_name(self) -> str:
        """Return agent name for metrics."""
        return self.agent.name

    def take_action(self, action: GameAction) -> FrameData:
        """
        Execute an action via the ARC API and return the resulting frame.

        This method:
        1. Calls the ARC API with the action
        2. Parses the response into a FrameData object
        3. Updates self.guid from the response
        4. Returns the frame (harness will call append_frame separately)
        """
        # Build action payload
        action_payload = {
            "game_id": self.game_id,
        }

        # Add card_id for RESET action
        if action == GameAction.RESET:
            action_payload["card_id"] = self.card_id

        # Add coordinates if present
        if hasattr(action, 'x') and hasattr(action, 'y'):
            action_payload["x"] = action.x
            action_payload["y"] = action.y

        # Add guid if we have one (for continuing a game)
        if self.guid:
            action_payload["guid"] = self.guid

        # Add reasoning (observation text) if available
        if self._last_reasoning:
            action_payload["reasoning"] = self._last_reasoning

        # Call ARC API using the correct endpoint format: /api/cmd/{action_name}
        action_name = action.name if hasattr(action, 'name') else str(action)
        response = self.session.post(
            f"{self.root_url}/api/cmd/{action_name}",
            json=action_payload,
            timeout=30
        )
        response.raise_for_status()

        # Parse response into FrameData
        data = response.json()

        # Create ActionInput from the action
        action_data = {}
        if hasattr(action, 'x') and hasattr(action, 'y'):
            action_data = {"x": action.x, "y": action.y}

        action_input = ActionInput(
            id=action,
            data=action_data
        )

        frame = FrameData(
            score=data.get("score", 0),
            state=GameState[data.get("state", "NOT_PLAYED")],
            frame=data.get("frame", []),
            action_input=action_input,
            guid=data.get("guid")
        )

        # Update our tracked guid
        if frame.guid:
            self.guid = frame.guid

        return frame

    def append_frame(self, frame: FrameData) -> None:
        """
        Add a frame to the history and update the BaseAgent.

        This method:
        1. Stores the frame in self.frames (for legacy interface)
        2. Converts frame to observation dict
        3. Calls agent.update_from_env() to update the BaseAgent
        """
        self.frames.append(frame)

        # Convert FrameData to observation dict for BaseAgent
        observation = {
            "state": frame.state.name,
            "score": frame.score,
            "frame": frame.frame
        }

        # Calculate reward as score delta
        reward = 0.0
        if len(self.frames) >= 2:
            reward = float(frame.score - self.frames[-2].score)

        # Check if done
        done = frame.state in (GameState.WIN, GameState.GAME_OVER)

        # Update the BaseAgent
        self.agent.update_from_env(
            observation=observation,
            reward=reward,
            done=done,
            info={"frame_data": frame}
        )

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """
        Get the next action from the BaseAgent.

        This method:
        1. Calls agent.update_from_model() to get Action
        2. Converts Action to GameAction for the harness
        3. Returns GameAction
        """
        # Get action from BaseAgent
        action: Action = self.agent.update_from_model(response="")

        # Convert Action to GameAction
        action_dict = action.action
        action_name = action_dict.get("name", "ACTION1")

        # Map action name to GameAction enum
        try:
            game_action = GameAction[action_name]
        except KeyError:
            # Fallback to ACTION1 if invalid
            game_action = GameAction.ACTION1

        # Add coordinates if present
        action_data = action_dict.get("data", {})
        if "x" in action_data and "y" in action_data:
            game_action.x = action_data["x"]
            game_action.y = action_data["y"]

        # Store reasoning (observation text) for API call
        self._last_reasoning = action_dict.get("reasoning")

        return game_action

    def cleanup(self) -> None:
        """
        Cleanup method called by the harness after evaluation.

        Currently a no-op since BaseAgent doesn't require cleanup.
        """
        pass
