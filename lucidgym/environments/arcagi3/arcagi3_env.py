"""
ARC-AGI-3 HTTP environment wrapper.

This adapter mirrors the remote evaluation loop defined in the ARC reference
agent (``agents/agent.py``) so LucidGym workflows can interact with ARC games
via the standard ``BaseEnv`` interface.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from rllm.environments.base.base_env import BaseEnv

from .client import ArcAgi3Client, ArcAgi3ClientError, ArcAgi3TransportError, TransportFn
from .mocks import StaticArcTransport
from .structs import (
    FrameData,
    GameAction,
    GameState,
    Scorecard,
    normalize_available_actions,
)

from lucidgym.utils.grid_processing import flatten_frame, downsample_4x4, frame_to_grid_text


@dataclass
class ArcAgi3EnvConfig:
    game_id: str
    root_url: str
    card_id: str | None = None
    api_key: str | None = None
    cookies: Mapping[str, str] | None = None
    max_actions: int = 80
    reward_mode: str = "delta_score"  # delta_score | score | binary
    reward_scale: float = 1.0
    include_grid_ascii: bool = True
    include_grid_flat: bool = False
    include_raw_frame: bool = False
    timeout: float = 30.0
    tags: tuple[str, ...] = ()


class ArcAgi3Env(BaseEnv):
    """
    Wraps the ARC REST service with Gym-compatible reset/step mechanics.
    """

    def __init__(
        self,
        game_id: str,
        root_url: str,
        *,
        api_key: str | None = None,
        cookies: Mapping[str, str] | None = None,
        max_actions: int = 80,
        reward_mode: str = "binary",
        reward_scale: float = 1.0,
        include_grid_ascii: bool = True,
        include_grid_flat: bool = False,
        include_raw_frame: bool = False,
        timeout: float = 30.0,
        transport: TransportFn | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        normalized_root = (root_url or "").rstrip("/")
        if not normalized_root:
            raise ValueError("root_url is required for ArcAgi3Env.")
        self.config = ArcAgi3EnvConfig(
            game_id=game_id,
            root_url=normalized_root,
            card_id=None,
            api_key=api_key,
            cookies=dict(cookies or {}),
            max_actions=max_actions,
            reward_mode=reward_mode,
            reward_scale=reward_scale,
            include_grid_ascii=include_grid_ascii,
            include_grid_flat=include_grid_flat,
            include_raw_frame=include_raw_frame,
            timeout=timeout,
            tags=tuple(tags or ()),
        )
        if self.config.reward_mode not in {"delta_score", "score", "binary"}:
            raise ValueError("reward_mode must be one of {'delta_score', 'score', 'binary'}.")
        self._transport = transport
        self._client: ArcAgi3Client | None = None
        self._last_frame: FrameData | None = None
        self._episode_frames: list[FrameData] = []
        self._actions_taken: int = 0
        self._episode_guid: str | None = None
        self._episode_card_id = None
        self._episode_game_id = game_id
        self._episode_root_url = self.config.root_url

    # ------------------------------------------------------------------ #
    def reset(self, task: dict | None = None) -> tuple[dict, dict]:
        # This will create a new client and open a new scorecard. You should rarely need to use this beside the 
        # very first initialization. This method is not meant to be called repeatedly during a single episode.
        task = task or {}
        game_id = task.get("game_id", self.config.game_id)
        if "card_id" in task:
            override_card = task.get("card_id")
            self.config.card_id = str(override_card) if override_card is not None else None
        root_url = str(task.get("root_url", self.config.root_url)).rstrip("/")
        runtime_cookies = task.get("cookies")
        cookies = runtime_cookies or self.config.cookies
        if "max_actions" in task:
            self.config.max_actions = int(task["max_actions"])
        if "reward_mode" in task:
            reward_mode = str(task["reward_mode"])
            if reward_mode not in {"delta_score", "score", "binary"}:
                raise ValueError("Invalid reward_mode override.")
            self.config.reward_mode = reward_mode
        if "tags" in task:
            override_tags = task.get("tags") or ()
            if isinstance(override_tags, Sequence) and not isinstance(override_tags, (str, bytes)):
                self.config.tags = tuple(str(tag) for tag in override_tags)
            else:
                raise ValueError("tags override must be a sequence of strings.")

        if hasattr(self, "_scorecard_cache"):
            delattr(self, "_scorecard_cache")

        if self._client is not None:
            self._client.close()

        self._client = ArcAgi3Client(
            root_url=root_url,
            api_key=self.config.api_key,
            default_game_id=game_id,
            default_card_id=None,
            timeout=self.config.timeout,
            cookies=cookies,
            transport=self._transport,
        )

        frame = self._client.reset(tags=self.config.tags)
        self._episode_game_id = game_id
        self._episode_root_url = root_url
        if self._episode_guid is None:
            self._episode_guid = frame.guid
        self._actions_taken = 0
        self._episode_frames = []
        self.config.card_id = self._client.default_card_id
        self._episode_card_id = self._client.default_card_id
        self._handle_new_frame(frame, is_reset=True)

        observation = self._format_observation(frame, step_idx=0)
        info = self._build_info(frame, done=False)
        return observation, info

    def step(self, action_payload: Any) -> tuple[dict, float, bool, dict]:
        # will be an rllm agent Action, not GameAction
        if self._client is None or self._last_frame is None:
            raise RuntimeError("ArcAgi3Env.step called before reset.")

        if self._actions_taken >= self.config.max_actions:
            raise RuntimeError("ArcAgi3Env has reached max_actions; call reset() for a new episode.")

        action, payload, reasoning = self._coerce_action(action_payload)
        if action == None:
            return {}, 0.0, False, {}

        try:
            frame = self._client.step(
                action,
                game_id=self._episode_game_id,
                payload=payload,
                reasoning=reasoning,
                guid=self._episode_guid,
            )
        except (ArcAgi3ClientError, ArcAgi3TransportError) as exc:
            info = {"arc": {"error": str(exc)}}
            return {}, 0.0, True, info

        self._actions_taken += 1
        self._handle_new_frame(frame)

        reward = self._compute_reward(frame)
        done = frame.is_terminal() or self._actions_taken >= self.config.max_actions
        observation = self._format_observation(frame, step_idx=self._actions_taken)
        info = self._build_info(frame, done=done)
        return observation, reward, done, info

    def close(self) -> None:
        if self._client is not None:
            try:
                # Fetch the scorecard so users can inspect aggregate stats.
                scorecard = self._client.scorecard(
                    game_id=self._episode_game_id,
                )
                self._scorecard_cache = scorecard.summary_for(self._episode_game_id)
            except Exception:
                self._scorecard_cache = {}
            finally:
                self._client.close()
                self.close_scorecard(self._episode_card_id)
                self._client = None
        self._last_frame = None
        self._episode_frames = []
        self._actions_taken = 0

    def open_scorecard(self, tags: Sequence[str] | None = None) -> str:
        return self._client.open_scorecard(tags=tags)

    def close_scorecard(self, card_id: str) -> Scorecard | None:
        return self._client.close_scorecard(card_id)


    @staticmethod
    def from_dict(env_args: dict) -> "ArcAgi3Env":
        transport = None
        mock_session_path = env_args.get("mock_session_path")
        if mock_session_path:
            session_path = Path(mock_session_path)
            session_data = json.loads(session_path.read_text())
            transport = StaticArcTransport(session_data)
        return ArcAgi3Env(
            game_id=env_args["game_id"],
            root_url=env_args["root_url"],
            api_key=env_args.get("api_key"),
            cookies=env_args.get("cookies"),
            max_actions=env_args.get("max_actions", 80),
            reward_mode=env_args.get("reward_mode", "delta_score"),
            reward_scale=env_args.get("reward_scale", 1.0),
            include_grid_ascii=env_args.get("include_grid_ascii", True),
            include_grid_flat=env_args.get("include_grid_flat", False),
            include_raw_frame=env_args.get("include_raw_frame", False),
            timeout=env_args.get("timeout", 30.0),
            transport=transport,
            tags=env_args.get("tags"),
        )

    # ------------------------------------------------------------------ #
    def _handle_new_frame(self, frame: FrameData, *, is_reset: bool = False) -> None:
        if frame.guid:
            self._episode_guid = frame.guid
        if is_reset:
            self._episode_frames = [frame]
        else:
            self._episode_frames.append(frame)
        self._last_frame = frame

    def _format_observation(self, frame: FrameData, *, step_idx: int) -> dict[str, Any]:
        observation: dict[str, Any] = {
            "card_id": self._episode_card_id,
            "game_id": frame.game_id or self._episode_game_id,
            "score": frame.score,
            "state": frame.state.value,
            "guid": frame.guid or self._episode_guid,
            "step": step_idx,
            "full_reset": frame.full_reset,
            "available_actions": normalize_available_actions(frame.available_actions),
            "dimensions": {
                "width": frame.width,
                "height": frame.height,
            },
        }
        if self.config.include_grid_ascii:
            observation["grid_ascii"] = frame_to_grid_text(frame.frame)
            # print(f"grid_ascii\n{observation['grid_ascii']}")
        if self.config.include_grid_flat:
            observation["grid_flat"] = flatten_frame(frame.frame)
        if self.config.include_raw_frame:
            observation["frame"] = frame.frame
        return observation

    def _build_info(self, frame: FrameData, *, done: bool) -> dict[str, Any]:
        info: dict[str, Any] = {
            "arc": {
                "card_id": self._episode_card_id,
                "game_id": self._episode_game_id,
                "guid": frame.guid or self._episode_guid,
                "state": frame.state.value,
                "score": frame.score,
                "done": done,
                "max_actions": self.config.max_actions,
                "actions_taken": self._actions_taken,
                "available_actions": normalize_available_actions(frame.available_actions),
            }
        }
        if hasattr(self, "_scorecard_cache"):
            info["arc"]["scorecard"] = getattr(self, "_scorecard_cache")
        if self._episode_guid and self._episode_root_url:
            info["arc"]["replay_url"] = f"{self._episode_root_url}/replay/{self._episode_game_id}/{self._episode_guid}"
        return info

    def _coerce_action(self, action_payload: Any) -> tuple[GameAction, MutableMapping[str, Any], Any | None]:
        if isinstance(action_payload, GameAction):
            return action_payload, {}, None
        if isinstance(action_payload, Mapping):
            action = action_payload.get("action")
            reasoning = action_payload.get("reasoning")
            action_payload = {k: v for k, v in action_payload.items() if k not in {"action", "reasoning"}}
            return action, action_payload, reasoning

        print(f"\n[DEBUG] _coerce_action failed!")
        print(f"[DEBUG] Payload Type: {type(action_payload)}")
        print(f"[DEBUG] Payload Value: {repr(action_payload)}")
        raise TypeError("Unsupported action payload type.")

    def _compute_reward(self, frame: FrameData) -> float:
        mode = self.config.reward_mode
        if mode == "score":
            base = frame.score
        elif mode == "binary":
            if frame.state == GameState.WIN:
                base = 1.0
            elif frame.is_terminal():
                base = 0.0
            else:
                base = 0.0
        format_reward = 0.15 # calling this funciton meaning the format is correct.

        
        return format_reward + float(base) * float(self.config.reward_scale)
