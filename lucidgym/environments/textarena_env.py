"""
TextArena environment wrapper for rllm.

The adapter instantiates TextArena games via ``textarena.make`` and exposes a
Gym-like ``reset``/``step`` API so existing workflows can treat TextArena games
as standard environments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from rllm.environments.base.base_env import BaseEnv

try:
    import textarena as ta
    from textarena.core import ObservationType
except Exception as exc:  # pragma: no cover - deferred import error handling
    ta = None
    ObservationType = None
    _TEXTARENA_IMPORT_ERROR = exc
else:
    _TEXTARENA_IMPORT_ERROR = None


def _require_textarena() -> None:
    if _TEXTARENA_IMPORT_ERROR is not None:
        raise ImportError("TextArenaEnv requires the `textarena` package. Install it via `pip install textarena`.") from _TEXTARENA_IMPORT_ERROR


def _aggregate_rewards(rewards: Mapping[int, Any] | None, mode: str) -> float:
    if not rewards:
        return 0.0

    values = [float(v) for v in rewards.values()]
    if mode == "sum":
        return float(sum(values))
    if mode == "max":
        return float(max(values))
    if mode == "first":
        return values[0]
    # Default to mean aggregation.
    return float(sum(values) / len(values))


def _serialize_messages(messages: Sequence[tuple[int, str, ObservationType]], role_mapping: Mapping[int, str]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for author_id, text, obs_type in messages:
        serialized.append(
            {
                "from_id": author_id,
                "from_role": role_mapping.get(author_id, f"Player {author_id}"),
                "type": obs_type.name if hasattr(obs_type, "name") else str(obs_type),
                "text": text,
            }
        )
    return serialized


@dataclass
class TextArenaEnvConfig:
    env_id: str
    num_players: int = 1
    seed: int | None = None
    reward_aggregation: str = "mean"  # sum|max|first|mean
    make_kwargs: dict[str, Any] | None = None


class TextArenaEnv(BaseEnv):
    """Wraps TextArena games so they can be used inside rllm workflows."""

    def __init__(
        self,
        env_id: str,
        num_players: int = 1,
        seed: int | None = None,
        reward_aggregation: str = "mean",
        make_kwargs: dict[str, Any] | None = None,
    ) -> None:
        _require_textarena()
        self.config = TextArenaEnvConfig(
            env_id=env_id,
            num_players=num_players,
            seed=seed,
            reward_aggregation=reward_aggregation,
            make_kwargs=make_kwargs or {},
        )
        self._env = None
        self._latest_rewards: Mapping[int, Any] | None = None

    def reset(self, task: dict | None = None) -> tuple[dict, dict]:
        """
        Reset the underlying TextArena environment.

        ``task`` can override ``env_id``, ``num_players``, or ``seed`` at runtime.
        """
        _require_textarena()
        task = task or {}
        env_id = task.get("env_id", self.config.env_id)
        num_players = task.get("num_players", self.config.num_players)
        seed = task.get("seed", self.config.seed)
        make_overrides = task.get("make_kwargs", {})

        # Recreate env to avoid stale state between episodes.
        self._env = ta.make(env_id=env_id, **self.config.make_kwargs, **make_overrides)
        self._env.reset(num_players=num_players, seed=seed)

        player_id, observation = self._env.get_observation()
        print(observation)
        obs_payload = self._format_observation(player_id, observation, env_id)
        info = {"arena": {"env_id": env_id, "num_players": num_players, "seed": seed}}
        return obs_payload, info

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        if self._env is None:
            raise RuntimeError("TextArenaEnv.step called before reset.")

        action_str = "" if action is None else str(action)
        done, step_info = self._env.step(action=action_str)
        info: dict[str, Any] = {"arena": {"step_info": step_info, "last_action": action_str}}
        reward = 0.0
        next_observation: dict[str, Any] = {}

        if done:
            rewards, game_info = self._env.close()
            self._latest_rewards = rewards
            info["arena"]["game_info"] = game_info
            reward = _aggregate_rewards(rewards, self.config.reward_aggregation)
        else:
            player_id, observation = self._env.get_observation()
            next_observation = self._format_observation(player_id, observation, self._env.env_id)

        return next_observation, reward, done, info

    def close(self) -> None:
        self._env = None
        self._latest_rewards = None

    @staticmethod
    def from_dict(env_args: dict) -> "TextArenaEnv":
        return TextArenaEnv(
            env_id=env_args["env_id"],
            num_players=env_args.get("num_players", 1),
            seed=env_args.get("seed"),
            reward_aggregation=env_args.get("reward_aggregation", "mean"),
            make_kwargs=env_args.get("make_kwargs"),
        )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _format_observation(self, player_id: int, observation: Sequence[tuple[int, str, ObservationType]], env_id: str) -> dict[str, Any]:
        role_mapping: Mapping[int, str] = getattr(getattr(self._env, "state", None), "role_mapping", {})
        serialized_messages = _serialize_messages(observation, role_mapping)
        return {
            "env_id": env_id,
            "current_player_id": player_id,
            "messages": serialized_messages,
            "raw_observation": list(observation),
        }
