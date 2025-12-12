"""
Train a TextArena countdown agent with LucidGym's env/agent adapters.

This mirrors the workflow from ``rllm/examples/countdown`` but routes the
episodes through TextArena while defaulting to the LucidGym-owned
`LucidCountdown-v0-raw` environment so that PPO sees real environment
transitions instead of single-turn reward shaping.
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from lucidgym import register_lucidgym_components
from lucidgym.agents.textarena_agent import TextArenaAgent
from lucidgym.environments.textarena_env import TextArenaEnv
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.cumulative_workflow import CumulativeWorkflow

DEFAULT_AGENT_ARGS = {
    "system_prompt": (
        "You are the single player in TextArena's Countdown numbers game.\n"
        "Legal moves must follow the format `[i j op]` where `i` and `j` are "
        "indexes into the current number list and `op` is one of + - * /.\n"
        "Always reason briefly before emitting exactly one action string."
    ),
    "name": "textarena_countdown_agent",
}

DEFAULT_ENV_ARGS = {
    "env_id": "LucidCountdown-v0-raw",
    "num_players": 1,
    "reward_aggregation": "mean",
    "seed": 42,
}


def _resolve_args(cfg_section: DictConfig | None, defaults: dict) -> dict:
    """
    Convert a nested OmegaConf section into a plain dict and merge over defaults.

    Args:
        cfg_section: The config subtree supplied via Hydra overrides.
        defaults: Baseline values applied when caller does not override anything.
    """
    merged = dict(defaults)
    if cfg_section is None:
        return merged
    overrides = OmegaConf.to_container(cfg_section, resolve=True) or {}
    if not isinstance(overrides, dict):
        raise TypeError("Expected mapping for args overrides.")
    merged.update(overrides)
    return merged


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config: DictConfig) -> None:
    register_lucidgym_components()

    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    val_dataset = DatasetRegistry.load_dataset("countdown", "test")
    if train_dataset is None or val_dataset is None:
        raise RuntimeError("Countdown dataset not found. Run `rllm/examples/countdown/prepare_countdown_data.py` first.")

    agent_args = _resolve_args(getattr(config.rllm.agent, "agent_args", None), DEFAULT_AGENT_ARGS)
    env_args = _resolve_args(getattr(config.rllm.env, "env_args", None), DEFAULT_ENV_ARGS)
    print(agent_args)
    print(env_args)

    trainer = AgentTrainer(
        workflow_class=CumulativeWorkflow,
        workflow_args={
            "agent_cls": TextArenaAgent,
            "agent_args": agent_args,
            "env_cls": TextArenaEnv,
            "env_args": env_args,
            "max_steps": 10,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
