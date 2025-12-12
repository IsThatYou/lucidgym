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
from lucidgym.agents.arcagi3_agent import ArcAgi3Agent
from lucidgym.environments.arcagi3.arcagi3_env import ArcAgi3Env
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.cumulative_workflow import CumulativeWorkflow

DEFAULT_AGENT_ARGS = {
    "name": "arcagi3_agent",
}

DEFAULT_ENV_ARGS = {
    "game_id": "as66-821a4dcad9c2",
    "root_url": "https://three.arcprize.org",
    "include_grid_ascii": True,
    "include_raw_frame": True,
    "max_actions": 10,
    "tags": ["train-debug", "arcagi_agent"],
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

    train_dataset = DatasetRegistry.load_dataset("arcagi3-as66", "train")
    val_dataset = DatasetRegistry.load_dataset("arcagi3-as66", "test")

    agent_args = _resolve_args(getattr(config.rllm.agent, "agent_args", None), DEFAULT_AGENT_ARGS)
    env_args = _resolve_args(getattr(config.rllm.env, "env_args", None), DEFAULT_ENV_ARGS)


    trainer = AgentTrainer(
        workflow_class=CumulativeWorkflow,
        workflow_args={
            "agent_cls": ArcAgi3Agent,
            "agent_args": agent_args,
            "env_cls": ArcAgi3Env,
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
