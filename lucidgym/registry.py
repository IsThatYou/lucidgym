"""
Utilities for registering LucidGym components with rllm's runtime registries.

The registry bootstrap performs three tasks, in order:

1. Import TextArena's registry so upstream env IDs are available.
2. Register LucidGym-owned TextArena env variants (e.g., ``LucidCountdown-v0``)
   so calling ``ta.make`` with those IDs resolves to LucidGym subclasses.
3. Merge LucidGym's agent/env/workflow classes into rllm's trainer mappings.

Hydra entry points should call :func:`register_lucidgym_components` before any
workflow spins up environments to ensure the TextArena overrides are active.
"""
from __future__ import annotations

from typing import Dict, Tuple, Type

from rllm.trainer import env_agent_mappings as base_registry

from .agents.arcagi3_agent import ArcAgi3Agent
from .agents.placeholder_agent import LucidGymPlaceholderAgent
from .agents.textarena_agent import TextArenaAgent
from .environments.arcagi3.arcagi3_env import ArcAgi3Env
from .environments.placeholder_env import LucidGymPlaceholderEnv
from .environments.textarena_env import TextArenaEnv
from .workflows.placeholder_workflow import LucidGymPlaceholderWorkflow
from .environments.textarena.registry import register_lucidgym_textarena_envs

LUCIDGYM_AGENT_CLASS_MAPPING: dict[str, type] = {
    "lucidgym_placeholder_agent": LucidGymPlaceholderAgent,
    "textarena_agent": TextArenaAgent,
    "arcagi3_agent": ArcAgi3Agent,
}

LUCIDGYM_ENV_CLASS_MAPPING: dict[str, type] = {
    "lucidgym_placeholder_env": LucidGymPlaceholderEnv,
    "textarena_env": TextArenaEnv,
    "arcagi3_env": ArcAgi3Env,
}

LUCIDGYM_WORKFLOW_CLASS_MAPPING: dict[str, type] = {
    "lucidgym_placeholder_workflow": LucidGymPlaceholderWorkflow,
}

_TEXTARENA_OVERRIDES_INSTALLED = False


def _ensure_textarena_overrides() -> None:
    """
    Install LucidGym's custom TextArena env registrations once per process.

    We only surface errors from this phase when TextArena itself is available.
    Import-related failures are ignored so the registry can still be used for
    non-TextArena scenarios.
    """

    global _TEXTARENA_OVERRIDES_INSTALLED
    if _TEXTARENA_OVERRIDES_INSTALLED:
        return

    try:
        register_lucidgym_textarena_envs()
    except ImportError:
        # TextArena is optional; defer the error to the first user of TextArenaEnv.
        return
    else:
        _TEXTARENA_OVERRIDES_INSTALLED = True


def _merge_dict(
    target: Dict[str, Type],
    additions: Dict[str, Type],
    *,
    allow_override: bool,
    mutate: bool,
) -> Dict[str, Type]:
    """Merge ``additions`` into ``target`` while guarding against key collisions."""
    destination: Dict[str, Type]
    destination = target if mutate else dict(target)
    for key, value in additions.items():
        if not allow_override and key in destination and destination[key] is not value:
            raise KeyError(f"Component key '{key}' already registered. Pass allow_override=True to replace it.")
        destination[key] = value
    return destination


def register_lucidgym_components(
    *,
    allow_override: bool = False,
    mutate_base_registry: bool = True,
) -> dict[str, Dict[str, Type]]:
    """
    Register LucidGym's classes with the rllm trainer registry.

    Args:
        allow_override: When False (default), raises if LucidGym tries to reuse an
            existing key from the upstream mappings. When True, LucidGym entries
            clobber prior registrations.
        mutate_base_registry: When True (default) the upstream mappings are updated
            in-place so subsequent imports see the new entries. When False a merged
            copy is returned without mutating global state.

    Returns:
        A dict carrying the merged agent/env/workflow mappings. This makes it easy
        for callers to inspect the effective registry even when ``mutate_base_registry``
        is False.
    """
    _ensure_textarena_overrides()

    agent_mapping = _merge_dict(
        base_registry.AGENT_CLASS_MAPPING,
        LUCIDGYM_AGENT_CLASS_MAPPING,
        allow_override=allow_override,
        mutate=mutate_base_registry,
    )

    env_mapping = _merge_dict(
        base_registry.ENV_CLASS_MAPPING,
        LUCIDGYM_ENV_CLASS_MAPPING,
        allow_override=allow_override,
        mutate=mutate_base_registry,
    )

    workflow_mapping = _merge_dict(
        base_registry.WORKFLOW_CLASS_MAPPING,
        LUCIDGYM_WORKFLOW_CLASS_MAPPING,
        allow_override=allow_override,
        mutate=mutate_base_registry,
    )

    return {
        "agents": agent_mapping,
        "environments": env_mapping,
        "workflows": workflow_mapping,
    }
