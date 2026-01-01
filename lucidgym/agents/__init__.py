"""LucidGym agent package."""

from .arcagi3_agent import ArcAgi3Agent
from .placeholder_agent import LucidGymPlaceholderAgent
from .textarena_agent import TextArenaAgent
from .variants.guided_text_16 import AS66GuidedAgent
from .variants.guided_text_64 import AS66GuidedAgent64
from .variants.hypothesis_agent import AS66MemoryAgent
from .variants.visual_hypothesis_agent import AS66VisualMemoryAgent
from .new_variants.basic_obs_action_agent import BasicObsActionAgent

# Agent registry for evaluation harness
AVAILABLE_AGENTS = {
    "arcagi3_agent": ArcAgi3Agent,
    "placeholder_agent": LucidGymPlaceholderAgent,
    "textarena_agent": TextArenaAgent,
    "as66_guided_agent": AS66GuidedAgent,
    "as66_guided_agent_64": AS66GuidedAgent64,
    "as66_memory_agent": AS66MemoryAgent,
    "as66_visual_memory_agent": AS66VisualMemoryAgent,
    "basic_obs_action_agent": BasicObsActionAgent,
}

__all__ = [
    "ArcAgi3Agent",
    "LucidGymPlaceholderAgent",
    "TextArenaAgent",
    "AS66GuidedAgent",
    "AS66GuidedAgent64",
    "AS66MemoryAgent",
    "AS66VisualMemoryAgent",
    "BasicObsActionAgent",
    "AVAILABLE_AGENTS",
]
