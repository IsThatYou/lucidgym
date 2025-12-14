"""LucidGym agent package."""

from .arcagi3_agent import ArcAgi3Agent
from .placeholder_agent import LucidGymPlaceholderAgent
from .textarena_agent import TextArenaAgent
from .variants.guided_text_16 import AS66GuidedAgent
from .variants.guided_text_64 import AS66GuidedAgent64
from .variants.hypothesis_agent import AS66MemoryAgent
from .variants.visual_hypothesis_agent import AS66VisualMemoryAgent

# Agent registry for evaluation harness
AVAILABLE_AGENTS = {
    "arcagi3_agent": ArcAgi3Agent,
    "placeholder_agent": LucidGymPlaceholderAgent,
    "textarena_agent": TextArenaAgent,
    "as66_guided_agent": AS66GuidedAgent,
    "as66guidedagent": AS66GuidedAgent,
    "as66_guided_agent_64": AS66GuidedAgent64,
    "as66guidedagent64": AS66GuidedAgent64,
    "as66_memory_agent": AS66MemoryAgent,
    "as66memoryagent": AS66MemoryAgent,
    "as66_visual_memory_agent": AS66VisualMemoryAgent,
    "as66visualmemoryagent": AS66VisualMemoryAgent,
}

__all__ = [
    "ArcAgi3Agent",
    "LucidGymPlaceholderAgent",
    "TextArenaAgent",
    "AS66GuidedAgent",
    "AS66GuidedAgent64",
    "AS66MemoryAgent",
    "AS66VisualMemoryAgent",
    "AVAILABLE_AGENTS",
]
