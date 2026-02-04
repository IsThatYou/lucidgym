"""LucidGym agent package."""

from .arcagi3_agent import ArcAgi3Agent
from .placeholder_agent import LucidGymPlaceholderAgent
from .textarena_agent import TextArenaAgent
from .variants.guided_text_16 import AS66GuidedAgent
from .variants.guided_text_64 import AS66GuidedAgent64
from .variants.hypothesis_agent import AS66MemoryAgent
from .variants.visual_hypothesis_agent import AS66VisualMemoryAgent
from .variants.meta_coding_harness_agent import MetaCodingHarnessAgent
from .new_variants.basic_obs_action_agent import BasicObsActionAgent
from .new_variants.basic_obs_action_agent_rolling_context import BasicObsActionAgentRollingContext
from .new_variants.basic_obs_action_agent_directed_graph_memory import BasicObsActionAgentDirectedGraphMemory
from .new_variants.basic_obs_action_agent_graph_memory import BasicObsActionAgentGraphMemory
from .new_variants.basic_obs_action_agent_cycle_detection import BasicObsActionAgentCycleDetection
from .new_variants.basic_obs_action_agent_hypothesis import BasicObsActionAgentHypothesis
from .new_variants.graph_explorer_agent import GraphExplorerAgent
from .new_variants.graph_explorer_obs_action_agent import GraphExplorerObsActionAgent

# Agent registry for evaluation harness
AVAILABLE_AGENTS = {
    "arcagi3_agent": ArcAgi3Agent,
    "placeholder_agent": LucidGymPlaceholderAgent,
    "textarena_agent": TextArenaAgent,
    "as66_guided_agent": AS66GuidedAgent,
    "as66_guided_agent_64": AS66GuidedAgent64,
    "as66_memory_agent": AS66MemoryAgent,
    "as66_visual_memory_agent": AS66VisualMemoryAgent,
    "meta_coding_harness_agent": MetaCodingHarnessAgent,
    "basic_obs_action_agent": BasicObsActionAgent,
    "basic_obs_action_agent_rolling_context": BasicObsActionAgentRollingContext,
    "basic_obs_action_agent_directed_graph_memory": BasicObsActionAgentDirectedGraphMemory,
    "basic_obs_action_agent_graph_memory": BasicObsActionAgentGraphMemory,
    "basic_obs_action_agent_cycle_detection": BasicObsActionAgentCycleDetection,
    "basic_obs_action_agent_hypothesis": BasicObsActionAgentHypothesis,
    "graph_explorer_agent": GraphExplorerAgent,
    "graph_explorer_obs_action_agent": GraphExplorerObsActionAgent,
}

__all__ = [
    "ArcAgi3Agent",
    "LucidGymPlaceholderAgent",
    "TextArenaAgent",
    "AS66GuidedAgent",
    "AS66GuidedAgent64",
    "AS66MemoryAgent",
    "AS66VisualMemoryAgent",
    "MetaCodingHarnessAgent",
    "BasicObsActionAgent",
    "BasicObsActionAgentRollingContext",
    "BasicObsActionAgentDirectedGraphMemory",
    "BasicObsActionAgentGraphMemory",
    "BasicObsActionAgentCycleDetection",
    "BasicObsActionAgentHypothesis",
    "GraphExplorerAgent",
    "GraphExplorerObsActionAgent",
    "AVAILABLE_AGENTS",
]
