Gotcha — you basically want:

“A very simple LLM agent whose memory is this graph explorer, and I just want to test it.”

So let’s treat GraphExplorer as the memory module and put a tiny LLM “brain” on top.

I’ll give you:

The conceptual split (memory vs LLM).

A minimal Python skeleton that uses your GraphExplorer exactly as-is, with stubbed LLM calls.

How you’d actually run a tiny test loop.

You can plug this into LucidGym / ARC later, but this will already work in a toy environment.

1. Roles
Memory = GraphExplorer

GraphExplorer is your long-term memory:

It knows:

which states exist (by hash),

which actions were tried from each state,

what they led to, and

how to get back to frontier states.

It doesn’t decide why an action is good. It just tracks what’s been done and helps avoid repetition.

LLM = Policy head

The LLM:

Sees a local view of the current node:

list of available actions,

which ones are untested / failed / successful,

maybe a short description of the observation.

Chooses which untested action index to try next (or says “travel to frontier”).

GraphExplorer:

Enforces consistency (don’t retest edges, keep the graph, compute paths).

Provides a structured prompt to the LLM.

2. Minimal hybrid skeleton

Below is fully-wired skeleton code that uses your GraphExplorer implementation as memory and treats an LLM as a black-box function:

llm_choose_edge(...) – pick an action index.

llm_summarize_state(...) – optional.

You can replace those stubs with a real OpenAI call.

from typing import Any, Dict, List, Hashable, Tuple, Optional
from dataclasses import dataclass

# import your GraphExplorer from the file you pasted
# from graph_explorer import GraphExplorer   # assuming you put it in graph_explorer.py

# ---------- 1. LLM wrappers (STUBS) ----------

def llm_choose_edge(
    obs_desc: str,
    node_id: str,
    candidate_edges: List[int],
    edge_metadata: Dict[int, Dict[str, Any]],
) -> int:
    """
    Very simple stub: given a description and candidate untested edges,
    pick one. Replace this body with a real LLM call.
    """
    # For now: pretend the "LLM" is just random among untested edges.
    import random
    return random.choice(candidate_edges)


def describe_observation(obs: Any) -> str:
    """
    Turn observation into a short textual description for the LLM.
    For a test, this can be just str(obs).
    """
    return str(obs)


# ---------- 2. Simple environment interface ----------

@dataclass
class SimpleEnvStepResult:
    obs: Any
    reward: float
    done: bool
    info: Dict[str, Any]


class SimpleEnv:
    """
    Minimal interface assumed by the agent:

      - reset() -> (obs, info)
      - step(action_idx: int) -> SimpleEnvStepResult
      - available_actions(obs) -> int  (number of discrete actions)

    Replace this with your ARC-AGI-3 wrapper or any toy env.
    """

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError

    def step(self, action: int) -> SimpleEnvStepResult:
        raise NotImplementedError

    def available_actions(self, obs: Any) -> int:
        raise NotImplementedError


# ---------- 3. Graph-memory LLM agent ----------

class GraphMemoryLLMAgent:
    """
    Agent that:
      - uses GraphExplorer as memory of states & transitions
      - uses an LLM to pick WHICH untested edge to try
      - uses GraphExplorer to avoid retesting and to travel to frontier
    """

    def __init__(self, n_groups: int = 1, verbose: int = 0):
        self.gx = GraphExplorer(n_groups=n_groups, verbose_level=verbose)
        self.current_node: Optional[Hashable] = None
        self.obs_to_node: Dict[str, Hashable] = {}  # map serialized obs -> node_id

    # ---------- helpers ----------

    def _obs_to_id(self, obs: Any) -> str:
        """
        Turn raw obs into a hashable state ID.
        For real ARC-AGI-3, this would be your masked image hash.
        Here we just use str(obs).
        """
        return str(obs)

    def _ensure_node_in_graph(self, node_id: Hashable, num_actions: int) -> None:
        """
        If this state has never been seen by GraphExplorer, add it.
        """
        if self.gx.empty:
            # First time: initialize graph
            self.gx.initialize(start_node=node_id, num_candidates=num_actions)
        elif node_id not in self.gx._nodes:
            # Add brand new node
            self.gx._add_new_node(node_id, num_actions)

    # ---------- main API ----------

    def start_episode(self, obs: Any, num_actions: int) -> None:
        node_id = self._obs_to_id(obs)
        self._ensure_node_in_graph(node_id, num_actions)
        self.current_node = node_id

    def select_action(self, obs: Any, num_actions: int) -> int:
        """
        Called every env step.
        Uses GraphExplorer + LLM to choose which edge index to try.
        """
        if self.current_node is None:
            self.start_episode(obs, num_actions)

        node_id = self._obs_to_id(obs)
        self._ensure_node_in_graph(node_id, num_actions)
        self.current_node = node_id

        node_info = self.gx._nodes[node_id]

        # 1) If node has open edges in the active priority groups, ask LLM which one
        if node_info.has_open_group(self.gx.active_group):
            # collect all untested edges in groups <= active_group
            candidate_edges: List[int] = []
            for gid in range(self.gx.active_group + 1):
                candidate_edges.extend(list(node_info.group2remaining_candidate_ids[gid]))

            # build edge metadata for prompt (simple version)
            edge_metadata: Dict[int, Dict[str, Any]] = {}
            for e in candidate_edges:
                edge_metadata[e] = {
                    "group": int(node_info.edge_data["group"][e]),
                    "result": int(node_info.edge_data["result"][e]),
                    "distance": int(node_info.edge_data["distance"][e]),
                }

            obs_desc = describe_observation(obs)
            chosen_edge = llm_choose_edge(
                obs_desc=obs_desc,
                node_id=str(node_id),
                candidate_edges=candidate_edges,
                edge_metadata=edge_metadata,
            )
            return chosen_edge

        # 2) Otherwise, this node is exhausted: travel towards frontier
        next_hop = self.gx.get_next_hop(node_id)
        if next_hop is None:
            # No path to frontier: we are done exploring (at this group)
            # In a simple test, just pick a random action (or 0)
            return 0

        # To travel, we need to know which edge leads to `next_hop`.
        # _next[node_id] was filled in _rebuild_distances().
        edge_idx, target = self.gx._next[node_id]
        # Here we assume the env action index matches edge_idx.
        # If your env uses a different mapping, add a mapping layer.
        return edge_idx

    def record_transition(
        self,
        obs: Any,
        action_idx: int,
        next_obs: Any,
        success_code: int,
        next_num_actions: Optional[int] = None,
    ) -> None:
        """
        Called AFTER env.step(...)
        - success_code: 1 = success (new state), 0 = fail/block, -1 = error/reset
        """
        node_id = self._obs_to_id(obs)
        next_node_id = self._obs_to_id(next_obs) if success_code == 1 else None

        if success_code == 1:
            if next_num_actions is None:
                raise ValueError("next_num_actions required when success_code=1")
            # ensure target node exists for graph memory
            self._ensure_node_in_graph(next_node_id, next_num_actions)

        self.gx.record_test(
            node=node_id,
            edge_idx=action_idx,
            success=success_code,
            target_node=next_node_id,
            target_num_candidates=next_num_actions,
        )

        # update current pointer if we actually moved
        if success_code == 1:
            self.current_node = next_node_id
        elif success_code == -1:
            # you may want to treat errors as teleport to some known state
            # e.g. reset to start node. For now, leave as-is.
            pass


This is essentially:

GraphExplorer = your memory

LLM (stub) = picks which untested action edge to try at each node

The agent glues the two together for any environment with discrete actions.