"""
Meta-Coding Harness Agent - Harness-compatible wrapper for code generation paradigm.

This agent:
- Externally conforms to the BaseAgent interface (reset, call_llm, update_from_model, update_from_env)
- Internally manages code generation and uses generated heuristic agents
- Preserves the meta-agent paradigm of generating and improving Python code
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import logging
import os
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from rllm.agents.agent import Step, Trajectory
from lucidgym.agents.arcagi3_agent import ArcAgi3Agent
from arcengine import GameAction
from lucidgym.utils.grid_processing import downsample_4x4, matrix16_to_lines, format_grid
from lucidgym.prompts.meta_prompts import (
    PROMPT_AS66_RULES,
    PROMPT_GENERAL_ARC_RULES,
    PROMPT_SYSTEM_INSTRUCTION_16,
    PROMPT_SYSTEM_INSTRUCTION_64,
    PROMPT_PROGRESSIVE_INSTRUCTION,
    PROMPT_CONDENSER_SYSTEM,
)

log = logging.getLogger(__name__)

# Configuration constants
EPISODES_PER_ITERATION = 5
CODER_MODEL = "gpt-5.1"
REASONING_EFFORT = "low"
STUCK_PATIENCE = 2
CONTEXT_TOKEN_LIMIT = 50000
SLIDING_WINDOW_SIZE = 3
ACTION_GROWTH_FACTOR = 1.5

# Directory for meta-agent logs
META_MEMORY_DIR = Path(__file__).resolve().parents[2] / "evaluation_results" / "meta_agent_logs"


def get_bootstrap_code(use_64x64: bool) -> str:
    """Returns the initial random agent code, adapted for the grid resolution."""
    if use_64x64:
        return r'''import random
from typing import Any, Dict, List, Optional
import logging

log = logging.getLogger(__name__)

class GeneratedHeuristicAgent:
    """Bootstrap Random Agent (64x64 Mode)"""
    def __init__(self):
        self.turn_count = 0
        self.scripted_moves = []
        log.info("Bootstrap Heuristic Agent (Random 64x64) initialized.")

    def choose_action(self, frame_data: dict) -> dict:
        self.turn_count += 1
        current_state = frame_data.get('state', 'NOT_PLAYED')

        if current_state in ("GAME_OVER", "NOT_PLAYED"):
            self.turn_count = 0
            return {'name': 'RESET', 'data': {}}

        if self.turn_count <= len(self.scripted_moves):
            return {'name': self.scripted_moves[self.turn_count - 1], 'data': {}}

        # Random fallback
        possible_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        return {'name': random.choice(possible_actions), 'data': {}}
'''
    else:
        return r'''import random
from typing import Any, Dict, List, Optional
import logging
from lucidgym.utils.grid_processing import downsample_4x4, matrix16_to_lines

log = logging.getLogger(__name__)

class GeneratedHeuristicAgent:
    """Bootstrap Random Agent (16x16 Mode)"""
    def __init__(self):
        self.turn_count = 0
        self.scripted_moves = []
        log.info("Bootstrap Heuristic Agent (Random 16x16) initialized.")

    def choose_action(self, frame_data: dict) -> dict:
        self.turn_count += 1
        current_state = frame_data.get('state', 'NOT_PLAYED')

        if current_state in ("GAME_OVER", "NOT_PLAYED"):
            self.turn_count = 0
            return {'name': 'RESET', 'data': {}}

        if self.turn_count <= len(self.scripted_moves):
            return {'name': self.scripted_moves[self.turn_count - 1], 'data': {}}

        # Downsample check (to ensure imports work)
        full_frame_3d = frame_data.get('frame', [])
        if full_frame_3d:
            try:
                _ = downsample_4x4(full_frame_3d, take_last_grid=True, round_to_int=True)
            except Exception: pass

        possible_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        return {'name': random.choice(possible_actions), 'data': {}}
'''


class MetaCodingHarnessAgent(ArcAgi3Agent):
    """
    Harness-compatible wrapper for the meta-coding paradigm.

    This agent generates and iteratively improves Python code for action selection.
    It conforms to the standard harness interface while internally managing
    a GeneratedHeuristicAgent instance.
    """

    def __init__(
        self,
        name: str = "meta_coding_harness_agent",
        system_prompt: str | None = None,
        game_id: str | None = None,
        use_64x64: bool = False,
        use_general_prompts: bool = False,
        no_progressive: bool = False,
        coder_model: str = CODER_MODEL,
        reasoning_effort: str = REASONING_EFFORT,
        representation: "RepresentationConfig | None" = None,
    ) -> None:
        """
        Initialize the meta-coding harness agent.

        Args:
            name: Agent name
            system_prompt: Override system prompt (optional)
            game_id: Game ID for prompt selection
            use_64x64: Whether to use 64x64 grid (vs 16x16 downsampled)
            use_general_prompts: Use general ARC prompts instead of AS66-specific
            no_progressive: Disable progressive hardcoding mode
            coder_model: Model to use for code generation
            reasoning_effort: Reasoning effort for code generation
            representation: Grid representation configuration
        """
        from lucidgym.utils.representation import RepresentationConfig
        self.representation = representation or RepresentationConfig(downsample=not use_64x64)
        # Initialize meta-coding state BEFORE calling parent __init__
        # because parent's __init__ calls reset() which needs these attributes
        self._generated_agent: Any = None
        self._generated_code: str = ""
        self._episode_action_log: List[Dict[str, Any]] = []
        self._episode_count: int = 0
        self._iteration_count: int = 0
        self._meta_memory: List[Dict[str, Any]] = []
        self._current_frame_data: Dict[str, Any] = {}
        self._last_action_dict: Dict[str, Any] | None = None
        self._pending_action: Dict[str, Any] | None = None

        # Progressive mode state
        self._best_level_solved: int = 0
        self._stuck_counter: int = 0
        self._stuck_multiplier_level: int = 0
        self._hardcoded_moves: List[str] = []

        # Configuration attributes - must be set before super().__init__()
        # because parent's __init__ calls reset() which may need these
        self.game_id = game_id
        self.use_64x64 = use_64x64
        self.use_general_prompts = use_general_prompts
        self.no_progressive = no_progressive
        self.coder_model = coder_model
        self.reasoning_effort = reasoning_effort

        # OpenAI client for code generation
        self._openai_client = OpenAI()

        # Set up logging directory - must be before super().__init__()
        # because reset() -> _bootstrap_agent() needs these
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        pid = os.getpid()
        self.run_id = f"{name}_{timestamp}_{pid}"
        self.log_dir = META_MEMORY_DIR / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.generated_agent_path = self.log_dir / "current_agent.py"

        # Select prompts based on config - must be before super().__init__()
        if self.game_id and self.game_id.startswith("as66") and not self.use_general_prompts:
            self.rules_prompt = PROMPT_AS66_RULES
        else:
            self.rules_prompt = PROMPT_GENERAL_ARC_RULES

        if self.use_64x64:
            self.system_instruction = PROMPT_SYSTEM_INSTRUCTION_64
        else:
            self.system_instruction = PROMPT_SYSTEM_INSTRUCTION_16

        super().__init__(system_prompt=system_prompt, name=name, representation=self.representation)

        log.info(f"MetaCodingHarnessAgent initialized: {self.run_id}")
        log.info(f"Log directory: {self.log_dir}")
        log.info(f"64x64 mode: {self.use_64x64}, General prompts: {self.use_general_prompts}")

        # Note: parent's __init__ already called reset(), no need to call again

    def reset(self) -> None:
        """Reset agent state for new episode and handle code regeneration."""
        super().reset()

        # Check if we need to regenerate code
        if self._should_regenerate_code():
            self._regenerate_code()

        # Ensure generated agent exists
        if self._generated_agent is None:
            self._bootstrap_agent()

        # Reset the generated agent
        if hasattr(self._generated_agent, 'turn_count'):
            self._generated_agent.turn_count = 0

        # Reset episode-specific state
        self._episode_action_log = []
        self._current_frame_data = {}
        self._last_action_dict = None
        self._pending_action = None

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **_: Any) -> None:
        """Process environment observation and log for meta-analysis."""
        self._last_observation = observation

        # Convert to frame_data format expected by GeneratedHeuristicAgent
        self._current_frame_data = self._build_frame_data(observation)

        # Log action result for meta-memory
        if self._last_action_dict is not None:
            self._log_action_result(observation, reward, done, info)

        # Track episode completion
        if done:
            self._on_episode_complete()

        # Standard trajectory tracking
        step = Step(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            chat_completions=self.chat_completions.copy()
        )
        self._trajectory.steps.append(step)

    async def call_llm(self, rollout_engine=None) -> dict:
        """
        Get next action from generated heuristic agent.

        Note: This doesn't call an LLM for action selection - it delegates
        to the generated Python code's choose_action() method.
        """
        obs = self._last_observation or {}
        state = obs.get("state", "NOT_PLAYED")

        # Handle RESET states
        if state in ("NOT_PLAYED", "GAME_OVER"):
            action_dict = {"name": "RESET", "data": {}, "obs_text": "Resetting game", "action_text": ""}
            self._pending_action = action_dict
            return action_dict

        # Delegate to generated agent
        try:
            action_dict = self._generated_agent.choose_action(self._current_frame_data)
            if "data" not in action_dict:
                action_dict["data"] = {}
            action_dict["obs_text"] = f"Generated agent turn {getattr(self._generated_agent, 'turn_count', 'N/A')}"
            action_dict["action_text"] = ""
        except Exception as e:
            log.error(f"Generated agent crashed: {e}")
            action_dict = {"name": "ACTION5", "data": {}, "obs_text": f"Error: {e}", "action_text": ""}

        self._pending_action = action_dict
        return action_dict

    def update_from_model(self, action_payload: dict | None = None, **_: Any) -> dict:
        """Convert generated agent's action to harness format."""
        action_dict = action_payload or self._pending_action

        if action_dict is None:
            action_dict = {"name": "ACTION5", "data": {}}

        obs_text = action_dict.get("obs_text", "")
        action_text = action_dict.get("action_text", "")
        response_text = f"Observation: {obs_text}\nAction: {action_dict['name']}"

        # Record in trajectory
        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = response_text
            self._trajectory.steps[-1].action = action_dict

        # Store for logging
        self._last_action_dict = action_dict
        self._pending_action = None

        # Convert to GameAction format
        action = GameAction.from_name(action_dict["name"])
        result = {"action": action, "reasoning": response_text}

        if action.requires_coordinates():
            result["x"] = action_dict.get("data", {}).get("x", 0)
            result["y"] = action_dict.get("data", {}).get("y", 0)

        return result

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return message history formatted for chat API."""
        system_msg = self.system_instruction + "\n\n" + self.rules_prompt
        messages: list[dict] = [{"role": "system", "content": system_msg}]
        messages.extend(self._chat_history)
        return messages

    # --- Internal Methods: Code Generation ---

    def _should_regenerate_code(self) -> bool:
        """Determine if code should be regenerated."""
        if self._generated_agent is None:
            return True

        # Regenerate after every N episodes
        if self._episode_count > 0 and self._episode_count % EPISODES_PER_ITERATION == 0:
            return True

        return False

    def _regenerate_code(self) -> None:
        """Generate new agent code based on meta-memory."""
        if not self._meta_memory:
            log.info("No meta-memory yet, skipping code regeneration")
            return

        log.info(f"Iteration {self._iteration_count}: Regenerating code...")

        try:
            # Build prompt from meta-memory
            messages = self._build_coder_prompt()

            # Call coder model
            new_code = self._call_coder_model(messages)

            # Validate and clean code
            new_code = self._clean_generated_code(new_code)

            if "GeneratedHeuristicAgent" not in new_code:
                log.error("Generated code missing required class, keeping old code")
                return

            # Save and load new agent
            self._save_generated_code(new_code, self._iteration_count)
            self._load_agent_from_code(new_code)

            self._iteration_count += 1
            log.info(f"Successfully loaded new agent code (iteration {self._iteration_count})")

        except Exception as e:
            log.error(f"Code regeneration failed: {e}")

    def _bootstrap_agent(self) -> None:
        """Initialize with bootstrap random agent."""
        log.info("Bootstrapping with initial random agent...")
        bootstrap_code = get_bootstrap_code(self.use_64x64)
        self._save_generated_code(bootstrap_code, 0)
        self._load_agent_from_code(bootstrap_code)
        self._generated_code = bootstrap_code

    def _load_agent_from_code(self, code: str) -> None:
        """Load GeneratedHeuristicAgent from code string."""
        # Create unique module name to avoid caching
        module_name = f"generated_agent_{self.run_id}_{self._iteration_count}_{len(self._meta_memory)}"

        # Remove old module from cache if exists
        if module_name in sys.modules:
            del sys.modules[module_name]

        try:
            spec = importlib.util.spec_from_file_location(module_name, self.generated_agent_path)
            if not spec or not spec.loader:
                raise ImportError(f"Could not create module spec from {self.generated_agent_path}")

            gen_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = gen_module
            spec.loader.exec_module(gen_module)

            AgentClass = getattr(gen_module, "GeneratedHeuristicAgent")
            self._generated_agent = AgentClass()
            self._generated_code = code
            log.info(f"Loaded GeneratedHeuristicAgent from {self.generated_agent_path}")

        except Exception as e:
            log.error(f"Failed to load agent: {e}")
            raise

    def _save_generated_code(self, code: str, iteration: int) -> None:
        """Save generated code to files."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Save current agent
        self.generated_agent_path.write_text(code, encoding="utf-8")

        # Save iteration copy
        iter_path = self.log_dir / f"generated_agent_iter_{iteration}.py"
        iter_path.write_text(code, encoding="utf-8")

        log.info(f"Saved generated code to {self.generated_agent_path}")

    def _build_coder_prompt(self) -> List[Dict[str, str]]:
        """Build prompt for code generation from meta-memory."""
        system_prompt = self.system_instruction + "\n\n" + self.rules_prompt
        messages = [{"role": "system", "content": system_prompt}]

        # Add progressive mode instruction if applicable
        if not self.no_progressive and self._hardcoded_moves:
            prog_msg = PROMPT_PROGRESSIVE_INSTRUCTION + f"\n```json\n{json.dumps(self._hardcoded_moves)}\n```"
            messages.append({"role": "system", "content": prog_msg})

        # Add sliding window of meta-memory
        window = self._meta_memory[-SLIDING_WINDOW_SIZE:] if self._meta_memory else []
        for i, entry in enumerate(window):
            try:
                code_path = entry.get("code_file_path", "")
                if code_path and Path(code_path).exists():
                    code = Path(code_path).read_text(encoding="utf-8")
                else:
                    code = entry.get("code", "# Code not found")
            except Exception:
                code = "# Code not found"

            log_section = ""
            if i == len(window) - 1:
                log_section = f"\nExecution Log:\n{entry.get('action_log', '')}"

            content = (
                f"--- Iteration {entry.get('iteration', i)} ---\n"
                f"Code:\n```python\n{code}\n```\n"
                f"Result: {json.dumps(entry.get('summary', {}))}{log_section}"
            )
            messages.append({"role": "user", "content": content})

        messages.append({"role": "user", "content": "Analyze the log. Write the next iteration of the agent code."})
        return messages

    def _call_coder_model(self, messages: List[Dict[str, str]]) -> str:
        """Call the coder model to generate new agent code."""
        try:
            # Try using responses API for reasoning models
            instructions = messages[0]['content']
            history = "\n".join([m['content'] for m in messages[1:]])

            response = self._openai_client.responses.create(
                model=self.coder_model,
                instructions=instructions,
                input=history,
                reasoning={"effort": self.reasoning_effort},
            )

            if hasattr(response, 'output_text'):
                return response.output_text
            if response.output:
                for item in response.output:
                    if item.type == 'message':
                        return "".join([p.text for p in item.content if p.type == 'output_text'])
            return ""

        except Exception as e:
            log.warning(f"Responses API failed, trying chat completions: {e}")

            # Fallback to chat completions
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.coder_model,
                    messages=messages,
                )
                return response.choices[0].message.content or ""
            except Exception as e2:
                log.error(f"Chat completions also failed: {e2}")
                raise

    def _clean_generated_code(self, code: str) -> str:
        """Extract and clean code from LLM response."""
        # Extract code from markdown blocks
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        return textwrap.dedent(code).strip()

    # --- Internal Methods: Frame Data and Logging ---

    def _build_frame_data(self, observation: dict) -> dict:
        """Convert harness observation to frame_data format expected by generated agent."""
        return {
            "frame": observation.get("frame", []),
            "state": observation.get("state", "NOT_PLAYED"),
            "score": observation.get("score", 0),
        }

    def _hash_frame(self, frame_data: Dict[str, Any]) -> str:
        """Create hash of frame for state tracking."""
        frame_json = json.dumps(frame_data.get('frame', []))
        return hashlib.md5(frame_json.encode()).hexdigest()

    def _log_action_result(self, observation: dict, reward: float, done: bool, info: dict) -> None:
        """Log action for meta-analysis."""
        # Get grid view
        try:
            if self.use_64x64:
                frame_3d = observation.get('frame', [])
                grid = frame_3d[-1] if frame_3d else []
                grid_view = "\n".join([str(row) for row in grid[:8]]) + "\n..." if grid else "(no grid)"
            else:
                frame_3d = observation.get('frame', [])
                grid_16x16 = downsample_4x4(frame_3d, take_last_grid=True, round_to_int=True) if frame_3d else []
                grid_view = format_grid(grid_16x16, self.representation) if grid_16x16 else "(no grid)"
        except Exception:
            grid_view = "(grid error)"

        # Create state hash
        state_hash = self._hash_frame(self._current_frame_data)
        action_hash = f"{self._last_action_dict.get('name', 'NONE')}:{json.dumps(self._last_action_dict.get('data', {}), sort_keys=True)}"
        state_action_key = f"{state_hash}|{action_hash}"

        entry = {
            "turn": len(self._episode_action_log) + 1,
            "action": self._last_action_dict,
            "score": observation.get("score", 0),
            "state": observation.get("state", "UNKNOWN"),
            "grid_view": grid_view,
            "state_action_key": state_action_key,
            "level_at_turn": observation.get("score", 0) + 1,
            "is_game_over": observation.get("state") == "GAME_OVER",
        }
        self._episode_action_log.append(entry)

    def _compress_action_log(self) -> str:
        """Compress action log for meta-memory."""
        if not self._episode_action_log:
            return ""

        # Group by state-action key
        seen_states: Dict[str, List[int]] = {}
        for entry in self._episode_action_log:
            key = entry["state_action_key"]
            seen_states.setdefault(key, []).append(entry["turn"])

        compressed_lines = []
        processed_turns = set()

        for entry in self._episode_action_log:
            turn = entry["turn"]
            if turn in processed_turns:
                continue

            state_action_key = entry["state_action_key"]
            all_turns = seen_states.get(state_action_key, [turn])

            turn_str = f"{min(all_turns)}-{max(all_turns)}" if len(all_turns) > 1 else str(turn)

            line = (
                f"Turns [{turn_str}]:\n"
                f"  State (Grid):\n{textwrap.indent(entry['grid_view'], '    ')}\n"
                f"  Action: {json.dumps(entry['action'])}\n"
                f"  Result: State={entry['state']}, Score={entry['score']}"
            )
            compressed_lines.append(line)
            processed_turns.update(all_turns)

        return "\n".join(compressed_lines)

    def _on_episode_complete(self) -> None:
        """Handle episode completion - update meta-memory."""
        self._episode_count += 1

        # Calculate episode summary
        max_score = max((e["score"] for e in self._episode_action_log), default=0)
        summary = {
            "episode": self._episode_count,
            "max_score": max_score,
            "total_actions": len(self._episode_action_log),
            "status": self._episode_action_log[-1]["state"] if self._episode_action_log else "UNKNOWN",
        }

        # Add to meta-memory
        self._meta_memory.append({
            "iteration": self._iteration_count,
            "episode": self._episode_count,
            "code_file_path": str(self.generated_agent_path),
            "code": self._generated_code,
            "summary": summary,
            "action_log": self._compress_action_log(),
            "status": "SUCCESS",
        })

        log.info(f"Episode {self._episode_count} complete: max_score={max_score}, actions={len(self._episode_action_log)}")

        # Check for progressive mode updates
        if not self.no_progressive:
            self._check_progressive_update(max_score)

        # Persist meta-memory
        self._persist_meta_memory()

    def _check_progressive_update(self, episode_best_score: int) -> None:
        """Check if we solved a new level and should hardcode moves."""
        if episode_best_score > self._best_level_solved:
            log.info(f"New best score: {episode_best_score} (previous: {self._best_level_solved})")

            # Extract winning moves
            moves = self._run_condenser_loop(
                self._episode_action_log,
                self._best_level_solved,
                episode_best_score
            )

            if moves:
                self._hardcoded_moves.extend(moves)
                self._best_level_solved = episode_best_score
                self._stuck_counter = 0
                self._stuck_multiplier_level = 0
                log.info(f"Updated hardcoded moves: {len(self._hardcoded_moves)} total")
            else:
                log.warning("Condenser failed to extract moves")
        else:
            self._stuck_counter += 1
            log.info(f"Stuck counter: {self._stuck_counter}/{STUCK_PATIENCE}")

            if self._stuck_counter >= STUCK_PATIENCE:
                self._stuck_multiplier_level += 1
                self._stuck_counter = 0
                log.info(f"Increased stuck multiplier to {self._stuck_multiplier_level}")

    def _run_condenser_loop(self, action_log: List[Dict[str, Any]], start_score: int, target_score: int) -> List[str]:
        """Extract minimal action sequence for the solved level segment."""
        log.info(f"Condenser: Extracting moves for Score {start_score} -> {target_score}")

        # Find segment that achieved the score increase
        segment_log = []
        for entry in action_log:
            if entry['score'] >= start_score:
                segment_log.append(entry)
            if entry['score'] >= target_score:
                break

        if not segment_log:
            return []

        # Extract action names
        moves = [e['action']['name'] for e in segment_log if e['action'].get('name') not in ('RESET', None)]

        # Try to use condenser model for refinement
        try:
            log_content = "\n".join([
                f"Turn {e['turn']}: {e['action']['name']} (Score: {e['score']})"
                for e in segment_log
            ])

            messages = [
                {"role": "system", "content": PROMPT_CONDENSER_SYSTEM},
                {"role": "user", "content": f"Segment: Score {start_score} -> {target_score}.\n\n{log_content}"}
            ]

            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )

            content = response.choices[0].message.content or "[]"
            if "[" in content and "]" in content:
                json_str = content[content.find("["):content.rfind("]")+1]
                refined_moves = json.loads(json_str)
                if refined_moves:
                    log.info(f"Condenser refined {len(moves)} moves to {len(refined_moves)}")
                    return refined_moves
        except Exception as e:
            log.warning(f"Condenser refinement failed: {e}")

        return moves

    def _persist_meta_memory(self) -> None:
        """Persist meta-memory to disk."""
        try:
            meta_memory_path = self.log_dir / "meta_memory.jsonl"
            with open(meta_memory_path, "a", encoding="utf-8") as f:
                if self._meta_memory:
                    f.write(json.dumps(self._meta_memory[-1]) + "\n")
        except Exception as e:
            log.warning(f"Failed to persist meta-memory: {e}")
