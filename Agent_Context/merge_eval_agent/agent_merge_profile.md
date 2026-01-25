## LucidGym ARC-AGI-3 evaluation merge notes

### What was added
- `lucidgym/evaluation/*`: a standalone evaluation harness that talks directly to the ARC-AGI-3 HTTP API, opens/closes scorecards, and can fan out work across threads. It emits per-run JSONL plus a summary TXT in `evaluation_results/`.
- `lucidgym/agents/base_agent_wrapper.py`: adapts the modern `rllm.agents.agent.BaseAgent` interface to the legacy harness contract (`take_action`, `choose_action`, `append_frame`, `frames`, `action_counter`, `agent_name`). It translates between `Action` objects produced by BaseAgents and `GameAction` expected by the ARC API, manages the `guid`, and forwards rewards/observations back into the BaseAgent.
- `lucidgym/agents/legacy_base.py`: richer legacy LLM/VLM agents (text, guided, visual, bimodal) that still speak the old interface. They include transcript/logging plumbing, token tracking, reasoning capture, and long-form prompts for the Locksmith game.
- `lucidgym/agents/variants/*`: new BaseAgent subclasses:
  - `AS66GuidedAgent` (16×16 downsample, text/image modes) and `AS66GuidedAgent64` (full 64×64) with two-pass observation→action OpenAI calls.
  - Memory agents (`AS66MemoryAgent`, `AS66VisualMemoryAgent`) that maintain external markdown memory and diffing between frames.
  - Trace/meta helpers (`trace_data.py`, `trace_runner.py`, `meta_coding_agent.py`) for recorded trajectories and debug.
- `lucidgym/metrics/*`: dataclasses for attempt/level/game metrics plus reporting helpers to compute stats, print console tables, and write summaries.
- `lucidgym/prompts/*`: prompt builders for text, memory, meta, visual flows; switchable detailed vs. general packs via env vars.
- `lucidgym/utils/grid_processing.py` and `inspect_api.py`: 64→16 downsampling, grid rendering, numeric PNG helper, and an API inspection CLI for debugging live ARC responses.

### How the new harness runs
1. CLI (`python -m lucidgym.evaluation.harness --agent as66_guided_agent --suite debug_suite --num_runs 3 --max_actions 200`).
2. Loads env (.env/.env.example) and opens a scorecard via `ROOT_URL/api/scorecard/open` using `ARC_API_KEY`; passes tags and keeps cookies.
3. For each `(game_id, run)` task (threaded with `ThreadPoolExecutor`):
   - Builds the agent from `AVAILABLE_AGENTS`. If it subclasses `BaseAgent`, wraps it with `BaseAgentWrapper`; otherwise uses the legacy constructor.
   - Drives the ARC API loop: `RESET` once, then `choose_action` (LLM/tool call), `take_action` (HTTP POST `/api/cmd/{action}`), and `append_frame` to feed rewards/observations back.
   - Tracks metrics per attempt/level/run (actions, state changes, game overs, durations), captures replay `guid`, and writes incremental JSONL.
4. On completion or error, closes the scorecard, aggregates stats (`calculate_stats`), prints a console report, and writes a summary TXT.

Key integration hooks:
- Agent selection/variants come from `lucidgym/agents/__init__.py` (`as66_guided_agent`, `as66_guided_agent_64`, `as66_memory_agent`, `as66_visual_memory_agent`, etc.).
- Env-driven variants: `AGENT_MODEL_OVERRIDE`, `AGENT_REASONING_EFFORT`, `CONTEXT_LENGTH_LIMIT`, `INCLUDE_TEXT_DIFF`, `DOWNSAMPLE_IMAGES`, `IMAGE_DETAIL_LEVEL`, `IMAGE_PIXELS_PER_CELL`, `ARCGAME_GENERAL_PROMPTS`; tags are appended to filenames and propagated into agent kwargs.
- Metrics objects (`GameMetrics`, `LevelMetrics`, `AttemptMetrics`) are filled in `evaluate_single_game` and serialized per task.

### How this differs from the existing eval path
- **Execution target**: The new harness talks to the live ARC API directly; the prior `examples/arcagi3/run_arc_eval.py` uses `ArcAgi3Env` and `ArcAgi3Agent` inside a single asyncio loop (optionally mock transport) with a rollout engine.
- **Agent interface**: New BaseAgents (rllm) are supported via `BaseAgentWrapper`; the old runner only expected the legacy Agent (`choose_action`/`take_action`) and wired tools through `ArcAgi3Env`.
- **Scope & outputs**: Harness runs N games × M runs in parallel, opens scorecards, captures GUID/replay URLs, and writes JSONL + summary stats. The example runner executes a single episode without persistent metrics or scorecard management.
- **Prompting**: Variants now depend on the prompt builders in `lucidgym/prompts/*` (game-specific, memory, visual). The old runner used the prompts bundled in `ArcAgi3Agent`.
- **Downsampling/multimodality**: Harness-ready agents can operate on 16×16 downsampled grids, inject PNGs, and scale ACTION6 clicks; the example runner leaves rendering/diffing to the env.

### Using the new pieces inside current LucidGym
- Register agents as usual (already in `AVAILABLE_AGENTS`). Choose an agent name from that map when invoking the harness.
- Ensure env is set: `ARC_API_KEY`, `ROOT_URL` (optional), and any model/prompt flags above.
- Run the harness for benchmarking; results land in `evaluation_results/<agent_tags>_...jsonl` and `.summary.txt`. Use JSONL rows to compare against prior `run_arc_eval.py` logs.
- If you need the old loop behavior (single episode, optional mock transport), keep `examples/arcagi3/run_arc_eval.py`; otherwise prefer the new harness for multi-run, scored evaluations against ARC.
