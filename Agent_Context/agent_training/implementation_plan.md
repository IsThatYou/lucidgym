# Plan: ARC-AGI-3 Environment Integration in LucidGym

## Goals
- Add an HTTP-backed `ArcAgi3Env` that mirrors ARC-AGI-3’s remote REST loop from `/home/jwang/Arc/Trying-ARC-AGI-3-Agents/agents/agent.py`, including session cookies, API-key headers, and the `/api/cmd/{GameAction}` contract.
- Provide an accompanying `ArcAgi3Agent` façade so LucidGym policies can either wrap the reference ARC agents or emit LLM-generated `GameAction` commands compatible with the evaluation harness.
- Register the new env/agent with `lucidgym/registry.py`, ship example configs, and supply automated tests + mocks so we can iterate without live network calls.
- Keep the integration additive: no invasive edits to upstream ARC repo, and LucidGym continues to function for TextArena/Gym workloads.

## Current Foundations
- `lucidgym/environments/textarena_env.py` already demonstrates how to adapt an external simulator into the `BaseEnv` API; we can mirror its structure (config dataclass, `_format_observation`, aggregation helpers) for ARC.
- `lucidgym/registry.py` centralizes component registration, so we only need to extend `LUCIDGYM_ENV_CLASS_MAPPING`/`LUCIDGYM_AGENT_CLASS_MAPPING`.
- The ARC repo (`/home/jwang/Arc/Trying-ARC-AGI-3-Agents`) exposes `GameAction`, `FrameData`, and the evaluation loop we must reproduce; we can vendor the minimal pydantic models or import them via a light-weight client module.
- The reference ARC `Agent` (see `agents/agent.py`) already implements the remote loop: it keeps a persistent `requests.Session`, injects `X-API-Key`, posts to `/api/cmd/{action.name}` with serialized `ActionInput`, parses `FrameData` responses, enforces `MAX_ACTIONS`, and fetches scorecards/replays. Our env should encapsulate the same lifecycle so LucidGym workflows just call `reset`/`step`.
- Network access is restricted inside the harness, meaning unit tests must stub requests; live end-to-end checks can run manually once credentials/ARC_API_KEY are configured.

## Implementation Phases

### Phase 1 – Types, client, and dependency scaffolding
1. Create `lucidgym/integrations/arcagi3/types.py` with trimmed copies of `GameAction`, `FrameData`, `GameState`, and helpers (`SimpleAction`, `ComplexAction`, `ActionInput`). Keep them in sync with upstream (link to commit) and add conversion utilities (e.g., `frame_to_grid_text(frame.frame)`).
2. Implement `lucidgym/integrations/arcagi3/client.py`:
   - Wrap a persistent `requests.Session`, load `ROOT_URL`, `ARC_API_KEY`, and optional cookies from env/config.
   - Provide high-level methods (`reset(card_id, game_id)`, `step(action: GameAction)`, `scorecard(...)`) mirroring `Agent.do_action_request`/`take_action`, including `guid` threading and optional `reasoning`.
   - Support local mock adapters (pass in a transport or fake responses) to unblock tests.
3. Update `pyproject.toml` if extra deps (e.g., `pydantic`) are required in LucidGym directly; otherwise reuse rllm’s pinned versions.

### Phase 2 – `ArcAgi3Env` (lucidgym/environments/arcagi3_env.py)
1. Define a config dataclass capturing `card_id`, `game_id`, `root_url`, `max_actions`, `cookies`, aggregation strategy, and observation rendering options (raw pixels vs. ASCII vs. summary stats).
2. `reset(task: dict | None)` should:
   - Allow runtime overrides for `card_id`, `game_id`, and `root_url`.
   - Call `client.reset(...)` (HTTP `POST /api/cmd/RESET`), validate `FrameData`, seed env state (`self._last_frame`, `self._episode_frames`), and emit an observation dict with:
     - Serialized grid (e.g., `grid_ascii`, `grid_flat`),
     - Score/state metadata,
     - `available_actions` list with action IDs + whether complex coordinates are needed (mirrors `FrameData.available_actions`).
   - Populate `info["arc"]` with identifiers (card_id, game_id, guid placeholder).
3. `step(action_payload: dict | str)` should accept either a ready-made `GameAction` name or a structured dict (`{"action": "ACTION6", "x": 12, "y": 3, "reasoning": {...}}`), normalize it, call `client.step`, and return `(observation, reward, done, info)` where:
   - Reward shaping defaults to delta score (current - previous) but is configurable (raw score, binary win/loss).
   - `done` flips when `FrameData.state in {WIN, GAME_OVER}` or when `max_actions` is exceeded.
   - `info["arc"]` carries `guid`, `fps`, `available_actions`, and replay URLs (computed via `f"{root}/replay/{game_id}/{guid}"` when available).
4. Implement `close()` to optionally fetch the final scorecard via `client.scorecard()` (equivalent to `Agent.get_scorecard`), aggregate metrics, and release the session.
5. Ensure `from_dict` matches rllm expectations for remote instantiation.

### Phase 3 – Agent façade & action serialization
1. Add `lucidgym/agents/arcagi3_agent.py` that subclasses `BaseAgent` (or `TextArenaAgent` style) and:
   - Maintains the latest `FrameData` observation, tracks remaining actions, and surfaces helper methods (`_format_prompt`, `_parse_action`).
   - Supports two modes:
     - **LLM mode**: convert observation dicts into chat prompts and parse the model’s reply into `GameAction` + parameters.
     - **Heuristic passthrough**: optionally wrap an existing ARC agent (imported from `/home/jwang/Arc/...`) for bootstrapping; the plan should specify using dependency injection so the same agent can be swapped later.
2. Provide validation errors that map cleanly back to workflows (e.g., raise `ValueError` when the agent outputs invalid `x/y` coordinates).
3. Log reasoning snippets and attach them to `GameAction.reasoning` so the backend stores them.

### Phase 4 – Registry, configs, and CLI entry points
1. Update `lucidgym/registry.py` to register `ArcAgi3Env` and `ArcAgi3Agent` keys (e.g., `"arcagi3_env"`, `"arcagi3_agent"`). Guard imports so environments without HTTP deps still load.
2. Add Hydra/JSON config templates under `lucidgym/configs/examples/arcagi3_*.yaml` covering:
   - Minimal local run (single card/game, mocked HTTP).
   - Batch evaluation (list of `game_id`s, concurrency hints).
3. Provide a reference training/eval script (e.g., `examples/arcagi3/run_arc_eval.py`) that calls `register_lucidgym_components()`, instantiates the env/agent, and runs a workflow episode for smoke testing.

### Phase 5 – Testing, mocks, and documentation
1. Write unit tests in `tests/lucidgym` that:
   - Mock the HTTP client (using `responses` or a stub class) to emulate reset/step/scorecard flows.
   - Assert observation formatting, reward computation, done conditions, and error handling (bad action payloads, HTTP failures, `MAX_ACTIONS` guard).
2. Document usage in `Agent_Context/CUR_IMPLEMENTATION_PLAN.md` and a new `Agent_Context/arcagi3_env.md`: environment variables, expected credentials, replay URL interpretation, and how to plug in existing ARC agents.
3. Capture manual verification steps (running against the real ARC server) in `Agent_Context/Training_Log.md`.
4. Ensure CI (or local `pytest`) skips ARC tests when network variables are missing by using `pytest.mark.skipif`.

## Risk & Mitigation Notes
- **Network restrictions**: Provide a mock transport layer so development and CI do not rely on live ARC endpoints.
- **Schema drift**: By vendoring type definitions and linking to the upstream commit, diffs are easy to re-sync; add a small script to compare enums/fields against the source repo.
- **Action parsing**: Introduce a JSON schema or pydantic model for agent outputs to avoid runtime surprises.
- **Long episodes**: Enforce `MAX_ACTIONS` both in the agent and env to prevent workflows from hanging if the ARC service returns repetitive frames.

## Key Reference Files
- `/home/jwang/Arc/Trying-ARC-AGI-3-Agents/agents/agent.py` — canonical HTTP gameplay loop: session setup, `GameAction` serialization, replay logging, and scorecard fetch logic.
- `/home/jwang/Arc/Trying-ARC-AGI-3-Agents/agents/structs.py` — pydantic definitions for `GameAction`, `ActionInput`, `FrameData`, `GameState`, and helper models we must mirror.
- `/home/jwang/Arc/Trying-ARC-AGI-3-Agents/evaluation/evaluate.py` — multi-game evaluation flow showing how frames, resets, and metrics interact; informs workflow integration.
- `/home/jwang/Arc/Trying-ARC-AGI-3-Agents/evaluation/metrics.py` — `GameMetrics`/`LevelMetrics` schema for tracking actions, durations, and replay URLs.
- `/home/jwang/Agents/lucidgym/lucidgym/environments/textarena_env.py` — existing external-env adapter to emulate when designing `ArcAgi3Env`.
- `/home/jwang/Agents/lucidgym/lucidgym/registry.py` — centralized registration helper that must expose the new ARC env/agent to rllm workflows.
