# Current Implementation & Extension Plan

## 1. How rllm currently organizes agents, environments, and workflows

### Agents (`rllm/rllm/agents`)
- `agent.py` defines the core dataclasses (`Step`, `Trajectory`, `Episode`) plus `BaseAgent`, which every concrete agent extends. Each agent must implement `reset`, `update_from_env`, and `update_from_model`, and expose a `trajectory` that workflows can read/commit. Steps capture messages (`chat_completions`), intermediate thoughts/actions, model outputs, rewards, and metadata so the trainer can compute returns and correctness.
- Concrete agents (e.g., `tool_agent.py`, `swe_agent.py`, `code_agent.py`) mainly differ by how they format prompts (`chat_completions`), how they parse model responses into domain-specific actions, and how they log info into the `Trajectory`.
- Agents typically retain environment observations inside their own state, then build the next prompt before calling the rollout engine. They rely on workflows to handle looping/termination.

### Environments (`rllm/rllm/environments`)
- `BaseEnv` (`base/base_env.py`) mirrors the Gym API (`reset -> (obs, info)`, `step -> (obs, reward, done, info)`), plus a serializable `from_dict` factory so Hydra/Ray configs can spawn envs remotely. `MultiTurnEnvironment` and `SingleTurnEnvironment` add turn accounting and reward hooks for instruction-following tasks. `ToolEnvironment` wraps the tools subsystem, executing tool calls until a finish call or max steps.
- Domain envs (browsergym, SWE, competition coding, AppWorld, etc.) live in sibling packages. `environments/__init__.py` lazy-imports them into the public namespace and updates `__all__`. Each env can declare whether it is multithreading-safe by overriding `is_multithread_safe`.
- `env_agent_mappings.py` connects string keys (e.g., `"browsergym"`) to env classes so workflows/trainers can instantiate envs from config.

### Workflows (`rllm/rllm/workflows`)
- `Workflow` orchestrates the loop: it stores the injected `RolloutEngine`, executor, timeout, and discount settings, manages trajectory commits, applies reward shaping/Monte Carlo returns, flags correctness, and packages metrics. It also enforces termination handling (`TerminationReason`/`TerminationEvent`) and ensures agents/envs are reset with a shared `task` payload and `uid`.
- `SingleTurnWorkflow`, `MultiTurnWorkflow`, and `CumulativeWorkflow` mix in `TimingTrackingMixin` to record LLM/env durations per step. Each workflow builds prompts from the agent, calls the rollout engine, feeds responses back to the agent, then advances the env until done/max turns/max prompt length.
- `TimingTrackingMixin` wraps env/model calls to emit per-step timing metadata that gets attached to the episode/trajectory on postprocess.

### Trainer hook points
- `AgentTrainer` (`rllm/rllm/trainer/agent_trainer.py`) is the entry point for running PPO fine-tuning via the `verl` backend. Callers can pass either a workflow class (preferred) or explicit agent/env classes plus kwargs. It uses `env_agent_mappings.py` registries to resolve names, then spins up Ray workers that import the same modules, so any new agent/env/workflow just needs to follow the same registration pattern to be discoverable.

## 2. Reuse strategy for this project
1. **Mirror rllm’s module layout under `lucidgym/`**: add `lucidgym/agents`, `lucidgym/environments`, `lucidgym/workflows` packages that subclass rllm bases but keep project-specific defaults/prompts/action parsers. Keep constructors signature-compatible so they can drop into existing workflows/AgentTrainer.
2. **Extend registries instead of forking core code**: provide a `lucidgym.registry` helper that imports custom classes and updates `AGENT_CLASS_MAPPING`, `ENV_CLASS_MAPPING`, and `WORKFLOW_CLASS_MAPPING` (or exposes merged dicts) without modifying upstream files. Downstream configs can set `agent_cls="lucidgym.gym_agent"` (or similar) and the trainer will pick up the custom implementations.
3. **Leverage existing workflows**: reuse `MultiTurnWorkflow`/`CumulativeWorkflow` for Gym/TextArena/ARC tasks by ensuring new envs obey the `BaseEnv` contract and new agents expose the right prompt style. Only add new workflow types if a target env requires non-standard scheduling (e.g., parallel rollouts or multi-agent negotiation).
4. **Keep training script stock**: continue instantiating `AgentTrainer` but feed it custom classes/args via CLI/config. This preserves all Ray/VERL glue, dataset handling, and reward shaping pipelines from upstream.

## 3. Phases for implementing new environments

### Phase 1 – Registry & packaging
- Create `lucidgym/__init__.py` that imports custom agents, envs, workflows solely for side effects (registration). Provide a `register_lucidgym_components()` that merges new mappings into `rllm.trainer.env_agent_mappings` or returns a combined dict the CLI can pass to the trainer.
- Add base config templates (Hydra or JSON/YAML) that demonstrate selecting custom classes while still using stock workflows/trainers.

### Phase 2 – Gymnasium compatibility layer
- Implement `lucidgym/environments/gymnasium_env.py` that subclasses `BaseEnv`. Responsibilities:
  - Accept a `task`/`env_id` plus optional wrappers/seeds, call `gymnasium.make`, and forward `reset`/`step`.
  - Convert Gym’s `(obs, reward, terminated, truncated, info)` into rllm’s `(obs, reward, done=terminated or truncated, info)`.
  - Handle serialization via `from_dict`, including `env_id`, `gym_kwargs`, and optional observation/action adapters.
- Pair it with a lightweight `GymAgent` (if needed) that formats observations into prompts (e.g., describe numeric/vector obs as JSON) and parses text actions back into env-compatible actions (could reuse `ToolAgent`-style finish calls). Start by reusing `SingleTurnWorkflow` or `MultiTurnWorkflow` depending on per-episode turn count.

### Phase 3 – TextArena wrapper
- Implement `lucidgym/environments/textarena_env.py` that wraps the official `textarena.make(...)` API (`textarena/envs/registration.py`) and adheres to rllm’s `BaseEnv`.
  - Accept `env_id`, `num_players`, `seed`, wrapper/version flags (`-raw`, `-train`, etc.), and optional `textarena.wrappers.*` lists so configs can mirror the registry entries in `textarena/envs/README.md`.
  - On `reset`, call `env.reset(num_players=...)` and immediately fetch the first `(player_id, observation)` via `env.get_observation()`. Observations arrive as `List[Message]` tuples (actor id, text, `ObservationType` from `textarena.core`) and must be serialized into an rllm-friendly dict (e.g., role-tagged transcripts plus any board state returned by wrappers such as `LLMObservationWrapper`).
  - On `step`, forward the agent’s string action to `env.step(action=...)`, capture the returned `(done: bool, step_info: dict)`, and if not done, continue polling `env.get_observation()` for the next player. When the episode ends, call `env.close()` to retrieve `rewards` and `game_info`, and convert those into the `(reward, info)` pair rllm workflows expect (e.g., aggregate per-player rewards or keep player-specific info in `info["arena"]`).
  - Surface TextArena metadata—current player, ObservationType tags, `step_info`, `game_info`, replay URLs if provided—in the `info` dict for reward shaping/debugging. Make sure the env advertises whether it is multithread-safe, since many TextArena envs rely on Python-level RNG.
- Provide utilities for offline vs. online play (the repo exposes `textarena.api.make_online`) but keep the initial wrapper focused on offline `ta.make` paths; online support can follow once credential handling is designed.
- Build a `TextArenaAgent` that understands the serialized observation schema: it should rehydrate the message list into `chat_completions`, optionally condition on `ObservationType` (PROMPT vs. GAME_MESSAGE), and output plain-text actions that TextArena validators accept (wrappers like `ActionFormattingWrapper` already enforce the bracketed syntax). Multi-player games can be handled either by cloning an agent per role or by giving the agent awareness of `player_id` so it can reason about opposing turns.

### Phase 4 – ARC-AGI-3 environment prep
- Model ARC-AGI-3 runs as a `BaseEnv` subclass that proxies HTTP requests to `ROOT_URL` similar to `/agents/agent.py` in the ARC repo. Key duties:
  - Hold session state (`card_id`, `game_id`, `guid`, cookies) and expose `reset`/`step` that call `GameAction.RESET` or the selected action via the `/api/cmd/{action}` endpoints, returning the new `FrameData` fields as the observation.
  - Translate `FrameData` (pixel grid, score, `available_actions`) into something an rllm agent can reason over (e.g., textual summaries or serialized matrices).
  - Pump metrics (level status, replay URL, GUID) into `info` so workflows can set `TerminationReason.ENV_DONE` with richer context.
- Craft an `ARCAgent` that wraps the existing ARC agent abstractions (choose_action/is_done) but adapts them to produce `chat_completions` so we can fine-tune via LLM feedback. Initially, we can embed ARC heuristics inside `update_from_env`/`update_from_model` before moving to full LLM prompting.

## 4. ARC-AGI-3 repository analysis (agents & evaluation)
- Base agent (`agents/agent.py`) manages HTTP sessions against `ROOT_URL`, maintains a rolling list of `FrameData`, and repeatedly calls `choose_action` until `is_done`. Actions are serialized via `GameAction` enums defined in `agents/structs.py`, which also define `FrameData` (3D frame tensor, score, game state, action metadata) and scorecard helpers. Any wrapper env must expose these objects or flattened equivalents to upstream agents.
- Evaluation (`evaluation/evaluate.py`) spins up concurrent runs per game ID: it instantiates an agent, repeatedly (a) chooses an action, (b) posts it to `/api/cmd/{action}`, (c) logs metrics per level, and (d) resets on `GameState.GAME_OVER`. Metrics classes in `evaluation/metrics.py` capture per-level action counts, durations, and replay URLs. This flow shows the minimal API surface we need to reproduce inside a `BaseEnv` wrapper: `RESET`, `ACTION{1-7}`, `FrameData`, and scoreboard lookups via `/api/scorecard/{card_id}/{game_id}`.
- Available agents (registered in `agents/__init__.py`) auto-discover subclasses of `Agent`, so our eventual rllm-compatible `ARCAgent` can reuse their logic or act as a façade around them.

## 5. Immediate next steps
1. Stand up the `lucidgym` package skeleton plus registry glue (Phase 1).
2. Implement and document the Gymnasium env/agent pair, validate via an rllm `MultiTurnWorkflow`.
3. Do the same for TextArena, capturing any async/network constraints early.
4. Design the ARC proxy environment, referencing the HTTP contract from `agents/agent.py` and the evaluation loop, even if we defer full implementation until after Gym/TextArena are stable.

## 6. ARC-AGI-3 integration deliverables (Nov 2024)
- **Vendored types and client**: `lucidgym/integrations/arcagi3/{types,client}.py` replicate the upstream `GameAction`, `FrameData`, and scorecard models plus an HTTP client that mirrors `agents/agent.py` (session reuse, headers, `/api/cmd/{ACTION}` loop, scorecard fetches). The client accepts a pluggable `transport` so CI/tests can replay local fixtures instead of calling the live ARC service. Dependencies (`requests`, `pydantic>=2`) were added to `pyproject.toml`.
- **Environment wrapper**: `ArcAgi3Env` (`lucidgym/environments/arcagi3_env.py`) subclasses `BaseEnv`, exposes runtime overrides via `reset(task=...)`, renders frames into ASCII/text observations, carries score/guid/replay metadata inside `info["arc"]`, enforces `max_actions`, and optionally fetches scorecards on `close()`. `from_dict` now accepts `mock_session_path` which auto-loads the shipped fixtures through `StaticArcTransport`.
- **Agent façade**: `ArcAgi3Agent` formats observations into prompts, parses JSON or inline actions back into `GameAction` payloads, and offers a `passthrough` mode via `passthrough_fn` for wrapping the reference ARC agents or scripted policies. A sentinel `__PASSTHROUGH__` response also triggers the fallback.
- **Registry/configs/scripts**: `lucidgym.registry` registers the new env/agent keys (`arcagi3_env`, `arcagi3_agent`). Example Hydra configs live under `lucidgym/configs/examples/arcagi3_{local,batch}.yaml`, and `examples/arcagi3/run_arc_eval.py` demonstrates running the env/agent loop with the offline mock session.
- **Mocks & fixtures**: `lucidgym/integrations/arcagi3/mocks.py` plus `examples/arcagi3/mock_session.json` provide deterministic frames/scorecards for both manual runs and tests.
- **Tests**: `tests/lucidgym/test_arcagi3.py` validates the client, env (reward/done/replay metadata), `from_dict` mock loader, and the agent’s parsing utilities. Run via `python -m pytest tests/lucidgym/test_arcagi3.py` after installing the `dev` extras.
