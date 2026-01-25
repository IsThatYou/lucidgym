# ARC-AGI-3 Integration Notes

## Overview
- Environment entry point: `lucidgym.environments.arcagi3.arcagi3_env.ArcAgi3Env`.
- HTTP client/types now live beside the env under `lucidgym.environments.arcagi3.*`; registry helpers hook them up through `register_lucidgym_components()`.
- Agent reference: `lucidgym.agents.arcagi3_agent.ArcAgi3Agent` (LLM or passthrough driving `ArcAgi3Env`).

## Environment configuration & lifecycle
```yaml
rllm:
  agent_class: "arcagi3_agent"
  env_class: "arcagi3_env"
  env_args:
    card_id: "${oc.env:ARC_CARD_ID}"
    game_id: "${oc.env:ARC_GAME_ID}"
    root_url: "${oc.env:ARC_ROOT_URL}"
    api_key: "${oc.env:ARC_API_KEY}"        # optional header injection
    max_actions: 80                         # mutable via reset(task={"max_actions": ...})
    reward_mode: "delta_score"              # delta_score | score | binary
    reward_scale: 1.0
    include_grid_ascii: true
    include_grid_flat: false
    include_raw_frame: false
    mock_session_path: "examples/arcagi3/mock_session.json"  # optional offline transport
```
- `ArcAgi3EnvConfig` stores the resolved values; runtime overrides via `reset(task=...)` support `card_id`, `game_id`, `root_url`, `max_actions`, `reward_mode`, `cookies`, and `tags`. Invalid reward modes raise early.
- Reset spins up an `ArcAgi3Client`, captures the first `FrameData`, and zeroes counters/episode metadata (`guid`, `action_count`, cached frames).
- Step loop:
  - `_coerce_action` accepts a `GameAction`, action name, or dict w/ `"action"` and optional `"reasoning"` + extra payload keys.
  - Each action increments `_actions_taken`, appends the returned `FrameData`, and computes reward based on mode (`delta_score` diff vs prior frame, raw score, or binary win signal). Reward is scaled by `reward_scale`.
  - Termination triggers on ARC terminal states (`WIN`/`GAME_OVER`) or `max_actions` exhaustion.
- Observations include card/game ids, score/state, step index, frame dimensions, available actions (normalized to `name/id/requires_coordinates`), ASCII grid, optional flattened RGB list, and optionally the raw 3-D frame array.
- `info["arc"]` mirrors the identifiers, current score/state, action counters, available actions, and `replay_url = {root_url}/replay/{game_id}/{guid}` once a guid exists. When the env has previously fetched a scorecard, its summary is cached on `info["arc"]["scorecard"]`.
- `open_scorecard(tags=None)`/`close_scorecard(card_id)` proxy the `/api/scorecard/*` endpoints using either the active client or a temporary one (via `_resolve_client_for_scorecard`).
- `close()` grabs the remote scorecard (best effort), caches the summary, and tears down the `requests.Session`. `ArcAgi3Env.from_dict()` supports Hydra/CLI wiring and optional `mock_session_path` to instantiate `StaticArcTransport`.

## HTTP client & transport
- `ArcAgi3Client` wraps the ARC REST API:
  - Auto-loads `root_url`/`api_key` from env vars if not provided, maintains a persistent `requests.Session`, and can bootstrap cookie jars from plain dicts.
  - Commands (`reset`, `step`) call `/api/cmd/{ACTION}` with payloads validated by `normalize_action_payload`.
  - Scorecard helpers cover `GET /api/scorecard/{card}/{game}`, `POST /api/scorecard/open`, and `POST /api/scorecard/close`.
  - All responses route through `_request_json` → `_parse_response`, raising `ArcAgi3TransportError` on request failures/status errors and `ArcAgi3ClientError` on invalid payloads or backend-declared errors.
- `TransportFn` allows dependency injection (used for deterministic mocks and replay). Either a transport or a custom `requests.Session` may be supplied, not both.

## Data model & helpers (`structs.py`)
- Pydantic models:
  - `GameState`, `GameAction` (RESET + ACTION1–ACTION7, with simple/complex variants enforcing `x/y` ranges).
  - `ActionInput` enforces `MAX_REASONING_BYTES` (16 KB JSON-encoded) and normalizes action ids/names.
  - `FrameData` (RGB frame, score, guid, available actions, computed width/height, terminal helpers).
  - Score tracking via `Card` and `Scorecard` (computed totals, `summary_for`, `get_json_for`).
- Utilities: `normalize_action_payload`, `normalize_available_actions`, `frame_to_grid_text` (ASCII shading), and `flatten_frame` (1-D pixel buffer).

## Mocking, offline runs, and tests
- `StaticArcTransport` replays a recorded session (`reset`, queued `steps`, scorecard) for deterministic tests/demos. Supply via `env_args.mock_session_path`.
- Recorder JSONL sessions live in `examples/arcagi3/mock_session.json` for turnkey demos (`examples/arcagi3/run_arc_eval.py` defaults to the mock and can be pointed at live ARC via `--no-mock`).
- Tests: `tests/lucidgym/test_arcagi3.py` cover the env, client, agent parsing, and mock wiring. Run `python -m pytest tests/lucidgym/test_arcagi3.py` after installing dev extras (`pip install -e .[dev]`).

## Recorder utility (`recorder.py`)
- `Recorder` writes timestamped JSONL events under `RECORDINGS_DIR` (auto-created). Filenames default to `{prefix}.{guid}.recording.jsonl`, but existing filenames preserve their GUID via `get_guid`.
- Helpers: `record(data)`, `get()` to load events, `list()` to enumerate recordings, plus prefix/GUID parsing helpers for custom tooling.

## AgentOps tracing (`tracing.py`)
- Provides optional AgentOps instrumentation:
  - `initialize(api_key, log_level)` sets up the SDK unless the module is missing or the key is blank/placeholder.
  - `trace_agent_session` decorator wraps an `Agent`'s main loop, creating a trace tagged with `agent_instance.tags`. On completion it marks status `Success` or `Indeterminate` (if action cap hit); exceptions propagate but set the trace status to the error message.
  - Falls back to `NoOpAgentOps` when the dependency is absent, so imports are safe in environments without AgentOps.

## Agent modes (context)
1. **LLM (`mode="llm"`)**: converts observations into prompts, expects structured replies containing `"action"` plus optional coordinates/reasoning (free-form strings like `ACTION6 x=12 y=3` also parse).
2. **Passthrough (`mode="passthrough"`)**: provides `passthrough_fn(observation, trajectory)` so scripted ARC solvers or legacy agents can feed actions directly. LLM agents may emit `"__PASSTHROUGH__"` to delegate mid-episode.

## Live verification checklist
1. Export `ARC_API_KEY`, `ARC_ROOT_URL`, `ARC_CARD_ID`, `ARC_GAME_ID` (and cookies if required).
2. Update Hydra config (e.g., `lucidgym/configs/examples/arcagi3_batch.yaml`) with the correct env/agent args or supply CLI overrides.
3. Launch via your workflow or `examples/arcagi3/run_arc_eval.py --no-mock`.
4. Inspect `info["arc"]["replay_url"]`, cached scorecards, and recorder outputs (if enabled) to confirm remote bookkeeping.
