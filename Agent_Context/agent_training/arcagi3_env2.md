# ARC-AGI-3 Decomposition Notes

## File analysis

**`lucidgym/environments/arcagi3/arcagi3_env.py`**
- Encapsulates all HTTP/game-loop mechanics behind the `BaseEnv` interface; initialization builds an `ArcAgi3EnvConfig` and normalizes reward/observation options (`arcagi3_env.py:30-99`).
- `reset()` reconstructs the `ArcAgi3Client`, applies runtime overrides (card/game IDs, reward mode, tags, cookies), caches the opening `FrameData`, and returns an observation/info tuple formatted for LucidGym (`arcagi3_env.py:101-180`).
- `step()` coerces arbitrary agent payloads into `GameAction` objects, executes `/api/cmd/{ACTION}`, performs reward computation (delta, raw score, or binary), and accumulates per-episode metadata including `guid`, scorecard cache, and replay URL (`arcagi3_env.py:151-325`).
- Helper APIs cover scorecard admin (`open_scorecard`, `close_scorecard`), offline mocking via `from_dict(... mock_session_path=...)`, and observation serialization (`_format_observation`, `_build_info`) so downstream agents receive ASCII grids, flattened frames, and action metadata (`arcagi3_env.py:200-293`).

**`lucidgym/agents/arcagi3_agent.py`**
- Provides a thin `BaseAgent` implementation that turns environment observations into chat prompts and parses LLM or passthrough outputs back into ARC action dicts (`arcagi3_agent.py:31-175`).
- Maintains conversation history (`chat_completions`), collects a `Trajectory` of `Step` objects, and supports two execution modes: `"llm"` (default) and `"passthrough"` via a callable or the special `__PASSTHROUGH__` token (`arcagi3_agent.py:36-93`).
- Observation formatting emphasizes low context (card/game IDs, score/state summary, ASCII grid, and available actions with coordinate hints) to guide model outputs (`arcagi3_agent.py:96-113`).
- Response parsing accepts JSON or inline text, enforces coordinate requirements per `GameAction`, and validates the requested action against the most recent `available_actions` list before handing it back to the environment (`arcagi3_agent.py:115-169`).

**`/home/jwang/Arc/Trying-ARC-AGI-3-Agents/agents/agent.py`**
- Legacy monolithic agent that embeds environment + policy logic: manages the HTTP session, headers, recorder, scorecard fetching, timing/FPS stats, and the main action loop inside one abstract base class (`agent.py:22-210`).
- `main()` (decorated with `trace_agent_session`) repeatedly calls subclass `choose_action()` until `is_done()` or `MAX_ACTIONS`; each action goes through `do_action_request()`/`take_action()`, and frames are appended/recorded with GUID updates (`agent.py:74-164`).
- Cleanup finalizes recordings, logs performance metrics, and closes the session, while `Playback` replays stored recordings by synthesizing `GameAction` objects from JSONL files (`agent.py:165-289`).

## How the split maps the legacy agent

- **Transport/session management** – `Agent.do_action_request()` and raw `requests.Session` usage are now concentrated in `ArcAgi3Client`, which `ArcAgi3Env` owns for each episode (`arcagi3_env.py:128-170`). Agents no longer touch HTTP directly.
- **Frame bookkeeping** – `Agent.frames`, `guid`, and `_cleanup` state have become `_episode_frames`, `_episode_guid`, and `_actions_taken` within the environment (`arcagi3_env.py:90-199`), while per-step metadata is surfaced to the agent through observations and the `Trajectory` object.
- **Action selection contract** – Instead of overriding `Agent.choose_action`, LucidGym agents receive the latest observation/reward/done via `update_from_env()` and return a payload through `update_from_model()`; `ArcAgi3Env._coerce_action()` fulfills the legacy role of attaching coords/reasoning and enforcing action IDs (`arcagi3_env.py:295-308`).
- **Scorecard & lifecycle** – `Agent.cleanup()` fetched `/api/scorecard/...` and recorded results; now `ArcAgi3Env.close()` performs the fetch and injects the summary into subsequent `info["arc"]["scorecard"]`, while agents can optionally call `open_scorecard/close_scorecard` on the env for card management (`arcagi3_env.py:182-215`).
- **Prompting & parsing** – Responsibilities that previously lived in concrete `Agent` subclasses (LLM prompts, scripted heuristics, playback) are centralized in `ArcAgi3Agent`, making the interface compatible with ReAct-style pipelines rather than the bespoke `main()` loop (`arcagi3_agent.py:36-175`).

## Missing parity & recommended additions

1. **Recording hooks** – The recorder functionality (`agent.py:120-189`) is not represented in the LucidGym split. To regain audit trails and playback support, expose opt-in logging from `ArcAgi3Env.step()`/`reset()` (e.g., `enable_recording(recorder: Recorder)` that JSONL-dumps each `FrameData`) or add a middleware wrapper that calls the existing `Recorder` helpers.
2. **Playback agent** – Legacy workflows rely on `Playback` to deterministically replay sessions (`agent.py:214-289`). Implement an equivalent `ArcAgi3PlaybackAgent` or a `passthrough_fn` helper that streams recorded `action_input` blobs so experiments can regress against prior runs.
3. **AgentOps tracing integration** – The new `lucidgym/environments/arcagi3/tracing.py` module is currently unused. Mirror the old `@trace_agent_session` behavior by wrapping the outer control loop (e.g., the runner that alternates `env.step`/`agent.update_from_model`) or by adding hooks inside `ArcAgi3Agent` to start/end traces per episode.
4. **Scorecard-to-agent plumbing** – Although `ArcAgi3Env` caches scorecard summaries, agents never surface them. Consider adding a helper like `ArcAgi3Env.latest_scorecard()` or including the cached summary in `info` earlier (at `done=True`) so policies/tools can log results during evaluation.
5. **Runtime metrics** – The legacy agent tracked elapsed time and FPS to detect stalls (`agent.py:100-110`). If comparable telemetry is needed, add optional timing callbacks in the environment or a monitoring wrapper that records `actions_taken / wall_time`.

## Recent fixes

- `lucidgym/environments/arcagi3/client.py:82-200` – Added a `card_id` kwarg to `_command`, threaded it through `reset()`/`step()`, and ensured resets bootstrap a scorecard when neither a default nor override card ID is provided. `ArcAgi3Env.reset()` can now reach the client without `TypeError`.
- `lucidgym/environments/arcagi3/arcagi3_env.py:31-330` – `ArcAgi3EnvConfig` now owns an optional `card_id`, the constructor accepts it, runtime overrides are honored via `reset(task=...)`, and `_resolve_client_for_scorecard()` no longer references a missing attribute. Scorecard helpers work both during and after an episode.
- `lucidgym/agents/arcagi3_agent.py:160-169` – `_parse_inline_action` now splits on `r"[\s,]+"`, so plain-text responses like `ACTION6 x=3 y=5` parse correctly and coordinate validation is triggered.
- `examples/arcagi3/run_arc_eval.py:17-84` – Updated the `ArcAgi3Env` import to `lucidgym.environments.arcagi3.arcagi3_env`, honored `--no-mock`, passed the new `card_id` parameter, and delayed score printing until after `env.close()` populates the cached scorecard. The CLI runner can now target either the built-in mock recording or a live ARC endpoint without crashing.
These additions would close the remaining gaps between the original single-class implementation and the new env/agent split, ensuring feature parity (recordings, tracing, deterministic playback, and richer instrumentation) while preserving the cleaner separation of concerns.
