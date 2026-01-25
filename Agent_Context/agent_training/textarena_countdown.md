# Plan – TextArena Countdown RL wiring

## Objectives
- Reuse the PPO config/hyper-params from `rllm/examples/countdown` but route gameplay through TextArena’s official `Countdown-v0` env.
- Ensure LucidGym’s `TextArenaEnv`/`TextArenaAgent` pair plugs into the VERL trainer without touching upstream RLLM files.
- Provide a turnkey shell launcher so researchers can kick off training with the same flags used by the native countdown run.

## Implementation steps
1. **Study upstream refs** – inspect `rllm/examples/countdown/train_countdown.py` + `.sh` plus the LucidGym adapters (`lucidgym/environments/textarena_env.py`, `lucidgym/agents/textarena_agent.py`) to understand required ctor args and PPO expectations.
2. **Hydra entrypoint** – add `examples/textarena/train_textarena_countdown.py` that:
   - registers LucidGym components,
   - loads the countdown datasets via `DatasetRegistry`,
   - merges sensible default `agent_args`/`env_args` (env_id=`Countdown-v0-train`, single player) with any Hydra overrides,
   - instantiates `AgentTrainer` with `TextArenaAgent`/`TextArenaEnv` and starts training.
3. **Shell launcher** – implement `examples/textarena/countdown.sh` mirroring the upstream script’s environment exports + PPO overrides, but point the module to `examples.textarena.train_textarena_countdown` and set `rllm.agent/env` names + env args for TextArena.
4. **Usage notes** – remind users (in commit message or future docs) that they must run `prepare_countdown_data.py` beforehand so the dataset registry exists, and that TextArena must be installed/accessible on the Python path.

## Open questions / next steps
- Add optional Hydra config group for LucidGym if we need richer defaults than the inline dicts.
- Consider exposing different TextArena env_ids (train/test) via CLI flags or dataset metadata once multi-env curriculum is required.
