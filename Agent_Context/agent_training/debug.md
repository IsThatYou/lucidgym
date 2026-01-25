# TextArena Countdown Debug Notes

## Execution Path
- `examples/textarena/countdown.sh` launches `python3 -m train_textarena_countdown` with overrides that set the `rllm.env.name` to `textarena_env` and pass `LucidCountdown-v0-raw` through `+rllm.env.env_args.env_id` (`examples/textarena/countdown.sh:12-48`).
- The Hydra entrypoint wires those overrides into an `AgentTrainer`, explicitly choosing `TextArenaAgent` and `TextArenaEnv` via `workflow_args` so every rollout instantiates LucidGym's adapter (`examples/textarena/train_textarena_countdown.py:56-83`).

## What `TextArenaEnv` Does
- `TextArenaEnv.reset` recreates the arena every episode by calling `ta.make(env_id=env_id, **make_kwargs)` before immediately calling `self._env.get_observation()` so it can format messages into the structure RLLM expects (`lucidgym/environments/textarena_env.py:91-132`).
- `_format_observation` assumes it receives a sequence of `(from_id, text, ObservationType)` tuples coming directly from `textarena.core.State` (`lucidgym/environments/textarena_env.py:152-160`).

## Why Observations Turn Into Strings
- The `ta.make` helper resolves `env_id` inside TextArena's registry and eagerly applies any default wrappers declared for that ID (`textarena/envs/registration.py:30-95`). The wrappers list for `Countdown-v0` includes the `BOARDGAME_WRAPPERS` preset, and the `-train` suffix swaps in `[GameMessagesAndCurrentBoardObservationWrapper, ActionFormattingWrapper]` (`textarena/envs/__init__.py:40-41`).
- `GameMessagesAndCurrentBoardObservationWrapper` subclasses `ObservationWrapper`, overrides `observation`, and returns a concatenated string containing the prompt, intermediate messages, and current board (`textarena/wrappers/ObservationWrappers/llm_observation_wrapper.py:143-184`). Because wrappers wrap the base env before LucidGym ever sees it, `self._env.get_observation()` inside `TextArenaEnv` would yield `(player_id, "<formatted string>")` instead of raw message tuples if we reused the upstream IDs.
- Selecting a `-raw` variant avoids observation wrappers entirely so `textarena.core.Env.get_observation` returns `(player_id, List[Tuple[int, str, ObservationType]])` by forwarding `State.get_current_player_observation()` (`textarena/core.py:129-181`).

## LucidGym Env Overrides
- LucidGym ships `LucidGymCountdownEnv`, a thin subclass of TextArena’s `CountdownEnv` that overrides `_add_board_observation` to emit only `_render_board()` without the `Current progress score` suffix (`lucidgym/environments/textarena/countdown_env.py`).
- `lucidgym/environments/textarena/registry.py` mirrors TextArena’s registry entries and registers `LucidCountdown-v0`, `LucidCountdown-v0-train`, and `LucidCountdown-v0-raw` so `ta.make` can instantiate the subclass without patching upstream files.
- `register_lucidgym_components()` ensures these overrides are installed before merging agent/env/workflow mappings, which means Hydra configs can set `env_id=LucidCountdown-v0-raw` to receive tuple observations formatted for `TextArenaEnv`.
- When extending this pattern to another TextArena game, add a new subclass under `lucidgym/environments/textarena/`, register it through the helper module, and import the registration function inside `lucidgym/registry.py` so the IDs are live before any workflow calls `ta.make(...)`.
