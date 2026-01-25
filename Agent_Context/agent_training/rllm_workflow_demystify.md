# rLLM Workflow Demystify

This note walks through every concrete workflow module in `rllm/rllm/workflows` and highlights how they relate to one another.

## Core infrastructure (`workflow.py`)
- `Workflow` defines the contract shared by every workflow: `run(...)` must be implemented and is wrapped by `run_with_termination_handling(...)`, which standardizes timeout/error handling and converts any collected trajectories into an `Episode` with metrics and termination metadata.
- Common utilities include `commit(...)` for persisting finished trajectories, `collect_trajectories()` for aggregating agent state, reward shaping/discounting hooks (`compute_trajectory_reward`, `adjust_step_rewards`), per-episode correctness/metrics assignment, and lifecycle helpers such as `reset()` and `run_in_executor()` for async-safe env interaction.
- `TerminationReason`/`TerminationEvent` give every workflow a shared vocabulary for why execution stopped (env done, token/turn limits, timeout, etc.).

## Timing instrumentation (`timing_mixin.py`)
- `TimingTrackingMixin` can be combined with any workflow to record overall + per-step timing for LLM calls, env interactions, and reward computation.
- The mixin wraps env/model calls (`timed_env_call`, `timed_llm_call`), initializes timers during `reset()`, and injects the collected timing data back into the episode/trajectory/step info in `postprocess_episode`.

## SimpleWorkflow (`simple_workflow.py`)
- Minimal, single-shot workflow that does not rely on an external environment/agent pairing. It wraps a lightweight `SimpleAgent` whose only job is to hold a trajectory.
- Input normalization: accepts `messages`, `question`, `prompt`, or `problem` and always builds a single user message list before invoking the rollout engine.
- Produces exactly one assistant response, scores it via a pluggable `RewardFunction`, logs the step, and terminates immediately with either `MAX_RESPONSE_LENGTH_EXCEEDED` or `ENV_DONE`. No env loop, no timing mixin, no multi-step logic.

## SingleTurnWorkflow (`single_turn_workflow.py`)
- First workflow that integrates a concrete agent/environment pair. Both are built from the trainer registries, so experiment configs can pass either classes or their registered names.
- Execution is intentionally limited to one perception-action-update pass: reset env, inform the agent of the initial observation, collect exactly one LLM response, act once in the env, and finish. If the env does not signal `done`, the workflow still raises `MAX_TURNS_EXCEEDED` because single-turn execution was expected.
- Inherits `TimingTrackingMixin`, so every env/model call is timed and stored in the resulting episode.

## MultiTurnWorkflow (`multi_turn_workflow.py`)
- Expands on SingleTurn by looping up to `max_steps`. After each model inference the agent/environment exchange continues until either the environment signals termination or the step budget is exhausted.
- Shares the same timing instrumentation as SingleTurn but differs in termination policy: `ENV_DONE` stops early, whereas surviving the loop triggers `MAX_TURNS_EXCEEDED`. It also enforces finish-reason checks on every turn to catch truncated generations.

## CumulativeWorkflow (`cumulative_workflow.py`)
- Architecturally similar to MultiTurn (same agent/env construction, timing mixin, and step budget) but adds explicit control over prompt growth.
- Before each generation it rebuilds the full prompt, measures its token length, and dynamically shrinks the allowable response budget (`max_tokens`) so that the cumulative prompt + response stays within the rollout engine’s `max_response_length`. This makes it suitable for long-running conversations where context can creep toward the tokenizer limit.
- If the admissible `max_tokens` drops to zero it terminates early with `MAX_RESPONSE_LENGTH_EXCEEDED` rather than risking an over-limit API call.

## Putting the workflows in context
- **Lifecycle sophistication:** `SimpleWorkflow` is stateless beyond one trajectory; `SingleTurnWorkflow` introduces env dynamics; `MultiTurnWorkflow` reuses the same building blocks but loops; `CumulativeWorkflow` adds prompt-budget accounting on top of the multi-turn loop.
- **Timing data:** only the workflows that inherit `TimingTrackingMixin` (SingleTurn, MultiTurn, Cumulative) emit timing metrics. SimpleWorkflow uses the bare `Workflow` base and therefore does not collect timing info by default.
- **Token/turn management:** MultiTurn and Cumulative guard against both excessive turns and response truncation; Cumulative additionally guards prompt growth. SingleTurn only checks a single response and step, while SimpleWorkflow only enforces response-length limits coming straight from the model output.
- **Environment usage:** SimpleWorkflow never touches an environment; the other three workflows depend on env reset/step semantics and feed observations/rewards back through a richer agent interface.

These differences should make it clearer which workflow to choose for a given experiment: simple scoring tasks can live on `SimpleWorkflow`, env-based evaluations pick between `SingleTurn` and `MultiTurn` depending on turn count, and conversational tasks that risk prompt blow-up should use `CumulativeWorkflow`.
