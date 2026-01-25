## Merge Plan: Unify Eval Harness with Env + Agents (No BaseAgentWrapper)
- Guideline: when you complete a phase, append a new bullet under that phase reading `Implementation finished for Phase N` to make handoffs explicit.

### Phase 1: Env-Driven Walking Skeleton (Single + Multi-Thread Ready)
- **Goal:** Prove BaseAgent variants run via `ArcAgi3Env` without `BaseAgentWrapper`, supporting both single and threaded execution from the start.
- **Step-by-Step Tasks (use `lucidgym/environments/arcagi3/arcagi3_env.py` for env fields/rewards/actions):**
  - In `lucidgym/evaluation/harness.py::run_evaluation_task`, build a fresh `ArcAgi3Env` and agent per worker (including single-thread runs), pass CLI tags/kwargs into `env.reset`, and wrap both `env.reset` and `env.step` calls with the existing retry helper; ensure a `finally` block always calls `env.close()`.
  - In `evaluate_single_game` (same file), rewrite the episode loop to call `env.reset` → `agent.update_from_env` → `agent.update_from_model` → `env.step` until done; ensure the loop only uses env state/`info["arc"]` (card_id/guid/replay_url) and does not touch wrapper fields; store attempts/levels from loop counters for thread safety.
  - Keep JSONL/summary writes stubbed but ensure the new loop still returns `GameMetrics` and is safe to reuse inside the threaded executor; double-check the env/client scorecard open/close is solely driven by `ArcAgi3Env` (no manual HTTP calls here).
  - **Harness Modifications (how):** Directly modify `evaluate_single_game` and `run_evaluation_task` in `lucidgym/evaluation/harness.py` to use env-based stepping and per-task env/agent instantiation; remove wrapper-specific state usage. No new helper; existing functions are rewritten for env + thread safety.
  - **Verification:** Run `--agent as66_guided_agent --suite debug_suite --num_runs 1 --max_actions 50 --max_workers 1` and with `--max_workers 2`; confirm runs complete, env opens/closes scorecard, replay url is emitted, and there is no `BaseAgentWrapper` import.
  - **Potential Blockers:** Action schema mismatches (Action vs. GameAction), thread safety of env/client, rate limits during multi-worker calls, missing replay/guid propagation from env info.
  - Implementation finished for Phase 1

### Phase 2: Harness with rllm Rollout Engine
- **Goal:** Drive the env loop with an rllm rollout engine (OpenAI/Together) the same way `examples/arcagi3/run_arc_eval.py` does, so agents get real model outputs instead of stub responses.
- **Step-by-Step Tasks:**
  - In `lucidgym/evaluation/harness.py`, add a helper `_build_rollout_engine` (near the top of the file, after constants) that mirrors `examples/arcagi3/run_arc_eval.py::setup_rollout_engine` sampling defaults and switches OpenAI vs. Together + tokenizer.
  - In `run_evaluation_task` (same file, function body near the env creation), instantiate `rollout_engine = _build_rollout_engine(agent_kwargs["model"])` after creating the `agent_instance`, and pass it into `evaluate_single_game` via the existing call.
  - In `evaluate_single_game` (same file, loop where actions are produced), before `agent.update_from_model`, call `rollout_engine.get_model_response(agent.chat_completions, tools=agent_tools, accumulate_reasoning=bool(rollout_engine.tokenizer))` on a per-task event loop and feed the returned `.text` into `agent.update_from_model`; keep the rest of the retry/metrics logic unchanged.
  - Ensure the per-task event loop is created and closed inside `evaluate_single_game` (just before the main loop starts and in the `finally`) so threaded workers do not share loops.
- **Verification:** Run a short harness invocation and confirm model calls go through the rollout engine, actions are derived from model text, and JSONL/summary outputs still populate with env-derived metrics.
- **Potential Blockers:** Missing LLM API keys/tokenizer downloads, event-loop conflicts inside thread workers, tool payloads that differ from the agent's parser expectations.
