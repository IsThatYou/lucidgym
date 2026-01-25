# Training Log

## Pipeline Overview
- `examples/countdown/train_countdown.py:9-23` loads the countdown train/test splits from `DatasetRegistry`, wires `SimpleWorkflow` with `countdown_reward_fn`, and hands everything to `AgentTrainer`. The shell wrapper `examples/countdown/train_countdown.sh` overrides `trainer.test_freq=10`, so validation (logged as “test accuracy”) runs every ten PPO steps.
- `AgentTrainer` stays on the default VERL backend, so `rllm/trainer/agent_trainer.py:21-127` spins up the shared runtime and forwards control to the remote task runner defined in `rllm/trainer/verl/train_agent_ppo.py`.
- `TaskRunner.run` (`rllm/trainer/verl/train_agent_ppo.py:129-180`) instantiates `AgentWorkflowPPOTrainer` (`rllm/trainer/verl/agent_workflow_trainer.py`) which owns the PPO loop, the workflow execution, and when validation happens.

## Validation / “Test Accuracy”
- Inside `AgentWorkflowPPOTrainer.fit_agent`, the block at `rllm/trainer/verl/agent_workflow_trainer.py:349-410` checks `trainer.test_freq`. When the condition hits (every ten steps here), it calls `_validate_agent()` and merges the returned metrics into the logger output for that step.
- `_validate_agent` (`rllm/trainer/verl/agent_workflow_trainer.py:433-520`) iterates over the validation dataloader, reruns the workflow `actor_rollout_ref.rollout.val_kwargs.n` times per prompt (set to 1 in the script by default), and aggregates the episode-level `is_correct` flags.
- The reported metrics are `val/<data_source>/pass@1` (mean correctness over individual rollouts) and `val/<data_source>/pass@n` (whether any of the n rollouts succeeded per task). Those numbers are what end up in the logs as “test accuracy.”
- The `is_correct` signals come from the workflow: each rollout step invokes `countdown_reward_fn` (`rllm/rewards/countdown_reward.py:58-152`), `Workflow.postprocess_episode` sums the trajectory rewards and marks an episode correct if the total reward is positive (`rllm/workflows/workflow.py:136-227`), and `AgentWorkflowEngine.transform_results_for_verl` copies that boolean into the `DataProto` used by the trainer (`rllm/engine/agent_workflow_engine.py:215-352`).

## Critic Rewards & Scores
- Right after trajectories are unioned into the training batch, `AgentWorkflowPPOTrainer.fit_agent` fills `token_level_scores` with workflow rewards (`step_rewards` when stepwise mode is enabled, otherwise `traj_rewards`) and optionally applies a KL penalty to produce `token_level_rewards` (`rllm/trainer/verl/agent_workflow_trainer.py:318-338`).
- If a critic is enabled, `critic_wg.compute_values` runs before the advantage pass and attaches per-token value predictions to the batch (`rllm/trainer/verl/agent_workflow_trainer.py:315-319`). The PPO advantage computation then writes `advantages` and `returns` using those predictions along with the rewards (`rllm/trainer/verl/agent_workflow_trainer.py:349-358`).
- During the optimizer phase, `critic_wg.update_critic` receives the padded batch (containing `responses`, `response_mask`, `values`, and `returns`), runs several mini/micro-batches through the value head, minimizes the clipped value loss, and emits metrics such as `critic/vf_loss`, `critic/vf_clipfrac`, `critic/vpred_mean`, and `critic/grad_norm` (`verl/workers/critic/dp_critic.py:190-232`). Those metrics are merged back into the training logger at the driver (`rllm/trainer/verl/agent_workflow_trainer.py:389-394`).
- After each step, `compute_data_metrics` rolls up “critic” telemetry: it sums `token_level_scores` to produce `critic/score/*`, sums `token_level_rewards` for `critic/rewards/*`, and reports stats for `advantages`, `returns`, and (when the critic is on) `values` plus the value-function explained variance (`verl/trainer/ppo/metric_utils.py:80-138`). This is what shows up in TensorBoard/W&B under the `critic/...` namespaces.

## Validation Sample Trajectories
- The training loop randomly prints a couple of trajectories per step via `visualize_trajectory_last_step` (`rllm/trainer/verl/agent_workflow_trainer.py:415-753`), but `_validate_agent` does not call that helper, so validation currently ends with aggregate metrics only.
- If you enable `trainer.log_episodes`, `init_workers` wires up an `EpisodeLogger` (`rllm/trainer/verl/agent_workflow_trainer.py:80-96`). The `AgentWorkflowEngine` tags each rollout with the current mode via `set_training_step(...)`, and when `_validate_agent` runs it switches the mode to `"val"` before invoking `generate_trajectories` (`rllm/engine/agent_workflow_engine.py:60-212`). As a result, every validation episode is already written to disk (JSON files under `logs/<project>/<experiment>/episodes/val_step_*`) and can be used as sample trajectories.
- If you also need console-visible samples during validation, the minimal plan is to (1) gate a call to `visualize_trajectory_last_step` inside `_validate_agent` behind a small `trainer.log_val_samples` flag, (2) select a few episodes from the deduplicated `test_batch` right before metrics are aggregated, and (3) reuse the existing visualization helper so formatting stays consistent with the training prints.

## W&B Table Visualization Plan
Goal: log a few representative trajectories per validation step into a W&B Table for later browsing.

1. **Config Flag & Logger Access**  
   - Add a trainer flag such as `trainer.log_val_tables` and, if needed, a limit (e.g., `trainer.val_table_samples`).  
   - `AgentWorkflowPPOTrainer.fit_agent` already instantiates `Tracking`; extend `verl/utils/tracking.py` if necessary so you can access the underlying W&B run via `logger.backend` or inject a helper that exposes `wandb` primitives when `trainer.logger == "wandb"`.

2. **Capture Episodes in `_validate_agent`**  
   - Right after `test_batch = test_batch.select_idxs(selected_idxs)`, slice out `sample_idxs = np.random.choice(len(test_batch), size=limit, replace=False)` and build a lightweight struct with prompts/responses (`tokenizer.batch_decode` on `test_batch.batch["prompts"]` / `["responses"]`), reward tokens (`traj_rewards` or `step_rewards`), `is_correct`, `data_source`, and any workflow metrics.
   - Convert tensors to python strings so they’re JSON/W&B safe; strip padding using the attention masks before decoding.

3. **Create the Table**  
   - Inside `_validate_agent`, guard on `trainer.log_val_tables` and `self.config.trainer.logger == "wandb"`.  
   - Import `wandb` lazily, then build `table = wandb.Table(columns=["prompt", "response", "reward", "is_correct", "data_source", "metrics_json"])` and append one row per sampled episode (`metrics_json` can be `json.dumps(episode_metrics)`).

4. **Log the Table**  
   - Return both the usual scalar metrics and the table handle. One option is to stash the table in `metrics["val/sample_table"] = table`. W&B `Tracking` already forwards dict values to `wandb.log`, so the table will appear alongside scalar metrics for that `global_step`.  
   - Consider logging train tables every `k` steps using the same helper to keep parity between modes.

5. **Tidy Up & Limits**  
   - Keep samples small (≤10 rows) to avoid large payloads, and sanitize any PII in prompts/responses before logging.  
   - Document the flag and table schema in `README.md` so downstream consumers know where to find the visualization (`Artifacts -> Media -> Tables` in the W&B UI).

## Customization Tips
- Change when validation runs by overriding `trainer.test_freq` (setting it to `0` disables periodic evaluation).
- Change what “correct” means by editing `countdown_reward_fn` or `Workflow.assign_episode_correctness`; the new semantics automatically flow into both the rewards seen by PPO and the validation accuracy numbers.
- Change how accuracy is aggregated or logged by modifying `_validate_agent` (e.g., compute different `pass@k` metrics, bucket by custom fields, emit richer workflow metrics). All of that lives in the RLLM layer; VERL just runs the scheduling/logging scaffolding.

## ARC-AGI-3 smoke tests (2024-11-xx)
- Added `examples/arcagi3/mock_session.json` plus `StaticArcTransport` so the env/agent/client can run without network access.
- Verified the mock loop manually via `examples/arcagi3/run_arc_eval.py` (default options) which plays through the two-step win trajectory and prints the final reward/state.
- Test suite: `python -m pytest tests/lucidgym/test_arcagi3.py` exercises the client, env, mock loader, and agent parser. The current harness is missing `pytest`, so install the `dev` extra (`python -m pip install -e .[dev]`) before re-running the command locally.
- Live ARC verification is pending API credentials; once available, rerun the example script with `--no-mock` and confirm scorecards/replay URLs match the ARC dashboard.
