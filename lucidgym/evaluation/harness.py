import argparse
import asyncio
import logging
import os
import sys
import time
import threading
import json
import dataclasses
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Type, List, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import openai
from dotenv import load_dotenv
from transformers import AutoTokenizer

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env file explicitly
load_dotenv(dotenv_path=project_root / ".env.example")
load_dotenv(dotenv_path=project_root / ".env", override=True)

from lucidgym.agents import AVAILABLE_AGENTS
from lucidgym.environments import ArcAgi3Env
from lucidgym.environments.arcagi3.structs import GameAction, GameState
from lucidgym.utils.representation import RepresentationConfig, GridFormat
from lucidgym.evaluation.config import EVALUATION_GAMES
from lucidgym.metrics.structures import GameMetrics, LevelMetrics, AttemptMetrics
from lucidgym.metrics.reporting import generate_console_report, save_summary_report, calculate_stats
from rllm.engine.rollout import OpenAIEngine
from rllm.agents.agent import BaseAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT_URL = os.environ.get("ROOT_URL", "https://three.arcprize.org")

def _build_rollout_engine(model_name: str, reasoning_effort: str = "low") -> OpenAIEngine:
    """
    Build a rollout engine mirroring the ARC reference runner defaults.
    """
    together_api_key = os.getenv("TOGETHER_API_KEY", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    # If model has a provider prefix (contains "/"), force OpenRouter
    has_provider_prefix = "/" in model_name

    tokenizer = None
    if has_provider_prefix:
        # Use OpenRouter for provider-prefixed models
        api_key = openrouter_api_key
        base_url = "https://openrouter.ai/api/v1"
    else:
        # Default priority: OPENAI_API_KEY > OPENROUTER_API_KEY
        api_key = openai_api_key or openrouter_api_key
        base_url = "https://openrouter.ai/api/v1" if openrouter_api_key and not openai_api_key else "https://api.openai.com/v1"

    # Use Together/Qwen tokenizer when the model is not an OpenAI model
    # Check both "gpt-" and "openai/gpt-" formats, as well as o1/o3 models
    model_base = model_name.split("/")[-1]  # Get model name after provider prefix
    is_openai_model = model_base.startswith("gpt-") or model_base.startswith("o1-") or model_base.startswith("o3-")

    if not is_openai_model:
        if has_provider_prefix:
            # Already using OpenRouter, don't load tokenizer
            pass
        else:
            # Non-OpenAI model without prefix, use Together or OpenRouter with tokenizer
            api_key = together_api_key or openrouter_api_key
            base_url = "https://openrouter.ai/api/v1" if openrouter_api_key and not together_api_key else "https://api.together.xyz/v1"
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B-Instruct-2507")

    if not api_key:
        raise RuntimeError("No API key found for rollout engine. Set OPENAI_API_KEY, TOGETHER_API_KEY, or OPENROUTER_API_KEY.")

    sampling_params = {
        "temperature": 1,
        "max_completion_tokens": 8192,
    }

    # Only add reasoning_effort for models that support it (gpt-5.x and o-series models)
    supports_reasoning = model_base.startswith("gpt-5") or model_base.startswith("o1") or model_base.startswith("o3")
    if supports_reasoning and reasoning_effort and reasoning_effort != "none":
        sampling_params["reasoning_effort"] = reasoning_effort

    log.info(f"OpenAIEngine config: model={model_name}, base_url={base_url}, tokenizer={'loaded' if tokenizer else 'None'}, sampling_params={sampling_params}")

    return OpenAIEngine(
        model=model_name,
        tokenizer=tokenizer,
        base_url=base_url,
        api_key=api_key,
        max_prompt_length=65536 * 2,
        sampling_params=sampling_params,
    )

# --- Agent Variant Helper ---
def _get_agent_tags(agent_name: str, args: argparse.Namespace) -> List[str]:
    """
    Checks for known CLI args that create agent "variants"
    and returns a list of tags.
    """
    tags = []

    #
    if agent_name.startswith("as66"):
        # Check for the general prompts env var you mentioned
        if args.arcgame_general_prompts.strip().lower() in ("1", "true", "yes", "on"):
            tags.append("general")
    
    
    
    # Model/Reasoning
    model_override = args.agent_model_override
    if model_override:
        # Sanitize model name for filename
        sanitized_model = model_override.split('/')[-1].replace('.', '_')
        tags.append(sanitized_model)
        
    reasoning_effort = args.agent_reasoning_effort
    if reasoning_effort:
        tags.append(f"reason-{reasoning_effort}")

    # Text Diff
    include_text_diff = args.include_text_diff.lower() == "true"
    if not include_text_diff:
        tags.append("noDiff")

    # Context Limit
    context_limit = args.context_length_limit
    if context_limit != -1:
        tags.append(f"ctx{context_limit // 1000}k")

    # Crop Border
    crop_border = args.crop_border
    if crop_border > 0:
        tags.append(f"crop{crop_border}")

    # Downsample
    downsample_images = args.downsample_images.lower() == "true"
    if not downsample_images and "as66visualmemoryagent" in agent_name:
        tags.append("64x64")

    # Image Detail
    image_detail = args.image_detail_level.lower()
    if image_detail != "low" and "as66visualmemoryagent" in agent_name:
        tags.append(f"detail-{image_detail}")

    # Pixels Per Cell
    pixels_per_cell = args.image_pixels_per_cell
    if pixels_per_cell != 24 and "as66visualmemoryagent" in agent_name:
        tags.append(f"cell{pixels_per_cell}")

    return tags

# --- Retry Helper ---
MAX_RETRIES = 5
INITIAL_BACKOFF = 1 # in seconds

def _run_with_retries(func_to_run: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Runs a function with exponential backoff for specific retriable API errors.
    """
    retries = 0
    backoff = INITIAL_BACKOFF
    while True:
        try:
            return func_to_run(*args, **kwargs)
        except (openai.RateLimitError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if retries >= MAX_RETRIES:
                log.error(f"Final attempt failed for {func_to_run.__name__} after {retries} retries. Raising error.")
                raise e # Re-raise the error to be caught by the main try/except
            
            log.warning(f"API error detected for {func_to_run.__name__}: {type(e).__name__}. Retrying in {backoff}s... (Attempt {retries + 1}/{MAX_RETRIES})")
            time.sleep(backoff)
            retries += 1
            backoff *= 2 # Exponential backoff
        # Any other exception (e.g., logic error, validation error) should not be retried and will be caught by the main handler.
        except Exception as e:
            log.error(f"Non-retriable error in {func_to_run.__name__}: {e}", exc_info=False) # Log minimal info
            raise e # Re-raise

# --- Function to run a single game attempt ---
def evaluate_single_game(
    agent,
    env: ArcAgi3Env,
    game_id: str,
    agent_name: str,
    max_actions_per_game: int,
    run_index: int,
    tags: Optional[List[str]] = None,
    rollout_engine: Optional[OpenAIEngine] = None,
    prompts_log_path: Optional[Path] = None,
) -> GameMetrics:
    """Run a single env-driven game loop without the BaseAgentWrapper."""


    run_metrics = GameMetrics(
        game_id=game_id,
        agent_name=agent_name,
        run_index=run_index,
        start_time=time.time(),
    )
    run_metrics.status = "IN_PROGRESS"

    current_level_number = 1
    current_level_metrics = LevelMetrics(level_number=current_level_number)
    current_attempt_number = 1
    current_attempt_metrics = AttemptMetrics(attempt_number=current_attempt_number)
    attempt_start_time = run_metrics.start_time

    max_score = 0
    total_actions_this_run = 0
    arc_state: Optional[GameState] = None
    arc_score = 0

    agent_tools: List[dict[str, Any]] = []
    supports_accumulated_reasoning = bool(rollout_engine and getattr(rollout_engine, "tokenizer", None))
    rollout_loop = asyncio.new_event_loop() if rollout_engine is not None else None

    try:
        if hasattr(agent, "build_tools"):
            try:
                agent_tools = agent.build_tools()  # type: ignore[attr-defined]
            except Exception as tool_err:
                log.warning(f"[{game_id} Run {run_index}] Failed to build agent tools: {tool_err}")

        if hasattr(agent, "reset"):
            agent.reset()

        def _reset_game_state(env, agent, run_metrics, initial_reset=True):
            observation, info = _run_with_retries(
                env.reset,
                task={"game_id": game_id, "max_actions": max_actions_per_game, "tags": tags},
            )

            arc_info = info.get("arc", {})
            arc_state = GameState(arc_info.get("state") or GameState.NOT_PLAYED)
            arc_score = arc_info.get("score", 0) or 0
            run_metrics.guid = arc_info.get("guid")
            run_metrics.replay_url = arc_info.get("replay_url")

            agent.update_from_env(observation=observation, reward=0.0, done=False, info=info)

            return arc_state, arc_score
        arc_state, arc_score = _reset_game_state(env, agent, run_metrics)

        def _record_attempt(status: str, current_attempt_metrics, current_level_metric, game_over: bool = False) -> float:
            attempt_end_time = time.time()
            current_attempt_metrics.duration_seconds = attempt_end_time - attempt_start_time
            current_attempt_metrics.status = status
            if game_over:
                current_attempt_metrics.game_overs += 1
            current_level_metrics.attempts.append(current_attempt_metrics)
            return attempt_end_time

        while total_actions_this_run < max_actions_per_game:
            action_dict = rollout_loop.run_until_complete(agent.call_llm(rollout_engine=rollout_engine))
            action_obj = agent.update_from_model(response=action_dict)
            print(f"[DEBUG]:harness:action_obj={action_obj}")
            observation, reward, done, info = _run_with_retries(env.step, action_obj)
            print(f"[DEBUG]:harness:reward={reward}, done={done}, total actions:{total_actions_this_run+1}, _episode_guid:{env._episode_guid}")

            total_actions_this_run += 1
            current_attempt_metrics.actions += 1

            previous_arc_state = arc_state
            previous_arc_score = arc_score
            arc_info = info.get("arc", {})
            new_arc_state = GameState(arc_info.get("state") or GameState.NOT_PLAYED)
            new_arc_score = arc_info.get("score", 0) or 0
            run_metrics.guid = run_metrics.guid or arc_info.get("guid")
            run_metrics.replay_url = run_metrics.replay_url or arc_info.get("replay_url")

            # Detect if game reset to an earlier level (score dropped)
            if new_arc_score < previous_arc_score:
                # Game was reset, sync current_level_number to actual game level
                current_level_number = new_arc_score + 1
                current_level_metrics = LevelMetrics(level_number=current_level_number)
                current_attempt_number = 1
                current_attempt_metrics = AttemptMetrics(attempt_number=current_attempt_number)
                attempt_start_time = time.time()
                log.info(f"[{game_id} Run {run_index}] Game reset detected (score {previous_arc_score} -> {new_arc_score}). Reset to Level {current_level_number}.")

            if len(env._episode_frames) >= 2 and env._episode_frames[-2] != env._episode_frames[-1]:
                current_attempt_metrics.state_changes += 1
            arc_state = new_arc_state
            arc_score = new_arc_score
            max_score = max(max_score, arc_score)
            run_metrics.highest_level_reached = max(run_metrics.highest_level_reached, current_level_number)

            agent.update_from_env(observation=observation, reward=reward, done=done, info=info)

            # Log prompts and responses
            if prompts_log_path and hasattr(agent, 'trajectory') and agent.trajectory.steps:
                last_step = agent.trajectory.steps[-1]
                with open(prompts_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Action {total_actions_this_run} | Level {current_level_number} | Attempt {current_attempt_number}\n")
                    f.write(f"Score: {arc_score} | Highest Level Reached: {run_metrics.highest_level_reached}\n")
                    f.write(f"State: {arc_state.name if arc_state else 'UNKNOWN'}\n")
                    f.write(f"{'='*80}\n\n")

                    # Log chat completions (prompts and responses)
                    if hasattr(last_step, 'chat_completions') and last_step.chat_completions:
                        for msg in last_step.chat_completions:
                            role = msg.get('role', 'unknown')
                            content = msg.get('content', '')

                            if role == 'observation_phase':
                                f.write("OBSERVATION PHASE PROMPT:\n")
                                f.write("-" * 80 + "\n")
                                f.write(f"{content}\n\n")
                            elif role == 'observation_response':
                                f.write("OBSERVATION RESPONSE:\n")
                                f.write("-" * 80 + "\n")
                                f.write(f"{content}\n\n")
                            elif role == 'action_phase':
                                f.write("ACTION PHASE PROMPT:\n")
                                f.write("-" * 80 + "\n")
                                f.write(f"{content}\n\n")
                            elif role == 'action_response':
                                f.write("ACTION RESPONSE:\n")
                                f.write("-" * 80 + "\n")
                                f.write(f"{content}\n\n")
                            else:
                                f.write(f"[{role.upper()}]\n{content}\n\n")

            # --- Handle Level Completion ---
            level_completed = (new_arc_score > previous_arc_score and 
                               new_arc_state not in (GameState.WIN, GameState.GAME_OVER))
            # print(f"[DEBUG]:harness: level_completed={level_completed}, new_arc_score={new_arc_score}, arc_score={previous_arc_score}, arc_state={previous_arc_state}, max_score={max_score}")

            if level_completed:
                attempt_end_time = _record_attempt("COMPLETED", current_attempt_metrics, current_level_metrics)
                current_level_metrics.status = "COMPLETED"
                run_metrics.level_metrics[current_level_number] = current_level_metrics

                log.info(f"[{game_id} Run {run_index}] Level {current_level_number} COMPLETED. Attempt {current_attempt_number} actions: {current_attempt_metrics.actions}. Score: {new_arc_score}.")

                current_level_number += 1
                current_level_metrics = LevelMetrics(level_number=current_level_number)
                current_attempt_number = 1
                current_attempt_metrics = AttemptMetrics(attempt_number=current_attempt_number)
                attempt_start_time = attempt_end_time
                continue

            if new_arc_state == GameState.GAME_OVER:
                _record_attempt("GAME_OVER", current_attempt_metrics, current_level_metrics, game_over=True)
                current_level_metrics.status = "GAME_OVER"
                run_metrics.level_metrics[current_level_number] = current_level_metrics
                run_metrics.status = "TIMEOUT"
                log.warning(f"[{game_id} Run {run_index}] Game Over on Level {current_level_number}, Attempt {current_attempt_number}. Actions this attempt: {current_attempt_metrics.actions}.")
                # Agent should call reset to start a new game on its own.
                current_attempt_number += 1
                current_attempt_metrics = AttemptMetrics(attempt_number=current_attempt_number)
                attempt_start_time = time.time()

            if new_arc_state == GameState.WIN:
                _record_attempt("COMPLETED", current_attempt_metrics, current_level_metrics)
                current_level_metrics.status = "COMPLETED"
                run_metrics.level_metrics[current_level_number] = current_level_metrics
                run_metrics.status = "COMPLETED_RUN"
                log.info(f"[{game_id} Run {run_index}] Game COMPLETED successfully! Final Level {current_level_number} actions: {current_attempt_metrics.actions}. Final Score: {new_arc_score}")
                break

    except Exception as e:
        run_metrics.status = "ERROR"
        run_metrics.error_message = str(e)
        current_attempt_metrics.status = "ERROR"
        current_level_metrics.status = "ERROR"
        log.error(f"[{game_id} Run {run_index}] Exception occurred: {e}", exc_info=True)

    finally:
        run_metrics.end_time = time.time()
        run_metrics.run_duration_seconds = run_metrics.end_time - run_metrics.start_time

        final_attempt_status = current_attempt_metrics.status
        if final_attempt_status == "IN_PROGRESS":
            if run_metrics.status == "ERROR":
                final_attempt_status = "ERROR"
            elif arc_state == GameState.WIN:
                final_attempt_status = "COMPLETED"
                run_metrics.status = "COMPLETED_RUN"
            elif total_actions_this_run >= max_actions_per_game:
                final_attempt_status = "TIMEOUT"
                run_metrics.status = "TIMEOUT"
            elif arc_state == GameState.GAME_OVER:
                final_attempt_status = "GAME_OVER"
                run_metrics.status = "TIMEOUT"
            else:
                final_attempt_status = "COMPLETED" if run_metrics.status != "ERROR" else "ERROR"
                if run_metrics.status == "IN_PROGRESS":
                    run_metrics.status = final_attempt_status

        if current_attempt_metrics.status == "IN_PROGRESS":
            current_attempt_metrics.duration_seconds = run_metrics.end_time - attempt_start_time
        current_attempt_metrics.status = final_attempt_status

        if not current_level_metrics.attempts or current_level_metrics.attempts[-1].attempt_number != current_attempt_metrics.attempt_number:
            current_level_metrics.attempts.append(current_attempt_metrics)
        if current_level_metrics.status == "IN_PROGRESS":
            current_level_metrics.status = final_attempt_status

        run_metrics.level_metrics[current_level_number] = current_level_metrics
        run_metrics.run_total_actions = sum(lm.total_actions for lm in run_metrics.level_metrics.values())
        run_metrics.total_game_overs_across_run = sum(lm.total_game_overs for lm in run_metrics.level_metrics.values())
        run_metrics.total_state_changes_across_run = sum(lm.total_state_changes for lm in run_metrics.level_metrics.values())
        run_metrics.total_actions_taken = total_actions_this_run
        run_metrics.final_score = max_score

        if run_metrics.guid and not run_metrics.replay_url:
            run_metrics.replay_url = f"{ROOT_URL}/replay/{game_id}/{run_metrics.guid}"


        if rollout_loop is not None:
            rollout_loop.close()

    return run_metrics


# --- Task Wrapper for Parallel Execution ---
def run_evaluation_task(
    args,
    game_id: str,
    run_index: int,
    agent_class,
    agent_name_cli: str,
    max_actions: int,
    agent_name_with_variant: str,
    eval_tags: Optional[List[str]] = None,
    rollout_engine: Optional[OpenAIEngine] = None,
    prompts_log_dir: Optional[Path] = None,
) -> GameMetrics:
    """Creates an agent and runs evaluate_single_game for one task."""

    log.debug(f"Task starting: Game {game_id}, Run {run_index}")

    env = ArcAgi3Env(
        game_id=game_id,
        root_url=ROOT_URL,
        api_key=os.getenv("ARC_API_KEY", ""),
        max_actions=max_actions,
        tags=eval_tags,
    )

    try:
        # Check if agent_class is a BaseAgent subclass
        if issubclass(agent_class, BaseAgent):
            # Create BaseAgent instance with appropriate kwargs
            agent_kwargs = {
                "name": agent_name_with_variant,
            }

            # Add agent-specific kwargs
            if "memory" in agent_name_cli.lower():
                agent_kwargs["game_id"] = game_id
                agent_kwargs["downsample"] = args.downsample_images.lower() == "true"
                agent_kwargs["include_text_diff"] = args.include_text_diff.lower() == "true"
                agent_kwargs["context_length_limit"] = args.context_length_limit

                if "visual" in agent_name_cli.lower():
                    agent_kwargs["image_detail_level"] = args.image_detail_level
                    agent_kwargs["pixels_per_cell"] = args.image_pixels_per_cell

            # Build representation config from CLI args (used by multiple agent types)
            rep_config = RepresentationConfig(
                format=GridFormat(getattr(args, 'grid_format', 'integers_spaced')),
                downsample=not getattr(args, 'no_downsample', False),
            )
            use_general = getattr(args, 'use_general_prompts', False)

            if "guided" in agent_name_cli.lower():
                agent_kwargs["input_mode"] = getattr(args, 'input_mode', 'text_only')
                agent_kwargs["game_id"] = game_id
                agent_kwargs["representation"] = rep_config
                agent_kwargs["use_general"] = use_general

            if "memory" in agent_name_cli.lower():
                # Memory/hypothesis agents also support representation
                agent_kwargs["representation"] = rep_config
                agent_kwargs["use_general"] = use_general

            if "meta_coding" in agent_name_cli.lower():
                agent_kwargs["game_id"] = game_id
                agent_kwargs["use_64x64"] = getattr(args, 'no_downsample', False)
                agent_kwargs["use_general_prompts"] = use_general
                agent_kwargs["no_progressive"] = getattr(args, 'no_progressive', False)
                agent_kwargs["representation"] = rep_config

            if "basic_obs_action" in agent_name_cli.lower():
                agent_kwargs["crop_border"] = args.crop_border

            agent_instance = agent_class(**agent_kwargs)
        else:
            raise ValueError(f"Agent class {agent_class} is not a subclass of BaseAgent.")

        # Create prompts log file
        prompts_log_path = None
        if prompts_log_dir:
            prompts_log_path = prompts_log_dir / f"{game_id}_run{run_index}.txt"

        metrics = evaluate_single_game(
            agent=agent_instance,
            env=env,
            game_id=game_id,
            agent_name=agent_name_with_variant,
            max_actions_per_game=max_actions,
            run_index=run_index,
            tags=eval_tags,
            rollout_engine=rollout_engine,
            prompts_log_path=prompts_log_path,
        )
        log.debug(f"Task finished: Game {game_id}, Run {run_index} -> Status: {metrics.status}")
        return metrics
    finally:
        try:
            env.close()
        except Exception as close_err:
            log.debug(f"[{game_id} Run {run_index}] Env close failed: {close_err}")


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Run parallel agent evaluation with reruns.")
    parser.add_argument("--agent", required=True, choices=list(AVAILABLE_AGENTS.keys()), help="The name of the agent to evaluate.")
    parser.add_argument("--suite", required=True, choices=list(EVALUATION_GAMES.keys()), help="The evaluation suite to run.")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum actions per game run before timeout.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of times to run each game.") 
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of parallel workers.") 
    parser.add_argument("--agent-model", dest="agent_model_override", default="gpt-5-nano", help="Override the agent's default model (e.g., 'gpt-5-nano').")
    parser.add_argument("--agent-reasoning-effort", dest="agent_reasoning_effort", default="low", help="Override the agent's default reasoning effort.")
    parser.add_argument("--include-text-diff", dest="include_text_diff", default="true", help="Value for INCLUDE_TEXT_DIFF.")
    parser.add_argument("--context-length-limit", dest="context_length_limit", type=int, default=-1, help="Value for CONTEXT_LENGTH_LIMIT.")
    parser.add_argument("--crop-border", dest="crop_border", type=int, default=0, help="Remove outer N pixels from 16x16 grid (e.g., 2 for scoring numbers).")
    parser.add_argument("--downsample-images", dest="downsample_images", default="true", help="Value for DOWNSAMPLE_IMAGES.")
    parser.add_argument("--image-detail-level", dest="image_detail_level", default="low", help="Value for IMAGE_DETAIL_LEVEL.")
    parser.add_argument("--image-pixels-per-cell", dest="image_pixels_per_cell", type=int, default=24, help="Value for IMAGE_PIXELS_PER_CELL.")
    parser.add_argument("--arcgame-general-prompts", dest="arcgame_general_prompts", default="0", help="Value for ARCGAME_GENERAL_PROMPTS.")
    # New representation config arguments
    parser.add_argument("--grid-format", dest="grid_format", default="integers_spaced",
        choices=["integers_spaced", "integers_compact", "integers_tuple", "hex", "symbolic", "ascii"],
        help="Grid representation format (default: integers_spaced for backward compatibility)")
    parser.add_argument("--input-mode", dest="input_mode", default="text_only",
        choices=["text_only", "image_only", "text_and_image"],
        help="Input modality for agents (default: text_only)")
    parser.add_argument("--no-downsample", dest="no_downsample", action="store_true",
        help="Use full 64x64 grid instead of 16x16 downsampled")
    parser.add_argument("--use-general-prompts", dest="use_general_prompts", action="store_true",
        help="Use general learning prompts instead of game-specific prompts")
    args = parser.parse_args()

    agent_name_cli = args.agent 
    agent_class = AVAILABLE_AGENTS[agent_name_cli]
    game_ids = EVALUATION_GAMES[args.suite]
    num_runs = args.num_runs
    max_workers = args.max_workers
    
    log.info(f"Agent: '{agent_name_cli}', Suite: '{args.suite}', Games: {len(game_ids)}, Runs per game: {num_runs}, Max workers: {max_workers}, Max Actions: {args.max_actions}")

    api_key = os.getenv("ARC_API_KEY", "") 
    if not api_key:
        log.error("ARC_API_KEY environment variable not found. Please set it in your .env file.")
        sys.exit(1) 
    
    # --- Setup Results Files (with new naming convention) ---
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") 


    #  Get all tags for filename AND for passing to threads 
    agent_tags = [agent_name_cli] + _get_agent_tags(agent_name_cli, args)
    agent_name_with_variant = "-".join(agent_tags)

    eval_tags = [f"eval-{agent_name_with_variant}", args.suite, f"runs-{num_runs}", f"workers-{max_workers}", f"max_actions-{args.max_actions}"]

    rollout_model = args.agent_model_override
    rollout_reasoning_effort = args.agent_reasoning_effort
    rollout_engine = _build_rollout_engine(rollout_model, reasoning_effort=rollout_reasoning_effort)
    
    # Build comprehensive filename (date first for natural sorting)
    base_dirname = f"{timestamp}_{agent_name_with_variant}_{args.suite}_runs{num_runs}_max{args.max_actions}"

    # Create a folder for this evaluation run
    run_dir = results_dir / base_dirname
    run_dir.mkdir(exist_ok=True)

    results_filepath_jsonl = run_dir / "results.jsonl"
    results_filepath_txt = run_dir / "summary.txt"
    prompts_log_dir = run_dir / "prompts"
    prompts_log_dir.mkdir(exist_ok=True)
    file_lock = threading.Lock()
    log.info(f"Detailed results (JSONL): {results_filepath_jsonl}")
    log.info(f"Summary report (TXT): {results_filepath_txt}")
    log.info(f"Prompts logs directory: {prompts_log_dir}")

    overall_start_time = time.time()
    game_metrics_objects_list: List[GameMetrics] = []

    try:
        # Create Task List
        tasks_to_run = [
            (game_id, run_idx) 
            for run_idx in range(1, num_runs + 1) 
            for game_id in game_ids
        ]
        total_tasks = len(tasks_to_run)
        log.info(f"Total evaluation tasks to run: {total_tasks}")

        # Execute in Parallel
        completed_tasks = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    run_evaluation_task,
                    args,
                    game_id,
                    run_index,
                    agent_class,
                    agent_name_cli, # Pass the base name
                    args.max_actions,
                    agent_name_with_variant, # Pass the full name
                    eval_tags,
                    rollout_engine,
                    prompts_log_dir,
                ): (game_id, run_index)
                for game_id, run_index in tasks_to_run
            }

            for future in as_completed(future_to_task):
                game_id, run_index = future_to_task[future]
                try:
                    result_metrics: GameMetrics = future.result()
                    # Store the object for final stats
                    game_metrics_objects_list.append(result_metrics)

                    # Print replay URL immediately after run completes
                    if result_metrics.replay_url:
                        print(f"[{game_id} Run {run_index}] Replay: {result_metrics.replay_url}")

                    # Convert to dict for JSONL saving
                    metrics_dict = dataclasses.asdict(result_metrics)
                    
                    # Incremental Saving to JSONL
                    with file_lock:
                        with open(results_filepath_jsonl, "a", encoding="utf-8") as f:
                            # Convert nested metrics properly for JSON
                            metrics_dict_serializable = deepcopy(metrics_dict)
                            metrics_dict_serializable['level_metrics'] = {
                                str(k): dataclasses.asdict(v) 
                                for k, v in result_metrics.level_metrics.items()
                            }
                            # Convert LevelMetrics.attempts
                            for k_lm, v_lm in metrics_dict_serializable['level_metrics'].items():
                                v_lm['attempts'] = [dataclasses.asdict(att) for att in result_metrics.level_metrics[int(k_lm)].attempts]
                                # Pop derived properties, they will be recalculated on load
                                v_lm.pop('total_actions', None)
                                v_lm.pop('total_game_overs', None)
                                v_lm.pop('total_state_changes', None)
                                v_lm.pop('actions_in_successful_attempt', None)
                                v_lm.pop('state_change_percentage', None)

                            metrics_dict_serializable['start_time_iso'] = datetime.fromtimestamp(metrics_dict['start_time'], timezone.utc).isoformat()
                            metrics_dict_serializable['end_time_iso'] = datetime.fromtimestamp(metrics_dict['end_time'], timezone.utc).isoformat()
                            f.write(json.dumps(metrics_dict_serializable) + "\n")
                            
                    completed_tasks += 1
                    log.info(f"Progress: {completed_tasks}/{total_tasks} tasks completed.")

                except Exception as exc:
                    log.error(f"Task {game_id} (Run {run_index}) generated an exception: {exc}", exc_info=True)
                    error_metric_obj = GameMetrics( 
                        game_id=game_id, agent_name=agent_name_with_variant, run_index=run_index, 
                        status="ERROR", start_time=time.time(),
                        error_message=str(exc) # <-- STORE THE EXCEPTION TEXT
                    )
                    error_metric_obj.end_time = time.time()
                    error_metric_obj.run_duration_seconds = error_metric_obj.end_time - error_metric_obj.start_time 
                    
                    # Store object for final stats
                    game_metrics_objects_list.append(error_metric_obj)
                    
                    err_dict = dataclasses.asdict(error_metric_obj)

                    with file_lock:
                        with open(results_filepath_jsonl, "a", encoding="utf-8") as f:
                            err_dict['start_time_iso'] = datetime.fromtimestamp(err_dict['start_time'], timezone.utc).isoformat()
                            err_dict['end_time_iso'] = datetime.fromtimestamp(err_dict['end_time'], timezone.utc).isoformat()
                            f.write(json.dumps(err_dict) + "\n")
                            
                    completed_tasks += 1 
                    log.info(f"Progress: {completed_tasks}/{total_tasks} tasks completed (including errors).")

    except KeyboardInterrupt:
        log.warning("Keyboard interrupt. Shutting down. Results saved so far are in JSONL.")
    except Exception as e:
        log.error(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        # Generate Final Reports
        overall_end_time = time.time()
        total_duration = overall_end_time - overall_start_time
        log.info(f"Total evaluation time: {total_duration:.2f} seconds.")
        
        if game_metrics_objects_list: 
            log.info("Calculating final statistics...")
            
            # Pass the list of GameMetrics objects
            game_stats, overall_summary = calculate_stats(game_metrics_objects_list) 

            log.info(f"Generating console report...")
            try:
                # Pass the list of GameMetrics objects
                generate_console_report(game_metrics_objects_list, args.suite, agent_name_with_variant, num_runs) 
            except Exception as report_err:
                log.error(f"Failed to generate console report: {report_err}", exc_info=True)

            log.info(f"Saving summary report to: {results_filepath_txt}")
            try:
                save_summary_report(
                    str(results_filepath_txt), 
                    game_stats, overall_summary, game_metrics_objects_list, # Pass objects
                    agent_name_with_variant, args.suite, num_runs
                )
            except Exception as save_err:
                log.error(f"Failed to save summary text report: {save_err}", exc_info=True)
        else: 
            log.error("No evaluation results were collected. Cannot generate reports.")
            print("\n--- Evaluation Summary (No Results) ---")
            print(f"Agent: {agent_name_with_variant}")
            print(f"Suite: {args.suite}")
            print(f"Total Runs Attempted: 0")
            print(f"Total Duration: {total_duration:.2f}s")
            print("---------------------------------------")

    log.info("Evaluation script finished.")

if __name__ == "__main__":
    main()
