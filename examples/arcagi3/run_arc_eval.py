#!/usr/bin/env python3
"""
Reference runner that wires the ARC-AGI-3 env/agent into a mini loop.

By default the script replays the mock session shipped with LucidGym so it can
run without network access. Pass ``--mock-session`` to point at a different
recording or ``--no-mock`` to target a live ARC endpoint.
"""
from __future__ import annotations

import os
import argparse
import json
from collections import deque
from pathlib import Path
import asyncio

from lucidgym.agents.arcagi3_agent import ArcAgi3Agent
from lucidgym.environments.arcagi3.arcagi3_env import ArcAgi3Env
from lucidgym.environments.arcagi3.mocks import StaticArcTransport
from lucidgym.registry import register_lucidgym_components
from rllm.engine.rollout import OpenAIEngine
from transformers import AutoTokenizer

def _build_passthrough_queue(session_data: dict) -> deque[dict]:
    queue: deque[dict] = deque()
    for step in session_data.get("steps", []):
        payload = {"action": step["action"]}
        payload.update(step.get("payload") or {})
        queue.append(payload)
    return queue


def _passthrough_fn_builder(queue: deque[dict]):
    def _fn(observation: dict, trajectory) -> dict:
        if queue:
            return queue.popleft()
        return {"action": "ACTION1"}

    return _fn

def setup_rollout_engine(model) -> OpenAIEngine:
    # Provider selection
    together_api_key = os.getenv("TOGETHER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    tokenizer = None
    if model.startswith("gpt-"):
        api_key = openai_api_key
        base_url = "https://api.openai.com/v1"
        model_name = model
    else:
        api_key = together_api_key
        base_url = "https://api.together.xyz/v1"
        model_name = model
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B-Instruct-2507")

    sampling_params = {
                "temperature": 1,
                "top_p": 0.95,
                "max_tokens": 2048,
            }
    return OpenAIEngine(
        model=model_name,
        tokenizer=tokenizer,
        base_url=base_url,
        api_key=api_key,
        max_prompt_length=65536*2,
        sampling_params=sampling_params,
    )

async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single ARC-AGI-3 episode.")
    parser.add_argument("--game-id", default="as66-821a4dcad9c2")
    parser.add_argument("--root-url", default="https://three.arcprize.org")
    parser.add_argument("--mock-session", default="examples/arcagi3/mock_session.json", help="Path to a recorded session JSON file.")
    parser.add_argument("--no-mock", action="store_true", help="Disable the local mock transport and hit the live ARC endpoint.")
    parser.add_argument("--model", default="gpt-5.1")
    args = parser.parse_args()

    register_lucidgym_components()

    transport = None
    passthrough_fn = None
    args.no_mock = True
    if not args.no_mock:
        session_data = json.loads(Path(args.mock_session).read_text())
        transport = StaticArcTransport(session_data)
        action_queue = _build_passthrough_queue(session_data)
        passthrough_fn = _passthrough_fn_builder(action_queue)

    tags = [f"eval-debug", "arcagi_agent"]
    env = ArcAgi3Env(
        game_id=args.game_id,
        root_url=args.root_url,
        transport=transport,
        include_grid_ascii=True,
        include_raw_frame=True,
        max_actions=10,
        tags=tags,
    )
    agent = ArcAgi3Agent(mode="passthrough" if passthrough_fn else "llm", passthrough_fn=passthrough_fn)
    rollout_engine = setup_rollout_engine(args.model)

    observation, info = env.reset()
    agent.reset()
    done = False
    reward = 0.0
    total_reward = 0.0

    print("Starting episode...")
    print(f"Initial observation keys: {list(observation.keys()) if observation else 'None'}")
    print(f"card_id: {observation.get('card_id', 'N/A')}")
    # print(f"Initial frame:\n{observation.get('frame', 'N/A')}")
    print(f"Info: {info}")
    tools = agent.build_tools()
    try:
        while not done:
            agent.update_from_env(observation, reward, done, info)
            prompt = rollout_engine.chat_parser.parse(agent.chat_completions, add_generation_prompt=True, is_first_msg=True, accumulate_reasoning=True)
            output = await rollout_engine.get_model_response(agent.chat_completions, accumulate_reasoning=True, tools=tools)
            wrapped_action = agent.update_from_model(output.text)
            observation, reward, done, info = env.step(wrapped_action)
            total_reward += reward
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        env.close()
    arc_info = info.get("arc", {}) if info else {}
    scorecard_summary = getattr(env, "_scorecard_cache", {})
    print(
        f"Episode complete. Final state={arc_info.get('state')} score={scorecard_summary.get('scores')} reward={total_reward}"
    )


if __name__ == "__main__":
    asyncio.run(main())
