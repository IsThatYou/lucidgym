#!/usr/bin/env bash

source ../../.venv/bin/activate

export TOGETHER_API_KEY=$Fede_TOGETHER_API

# python run_arc_eval.py --model "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
# python run_arc_eval.py --model "Qwen/Qwen3-235B-A22B-Thinking-2507"
python run_arc_eval.py --model "gpt-5-mini"
