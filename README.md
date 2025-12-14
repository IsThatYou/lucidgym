LucidGym
========

LucidGym hosts add-on agents, environments, and workflows that slot into the upstream
`rllm` stack. All agents now use `rllm.agents.agent.BaseAgent` with zero legacy dependencies.

## Quick Start: Running Evaluations

### Run AS66 Guided Agent on AS66 Game

```bash
# Ensure API keys are set in .env file
# Then run:
python3.12 -m lucidgym.evaluation.harness \
    --agent as66_guided_agent \
    --suite debug_suite \
    --num_runs 3 \
    --max_actions 200

# Results saved to evaluation_results/
```

### Available Agents (All BaseAgent)

- `as66_guided_agent` - 16x16 text-based 
- `as66_guided_agent_64` - 64x64 full resolution
- `as66_memory_agent` - Memory + hypothesis tracking
- `as66_visual_memory_agent` - Multimodal (images + text)

Editable install
----------------

You should reference the rllm guide to install everything first. Don't worry about textarena.


```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# got to rllm and install their dependencies
```
Then:
```bash
pip install -e rllm          # installs the core rllm dependency
pip install -e .             # installs the lucidgym package itself
```


To use
----------------
After this one-time setup you can run `examples/arcagi3/eval.sh` from any
working directory and `python` will still be able to resolve `lucidgym`.

You can run `examples/arcagi3/train.sh` to start training RL. Right now it will run but won't train successfully. Still debugging.


To implement
----------------
I would recommend to look at `lucidgym/agents/arcagi3_agent.py` and  `lucidgym/environments/arcagi3/` to develop our agents. I only port necessary files from the original arc-agi-3 repo and you can add more support files if you find core functionalities missing. The logic is for rllm so we can easily train an agent, and I separte the logic of agent and enviroment (not done in arc-agi-3 repo).
