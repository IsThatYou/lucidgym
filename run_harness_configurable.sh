#!/bin/bash
# Usage: ./run_harness_configurable.sh [agent] [suite] [num_runs] [max_workers] [max_actions] [model] [reasoning] [block_size] [ds_method]
# Example: ./run_harness_configurable.sh basic_obs_action_agent_rolling_context debug_suite 3 3 150 openai/gpt-5 low 4 mode

# Default values
AGENT=${1:-basic_obs_action_agent_rolling_context}
SUITE=${2:-debug_suite}
NUM_RUNS=${3:-1}
MAX_WORKERS=${4:-1}
MAX_ACTIONS=${5:-500}
MODEL=${6:-openai/gpt-5-nano}
REASONING=${7:-high}
BLOCK_SIZE=${8:-4}        # 4=16x16, 2=32x32, 1=64x64
DS_METHOD=${9:-mode}      # mode or mean

# Create logs directory if it doesn't exist
mkdir -p logs

# Print environment info for debugging
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "=========================================="
echo "Run Configuration:"
echo "  Agent: $AGENT"
echo "  Suite: $SUITE"
echo "  Num Runs: $NUM_RUNS"
echo "  Max Workers: $MAX_WORKERS"
echo "  Max Actions: $MAX_ACTIONS"
echo "  Model: $MODEL"
echo "  Reasoning Effort: $REASONING"
echo "  Block Size: $BLOCK_SIZE (4=16x16, 2=32x32, 1=64x64)"
echo "  Downsample Method: $DS_METHOD"
echo "=========================================="

# Set environment variables
export PYTHONUNBUFFERED=1

# Run the harness
python -m lucidgym.evaluation.harness \
    --agent "$AGENT" \
    --suite "$SUITE" \
    --num_runs "$NUM_RUNS" \
    --max_actions "$MAX_ACTIONS" \
    --max_workers "$MAX_WORKERS" \
    --agent-model "$MODEL" \
    --agent-reasoning-effort "$REASONING" \
    --downsample-block-size "$BLOCK_SIZE" \
    --downsample-method "$DS_METHOD" \
    --grid-format ascii \
    --wandb \
    --wandb-project lucidgym-evaluation \
    --weave \
    --weave-project lucidgym-eval

EXIT_CODE=$?

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
