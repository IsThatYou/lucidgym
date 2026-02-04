#!/bin/bash
# Usage: ./run_harness_configurable.sh [agent] [suite] [num_runs] [max_workers] [max_actions] [model] [reasoning] [block_size] [ds_method] [click_only]
# Example: ./run_harness_configurable.sh basic_obs_action_agent_rolling_context debug_suite 3 3 150 gpt-5 low 4 mode 0
# Example (click only): ./run_harness_configurable.sh basic_obs_action_agent_rolling_context debug_suite 3 3 150 gpt-5 low 4 mode 1

# Default values
AGENT=${1:-basic_obs_action_agent_rolling_context}
SUITE=${2:-debug_suite}
NUM_RUNS=${3:-3}
MAX_WORKERS=${4:-10}
MAX_ACTIONS=${5:-150}
MODEL=${6:-gpt-5}
REASONING=${7:-high}
BLOCK_SIZE=${8:-4}        # 4=16x16, 2=32x32, 1=64x64
DS_METHOD=${9:-mean}      # mode or mean
CLICK_ONLY=${10:-1}       # 1=click only, 0=all actions

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
echo "  Click Only: $CLICK_ONLY"
echo "=========================================="

# Set environment variables
export PYTHONUNBUFFERED=1

# Build click-only flag
CLICK_ONLY_FLAG=""
if [ "$CLICK_ONLY" = "1" ]; then
    CLICK_ONLY_FLAG="--click-only"
fi

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
    --weave-project lucidgym-eval \
    $CLICK_ONLY_FLAG

EXIT_CODE=$?

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
