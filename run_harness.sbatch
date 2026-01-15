#!/bin/bash
#SBATCH --job-name=lucidgym_eval
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate arcenv

# Print environment info for debugging
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# Set environment variables
export PYTHONUNBUFFERED=1

# Run the harness
python -m lucidgym.evaluation.harness \
    --agent basic_obs_action_agent_rolling_context \
    --suite debug_suite \
    --num_runs 1 \
    --max_actions 50 \
    --max_workers 1 \
    --agent-model openai/gpt-5-nano \
    --agent-reasoning-effort high

echo "Job finished at: $(date)"
