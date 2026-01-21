#!/bin/bash
#SBATCH --job-name=baseline_image
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

mkdir -p logs
source ~/.bashrc
conda activate arcenv

echo "Job started at: $(date)"
echo "Running: Image baseline"
export PYTHONUNBUFFERED=1

python -m lucidgym.evaluation.harness \
    --agent basic_obs_action_agent_rolling_context \
    --suite debug_suite \
    --num_runs 4 \
    --max_actions 300 \
    --max_workers 4 \
    --agent-model gpt-5-mini \
     \
     \
     \
    --use-as66-prompts \
    --input-mode image_only \
    --wandb \
    --weave

echo "Job finished at: $(date)"
