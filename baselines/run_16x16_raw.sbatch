#!/bin/bash
#SBATCH --job-name=baseline_16x16_raw
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
echo "Running: 16x16 raw integers baseline"
export PYTHONUNBUFFERED=1

python -m lucidgym.evaluation.harness \
    --agent basic_obs_action_agent_rolling_context \
    --suite debug_suite \
    --num_runs 4 \
    --max_actions 300 \
    --max_workers 4 \
    --agent-model gpt-5.2 \
    --agent-reasoning-effort high \
    --use-responses-api \
    --crop-border 2 \
    --use-as66-prompts \
    --grid-format integers_spaced \
    --input-mode text_only \
    --wandb \
    --weave

echo "Job finished at: $(date)"
