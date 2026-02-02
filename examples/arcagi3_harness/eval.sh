# Ensure API keys are set in .env file
source ../../.venv/bin/activate

export OPENAI_API_KEY=$OPENAI_ARCAGI_API_KEY

python -m lucidgym.evaluation.harness \
    --agent arcagi3_agent \
    --suite ls20_suite \
    --num_runs 1 \
    --max_actions 50 \
    --agent-model gpt-5-mini \
    --input-mode text_only \
    --grid-format ascii \
    --agent-reasoning-effort medium \
    --operation-mode online \
    --wandb --wandb-project arcagi3-harness --weave --weave-project arcagi3-harness

# Results saved to evaluation_results/