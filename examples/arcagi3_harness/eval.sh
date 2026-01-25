# Ensure API keys are set in .env file
source ../../.venv/bin/activate

export OPENAI_API_KEY=$OPENAI_ARCAGI_API_KEY

python -m lucidgym.evaluation.harness \
    --agent arcagi3_agent \
    --suite debug_suite \
    --num_runs 3 \
    --max_actions 200 \
    --agent-model gpt-5-mini \
    --agent-reasoning-effort medium \
    --input-mode text_only \
    --grid-format ascii \
    # --no-downsample 

# Results saved to evaluation_results/