# Ensure API keys are set in .env file
source ../../.venv/bin/activate

export OPENAI_API_KEY=$OPENAI_ARCAGI_API_KEY

export VLLM_ATTENTION_BACKEND=FLASH_ATTN # FLASH_ATTN, FLASHINFER, XFormers
export VLLM_LOGGING_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
PORT=8002
TENSOR_PARALLEL_SIZE=8
GPU_MEMORY_UTILIZATION=0.8
MAX_MODEL_LEN=262144 # Adjust based on your VRAM; Qwen usually supports long context
# MAX_MODEL_LEN=32768

echo "Starting vLLM server for $MODEL_ID..."
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Port: $PORT"

# Run the server
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_ID \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    --port $PORT \
    --host 0.0.0.0 \
    --dtype auto \
    --served-model-name "$MODEL_ID" &

VLLM_PID=$!
trap "echo 'Stopping vLLM (PID ${VLLM_PID})'; kill ${VLLM_PID} || true" EXIT

echo "[`date`] Waiting for vLLM to become ready on localhost:8002..."
for i in {1..60}; do
  if curl -fsS "http://127.0.0.1:8002/v1/models" >/dev/null 2>&1; then
    echo "[`date`] vLLM is ready."
    break
  fi
  sleep 5
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "vLLM server exited unexpectedly. Check vllm_server.log"
    exit 1
  fi
  if [ $i -eq 60 ]; then
    echo "Timed out waiting for vLLM to start."
    exit 1
  fi
done

export VLLM_URL="http://localhost:$PORT/v1"
python -m lucidgym.evaluation.harness \
    --agent basic_obs_action_agent \
    --suite ls20_suite \
    --num_runs 1 \
    --max_actions 200 \
    --agent-model $MODEL_ID \
    --input-mode text_only \
    --grid-format ascii \
    --agent-reasoning-effort medium \
    --operation-mode online \
    --wandb --wandb-project arcagi3-harness --weave --weave-project arcagi3-harness \


# Results saved to evaluation_results/