
# export TORCHINDUCTOR_CACHE_DIR=~/.triton

# CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server \
#     --model-path OpenGVLab/InternVL3-1B \
#     --chat-template internvl-2-5 \
#     --mem-fraction-static 0.5 \
#     --disaggregation-mode decode \
#     --port 30001 \
#     --disaggregation-transfer-backend nixl 


export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export HOST_IP=0.0.0.0
export PORT=23335
export SGLANG_VLM_CACHE_SIZE_MB=40960
export TENSOR_PARALLEL_SIZE=1
export CHUNKED_PREFILL_SIZE=81920
export MAX_RUNNING_REQUESTS=256
export MEM_FRACTION_STATIC=0.85
export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=256
export SGLANG_EMBEDDING_CACHE_BLOCK_SIZE=16384

# Language: text generation
# Qwen2.5-VL: "architectures": ["Qwen2ForCausalLM"]
CUDA_VISIBLE_DEVICES=6 python3 -m sglang.launch_server --model-path ${MODEL_PATH} --enable-torch-compile --disable-radix-cache \
        --host $HOST_IP --port $PORT --trust-remote-code --tp-size ${TENSOR_PARALLEL_SIZE} --served-model-name "qwen3-vl" \
        --enable-cache-report --log-level info --max-running-requests ${MAX_RUNNING_REQUESTS} --json-model-override-args '{"architectures": ["Qwen3MoeForCausalLM"]}' \
        --mem-fraction-static ${MEM_FRACTION_STATIC} --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} --attention-backend fa3 \
        --disaggregation-mode language