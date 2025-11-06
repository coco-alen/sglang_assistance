# export TORCHINDUCTOR_CACHE_DIR=~/.triton

# CUDA_VISIBLE_DEVICES=6 python -m sglang.launch_server \
#     --model-path OpenGVLab/InternVL3-1B \
#     --chat-template internvl-2-5 \
#     --mem-fraction-static 0.5 \
#     --disaggregation-mode prefill \
#     --port 30000 \
#     --disaggregation-transfer-backend nixl 

export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export HOST_IP=0.0.0.0
export PORT=23334
export SGLANG_VLM_CACHE_SIZE_MB=40960
export TENSOR_PARALLEL_SIZE=1
export CHUNKED_PREFILL_SIZE=81920
export MAX_RUNNING_REQUESTS=256
export MEM_FRACTION_STATIC=0.85
export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=256
export SGLANG_EMBEDDING_CACHE_BLOCK_SIZE=16384
# Encode: vision encoding
CUDA_VISIBLE_DEVICES=6 python3 -m sglang.launch_server --model-path ${MODEL_PATH} --enable-torch-compile --max-prefill-tokens $CHUNKED_PREFILL_SIZE \
        --host $HOST_IP --port $PORT --trust-remote-code --tp-size ${TENSOR_PARALLEL_SIZE} --mem-fraction-static ${MEM_FRACTION_STATIC} \
        --enable-cache-report --log-level info --max-running-requests ${MAX_RUNNING_REQUESTS} \
        --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} --attention-backend fa3 --json-model-override-args '{"is_multimodal_embedding": true}' \
        --mm-attention-backend fa3 --disaggregation-mode encode