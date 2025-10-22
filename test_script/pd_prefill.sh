export TORCHINDUCTOR_CACHE_DIR=~/.triton

CUDA_VISIBLE_DEVICES=6 python -m sglang.launch_server \
    --model-path OpenGVLab/InternVL3-1B \
    --chat-template internvl-2-5 \
    --mem-fraction-static 0.5 \
    --disaggregation-mode prefill \
    --port 30000 \
    --disaggregation-transfer-backend nixl 