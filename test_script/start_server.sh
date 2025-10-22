

export TORCHINDUCTOR_CACHE_DIR=~/.triton
CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server \
   --model-path OpenGVLab/InternVL3_5-8B-Flash \
   --chat-template internvl-2-5 \
   --port 23333 \
   --mem-fraction-static 0.7 \
   --mm-attention-backend fa3 \
   --attention-backend fa3 \
   --log-level info \
   --quantization fp8 \
   --enable-torch-compile

# export TORCHINDUCTOR_CACHE_DIR=~/.triton

# export SGLANG_TORCH_PROFILER_DIR="/data/yipin/project/sglang-dev/sglang_assistance"
# CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
#    --model-path Qwen/Qwen3-1.7B-FP8 \
#    --mem-fraction-static 0.5 \
#    --quantization fp8 \
#    --attention-backend fa3 \
#    --host 0.0.0.0 \
#    --port 23333 \
#    --chat-template ./qwen3_nonthinking.jinja \
   # --cuda-graph-max-bs 384
   # --stream-output
   # --disable-radix-cache

# CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
#    --model-path Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
#    --port 23334 \
#    --mem-fraction-static 0.5 \
#    --mm-attention-backend fa3 \
#    --attention-backend fa3 \
#    --log-level info \
#    --quantization fp8 \
#    --enable-torch-compile


# CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
#    --model-path Qwen/Qwen3-VL-8B-Instruct-FP8 \
#    --port 23334 \
#    --mem-fraction-static 0.7 \
#    --mm-attention-backend fa3 \
#    --attention-backend fa3 \
#    --log-level info \
#    --quantization fp8 \
#    --enable-torch-compile
