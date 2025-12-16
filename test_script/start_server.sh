
export SGLANG_TORCH_PROFILER_DIR="/data/home/yipin/project/sglang-dev/sglang_assistance"
export TORCHINDUCTOR_CACHE_DIR=~/.triton



# CUDA_VISIBLE_DEVICES=3 \
# nsys profile --trace-fork-before-exec=true \
#   --cuda-graph-trace=node \
#   --capture-range=cudaProfilerApi \
#   --capture-range-end=stop \
#   -o layerwise_profile \
# nsys launch --pytorch=autograd-nvtx \

   # --model-path /data/home/yipin/project/sglang-dev/sglang_assistance/InterVL3_5-8B-Flash \

CUDA_VISIBLE_DEVICES=4 python -m sglang.launch_server \
   --model-path /sgl-workspace/sglang/sglang_assistance/InterVL3_5-8B-Flash \
   --chat-template internvl-2-5 \
   --port 23333 \
   --mem-fraction-static 0.8 \
   --mm-attention-backend fa3 \
   --attention-backend fa3 \
   --log-level info \
   --quantization fp8 \
   --enable-broadcast-mm-inputs-process
   # --enable-piecewise-cuda-graph \
   # --piecewise-cuda-graph-compiler eager \

   # --enable-torch-compile
   # --torch-compile-max-bs 64 \
   # --enable-piecewise-cuda-graph \
   # --piecewise-cuda-graph-compiler eager \

   # --enable-piecewise-cuda-graph

   # --enable-layerwise-nvtx-marker

# export SGLANG_TORCH_PROFILER_DIR="/data/home/yipin/project/sglang-dev/sglang_assistance"
# export TORCHINDUCTOR_CACHE_DIR=~/.triton
# CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server \
#    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
#    --port 23333 \
#    --mem-fraction-static 0.7 \
#    --mm-attention-backend fa3 \
#    --attention-backend fa3 \
#    --log-level info \
#    --quantization fp8 \
#    --enable-broadcast-mm-inputs-process \
#    --keep-mm-feature-on-device

# export TORCHINDUCTOR_CACHE_DIR=~/.triton

# export SGLANG_TORCH_PROFILER_DIR="/data/yipin/project/sglang-dev/sglang_assistance"
# CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
#    --model-path Qwen/Qwen3-1.7B \
#    --mem-fraction-static 0.8 \
#    --quantization fp8 \
#    --attention-backend fa3 \
#    --host 0.0.0.0 \
#    --port 23333 \
#    --chat-template ./qwen3_nonthinking.jinja \
#    --cuda-graph-max-bs 384 \
#    --enable-dynamic-batch-tokenizer \
#    --dynamic-batch-tokenizer-batch-size 384 \
#    --dynamic-batch-tokenizer-batch-timeout 0.05 \
#    --schedule-conservativeness 0.3 \
#    --max-prefill-tokens 327680 \
#    --num-continuous-decode-steps 16
   # --speculative-draft-model-path AngelSlim/Qwen3-1.7B_eagle3 \
   # --speculative-num-steps 3 \
   # --speculative-eagle-topk 1 \
   # --speculative-num-draft-tokens 4


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
