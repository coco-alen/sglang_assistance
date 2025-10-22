# python -m sglang.bench_serving \
#     --backend sglang-oai \
#     --port 33212 \
#     --dataset-name random \
#     --random-input-len 8192 \
#     --random-output-len 128 \
#     --random-range-ratio 0.5 \
#     --max-concurrency 64 \
#     --num-prompts 500

python -m sglang.bench_serving --backend sglang-oai-chat --dataset-name random \
    --random-input-len 280 \
    --random-output-len 270 \
    --random-range-ratio 1 \
    --host 0.0.0.0 \
    --port 23333 \
    --apply-chat-template \
    --warmup-requests 5 \
    --num-prompts 500 \
    --max-concurrency 256


# python3 -m sglang.bench_one_batch_server \
#     --dataset-name random \
#     --model Qwen/Qwen3-1.7B-FP8 \
#     --base-url http://0.0.0.0:23333 \
#     --batch-size 256 \
#     --input-len 280 \
#     --output-len 270