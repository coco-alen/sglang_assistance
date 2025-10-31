

# evalscope perf \
#     --model OpenGVLab/InternVL3_5-8B-Flash \
#     --url http://0.0.0.0:23333/v1/chat/completions \
#     --parallel 64 \
#     --number 500 \
#     --api openai \
#     --dataset random_vl \
#     --min-tokens 64 \
#     --max-tokens 64 \
#     --prefix-length 0 \
#     --min-prompt-length 350 \
#     --max-prompt-length 350 \
#     --image-width 448 \
#     --image-height 448 \
#     --image-format RGB \
#     --image-num 1 \
#     --tokenizer-path OpenGVLab/InternVL3_5-8B-Flash


evalscope perf \
    --parallel 64 \
    --number 500 \
    --model OpenGVLab/InternVL3_5-8B-Flash \
    --url http://0.0.0.0:23333/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 64 \
    --min-tokens 64 \
    --prefix-length 0 \
    --min-prompt-length 600 \
    --max-prompt-length 600 \
    --tokenizer-path OpenGVLab/InternVL3_5-8B-Flash \
    --extra-args '{"ignore_eos": true}'