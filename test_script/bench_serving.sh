

# nsys start && \
# img_nums=(80 60 40 20 10)
# numbers=(100 100 200 300 500)

# while true; do
#     for i in "${!img_nums[@]}"; do
#         img_num=${img_nums[$i]}
#         number=${numbers[$i]}
#         evalscope perf \
#             --model OpenGVLab/InternVL3_5-8B-Flash \
#             --url http://0.0.0.0:23333/v1/chat/completions \
#             --parallel 64 \
#             --number $number \
#             --api openai \
#             --dataset random_vl \
#             --min-tokens 64 \
#             --max-tokens 64 \
#             --prefix-length 0 \
#             --min-prompt-length 350 \
#             --max-prompt-length 350 \
#             --image-width 448 \
#             --image-height 448 \
#             --image-format RGB \
#             --image-num $img_num \
#             --tokenizer-path OpenGVLab/InternVL3_5-8B-Flash
#     done
# done
# && nsys stop

evalscope perf \
    --model OpenGVLab/InternVL3_5-8B-Flash \
    --url http://0.0.0.0:23333/v1/chat/completions \
    --parallel 64 \
    --number 500 \
    --api openai \
    --dataset random_vl \
    --min-tokens 64 \
    --max-tokens 64 \
    --prefix-length 0 \
    --min-prompt-length 350 \
    --max-prompt-length 350 \
    --image-width 448 \
    --image-height 448 \
    --image-format RGB \
    --image-num 10 \
    --tokenizer-path OpenGVLab/InternVL3_5-8B-Flash

# evalscope perf \
#     --parallel 64 \
#     --number 500 \
#     --model OpenGVLab/InternVL3_5-8B-Flash \
#     --url http://0.0.0.0:23333/v1/chat/completions \
#     --api openai \
#     --dataset random \
#     --max-tokens 64 \
#     --min-tokens 64 \
#     --prefix-length 0 \
#     --min-prompt-length 1016 \
#     --max-prompt-length 1016 \
#     --tokenizer-path OpenGVLab/InternVL3_5-8B-Flash \
#     --extra-args '{"ignore_eos": true}'


# evalscope perf \
#     --parallel 384 \
#     --number 1500 \
#     --model Qwen/Qwen3-1.7B-FP8 \
#     --url http://0.0.0.0:23333/v1/chat/completions \
#     --api openai \
#     --dataset random \
#     --max-tokens 270 \
#     --min-tokens 270 \
#     --prefix-length 0 \
#     --min-prompt-length 280 \
#     --max-prompt-length 280 \
#     --tokenizer-path Qwen/Qwen3-1.7B-FP8 \
#     --extra-args '{"ignore_eos": true}' \
#     --stream



# evalscope perf \
#     --parallel 1 \
#     --number 500 \
#     --model gpt-oss \
#     --api-key amsk_live_Cg2-dJW8_Cg2-dJW8dwrFO6b3uCNuau10kK1JZyffBtFpcQqaAig \
#     --url https://app.eigenai.com/api/v1/chat/completions \
#     --api openai \
#     --dataset longalpaca \
#     --max-tokens 1000 \
#     --min-tokens 1000 \
#     --prefix-length 0 \
#     --min-prompt-length 1000 \
#     --max-prompt-length 1000 \
#     --tokenizer-path openai-mirror/gpt-oss-120b \
#     --extra-args '{"ignore_eos": true}'