#!/bin/bash

# ======================
# Default configuration
# ======================
CONTAINER_NAME="eigen_wyze"

MODEL="OpenGVLab/InternVL3_5-8B-Flash"
PORT="23333"
PARALLEL=64
NUMBER=500
MAX_TOKENS=64
PROMPT_LENGTH=350
IMAGE_NUM=10

# ======================
# Help info
# ======================
show_help() {
    echo "Usage: $0 [options] [extra evalscope args]"
    echo ""
    echo "Options:"
    echo "  -c, --container NAME      Container name (default: eigen_wyze)"
    echo "  -m, --model PATH          Model path"
    echo "  -p, --port PORT           Server port"
    echo "  -P, --parallel N          Parallel requests"
    echo "  -n, --number N            Total requests"
    echo "  -t, --max-tokens N        Max tokens"
    echo "  -l, --prompt-length N     Prompt length"
    echo "  -i, --image-num N         Image number"
    echo "  -h, --help                Show help"
    echo ""
    echo "All other arguments will be passed directly to eigen-bench."
    echo ""
    echo "Example:"
    echo "  $0 -n 80 -i 80"
}

# ======================
# Parse arguments
# ======================
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--container)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -P|--parallel)
            PARALLEL="$2"
            shift 2
            ;;
        -n|--number)
            NUMBER="$2"
            shift 2
            ;;
        -t|--max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -l|--prompt-length)
            PROMPT_LENGTH="$2"
            shift 2
            ;;
        -i|--image-num)
            IMAGE_NUM="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ======================
# Display configuration
# ======================
echo "=== Run eigen-bench ==="
echo "Container:      $CONTAINER_NAME"
echo "Model:          $MODEL"
echo "Port:           $PORT"
echo "Parallel:       $PARALLEL"
echo "Number:         $NUMBER"
echo "Max Tokens:     $MAX_TOKENS"
echo "Prompt Length:  $PROMPT_LENGTH"
echo "Image Num:      $IMAGE_NUM"
echo "Extra Args:     ${EXTRA_ARGS[*]}"
echo "======================="
echo ""

# ======================
# Compose bench command
# ======================
BENCH_CMD=(
    ./eigen-bench
    --model "$MODEL"
    --port "$PORT"
    --parallel "$PARALLEL"
    --number "$NUMBER"
    --max-tokens "$MAX_TOKENS"
    --prompt-length "$PROMPT_LENGTH"
    --image-num "$IMAGE_NUM"
    "${EXTRA_ARGS[@]}"
)

echo "Executing inside container:"
printf '  %q ' "${BENCH_CMD[@]}"
echo -e "\n"

# ======================
# Run bench in container
# ======================
docker exec -it \
    "$CONTAINER_NAME" \
    bash -c "$(printf '%q ' "${BENCH_CMD[@]}")"