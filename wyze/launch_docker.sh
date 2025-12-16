#!/bin/bash

# ======================
# Default configuration
# ======================
IMAGE_NAME="wyze:251216"
CONTAINER_NAME="eigen_wyze"
SCRIPT_PATH="./eigen-serve"
CUDA_DEVICES="0"
MODEL_PATH="OpenGVLab/InternVL3_5-8B-Flash"
PORT="23333"
CACHE_DIR="/data/home/yipin/.cache/huggingface"

# ======================
# Help info
# ======================
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i, --image IMAGE        Docker image name"
    echo "  -g, --gpu DEVICES       CUDA devices (default: 0)"
    echo "  -m, --model PATH         Model path (default: OpenGVLab/InternVL3_5-8B)"
    echo "  -p, --port PORT          Port (default: 23333)"
    echo "  -c, --cache DIR          Huggingface cache directory (default: /cache/huggingface, should be absolute path)"
    echo "  -h, --help               Show help information"
    echo ""
    echo "All other arguments will be passed directly to the container command."
    echo ""
    echo "Example:"
    echo "  $0 -c 7 -m ./models/InternVL3_5-2B --decode-log-interval=1 --trust-remote-code"
}

# ======================
# Parse arguments
# ======================
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -g|--gpu)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -c|--cache)
            CACHE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --*=*)  # e.g. --decode-log-interval=1
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            echo "⚠️ Unknown or extra parameter: $1"
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ======================
# Display configuration
# ======================
echo "=== Launch Docker Container ==="
echo "Image:         $IMAGE_NAME"
echo "Container:     $CONTAINER_NAME"
echo "CUDA Devices:  $CUDA_DEVICES"
echo "Model Path:    $MODEL_PATH"
echo "Port:          $PORT"
echo "Cache Dir:     $CACHE_DIR"
echo "Extra Args:    ${EXTRA_ARGS[*]}"
echo "================================"
echo ""

# ======================
# Cleanup old container
# ======================
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

# ======================
# Prepare volumes
# ======================
DOCKER_CACHE_ARG="-v $CACHE_DIR:/root/.cache/huggingface"

if [[ "$MODEL_PATH" == OpenGVLab* ]]; then
    DOCKER_MODEL_ARG="$MODEL_PATH"
    DOCKER_VOLUME_ARG=""
else
    ABS_MODEL_PATH=$(readlink -f "$MODEL_PATH")
    CONTAINER_MODEL_PATH="/models/$(basename "$ABS_MODEL_PATH")"
    DOCKER_MODEL_ARG="$CONTAINER_MODEL_PATH"
    DOCKER_VOLUME_ARG="-v $ABS_MODEL_PATH:$CONTAINER_MODEL_PATH"
fi

# ======================
# Compose container command
# ======================
CMD=(
    "$SCRIPT_PATH"
    -c "$CUDA_DEVICES"
    -m "$DOCKER_MODEL_ARG"
    -p "$PORT"
    "${EXTRA_ARGS[@]}"
)

# ======================
# Run container
# ======================
echo "Executing inside container:"
printf '  %q ' "${CMD[@]}"
echo -e "\n"

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --network=host \
    $DOCKER_VOLUME_ARG \
    $DOCKER_CACHE_ARG \
    "$IMAGE_NAME" \
    bash -c "$(printf '%q ' "${CMD[@]}")"

# ======================
# Wait for server ready
# ======================
echo "⏳ Waiting for server to be ready on port $PORT ..."

MAX_RETRY=100      # max 100 retries
SLEEP_INTERVAL=5  # sleep 5 seconds between retries

READY=0
for ((i=1; i<=MAX_RETRY; i++)); do
    if docker exec "$CONTAINER_NAME" bash -c "nc -z localhost $PORT" >/dev/null 2>&1; then
        READY=1
        echo "✅ Server is ready (port $PORT is listening)"
        break
    fi
    echo "  [$i/$MAX_RETRY] Server not ready yet..."
    sleep $SLEEP_INTERVAL
done

if [[ "$READY" -ne 1 ]]; then
    echo "❌ Server did not become ready within timeout"
    exit 1
fi

# ======================
# Run eigen-bench inside container
# ======================
echo "🚀 Running eigen-bench with server health monitoring..."

docker exec "$CONTAINER_NAME" bash -c '
set -e

PORT='"$PORT"'
BENCH_CMD="./eigen-bench -n 80 -i 80"

echo "[bench] starting eigen-bench..."
$BENCH_CMD &
BENCH_PID=$!

while kill -0 $BENCH_PID 2>/dev/null; do
    if ! nc -z localhost $PORT >/dev/null 2>&1; then
        echo "[error] server port $PORT is down during eigen-bench"
        kill -9 $BENCH_PID 2>/dev/null || true
        exit 100
    fi
    sleep 1
done

wait $BENCH_PID
'
BENCH_EXIT_CODE=$?

if [[ "$BENCH_EXIT_CODE" -eq 0 ]]; then
    echo "✅ Server passed eigen-bench health check"
    echo "🎉 Server configuration looks GOOD"
elif [[ "$BENCH_EXIT_CODE" -eq 100 ]]; then
    echo "❌ Server crashed or port went down during eigen-bench"
    echo ""
    echo "👉 This usually indicates an invalid server configuration."
    echo "👉 Suggested actions:"
    echo "   - Reduce --mem-fraction-static"
    echo ""
    echo "🔍 Check logs:"
    echo "   docker logs -f $CONTAINER_NAME"
    exit 1
else
    echo "❌ eigen-bench failed with exit code $BENCH_EXIT_CODE"
    echo "👉 Server may be up, but inference is unstable"
    echo "👉 Suggested actions:"
    echo "   - Reduce --mem-fraction-static"
    exit 1
fi

echo "✅ Container started with name: $CONTAINER_NAME"
echo "🪶 View logs: docker logs -f $CONTAINER_NAME"
