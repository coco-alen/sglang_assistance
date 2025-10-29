#!/bin/bash
set -e

# 默认值
CUDA_DEVICES="0"
MODEL_PATH="OpenGVLab/InternVL3_5-2B"
PORT="23333"

# 参数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cuda)
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
        -h|--help)
            echo "Usage: $0 [--cuda 0,1] [--model path_or_name] [--port 23333]"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

echo "==== Launching SGLang Server ===="
echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
echo "MODEL_PATH=$MODEL_PATH"
echo "PORT=$PORT"
echo "================================="

# 设置CUDA环境变量
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# 启动服务
python -m sglang.launch_server \
   --model-path "$MODEL_PATH" \
   --chat-template internvl-2-5 \
   --port "$PORT" \
   --mem-fraction-static 0.8 \
   --mm-attention-backend fa3 \
   --attention-backend fa3 \
   --log-level info \
   --quantization fp8 \
   --enable-torch-compile \
   --disable-radix-cache \
   --decode-log-interval 1 \
   --enable-multimodal



LOG_FILE="/logs/start_server_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /logs
exec > >(tee -a "$LOG_FILE") 2>&1