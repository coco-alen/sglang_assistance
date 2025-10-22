#!/bin/bash

# 设置默认值
IMAGE_NAME="eigen_vlm/yipin:wyze"
CONTAINER_NAME="wyze"
SCRIPT_PATH="./start_servers.sh"

# 默认参数
CUDA_DEVICES="0"
MODEL_PATH="OpenGVLab/InternVL3-8B-Instruct"
PORT="23334"

# 显示帮助信息
show_help() {
    echo "How to use: $0 [options]"
    echo "Options:"
    echo "  -i, --image IMAGE      Docker image name"
    echo "  -c, --cuda DEVICES     CUDA devices (default: 0)"
    echo "  -m, --model PATH       Model path (default: OpenGVLab/InternVL3_5-8B)"
    echo "  -p, --port PORT        Port (default: 23334)"
    echo "  -h, --help             Show help information"
    echo ""
    echo "示例:"
    echo "  $0 -i myimage:latest -c 1 -p 8080"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
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
            show_help
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "Launch Docker Container..."
echo "Image: $IMAGE_NAME"
echo "CUDA Devices: $CUDA_DEVICES"
echo "Model Path: $MODEL_PATH"
echo "Port: $PORT"
echo ""

# 停止并删除已存在的同名容器
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true


# 判断MODEL_PATH是否以OpenGVLab开头
if [[ "$MODEL_PATH" == OpenGVLab* ]]; then
    # 直接传递模型名
    DOCKER_MODEL_ARG="$MODEL_PATH"
    DOCKER_VOLUME_ARG=""
else
    # 需要挂载本地模型路径
    # 获取绝对路径
    ABS_MODEL_PATH=$(readlink -f "$MODEL_PATH")
    # 容器内路径
    CONTAINER_MODEL_PATH="/models/$(basename "$ABS_MODEL_PATH")"
    DOCKER_MODEL_ARG="$CONTAINER_MODEL_PATH"
    DOCKER_VOLUME_ARG="-v $ABS_MODEL_PATH:$CONTAINER_MODEL_PATH"
fi

docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    --ipc=host \
    --network=host \
    $DOCKER_VOLUME_ARG \
    $IMAGE_NAME \
    bash -c "$SCRIPT_PATH -c $CUDA_DEVICES -m $DOCKER_MODEL_ARG -p $PORT"

echo "Container started with name: $CONTAINER_NAME"
echo "View logs: docker logs -f $CONTAINER_NAME"