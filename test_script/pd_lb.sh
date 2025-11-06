# python3 -m sglang.srt.disaggregation.mini_lb \
#     --prefill http://0.0.0.0:30000 \
#     --decode http://0.0.0.0:30001 \
#     --host 0.0.0.0 \
#     --port 23333


export HOST_IP=0.0.0.0
export SERVER_PORT=23333
export EMBEDDING_IP=0.0.0.0
export EMBEDDING_PORT=23334
export LANGUAGE_IP=0.0.0.0
export LANGUAGE_PORT=23335
# Launch the bootstrap server on a control node
python3 -m sglang.srt.disaggregation.mini_lb --host $HOST_IP \
    --port $SERVER_PORT --vision http://${EMBEDDING_IP}:${EMBEDDING_PORT} \
    --prefill http://${LANGUAGE_IP}:${LANGUAGE_PORT} \
    --enable-multimodal-disagg