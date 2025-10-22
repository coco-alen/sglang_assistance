

export SGLANG_TORCH_PROFILER_DIR="/data/yipin/project/sglang-dev/sglang_assistance"

python api_test.py
python api_test.py
python api_test.py

curl http://localhost:23333/start_profile -H "Content-Type: application/json"

python api_test.py

curl http://localhost:23333/stop_profile -H "Content-Type: application/json"