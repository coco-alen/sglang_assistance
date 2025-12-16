#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
    char *cuda_devices = "0";
    char *model_path = "OpenGVLab/InternVL3_5-2B";
    char *port = "23333";

    // 日志路径
    char log_dir[] = "/logs";
    mkdir(log_dir, 0755);

    // 解析参数
    int opt;
    int option_index = 0;
    static struct option long_options[] = {
        {"cuda", required_argument, 0, 'c'},
        {"model", required_argument, 0, 'm'},
        {"port", required_argument, 0, 'p'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    // 用于收集未知参数
    char extra_args[4096] = {0};

    while ((opt = getopt_long(argc, argv, "c:m:p:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'c':
                cuda_devices = optarg;
                break;
            case 'm':
                model_path = optarg;
                break;
            case 'p':
                port = optarg;
                break;
            case 'h':
                printf("Usage: %s [--cuda 0,1] [--model path_or_name] [--port 23333] [extra args...]\n", argv[0]);
                return 0;
            case '?': // 未知参数
            default:
                // 把未知参数拼接起来
                strcat(extra_args, " ");
                strcat(extra_args, argv[optind - 1]);
                if (optarg && optarg[0] != '-') {
                    strcat(extra_args, " ");
                    strcat(extra_args, optarg);
                }
                break;
        }
    }

    printf("==== Launching Server ====\n");
    printf("CUDA_VISIBLE_DEVICES=%s\n", cuda_devices);
    printf("MODEL_PATH=%s\n", model_path);
    printf("PORT=%s\n", port);
    printf("=================================\n");

    // 设置环境变量
    setenv("CUDA_VISIBLE_DEVICES", cuda_devices, 1);
    setenv("TORCHINDUCTOR_CACHE_DIR", "~/.triton", 1);

    // 日志文件
    char log_file[256];
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    snprintf(log_file, sizeof(log_file), "%s/start_server_%04d%02d%02d_%02d%02d%02d.log",
             log_dir, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec);

    // 构建 python 命令
    char cmd[8192];
    snprintf(cmd, sizeof(cmd),
             "python -m sglang.launch_server "
             "--model-path \"%s\" "
             "--chat-template internvl-2-5 "
             "--port \"%s\" "
             "--mem-fraction-static 0.65 "
             "--mm-attention-backend fa3 "
             "--attention-backend fa3 "
             "--log-level info "
             "--quantization fp8 "
             "--enable-broadcast-mm-inputs-process "
             "--enable-torch-compile "
             "--enable-multimodal "
             "%s "
             "2>&1 | tee -a \"%s\"",
             model_path, port, extra_args, log_file);

    // printf("Executing:\n%s\n", cmd);

    int ret = system(cmd);
    return ret;
}