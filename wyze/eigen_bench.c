#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
    /* ===== 默认参数 ===== */
    char *model = "OpenGVLab/InternVL3_5-8B-Flash";
    char *tokenizer = "OpenGVLab/InternVL3_5-8B-Flash";
    char *port = "23333";

    int parallel = 64;
    int number = 500;
    int max_tokens = 64;
    int prompt_length = 350;
    int image_num = 10;

    /* 日志目录 */
    char log_dir[] = "/logs";
    mkdir(log_dir, 0755);

    /* 解析参数 */
    int opt, option_index = 0;
    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"port", required_argument, 0, 'p'},
        {"parallel", required_argument, 0, 'P'},
        {"number", required_argument, 0, 'n'},
        {"max-tokens", required_argument, 0, 't'},
        {"prompt-length", required_argument, 0, 'l'},
        {"image-num", required_argument, 0, 'i'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    /* 未知参数透传 */
    char extra_args[4096] = {0};

    while ((opt = getopt_long(argc, argv, "m:p:P:n:t:l:i:h",
                              long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm': model = optarg; tokenizer = optarg; break;
            case 'p': port = optarg; break;
            case 'P': parallel = atoi(optarg); break;
            case 'n': number = atoi(optarg); break;
            case 't': max_tokens = atoi(optarg); break;
            case 'l': prompt_length = atoi(optarg); break;
            case 'i': image_num = atoi(optarg); break;
            case 'h':
                printf(
                    "Usage: %s [options]\n"
                    "  --model PATH\n"
                    "  --port PORT\n"
                    "  --parallel N\n"
                    "  --number N\n"
                    "  --max-tokens N\n"
                    "  --prompt-length N\n"
                    "  --image-num N\n"
                    "  [extra args...]\n",
                    argv[0]);
                return 0;
            default:
                strcat(extra_args, " ");
                strcat(extra_args, argv[optind - 1]);
                break;
        }
    }

    /* URL */
    char url[256];
    snprintf(url, sizeof(url),
             "http://0.0.0.0:%s/v1/chat/completions", port);

    /* 日志文件 */
    char log_file[256];
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    snprintf(log_file, sizeof(log_file),
             "%s/evalscope_%04d%02d%02d_%02d%02d%02d.log",
             log_dir,
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec);

    printf("==== Running evalscope perf ====\n");
    printf("MODEL=%s\n", model);
    printf("URL=%s\n", url);
    printf("PARALLEL=%d\n", parallel);
    printf("NUMBER=%d\n", number);
    printf("PROMPT_LEN=%d\n", prompt_length);
    printf("MAX_TOKENS=%d\n", max_tokens);
    printf("IMAGE_NUM=%d\n", image_num);
    printf("================================\n");

    /* 构建命令 */
    char cmd[8192];
    snprintf(cmd, sizeof(cmd),
        "evalscope perf "
        "--model \"%s\" "
        "--url \"%s\" "
        "--parallel %d "
        "--number %d "
        "--api openai "
        "--dataset random_vl "
        "--min-tokens %d "
        "--max-tokens %d "
        "--prefix-length 0 "
        "--min-prompt-length %d "
        "--max-prompt-length %d "
        "--image-width 448 "
        "--image-height 448 "
        "--image-format RGB "
        "--image-num %d "
        "--tokenizer-path \"%s\" "
        "%s "
        "2>&1 | tee -a \"%s\"",
        model, url, parallel, number,
        max_tokens, max_tokens,
        prompt_length, prompt_length,
        image_num,
        tokenizer,
        extra_args,
        log_file
    );

    /* 执行 */
    int ret = system(cmd);
    return ret;
}