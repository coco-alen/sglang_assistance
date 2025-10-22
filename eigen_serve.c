#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void ensure_env(const char *key, const char *value) {
    const char *existing = getenv(key);
    if (existing && existing[0] != '\0') {
        return;
    }
    if (setenv(key, value, 1) != 0) {
        fprintf(stderr, "eigen-serve: failed to set %s: %s\n", key, strerror(errno));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    ensure_env("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1");

    const char *defaults[] = {
        "/usr/bin/python3",
        "-m", "sglang.launch_server",
        "--model-path", "OpenGVLab/InternVL3_5-8B-Flash",
        "--chat-template", "internvl-2-5",
        "--port", "23333",
        "--mem-fraction-static", "0.7",
        "--mm-attention-backend", "fa3",
        "--attention-backend", "fa3",
        "--quantization", "fp8",
        "--enable-torch-compile",
        NULL
    };

    int default_count = 0;
    while (defaults[default_count] != NULL) {
        default_count++;
    }

    int passthrough_count = argc > 0 ? argc - 1 : 0;
    int total_args = default_count + passthrough_count;
    char **cmd = calloc((size_t)total_args + 1, sizeof(char *));
    if (cmd == NULL) {
        perror("eigen-serve: calloc");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < default_count; ++i) {
        cmd[i] = (char *)defaults[i];
    }

    for (int i = 1; i < argc; ++i) {
        cmd[default_count + i - 1] = argv[i];
    }

    cmd[total_args] = NULL;

    execv(cmd[0], cmd);
    perror("eigen-serve: execv");
    free(cmd);
    return EXIT_FAILURE;
}