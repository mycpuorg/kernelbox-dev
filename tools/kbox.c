// kbox — KernelBox CLI dispatcher.

#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define VERSION "0.2.0"

static char g_exec_dir[4096];
static char g_python_cmd[4096] = "python3";

static int join2(char *out, size_t out_sz, const char *a, const char *b) {
    size_t a_len = strlen(a);
    size_t b_len = strlen(b);

    if (a_len + b_len + 1 > out_sz)
        return -1;

    memcpy(out, a, a_len);
    memcpy(out + a_len, b, b_len + 1);
    return 0;
}

static int join3(char *out, size_t out_sz,
                 const char *a, const char *b, const char *c) {
    size_t a_len = strlen(a);
    size_t b_len = strlen(b);
    size_t c_len = strlen(c);

    if (a_len + b_len + c_len + 1 > out_sz)
        return -1;

    memcpy(out, a, a_len);
    memcpy(out + a_len, b, b_len);
    memcpy(out + a_len + b_len, c, c_len + 1);
    return 0;
}

static void resolve_exec_dir(const char *argv0) {
    char buf[4096];
    const char *dir;

    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        dir = dirname(buf);
    } else if (argv0 && strchr(argv0, '/')) {
        char tmp[4096];
        snprintf(tmp, sizeof(tmp), "%s", argv0);
        dir = dirname(tmp);
    } else {
        dir = ".";
    }

    snprintf(g_exec_dir, sizeof(g_exec_dir), "%s", dir);
}

static void set_python_cmd(const char *value) {
    size_t len;

    if (!value || !*value)
        return;
    len = strlen(value);
    if (len >= sizeof(g_python_cmd))
        len = sizeof(g_python_cmd) - 1;
    memcpy(g_python_cmd, value, len);
    g_python_cmd[len] = '\0';
}

static void prepend_path_dir(const char *dir) {
    const char *old_path = getenv("PATH");
    size_t old_len = (old_path && *old_path) ? strlen(old_path) : 0;
    size_t new_len = strlen(dir) + (old_len ? old_len + 1 : 0) + 1;
    char *new_path = malloc(new_len);

    if (!new_path)
        return;

    if (old_len)
        snprintf(new_path, new_len, "%s:%s", dir, old_path);
    else
        snprintf(new_path, new_len, "%s", dir);

    setenv("PATH", new_path, 1);
    free(new_path);
}

static void setup_python_cmd(void) {
    char dir[4096];
    const char *override = getenv("KBOX_PYTHON");

    if (override && *override) {
        set_python_cmd(override);
        return;
    }

    snprintf(dir, sizeof(dir), "%s", g_exec_dir);
    for (int depth = 0; depth < 4; depth++) {
        const char *python_names[] = {"python3", "python"};
        for (size_t i = 0; i < sizeof(python_names) / sizeof(python_names[0]); i++) {
            char probe[4096];
            char venv_bin[4096];
            if (join3(probe, sizeof(probe), dir, "/.venv/bin/", python_names[i]) != 0)
                continue;
            if (access(probe, X_OK) == 0) {
                if (join2(venv_bin, sizeof(venv_bin), dir, "/.venv/bin") != 0)
                    continue;
                set_python_cmd(probe);
                prepend_path_dir(venv_bin);
                return;
            }
        }
        char *slash = strrchr(dir, '/');
        if (!slash || slash == dir) break;
        *slash = '\0';
    }
}

static int find_script(const char *name, char *out, size_t out_sz) {
    const char *patterns[] = {
        "%s/%s",
        "%s/../libexec/kbox/%s",
        "%s/../../tools/%s",
    };
    for (size_t i = 0; i < sizeof(patterns) / sizeof(patterns[0]); i++) {
        snprintf(out, out_sz, patterns[i], g_exec_dir, name);
        if (access(out, R_OK) == 0)
            return 0;
    }
    return -1;
}

static void print_main_help(void) {
    fprintf(stderr,
        "KernelBox %s — GPU kernel iteration and task evaluation\n\n"
        "Usage:\n"
        "  kbox iterate <test_file.py> [options]\n"
        "  kbox mcp\n"
        "  kbox <test_file.py> [options]\n"
        "  kbox version\n\n"
        "Commands:\n"
        "  iterate   Run the hot-reload iterate workflow\n"
        "  mcp       Run the MCP task server on stdio\n"
        "  version   Print version\n\n"
        "Examples:\n"
        "  kbox iterate examples/dev/test_scale.py\n"
        "  kbox iterate examples/dev/test_mlp_suite.py --once\n"
        "  kbox iterate examples/dev/test_kernel_mode_cuda.py --once --bench --isolated-kernel-benchmark\n"
        "  kbox mcp\n"
        "  kbox examples/dev/test_triton_add.py --once\n",
        VERSION);
}

static int exec_script(const char *script_name, int argc, char **argv, int offset) {
    char script[4096];
    if (find_script(script_name, script, sizeof(script)) != 0) {
        fprintf(stderr,
                "kbox: could not find %s in the install or source tree\n",
                script_name);
        return 1;
    }

    int forwarded = argc - offset;
    char **cmd = calloc((size_t)forwarded + 3, sizeof(char *));
    if (!cmd) {
        perror("calloc");
        return 1;
    }

    cmd[0] = g_python_cmd;
    cmd[1] = script;
    for (int i = 0; i < forwarded; i++)
        cmd[i + 2] = argv[i + offset];
    cmd[forwarded + 2] = NULL;

    if (strchr(g_python_cmd, '/'))
        execv(g_python_cmd, cmd);
    else
        execvp(g_python_cmd, cmd);
    perror("exec");
    free(cmd);
    return 1;
}

static int looks_like_test_file(const char *arg) {
    size_t len = strlen(arg);
    return len > 3 && strcmp(arg + len - 3, ".py") == 0;
}

int main(int argc, char **argv) {
    resolve_exec_dir(argv[0]);
    setup_python_cmd();

    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_main_help();
        return 0;
    }

    if (strcmp(argv[1], "version") == 0) {
        printf("kbox %s\n", VERSION);
        return 0;
    }

    if (strcmp(argv[1], "iterate") == 0)
        return exec_script("kbox_iterate.py", argc, argv, 2);

    if (strcmp(argv[1], "mcp") == 0)
        return exec_script("kbox_mcp.py", argc, argv, 2);

    if (looks_like_test_file(argv[1]))
        return exec_script("kbox_iterate.py", argc, argv, 1);

    fprintf(stderr, "kbox: unknown command '%s'\n\n", argv[1]);
    print_main_help();
    return 1;
}
