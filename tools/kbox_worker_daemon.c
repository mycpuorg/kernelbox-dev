// kbox_worker_daemon — private iterate-mode CUDA worker.
//
// The Python iterate runtime owns allocation/compilation state and talks to
// this daemon over a Unix socket. The daemon imports VMM handles from the
// manager side, keeps them mapped across launches, and executes kernels in a
// crash-contained CUDA context.

#include "cuda_check.h"
#include "ipc.h"
#include "kbox_protocol.h"
#include "va.h"
#include "vmm.h"

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <nvrtc.h>
#include <unistd.h>

#define CACHE_SIZE 64
#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME  0x100000001b3ULL

typedef struct {
    uint64_t hash;
    CUmodule module;
    CUfunction func;      /* cached GetFunction result */
    char func_name[128];  /* function name for func */
    uint32_t lru_tick;    /* higher = more recently used */
    int valid;
} cache_entry_t;

static cache_entry_t g_cache[CACHE_SIZE];
static uint32_t g_lru_clock = 0;

static CUmemGenericAllocationHandle g_handles[MAX_BUFS * 2];
static CUdeviceptr g_ptrs[MAX_BUFS * 2];
static int g_count = 0;
static int g_num_inputs = 0;
static int g_num_outputs = 0;
static size_t g_chunk_size = 0;
static int g_active = 0;

static CUfunction g_last_func = NULL;  /* reused when kernel_data_len == 0 */

static CUstream g_stream = NULL;
static CUevent g_timing_start = NULL;
static CUevent g_timing_end = NULL;
static int g_timing_active = 0;

static CUdeviceptr g_scratch_alloc_base = 0;
static CUdeviceptr g_scratch_ptr = 0;
static size_t g_scratch_alloc_size = 0;

static CUmodule g_l2_read_module = NULL;
static CUfunction g_l2_read_func = NULL;
static int g_l2_read_failed = 0;  /* don't retry NVRTC compilation after failure */

/* CUDA C source for the L2 read kernel (compiled via NVRTC at runtime).
 * Uses 128-bit vectorized loads for maximum memory bandwidth. */
static const char g_l2_read_src[] =
"extern \"C\" __global__ void l2_read(unsigned int *buf, unsigned int n) {\n"
"    unsigned int n4 = n >> 2;\n"
"    unsigned int stride = gridDim.x * blockDim.x;\n"
"    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    const uint4 *buf4 = (const uint4 *)buf;\n"
"    uint4 v = {0, 0, 0, 0};\n"
"    for (; i < n4; i += stride) {\n"
"        uint4 t = buf4[i];\n"
"        v.x ^= t.x; v.y ^= t.y; v.z ^= t.z; v.w ^= t.w;\n"
"    }\n"
"    if (threadIdx.x == 0 && blockIdx.x == 0)\n"
"        buf[0] = v.x ^ v.y ^ v.z ^ v.w;\n"
"}\n";

static uint64_t fnv1a(const void *data, size_t len) {
    uint64_t h = FNV_OFFSET;
    const unsigned char *p = data;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
    return h;
}

/* LRU cache: linear scan is fine for 64 entries */
static cache_entry_t *cache_find(uint64_t hash) {
    for (int i = 0; i < CACHE_SIZE; i++) {
        if (g_cache[i].valid && g_cache[i].hash == hash) {
            g_cache[i].lru_tick = ++g_lru_clock;
            return &g_cache[i];
        }
    }
    return NULL;
}

static cache_entry_t *cache_evict_slot(void) {
    /* Find empty slot or LRU entry */
    int lru_idx = 0;
    uint32_t lru_min = UINT32_MAX;
    for (int i = 0; i < CACHE_SIZE; i++) {
        if (!g_cache[i].valid) return &g_cache[i];
        if (g_cache[i].lru_tick < lru_min) {
            lru_min = g_cache[i].lru_tick;
            lru_idx = i;
        }
    }
    cache_entry_t *e = &g_cache[lru_idx];
    cuModuleUnload(e->module);
    e->valid = 0;
    e->func = NULL;
    return e;
}

static void cache_insert(uint64_t hash, CUmodule module) {
    cache_entry_t *e = cache_find(hash);
    if (e) {
        /* Already cached (shouldn't happen normally, but handle it) */
        if (e->module != module) {
            cuModuleUnload(e->module);
            e->module = module;
            e->func = NULL;
            e->func_name[0] = '\0';
        }
        return;
    }
    e = cache_evict_slot();
    e->hash = hash;
    e->module = module;
    e->func = NULL;
    e->func_name[0] = '\0';
    e->lru_tick = ++g_lru_clock;
    e->valid = 1;
}

static void cache_clear(void) {
    for (int i = 0; i < CACHE_SIZE; i++) {
        if (g_cache[i].valid) {
            cuModuleUnload(g_cache[i].module);
            g_cache[i].valid = 0;
            g_cache[i].func = NULL;
        }
    }
    g_lru_clock = 0;
}

static void send_status(int sock, uint32_t status, float elapsed_ms) {
    ipc_send_bytes(sock, &status, sizeof(status));
    ipc_send_bytes(sock, &elapsed_ms, sizeof(elapsed_ms));
}

static int ensure_l2_read_kernel(CUdevice dev) {
    if (g_l2_read_func) return 0;
    if (g_l2_read_failed) return -1;

    /* Detect GPU compute capability for NVRTC. */
    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    char arch_opt[64];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d", major, minor);

    nvrtcProgram prog;
    nvrtcResult nrc = nvrtcCreateProgram(&prog, g_l2_read_src, "l2_read.cu",
                                         0, NULL, NULL);
    if (nrc != NVRTC_SUCCESS) {
        fprintf(stderr, "[worker] nvrtcCreateProgram failed: %s\n",
                nvrtcGetErrorString(nrc));
        g_l2_read_failed = 1;
        return -1;
    }
    const char *opts[] = { arch_opt };
    nrc = nvrtcCompileProgram(prog, 1, opts);
    if (nrc != NVRTC_SUCCESS) {
        size_t log_sz = 0;
        nvrtcGetProgramLogSize(prog, &log_sz);
        if (log_sz > 1) {
            char *log = malloc(log_sz);
            if (log) {
                nvrtcGetProgramLog(prog, log);
                fprintf(stderr, "[worker] l2_read compile log:\n%s\n", log);
                free(log);
            }
        }
        nvrtcDestroyProgram(&prog);
        g_l2_read_failed = 1;
        return -1;
    }
    size_t ptx_sz = 0;
    nvrtcGetPTXSize(prog, &ptx_sz);
    char *ptx = malloc(ptx_sz);
    if (!ptx) { nvrtcDestroyProgram(&prog); g_l2_read_failed = 1; return -1; }
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    CUresult err = cuModuleLoadData(&g_l2_read_module, ptx);
    free(ptx);
    if (err != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(err, &name);
        fprintf(stderr, "[worker] l2_read cuModuleLoadData failed: %s\n",
                name ? name : "?");
        g_l2_read_failed = 1;
        return -1;
    }
    err = cuModuleGetFunction(&g_l2_read_func, g_l2_read_module, "l2_read");
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[worker] l2_read entry not found\n");
        cuModuleUnload(g_l2_read_module);
        g_l2_read_module = NULL;
        g_l2_read_failed = 1;
        return -1;
    }
    return 0;
}

static int ensure_scratch(uint32_t min_mib) {
    size_t min_bytes = (size_t)(min_mib > 256 ? min_mib : 256) * 1024 * 1024;
    size_t gap_pages = 256;
    size_t page_size = 64 * 1024;
    size_t need = min_bytes + gap_pages * page_size;
    if (g_scratch_alloc_base && g_scratch_alloc_size >= need)
        return 0;

    if (g_scratch_alloc_base) {
        cuMemFree(g_scratch_alloc_base);
        g_scratch_alloc_base = 0;
        g_scratch_ptr = 0;
        g_scratch_alloc_size = 0;
    }

    CUresult err = cuMemAlloc(&g_scratch_alloc_base, need);
    if (err != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(err, &name);
        fprintf(stderr, "[worker] scratch alloc failed: %s\n", name ? name : "?");
        return -1;
    }

    unsigned int rval = 0;
    FILE *urand = fopen("/dev/urandom", "r");
    if (urand && fread(&rval, sizeof(rval), 1, urand) == 1) {
        fclose(urand);
    } else {
        if (urand) fclose(urand);
        rval = (unsigned int)(uintptr_t)&rval ^ 0xA5A5A5A5u;
    }
    size_t offset_pages = 1 + (rval % (gap_pages - 1));
    size_t offset = offset_pages * page_size;

    g_scratch_alloc_size = need;
    g_scratch_ptr = g_scratch_alloc_base + offset;
    cuMemsetD32(g_scratch_alloc_base, 0, (unsigned int)(need / 4));
    return 0;
}

static void scratch_release(void) {
    if (g_scratch_alloc_base) {
        cuMemFree(g_scratch_alloc_base);
        g_scratch_alloc_base = 0;
        g_scratch_ptr = 0;
        g_scratch_alloc_size = 0;
    }
    if (g_l2_read_module) {
        cuModuleUnload(g_l2_read_module);
        g_l2_read_module = NULL;
        g_l2_read_func = NULL;
    }
}

static void persistent_release(void) {
    if (g_stream) {
        cuStreamDestroy(g_stream);
        g_stream = NULL;
    }
    if (g_timing_start) {
        cuEventDestroy(g_timing_start);
        g_timing_start = NULL;
    }
    if (g_timing_end) {
        cuEventDestroy(g_timing_end);
        g_timing_end = NULL;
    }
    g_timing_active = 0;

    if (!g_active) return;
    for (int i = 0; i < g_count; i++) {
        if (g_ptrs[i]) {
            vmm_unmap(g_ptrs[i], g_chunk_size);
            cuMemRelease(g_handles[i]);
            g_ptrs[i] = 0;
        }
    }
    g_count = 0;
    g_num_inputs = 0;
    g_num_outputs = 0;
    g_chunk_size = 0;
    g_active = 0;
}

static int context_reset(CUdevice dev, CUcontext *ctx_ptr) {
    persistent_release();
    scratch_release();
    if (*ctx_ptr)
        cuCtxDestroy(*ctx_ptr);
    cache_clear();
    if (cu_init_device(0, &dev, ctx_ptr) != 0) {
        fprintf(stderr, "[worker] Failed to recreate CUDA context\n");
        *ctx_ptr = NULL;
        return -1;
    }
    return 0;
}

/* Look up by hash only (cubin already cached). Returns CUfunction or NULL. */
static CUfunction cache_get_func(uint64_t hash, const char *func_name) {
    cache_entry_t *e = cache_find(hash);
    if (!e) return NULL;
    /* If func cached and name matches, skip cuModuleGetFunction */
    if (e->func && strcmp(e->func_name, func_name) == 0)
        return e->func;
    CUfunction f = NULL;
    if (cuModuleGetFunction(&f, e->module, func_name) != CUDA_SUCCESS)
        return NULL;
    e->func = f;
    strncpy(e->func_name, func_name, sizeof(e->func_name) - 1);
    e->func_name[sizeof(e->func_name) - 1] = '\0';
    return f;
}

static int load_and_get_func(const char *kernel_data,
                             size_t kernel_data_len,
                             const char *func_name,
                             CUfunction *out_func) {
    uint64_t hash = fnv1a(kernel_data, kernel_data_len);
    cache_entry_t *e = cache_find(hash);

    if (!e) {
        CUmodule module = NULL;
        CUresult err = cuModuleLoadData(&module, kernel_data);
        if (err != CUDA_SUCCESS) {
            const char *name = NULL;
            cuGetErrorName(err, &name);
            fprintf(stderr, "[worker] cuModuleLoadData failed: %s\n", name ? name : "?");
            return -1;
        }
        cache_insert(hash, module);
        e = cache_find(hash);
    }

    /* Check func cache */
    if (e->func && strcmp(e->func_name, func_name) == 0) {
        *out_func = e->func;
        return 0;
    }

    if (cuModuleGetFunction(out_func, e->module, func_name) != CUDA_SUCCESS) {
        fprintf(stderr, "[worker] Function '%s' not found\n", func_name);
        return -1;
    }
    e->func = *out_func;
    strncpy(e->func_name, func_name, sizeof(e->func_name) - 1);
    e->func_name[sizeof(e->func_name) - 1] = '\0';
    return 0;
}

static int handle_setup(int sock, worker_config_t *cfg, CUdevice dev) {
    int total = (int)(cfg->num_inputs + cfg->num_outputs);
    if (total <= 0 || total > MAX_BUFS * 2) {
        send_status(sock, 1, 0);
        return 1;
    }

    persistent_release();

    size_t granularity = 0;
    if (vmm_get_granularity(dev, &granularity) != 0) {
        send_status(sock, 1, 0);
        return -1;
    }
    size_t chunk_size = vmm_round_up(cfg->chunk_size, granularity);

    int fds[MAX_BUFS * 2];
    memset(fds, -1, sizeof(fds));
    for (int i = 0; i < total; i++) {
        fds[i] = ipc_recv_fd(sock);
        if (fds[i] < 0) {
            for (int j = 0; j < i; j++)
                if (fds[j] >= 0) close(fds[j]);
            send_status(sock, 1, 0);
            return -1;
        }
    }

    for (int i = 0; i < total; i++) {
        if (vmm_import_from_fd(fds[i], &g_handles[i]) != 0) {
            close(fds[i]);
            for (int j = 0; j < i; j++) {
                vmm_unmap(g_ptrs[j], chunk_size);
                cuMemRelease(g_handles[j]);
            }
            send_status(sock, 1, 0);
            return -1;
        }
        close(fds[i]);
        fds[i] = -1;

        CUdeviceptr hint = VA_BASE_ADDR + (CUdeviceptr)i * chunk_size;
        int read_only = i < (int)cfg->num_inputs;
        g_ptrs[i] = vmm_map(g_handles[i], chunk_size, hint, dev, read_only);
        if (!g_ptrs[i]) {
            cuMemRelease(g_handles[i]);
            for (int j = 0; j < i; j++) {
                vmm_unmap(g_ptrs[j], chunk_size);
                cuMemRelease(g_handles[j]);
            }
            send_status(sock, 1, 0);
            return -1;
        }
    }

    g_count = total;
    g_num_inputs = (int)cfg->num_inputs;
    g_num_outputs = (int)cfg->num_outputs;
    g_chunk_size = chunk_size;
    g_active = 1;

    if (ensure_scratch(cfg->scratch_mib) != 0)
        fprintf(stderr, "[worker] Warning: scratch alloc failed\n");

    /* Pre-warm the l2_read JIT so first clean flush isn't slow. */
    ensure_l2_read_kernel(dev);

    send_status(sock, 0, 0);
    ipc_send_bytes(sock, g_ptrs, (size_t)total * sizeof(CUdeviceptr));
    ipc_send_bytes(sock, &g_scratch_ptr, sizeof(CUdeviceptr));
    return 0;
}

static int sync_stream(uint32_t timeout_ms, CUdevice dev, CUcontext *ctx_ptr) {
    if (!g_stream) return 0;

    (void)timeout_ms;  /* TODO: implement proper timeout (watchdog thread or CUDA events) */
    CUresult qr = CUDA_ERROR_NOT_READY;

    /* Spin briefly for sub-microsecond completions. */
    for (int spin = 0; spin < 100; spin++) {
        qr = cuStreamQuery(g_stream);
        if (qr != CUDA_ERROR_NOT_READY) break;
    }

    if (qr == CUDA_ERROR_NOT_READY) {
        /* Use cuStreamSynchronize with a reasonable timeout instead of
         * sleeping in 1ms increments (which adds up to 1ms+ latency for
         * any GPU work longer than the spin window). */
        qr = cuStreamSynchronize(g_stream);
        /* TODO: timeout handling — cuStreamSynchronize blocks indefinitely.
         * For proper timeout, could use a watchdog thread or CUDA events. */
    }

    if (qr == CUDA_SUCCESS)
        return 0;

    const char *name = NULL;
    cuGetErrorName(qr, &name);
    fprintf(stderr, "[worker] Stream failure: %s\n", name ? name : "?");

    if (context_reset(dev, ctx_ptr) != 0)
        return -1;
    return 1;
}

static float timing_get_elapsed(void) {
    float elapsed_ms = 0;
    if (g_timing_active == 2 && g_timing_start && g_timing_end) {
        cuEventElapsedTime(&elapsed_ms, g_timing_start, g_timing_end);
        g_timing_active = 0;
    }
    return elapsed_ms;
}

static int handle_run(int sock, worker_config_t *cfg, CUdevice dev, CUcontext *ctx_ptr) {
    if (!g_active) {
        fprintf(stderr, "[worker] RUN without prior SETUP\n");
        send_status(sock, 1, 0);
        return 1;
    }

    char *func_name = NULL;
    char *kernel_data = NULL;
    char *param_buf = NULL;
    CUfunction func = NULL;

    if (cfg->func_name_len == 0 && cfg->kernel_data_len == 0 && g_last_func) {
        /* Fastest path: same kernel as last call — no recv, no lookup */
        func = g_last_func;
    } else {
        /* Read func_name */
        func_name = malloc(cfg->func_name_len + 1);
        if (!func_name) { send_status(sock, 1, 0); return -1; }
        if (cfg->func_name_len > 0 &&
            ipc_recv_bytes(sock, func_name, cfg->func_name_len) != 0) {
            free(func_name); send_status(sock, 1, 0); return -1;
        }
        func_name[cfg->func_name_len] = '\0';

        if (cfg->kernel_data_len > 0) {
            /* Full cubin sent — load and cache */
            kernel_data = malloc(cfg->kernel_data_len + 1);
            if (!kernel_data) {
                free(func_name); send_status(sock, 1, 0); return -1;
            }
            if (ipc_recv_bytes(sock, kernel_data, cfg->kernel_data_len) != 0) {
                free(func_name); free(kernel_data);
                send_status(sock, 1, 0); return -1;
            }
            kernel_data[cfg->kernel_data_len] = '\0';
            if (load_and_get_func(kernel_data, cfg->kernel_data_len,
                                  func_name, &func) != 0) {
                free(func_name); free(kernel_data);
                send_status(sock, 1, 0); return 1;
            }
            free(kernel_data);
        } else {
            /* No cubin — hash is in _reserved[0..1], look up in cache */
            uint64_t hash;
            memcpy(&hash, cfg->_reserved, sizeof(hash));
            func = cache_get_func(hash, func_name);
            if (!func) {
                fprintf(stderr, "[worker] Cache miss for hash=%016llx func='%s'\n",
                        (unsigned long long)hash, func_name);
                free(func_name); send_status(sock, 1, 0); return 1;
            }
        }
        free(func_name);
        g_last_func = func;
    }

    if (cfg->param_buffer_len > 0) {
        param_buf = malloc(cfg->param_buffer_len);
        if (!param_buf || ipc_recv_bytes(sock, param_buf, cfg->param_buffer_len) != 0) {
            free(param_buf);
            send_status(sock, 1, 0);
            return -1;
        }
    }

    uint32_t flags = cfg->flags;
    if (!(flags & WORKER_FLAG_NO_MEMSET)) {
        for (int i = 0; i < g_num_outputs; i++) {
            CUdeviceptr out_ptr = g_ptrs[g_num_inputs + i];
            cuMemsetD8Async(out_ptr, 0, g_chunk_size, g_stream);
        }
    }

    if (!g_stream)
        cuStreamCreate(&g_stream, 0);

    if (cfg->scratch_zero_bytes > 0 && g_scratch_ptr) {
        uint32_t zero_u32 = (cfg->scratch_zero_bytes + 3) / 4;
        cuMemsetD32Async(g_scratch_ptr + cfg->scratch_zero_offset, 0, zero_u32, g_stream);
    }

    unsigned int gx = cfg->grid[0];
    unsigned int gy = cfg->grid[1];
    unsigned int gz = cfg->grid[2];
    unsigned int bx = cfg->block[0] ? cfg->block[0] : 256;
    unsigned int by = cfg->block[1] ? cfg->block[1] : 1;
    unsigned int bz = cfg->block[2] ? cfg->block[2] : 1;

    CUresult launch_rc;
    if (param_buf && (flags & WORKER_FLAG_KERNEL_PARAMS)) {
        uint32_t num_params = 0;
        memcpy(&num_params, param_buf, sizeof(num_params));
        if (num_params > 64) num_params = 64;
        uint32_t header_size = 4 + num_params * 4;
        char *data_start = param_buf + header_size;
        void *args[64];
        for (uint32_t i = 0; i < num_params; i++) {
            uint32_t offset = 0;
            memcpy(&offset, param_buf + 4 + i * 4, sizeof(offset));
            args[i] = data_start + offset;
        }
        if (gx == 0) {
            gx = 1;
            gy = 1;
            gz = 1;
        }
        launch_rc = cuLaunchKernel(
            func, gx, gy, gz, bx, by, bz, cfg->smem_bytes, g_stream, args, NULL);
    } else if (param_buf) {
        size_t param_size = cfg->param_buffer_len;
        void *extra[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER, param_buf,
            CU_LAUNCH_PARAM_BUFFER_SIZE, &param_size,
            CU_LAUNCH_PARAM_END
        };
        if (gx == 0) {
            gx = 1;
            gy = 1;
            gz = 1;
        }
        launch_rc = cuLaunchKernel(
            func, gx, gy, gz, bx, by, bz, cfg->smem_bytes, g_stream, NULL, extra);
    } else {
        unsigned int n_val = cfg->n;
        void *args[MAX_BUFS * 2 + 2];
        int ai = 0;
        if ((flags & WORKER_FLAG_PASS_SCRATCH) && g_scratch_ptr)
            args[ai++] = &g_scratch_ptr;
        for (int i = 0; i < g_count; i++)
            args[ai++] = &g_ptrs[i];
        args[ai++] = &n_val;
        if (gx == 0) {
            gx = (n_val + bx - 1) / bx;
            gy = 1;
            gz = 1;
        }
        launch_rc = cuLaunchKernel(
            func, gx, gy, gz, bx, by, bz, cfg->smem_bytes, g_stream, args, NULL);
    }

    free(param_buf);

    if (launch_rc != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(launch_rc, &name);
        fprintf(stderr, "[worker] Launch failed: %s\n", name ? name : "?");
        send_status(sock, 1, 0);
        return 1;
    }

    if (flags & WORKER_FLAG_NO_SYNC) {
        send_status(sock, 0, 0);
        return 0;
    }

    int rc = sync_stream(cfg->timeout_ms, dev, ctx_ptr);
    if (rc == -1) {
        send_status(sock, 1, 0);
        return -1;
    }
    if (rc != 0) {
        send_status(sock, 1, 0);
        return 1;
    }

    send_status(sock, 0, 0);
    return 0;
}

static int handle_sync(int sock, worker_config_t *cfg, CUdevice dev, CUcontext *ctx_ptr) {
    int rc = sync_stream(cfg->timeout_ms, dev, ctx_ptr);
    if (rc == -1) {
        send_status(sock, 1, 0);
        return -1;
    }
    if (rc != 0) {
        send_status(sock, 1, 0);
        return 1;
    }
    send_status(sock, 0, timing_get_elapsed());
    return 0;
}

static int handle_start_timing(int sock, worker_config_t *cfg, CUdevice dev, CUcontext *ctx_ptr) {
    if (!g_stream)
        cuStreamCreate(&g_stream, 0);

    if (cfg->flags & WORKER_FLAG_SYNC) {
        int rc = sync_stream(cfg->timeout_ms, dev, ctx_ptr);
        if (rc != 0) {
            send_status(sock, 1, 0);
            return rc;
        }
    }

    if (!g_timing_start)
        cuEventCreate(&g_timing_start, CU_EVENT_DEFAULT);
    cuEventRecord(g_timing_start, g_stream);
    g_timing_active = 1;
    send_status(sock, 0, 0);
    return 0;
}

static int handle_end_timing(int sock, worker_config_t *cfg, CUdevice dev, CUcontext *ctx_ptr) {
    if (!g_stream)
        cuStreamCreate(&g_stream, 0);

    if (!g_timing_end)
        cuEventCreate(&g_timing_end, CU_EVENT_DEFAULT);
    cuEventRecord(g_timing_end, g_stream);
    if (g_timing_active >= 1)
        g_timing_active = 2;

    float elapsed_ms = 0;
    if (cfg->flags & WORKER_FLAG_SYNC) {
        int rc = sync_stream(cfg->timeout_ms, dev, ctx_ptr);
        if (rc != 0) {
            send_status(sock, 1, 0);
            return rc;
        }
        elapsed_ms = timing_get_elapsed();
    }

    send_status(sock, 0, elapsed_ms);
    return 0;
}

static int handle_l2_flush(int sock, worker_config_t *cfg, CUdevice dev) {
    uint32_t count = cfg->n;
    uint32_t value = cfg->smem_bytes;
    uint32_t flags = cfg->flags;

    if (!g_scratch_alloc_base && ensure_scratch(0) != 0) {
        send_status(sock, 1, 0);
        return 1;
    }
    if (!g_stream)
        cuStreamCreate(&g_stream, 0);

    uint32_t num_u32 = (uint32_t)(g_scratch_alloc_size / 4);
    if (!(flags & L2_FLUSH_CLEAN_ONLY)) {
        for (uint32_t i = 0; i < count; i++) {
            uint32_t fill = (i < count - 1)
                ? (0xA5A5A5A5u ^ (i * 0x13579BDFu))
                : value;
            cuMemsetD32Async(g_scratch_alloc_base, fill, num_u32, g_stream);
        }
    }

    if (flags & (L2_FLUSH_CLEAN | L2_FLUSH_CLEAN_ONLY)) {
        if (ensure_l2_read_kernel(dev) == 0) {
            size_t read_bytes = g_scratch_alloc_size;
            if (read_bytes > 256 * 1024 * 1024)
                read_bytes = 256 * 1024 * 1024;
            uint32_t read_u32 = (uint32_t)(read_bytes / 4);
            uint32_t block = 256;
            uint32_t grid = 2048;
            void *args[] = { &g_scratch_alloc_base, &read_u32 };
            cuLaunchKernel(g_l2_read_func, grid, 1, 1, block, 1, 1,
                           0, g_stream, args, NULL);
        }
    }

    send_status(sock, 0, 0);
    return 0;
}

static int handle_release(int sock) {
    persistent_release();
    send_status(sock, 0, 0);
    return 0;
}

/* Returns: 0 = success, -1 = fatal error, -2 = shutdown, -3 = client disconnected */
static int handle_request(int sock, CUdevice dev, CUcontext *ctx_ptr) {
    worker_config_t cfg;
    if (ipc_recv_bytes(sock, &cfg, sizeof(cfg)) != 0)
        return -3;  /* client disconnected or recv error */

    if (cfg.n == WORKER_SHUTDOWN_N && cfg.func_name_len == 0 && cfg.kernel_data_len == 0)
        return -2;

    switch (cfg.request_type) {
    case WORKER_REQ_SETUP:
        return handle_setup(sock, &cfg, dev);
    case WORKER_REQ_RUN:
        return handle_run(sock, &cfg, dev, ctx_ptr);
    case WORKER_REQ_RELEASE:
        return handle_release(sock);
    case WORKER_REQ_SYNC:
        return handle_sync(sock, &cfg, dev, ctx_ptr);
    case WORKER_REQ_START_TIMING:
        return handle_start_timing(sock, &cfg, dev, ctx_ptr);
    case WORKER_REQ_END_TIMING:
        return handle_end_timing(sock, &cfg, dev, ctx_ptr);
    case WORKER_REQ_L2_FLUSH:
        return handle_l2_flush(sock, &cfg, dev);
    case WORKER_REQ_NOOP:
        if (cfg.n > 0) {
            /* Read and discard cfg.n extra payload bytes */
            char discard[4096];
            uint32_t remaining = cfg.n;
            while (remaining > 0) {
                uint32_t chunk = remaining < sizeof(discard) ? remaining : sizeof(discard);
                if (ipc_recv_bytes(sock, discard, chunk) != 0)
                    return -3;
                remaining -= chunk;
            }
        }
        send_status(sock, 0, 0);
        return 0;
    default:
        fprintf(stderr, "[worker] Unsupported request_type=%u\n", cfg.request_type);
        send_status(sock, 1, 0);
        return 1;
    }
}

static volatile sig_atomic_t g_shutdown = 0;

static void sig_handler(int sig) {
    (void)sig;
    g_shutdown = 1;
}

int main(int argc, char **argv) {
    int idle_timeout = 3600;
    char sock_path[256];
    snprintf(sock_path, sizeof(sock_path), "/tmp/kbox_worker_%d.sock", (int)getuid());

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--idle-timeout") == 0 && i + 1 < argc) {
            idle_timeout = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sock") == 0 && i + 1 < argc) {
            snprintf(sock_path, sizeof(sock_path), "%s", argv[++i]);
        }
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    CUdevice dev = 0;
    CUcontext ctx = NULL;
    if (cu_init_device(0, &dev, &ctx) != 0) {
        fprintf(stderr, "[worker] Failed to initialize CUDA\n");
        return 1;
    }

    unlink(sock_path);
    int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket");
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (strlen(sock_path) >= sizeof(addr.sun_path)) {
        fprintf(stderr, "[worker] Socket path too long: %s\n", sock_path);
        close(listen_fd);
        unlink(sock_path);
        cuCtxDestroy(ctx);
        return 1;
    }
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(listen_fd);
        return 1;
    }
    if (listen(listen_fd, 4) < 0) {
        perror("listen");
        close(listen_fd);
        unlink(sock_path);
        return 1;
    }

    fprintf(stderr, "[worker] Listening on %s (idle %ds, pid %d)\n",
            sock_path, idle_timeout, getpid());

    while (!g_shutdown) {
        struct pollfd pfd = { .fd = listen_fd, .events = POLLIN };
        int pr = poll(&pfd, 1, idle_timeout * 1000);
        if (pr == 0) {
            fprintf(stderr, "[worker] Idle timeout\n");
            break;
        }
        if (pr < 0) {
            if (errno == EINTR) continue;
            perror("poll");
            break;
        }

        int client = accept(listen_fd, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }

        /* Persistent connection: handle requests in a loop until
         * the client disconnects (-3) or a fatal/shutdown occurs. */
        int rc = 0;
        while (rc >= 0 && !g_shutdown) {
            rc = handle_request(client, dev, &ctx);
        }
        close(client);

        if (rc == -3) continue;  /* clean client disconnect, accept next */
        if (rc == -2) {
            fprintf(stderr, "[worker] Shutdown requested\n");
            break;
        }
        if (!ctx || rc < 0) {
            fprintf(stderr, "[worker] Fatal worker error, exiting\n");
            break;
        }
    }

    persistent_release();
    scratch_release();
    cache_clear();
    if (ctx)
        cuCtxDestroy(ctx);
    close(listen_fd);
    unlink(sock_path);
    fprintf(stderr, "[worker] Stopped\n");
    return 0;
}
