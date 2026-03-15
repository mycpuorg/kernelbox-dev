/* cuda_check.c — CUDA device initialization helper.
 * Wraps cuInit + cuDeviceGet + cuCtxCreate into a single call with
 * error reporting.  Used by tests and tools for one-line GPU setup. */
#include "cuda_check.h"
#include <string.h>

int cu_init_device(int ordinal, CUdevice *dev_out, CUcontext *ctx_out) {
    CU_CHECK_ERR(cuInit(0), -1);

    int count = 0;
    CU_CHECK_ERR(cuDeviceGetCount(&count), -1);
    if (ordinal < 0 || ordinal >= count) {
        fprintf(stderr, "cu_init_device: ordinal %d out of range (have %d devices)\n",
                ordinal, count);
        return -1;
    }

    CUdevice dev;
    CU_CHECK_ERR(cuDeviceGet(&dev, ordinal), -1);

    CUcontext ctx;
    CUctxCreateParams params;
    memset(&params, 0, sizeof(params));
    CU_CHECK_ERR(cuCtxCreate(&ctx, &params, 0, dev), -1);

    *dev_out = dev;
    *ctx_out = ctx;
    return 0;
}

int cu_mem_info(size_t *free_out, size_t *total_out) {
    CU_CHECK_ERR(cuMemGetInfo(free_out, total_out), -1);
    return 0;
}
