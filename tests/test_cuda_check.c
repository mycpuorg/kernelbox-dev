// Step 1: Verify CUDA driver API wrapper works
#include "cuda_check.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    CUdevice dev;
    CUcontext ctx;
    if (cu_init_device(0, &dev, &ctx) != 0) {
        fprintf(stderr, "SKIP: no GPU available\n");
        return 77;
    }

    // Verify device ordinal
    CUdevice check_dev;
    CU_CHECK(cuCtxGetDevice(&check_dev));
    assert(check_dev == dev);

    // Verify basic alloc/free works
    CUdeviceptr ptr;
    CU_CHECK(cuMemAlloc(&ptr, 4));
    assert(ptr != 0);
    CU_CHECK(cuMemFree(ptr));

    // Verify memory info
    size_t free_mem, total_mem;
    assert(cu_mem_info(&free_mem, &total_mem) == 0);
    assert(total_mem > 0);
    assert(free_mem > 0);
    assert(free_mem <= total_mem);

    CU_CHECK(cuCtxDestroy(ctx));
    printf("test_cuda_check: OK\n");
    return 0;
}
