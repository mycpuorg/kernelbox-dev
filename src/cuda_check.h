#ifndef KERNELBOX_CUDA_CHECK_H
#define KERNELBOX_CUDA_CHECK_H
/* cuda_check.h — CUDA error checking macros and device initialization.
 * CU_CHECK aborts on error; CU_CHECK_ERR returns a value.
 * cu_init_device() wraps cuInit + cuDeviceGet + cuCtxCreate. */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Abort on CUDA error with file/line info
#define CU_CHECK(expr) do { \
    CUresult _r = (expr); \
    if (_r != CUDA_SUCCESS) { \
        const char *_name = NULL; \
        cuGetErrorName(_r, &_name); \
        fprintf(stderr, "CUDA error %s (%d) at %s:%d: %s\n", \
                _name ? _name : "?", (int)_r, __FILE__, __LINE__, #expr); \
        abort(); \
    } \
} while(0)

// Return error value instead of aborting
#define CU_CHECK_ERR(expr, retval) do { \
    CUresult _r = (expr); \
    if (_r != CUDA_SUCCESS) { \
        const char *_name = NULL; \
        cuGetErrorName(_r, &_name); \
        fprintf(stderr, "CUDA error %s (%d) at %s:%d: %s\n", \
                _name ? _name : "?", (int)_r, __FILE__, __LINE__, #expr); \
        return (retval); \
    } \
} while(0)

// Initialize CUDA, create context on device ordinal.
// Returns 0 on success, -1 on failure.
// Outputs device and context.
int cu_init_device(int ordinal, CUdevice *dev_out, CUcontext *ctx_out);

// Get total and free device memory in bytes.
int cu_mem_info(size_t *free_out, size_t *total_out);

#endif // KERNELBOX_CUDA_CHECK_H
