/* vmm.c — CUDA VMM helpers for iterate-mode worker imports. */
#include "vmm.h"
#include "cuda_check.h"
#include <stdio.h>
#include <string.h>

size_t vmm_round_up(size_t size, size_t granularity) {
    return (size + granularity - 1) & ~(granularity - 1);
}

int vmm_get_granularity(CUdevice device, size_t *granularity_out) {
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = (int)device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    CU_CHECK_ERR(cuMemGetAllocationGranularity(
        granularity_out, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), -1);
    return 0;
}

int vmm_import_from_fd(int fd, CUmemGenericAllocationHandle *handle_out) {
    CU_CHECK_ERR(cuMemImportFromShareableHandle(
        handle_out, (void*)(intptr_t)fd,
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR), -1);
    return 0;
}

CUdeviceptr vmm_map(CUmemGenericAllocationHandle handle, size_t size,
                    CUdeviceptr va_hint, CUdevice device, int read_only) {
    size_t granularity;
    if (vmm_get_granularity(device, &granularity) != 0) return 0;

    size = vmm_round_up(size, granularity);

    CUdeviceptr ptr = 0;
    CUresult r = cuMemAddressReserve(&ptr, size, granularity, va_hint, 0);
    if (r != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(r, &name);
        fprintf(stderr, "cuMemAddressReserve failed: %s (hint=0x%llx, size=%zu)\n",
                name ? name : "?", (unsigned long long)va_hint, size);
        return 0;
    }

    if (va_hint != 0 && ptr != va_hint) {
        cuMemAddressFree(ptr, size);
        ptr = 0;
        r = cuMemAddressReserve(&ptr, size, granularity, 0, 0);
        if (r != CUDA_SUCCESS) {
            const char *name = NULL;
            cuGetErrorName(r, &name);
            fprintf(stderr, "cuMemAddressReserve fallback failed: %s\n",
                    name ? name : "?");
            return 0;
        }
    }

    r = cuMemMap(ptr, size, 0, handle, 0);
    if (r != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(r, &name);
        fprintf(stderr, "cuMemMap failed: %s\n", name ? name : "?");
        cuMemAddressFree(ptr, size);
        return 0;
    }

    CUmemAccessDesc access;
    memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = (int)device;
    access.flags = read_only ? CU_MEM_ACCESS_FLAGS_PROT_READ
                             : CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    r = cuMemSetAccess(ptr, size, &access, 1);
    if (r != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(r, &name);
        fprintf(stderr, "cuMemSetAccess failed: %s\n", name ? name : "?");
        cuMemUnmap(ptr, size);
        cuMemAddressFree(ptr, size);
        return 0;
    }

    return ptr;
}

void vmm_unmap(CUdeviceptr ptr, size_t size) {
    if (ptr == 0) return;
    cuMemUnmap(ptr, size);
    cuMemAddressFree(ptr, size);
}
