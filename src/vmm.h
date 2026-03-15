#ifndef KERNELBOX_VMM_H
#define KERNELBOX_VMM_H
/* vmm.h — CUDA VMM helpers used by the iterate worker daemon. */

#include <cuda.h>
#include <stddef.h>

// Import a VMM handle from a POSIX fd. Caller must eventually cuMemRelease it.
int vmm_import_from_fd(int fd, CUmemGenericAllocationHandle *handle_out);

// Map an imported handle into the current context.
CUdeviceptr vmm_map(CUmemGenericAllocationHandle handle, size_t size,
                    CUdeviceptr va_hint, CUdevice device, int read_only);

// Unmap a previously mapped region. Does not release the handle.
void vmm_unmap(CUdeviceptr ptr, size_t size);

int vmm_get_granularity(CUdevice device, size_t *granularity_out);
size_t vmm_round_up(size_t size, size_t granularity);

#endif
