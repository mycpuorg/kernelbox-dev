"""Python wrapper for CUDA Virtual Memory Management using cuda-python.

Provides VMMPool for the manager side (allocate, fill, export, backup/restore)
and import helpers for the worker side (import from fd, map, create tensors).
"""

import ctypes
import os
from cuda.bindings import driver as cu


def _check(err, msg="CUDA error"):
    if isinstance(err, tuple):
        err = err[0]
    code = int(err)
    if code != 0:
        raise RuntimeError(f"{msg}: CUDA error {code}")


def init_cuda(device_ordinal=0):
    """Initialize CUDA and create a context. Returns (device, context)."""
    _check(cu.cuInit(0), "cuInit")
    err, device = cu.cuDeviceGet(device_ordinal)
    _check(err, "cuDeviceGet")
    err, ctx = cu.cuCtxCreate(0, device)
    _check(err, "cuCtxCreate")
    return device, ctx


def _alloc_prop(device):
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = int(device)
    prop.requestedHandleTypes = (
        cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )
    return prop


def get_granularity(device):
    prop = _alloc_prop(device)
    err, gran = cu.cuMemGetAllocationGranularity(
        prop,
        cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
    )
    _check(err, "cuMemGetAllocationGranularity")
    return gran


def round_up(size, gran):
    return ((size + gran - 1) // gran) * gran


class VMMChunk:
    """One physical VMM allocation."""
    __slots__ = ('handle', 'size', 'mapped_ptr')

    def __init__(self, handle, size):
        self.handle = handle
        self.size = size
        self.mapped_ptr = 0


class VMMPool:
    """Allocates GPU memory as VMM chunks, manages mapping and lifecycle."""

    def __init__(self, device, chunk_size=2 * 1024 * 1024,
                 headroom=256 * 1024 * 1024, max_chunks=None):
        self.device = device
        self.granularity = get_granularity(device)
        self.chunk_size = round_up(chunk_size, self.granularity)
        self.headroom = headroom
        self.chunks = []
        self._alloc(max_chunks)

    def _alloc(self, max_chunks=None):
        err, free_mem, _total = cu.cuMemGetInfo()
        _check(err, "cuMemGetInfo")
        available = free_mem - self.headroom
        limit = int(available // self.chunk_size)
        if max_chunks is not None:
            limit = min(limit, max_chunks)
        prop = _alloc_prop(self.device)

        for _ in range(max(limit, 0)):
            err, handle = cu.cuMemCreate(self.chunk_size, prop, 0)
            if int(err) != 0:
                break
            self.chunks.append(VMMChunk(handle, self.chunk_size))

    def __len__(self):
        return len(self.chunks)

    def export_fd(self, index):
        """Export chunk handle as a POSIX file descriptor."""
        h = self.chunks[index].handle
        err, fd = cu.cuMemExportToShareableHandle(
            h,
            cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0,
        )
        _check(err, "cuMemExportToShareableHandle")
        return int(fd)

    def map_chunk(self, index, va_hint=0, read_only=False):
        """Reserve VA, map chunk, set access. Returns device pointer (int)."""
        chunk = self.chunks[index]
        ptr = _map_handle(self.device, chunk.handle, chunk.size,
                          self.granularity, va_hint, read_only)
        chunk.mapped_ptr = ptr
        return ptr

    def destroy(self):
        for chunk in self.chunks:
            if chunk.mapped_ptr:
                cu.cuMemUnmap(chunk.mapped_ptr, chunk.size)
                cu.cuMemAddressFree(chunk.mapped_ptr, chunk.size)
                chunk.mapped_ptr = 0
            cu.cuMemRelease(chunk.handle)
        self.chunks.clear()


def _map_handle(device, handle, size, granularity, va_hint=0, read_only=False):
    err, ptr = cu.cuMemAddressReserve(size, granularity, va_hint, 0)
    _check(err, "cuMemAddressReserve")

    err, = cu.cuMemMap(ptr, size, 0, handle, 0)
    _check(err, "cuMemMap")

    access = cu.CUmemAccessDesc()
    access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = int(device)
    access.flags = (
        cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ if read_only
        else cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    )
    err, = cu.cuMemSetAccess(ptr, size, [access], 1)
    _check(err, "cuMemSetAccess")
    return int(ptr)


def import_from_fd(fd):
    """Import a VMM handle from a POSIX file descriptor."""
    err, handle = cu.cuMemImportFromShareableHandle(
        fd,
        cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    )
    _check(err, "cuMemImportFromShareableHandle")
    return handle


def import_and_map(device, fd, size, granularity, va_hint=0, read_only=False):
    """Import from fd, map, set access. Returns (handle, device_ptr)."""
    handle = import_from_fd(fd)
    ptr = _map_handle(device, handle, size, granularity, va_hint, read_only)
    return handle, ptr


def unmap(ptr, size):
    cu.cuMemUnmap(ptr, size)
    cu.cuMemAddressFree(ptr, size)


def memset_d32(ptr, value, count):
    _check(cu.cuMemsetD32(ptr, value, count), "cuMemsetD32")


def memcpy_dtoh(host_buf, device_ptr, nbytes):
    """Copy device → host. host_buf must be a ctypes buffer or bytearray."""
    _check(
        cu.cuMemcpyDtoH(host_buf, device_ptr, nbytes),
        "cuMemcpyDtoH",
    )


def memcpy_htod(device_ptr, host_buf, nbytes):
    _check(
        cu.cuMemcpyHtoD(device_ptr, host_buf, nbytes),
        "cuMemcpyHtoD",
    )


def memcpy_dtod(dst, src, nbytes):
    _check(cu.cuMemcpyDtoD(dst, src, nbytes), "cuMemcpyDtoD")
