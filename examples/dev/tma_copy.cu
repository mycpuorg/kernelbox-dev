// Minimal TMA copy kernel: load via TMA into smem, store via regular writes.
// Requires sm_90+ (Hopper).
#include <cuda.h>

static constexpr int TILE_SIZE = 128;  // float32 elements per TMA tile
static constexpr int TILE_BYTES = TILE_SIZE * sizeof(float);

extern "C" __global__ void tma_copy(
    const __grid_constant__ CUtensorMap tma_in,
    float* __restrict__ output,
    unsigned int n
) {
    __shared__ alignas(128) float smem[TILE_SIZE];
    __shared__ alignas(8) uint64_t mbar;

    int tile_offset = blockIdx.x * TILE_SIZE;
    if (tile_offset >= (int)n) return;

    unsigned smem_addr = __cvta_generic_to_shared(smem);
    unsigned mbar_addr = __cvta_generic_to_shared(&mbar);

    // Initialize mbarrier: 1 arrival expected (the TMA engine itself)
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(mbar_addr), "r"(1));
    }
    __syncthreads();

    // Announce expected bytes from TMA
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
            :: "r"(mbar_addr), "r"(TILE_BYTES));
    }

    // Issue TMA load
    if (threadIdx.x == 0) {
        asm volatile(
            "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2}], [%3];"
            :: "r"(smem_addr),
               "l"(&tma_in),
               "r"(tile_offset),
               "r"(mbar_addr)
            : "memory");
    }

    // Wait for TMA load to complete (phase 0)
    if (threadIdx.x == 0) {
        asm volatile(
            "{\n\t"
            ".reg .pred P;\n\t"
            "WAIT_LOOP:\n\t"
            "mbarrier.try_wait.parity.shared.b64 P, [%0], 0;\n\t"
            "@!P bra WAIT_LOOP;\n\t"
            "}\n\t"
            :: "r"(mbar_addr) : "memory");
    }
    __syncthreads();

    // Regular store: shared → global
    for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
        int gidx = tile_offset + i;
        if (gidx < (int)n) {
            output[gidx] = smem[i];
        }
    }
}
