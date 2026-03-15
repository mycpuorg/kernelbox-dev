#ifndef KBOX_PROTOCOL_H
#define KBOX_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#define MAX_BUFS 16

// Request types used by the iterate worker protocol.
#define WORKER_REQ_SETUP        1
#define WORKER_REQ_RUN          2
#define WORKER_REQ_RELEASE      3
#define WORKER_REQ_SYNC         4
#define WORKER_REQ_START_TIMING 5
#define WORKER_REQ_END_TIMING   6
#define WORKER_REQ_L2_FLUSH     7
#define WORKER_REQ_NOOP         8

// Launch flags in worker_config_t.flags.
#define WORKER_FLAG_NO_MEMSET      (1u << 0)
#define WORKER_FLAG_NO_EVENTS      (1u << 1)
#define WORKER_FLAG_NO_HEALTH      (1u << 2)
#define WORKER_FLAG_KERNEL_PARAMS  (1u << 3)
#define WORKER_FLAG_NO_SYNC        (1u << 4)
#define WORKER_FLAG_SYNC           (1u << 5)
#define WORKER_FLAG_PASS_SCRATCH   (1u << 6)

// L2 flush flags share the upper bits of worker_config_t.flags.
#define L2_FLUSH_CLEAN             (1u << 8)
#define L2_FLUSH_CLEAN_ONLY        (1u << 9)

// Python packs this struct directly with "<24IQ" (see dev.py _WORKER_CONFIG_FMT).
// Any layout change must update both sides and bump WORKER_PROTOCOL_VERSION.
#define WORKER_PROTOCOL_VERSION 1
typedef struct {
    uint32_t dtype;
    uint32_t n;
    uint32_t is_cubin;
    uint32_t func_name_len;
    uint32_t kernel_data_len;
    uint32_t num_inputs;
    uint32_t num_outputs;
    uint32_t timeout_ms;
    uint32_t grid[3];
    uint32_t block[3];
    uint32_t smem_bytes;
    uint32_t _reserved[3];
    uint32_t request_type;
    uint32_t flags;
    uint32_t param_buffer_len;
    uint32_t scratch_mib;
    uint32_t scratch_zero_offset;
    uint32_t scratch_zero_bytes;
    size_t   chunk_size;
} worker_config_t;

_Static_assert(sizeof(worker_config_t) == 104,
               "worker_config_t layout changed — update Python _WORKER_CONFIG_FMT");

#define WORKER_SHUTDOWN_N 0xDEADu

#endif
