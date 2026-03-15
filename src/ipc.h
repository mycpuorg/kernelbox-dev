#ifndef KERNELBOX_IPC_H
#define KERNELBOX_IPC_H
/* ipc.h — Unix domain socket IPC primitives.
 * Socket pairs, SCM_RIGHTS fd passing, and blocking byte-stream I/O. */

#include <stddef.h>

// Create a Unix domain socket pair. Returns 0 on success.
int ipc_socketpair(int fds[2]);

// Send a file descriptor over a Unix socket. Returns 0 on success.
int ipc_send_fd(int socket, int fd_to_send);

// Receive a file descriptor from a Unix socket. Returns fd >= 0, or -1 on error.
int ipc_recv_fd(int socket);

// Send exactly len bytes (blocking, handles partial writes). Returns 0 on success.
int ipc_send_bytes(int socket, const void *buf, size_t len);

// Receive exactly len bytes (blocking, handles partial reads). Returns 0 on success.
int ipc_recv_bytes(int socket, void *buf, size_t len);

#endif // KERNELBOX_IPC_H
