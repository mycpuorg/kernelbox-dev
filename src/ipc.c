/* ipc.c — Unix domain socket IPC primitives.
 * Provides socket pairs, file descriptor passing (SCM_RIGHTS), and
 * reliable byte-stream send/recv for manager-worker communication. */
#include "ipc.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>

int ipc_socketpair(int fds[2]) {
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) < 0) {
        perror("socketpair");
        return -1;
    }
    return 0;
}

int ipc_send_fd(int socket, int fd_to_send) {
    char dummy = 'F';
    struct iovec iov = { .iov_base = &dummy, .iov_len = 1 };

    union {
        struct cmsghdr hdr;
        char buf[CMSG_SPACE(sizeof(int))];
    } cmsg_buf;
    memset(&cmsg_buf, 0, sizeof(cmsg_buf));

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.buf;
    msg.msg_controllen = sizeof(cmsg_buf.buf);

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &fd_to_send, sizeof(int));

    if (sendmsg(socket, &msg, 0) < 0) {
        perror("ipc_send_fd: sendmsg");
        return -1;
    }
    return 0;
}

int ipc_recv_fd(int socket) {
    char dummy;
    struct iovec iov = { .iov_base = &dummy, .iov_len = 1 };

    union {
        struct cmsghdr hdr;
        char buf[CMSG_SPACE(sizeof(int))];
    } cmsg_buf;
    memset(&cmsg_buf, 0, sizeof(cmsg_buf));

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.buf;
    msg.msg_controllen = sizeof(cmsg_buf.buf);

    ssize_t n = recvmsg(socket, &msg, 0);
    if (n < 0) {
        perror("ipc_recv_fd: recvmsg");
        return -1;
    }
    if (n == 0) {
        fprintf(stderr, "ipc_recv_fd: connection closed\n");
        return -1;
    }

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
        fprintf(stderr, "ipc_recv_fd: no fd in message\n");
        return -1;
    }

    int fd;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
    return fd;
}

int ipc_send_bytes(int socket, const void *buf, size_t len) {
    const char *p = buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = write(socket, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("ipc_send_bytes");
            return -1;
        }
        p += n;
        remaining -= n;
    }
    return 0;
}

int ipc_recv_bytes(int socket, void *buf, size_t len) {
    char *p = buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = read(socket, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("ipc_recv_bytes");
            return -1;
        }
        if (n == 0) {
            fprintf(stderr, "ipc_recv_bytes: connection closed (%zu bytes remaining)\n",
                    remaining);
            return -1;
        }
        p += n;
        remaining -= n;
    }
    return 0;
}
