// Step 3: IPC — socketpair, fd passing, byte transfer
#include "ipc.h"
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>

int main(void) {
    int fds[2];
    assert(ipc_socketpair(fds) == 0);

    pid_t pid = fork();
    assert(pid >= 0);

    if (pid == 0) {
        // Child: receive fd and data
        close(fds[0]);
        int sock = fds[1];

        // Receive a file descriptor
        int recv_fd = ipc_recv_fd(sock);
        assert(recv_fd >= 0);
        // Write to it (it's /dev/null)
        assert(write(recv_fd, "hello", 5) == 5);
        close(recv_fd);

        // Receive 8 bytes of data
        uint64_t data;
        assert(ipc_recv_bytes(sock, &data, sizeof(data)) == 0);
        assert(data == 0xDEADBEEFCAFEBABEULL);

        // Send confirmation
        uint32_t ok = 1;
        assert(ipc_send_bytes(sock, &ok, sizeof(ok)) == 0);

        close(sock);
        _exit(0);
    }

    // Parent: send fd and data
    close(fds[1]);
    int sock = fds[0];

    int null_fd = open("/dev/null", O_WRONLY);
    assert(null_fd >= 0);
    assert(ipc_send_fd(sock, null_fd) == 0);
    close(null_fd);

    uint64_t data = 0xDEADBEEFCAFEBABEULL;
    assert(ipc_send_bytes(sock, &data, sizeof(data)) == 0);

    // Receive confirmation
    uint32_t ok = 0;
    assert(ipc_recv_bytes(sock, &ok, sizeof(ok)) == 0);
    assert(ok == 1);

    close(sock);

    int status;
    waitpid(pid, &status, 0);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);

    printf("test_ipc_fd: OK\n");
    return 0;
}
