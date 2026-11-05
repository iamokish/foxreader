#define _GNU_SOURCE 1
#include "terminal_linux.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define HANDOFF_TIMEOUT_MS 10000
#define PEER_TIMEOUT_MS 1000

static long long now_ms(void)
{
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return -1;
    return (long long)value.tv_sec * 1000 + value.tv_nsec / 1000000;
}

static int wait_socket(int fd, short events, int timeout)
{
    struct pollfd item = {fd, events, 0};
    /* Interruption cancels the handoff rather than ignoring a logout/close. */
    return poll(&item, 1, timeout) > 0 && (item.revents & events);
}

static int same_user(int fd)
{
    struct ucred peer;
    socklen_t size = sizeof peer;
    return getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &size) == 0 &&
           size == sizeof peer && peer.uid == geteuid();
}

int fox_terminal_available(void)
{
    int fd, available;
    if (!isatty(STDIN_FILENO) || (!isatty(STDOUT_FILENO) && !isatty(STDERR_FILENO))) return 0;
    fd = open("/dev/tty", O_RDWR | O_NOCTTY | O_CLOEXEC);
    if (fd < 0) return 0;
    available = tcgetpgrp(fd) == getpgrp();
    close(fd);
    return available;
}

int fox_terminal_connect(const char *socket_path)
{
    struct sockaddr_un address = {0};
    int fd, error = 0, result = -1;
    socklen_t size = sizeof error;
    char answer;
    if (!fox_terminal_available() || !socket_path ||
        strlen(socket_path) >= sizeof address.sun_path) return -1;
    address.sun_family = AF_UNIX;
    strcpy(address.sun_path, socket_path);
    fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0) return -1;
    if (connect(fd, (struct sockaddr *)&address, sizeof address) != 0) {
        if (errno != EINPROGRESS || !wait_socket(fd, POLLOUT, PEER_TIMEOUT_MS) ||
            getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) != 0 || error) goto done;
    }
    if (same_user(fd) && send(fd, "R", 1, MSG_NOSIGNAL) == 1 &&
        wait_socket(fd, POLLIN, PEER_TIMEOUT_MS) && recv(fd, &answer, 1, 0) == 1 && answer == 'G')
        result = 0;
done:
    close(fd);
    return result;
}

/* Terminal servers may reuse an existing process and discard inherited FDs.
 * Use a private Unix socket and command arguments, rather than an inherited
 * pipe or the terminal emulator's exit code, to confirm the actual successor. */
static int wait_successor(int listener, pid_t terminal, int (*cancelled)(void))
{
    long long started = now_ms(), deadline = started + HANDOFF_TIMEOUT_MS;
    int status, reaped = 0;
    for (long long current = started; current >= 0 && current < deadline; current = now_ms()) {
        struct pollfd item = {listener, POLLIN, 0};
        int ready;
        if (cancelled && cancelled()) break;
        ready = poll(&item, 1, 100);
        if (ready < 0 || (cancelled && cancelled())) break;
        if (ready > 0 && (item.revents & POLLIN)) {
            int peer = accept4(listener, NULL, NULL, SOCK_CLOEXEC | SOCK_NONBLOCK);
            if (peer >= 0) {
                char request;
                int accepted = same_user(peer) && wait_socket(peer, POLLIN, PEER_TIMEOUT_MS) &&
                    recv(peer, &request, 1, 0) == 1 && request == 'R' &&
                    !(cancelled && cancelled()) &&
                    send(peer, "G", 1, MSG_NOSIGNAL) == 1;
                close(peer);
                if (accepted) return 1;
            }
        }
        if (!reaped) {
            pid_t ended = waitpid(terminal, &status, WNOHANG);
            if (ended == terminal) {
                reaped = 1;
                if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return 0;
                /* A successful server client can exit before its terminal
                 * child starts. Continue waiting for that child's handshake. */
            } else if (ended < 0 && errno != EINTR) {
                break;
            }
        }
    }
    if (!reaped) {
        /* Only terminate the launcher process we created, never an existing
         * terminal server or a session that already completed the handoff. */
        kill(terminal, SIGKILL);
        /* Do not extend the handoff deadline for a stuck emulator. The caller
         * exits after a failed handoff, which reparents any unreaped child. */
        (void)waitpid(terminal, NULL, WNOHANG);
    }
    return -1;
}

int fox_terminal_handoff(const char *executable, int (*cancelled)(void))
{
    /* Each terminal receives an argv vector. Paths containing spaces, quotes,
     * Unicode or shell metacharacters are never interpreted by a shell. */
    static const char *const terminals[][4] = {
        {"xdg-terminal-exec", "--", NULL},
        {"x-terminal-emulator", "-e", NULL},
        {"gnome-terminal", "--wait", "--", NULL},
        {"konsole", "--separate", "-e", NULL},
        {"xfce4-terminal", "--disable-server", "-x", NULL},
        {"mate-terminal", "--disable-factory", "-x", NULL},
        {"kgx", "--", NULL},
        {"foot", "--", NULL},
        {"xterm", "-e", NULL},
    };
    char directory[] = "/tmp/fox-reader-terminal-XXXXXX";
    struct sockaddr_un address = {0};
    int listener = -1, result = -1;
    size_t index;
    if (!executable || executable[0] != '/' || !mkdtemp(directory)) return -1;
    address.sun_family = AF_UNIX;
    snprintf(address.sun_path, sizeof address.sun_path, "%s/control", directory);
    listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (listener < 0 || bind(listener, (struct sockaddr *)&address, sizeof address) != 0 ||
        listen(listener, 4) != 0) goto done;
    for (index = 0; index < sizeof terminals / sizeof terminals[0]; ++index) {
        char *args[9];
        size_t at = 0;
        pid_t child;
        int accepted;
        if (cancelled && cancelled()) break;
        while (terminals[index][at]) {
            args[at] = (char *)terminals[index][at];
            ++at;
        }
        args[at++] = (char *)executable;
        args[at++] = "--terminal-manager";
        args[at++] = address.sun_path;
        args[at] = NULL;
        child = fork();
        if (child < 0) break;
        if (child == 0) {
            setsid();
            execvp(args[0], args);
            _exit(127);
        }
        accepted = wait_successor(listener, child, cancelled);
        if (accepted > 0) { result = 0; break; }
        /* A timeout may leave a delayed server request. Do not open another
         * terminal: remove the socket, so that late child cannot start a backend. */
        if (accepted < 0) break;
    }
done:
    if (listener >= 0) close(listener);
    unlink(address.sun_path);
    rmdir(directory);
    return result;
}

#ifndef FOX_NO_WINDOW
void fox_terminal_notify(const char *message)
{
    pid_t child = fork();
    if (child == 0) {
        int null_fd;
        setsid();
        alarm(5);
        null_fd = open("/dev/null", O_RDWR);
        if (null_fd >= 0) {
            dup2(null_fd, STDIN_FILENO);
            dup2(null_fd, STDOUT_FILENO);
            dup2(null_fd, STDERR_FILENO);
            if (null_fd > STDERR_FILENO) close(null_fd);
        }
        execlp("notify-send", "notify-send", "--app-name=Fox Reader", "--icon=dialog-error",
               "Fox Reader could not start", message, (char *)NULL);
        _exit(127);
    }
    /* This is best-effort feedback immediately before the launcher exits;
     * no backend has been started, even if notifications are unavailable. */
}
#endif
