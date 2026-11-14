/* Exercise real launcher deadlines with deterministic clocks and child state. */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#include <winsock2.h>
#include <windows.h>
#else
#define _GNU_SOURCE 1
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long long elapsed_ms;
static int sleep_calls, clock_calls, child_polls, last_sleep;
static int sleep_step = -1, clock_fail_at, exit_after_sleeps = -1;

static int read_clock(long long *value)
{
    if (++clock_calls == clock_fail_at) return 0;
    *value = elapsed_ms;
    return 1;
}

static void advance_clock(int requested)
{
    if (++sleep_calls > 1000) {
        fputs("child wait did not reach its deadline\n", stderr);
        exit(2);
    }
    last_sleep = requested;
    elapsed_ms += sleep_step < 0 ? requested : sleep_step;
}

static int child_exited(void)
{
    ++child_polls;
    return exit_after_sleeps >= 0 && sleep_calls >= exit_after_sleeps;
}

#ifdef _WIN32
static BOOL WINAPI timing_counter(LARGE_INTEGER *value)
{
    long long current;
    if (!read_clock(&current)) return FALSE;
    value->QuadPart = current;
    return TRUE;
}

static BOOL WINAPI timing_frequency(LARGE_INTEGER *value)
{
    value->QuadPart = 1000;
    return TRUE;
}

static void WINAPI timing_sleep(DWORD delay) { advance_clock((int)delay); }

static DWORD WINAPI timing_wait(HANDLE process, DWORD timeout)
{
    (void)process; (void)timeout;
    return child_exited() ? WAIT_OBJECT_0 : WAIT_TIMEOUT;
}

static BOOL WINAPI timing_exit_code(HANDLE process, LPDWORD code)
{
    (void)process;
    *code = 0;
    return TRUE;
}

#define QueryPerformanceCounter timing_counter
#define QueryPerformanceFrequency timing_frequency
#define Sleep timing_sleep
#define WaitForSingleObject timing_wait
#define GetExitCodeProcess timing_exit_code
#else
static int timing_clock(clockid_t clock_id, struct timespec *value)
{
    long long current;
    (void)clock_id;
    if (!read_clock(&current)) { errno = EINVAL; return -1; }
    value->tv_sec = (time_t)(current / 1000);
    value->tv_nsec = (long)(current % 1000) * 1000000L;
    return 0;
}

static int timing_sleep(const struct timespec *delay, struct timespec *remaining)
{
    (void)remaining;
    advance_clock((int)(delay->tv_sec * 1000 + delay->tv_nsec / 1000000));
    if (sleep_step == 1) { errno = EINTR; return -1; }
    return 0;
}

static pid_t timing_wait(pid_t process, int *status, int options)
{
    (void)options;
    if (!child_exited()) return 0;
    if (status) *status = 0;
    return process;
}

#define clock_gettime timing_clock
#define nanosleep timing_sleep
#define waitpid timing_wait
#endif

/* Include OS declarations before replacing API calls in the implementation. */
#define main fox_launcher_main
#include "../packaging/launcher/launcher.c"
#undef main

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: %s (time=%lld, sleeps=%d, polls=%d)\n", \
                mode, __LINE__, #condition, elapsed_ms, sleep_calls, child_polls); \
        return 1; \
    } \
} while (0)

int main(int argc, char **argv)
{
    child_t child;
    const char *mode = argc == 2 ? argv[1] : "";
    memset(&child, 0, sizeof child);
    child.started = 1;
#ifndef _WIN32
    child.pid = 123; /* waitpid returns zero while this fixture is still alive. */
#endif
    if (!strcmp(mode, "oversleep")) {
        sleep_step = 175;
        CHECK(child_wait(&child, 250) == 0);
        CHECK(elapsed_ms == 350 && sleep_calls == 2 && last_sleep == 75);
    } else if (!strcmp(mode, "interrupted-sleep")) {
        sleep_step = 1;
        CHECK(child_wait(&child, 250) == 0);
        CHECK(elapsed_ms == 250 && sleep_calls == 250 && last_sleep == 1);
    } else if (!strcmp(mode, "short-timeout")) {
        CHECK(child_wait(&child, 50) == 0);
        CHECK(elapsed_ms == 50 && sleep_calls == 1 && last_sleep == 50);
    } else if (!strcmp(mode, "zero-timeout")) {
        CHECK(child_wait(&child, 0) == 0);
        CHECK(child_polls == 1 && clock_calls == 0 && sleep_calls == 0);
    } else if (!strcmp(mode, "already-exited")) {
        child.exited = 1;
        clock_fail_at = 1;
        CHECK(child_wait(&child, 0) == 1);
        CHECK(child_polls == 0 && clock_calls == 0 && sleep_calls == 0);
    } else if (!strcmp(mode, "exit-during-wait")) {
        exit_after_sleeps = 1;
        CHECK(child_wait(&child, 250) == 1);
        CHECK(elapsed_ms == 100 && sleep_calls == 1 && child.exited && child.code == 0);
    } else if (!strcmp(mode, "clock-failure")) {
        clock_fail_at = 1;
        CHECK(child_wait(&child, 250) == 0);
        CHECK(child_polls == 1 && clock_calls == 1 && sleep_calls == 0);
    } else if (!strcmp(mode, "clock-failure-during-wait")) {
        clock_fail_at = 2;
        CHECK(child_wait(&child, 250) == 0);
        CHECK(clock_calls == 2 && elapsed_ms == 100 && sleep_calls == 1);
    } else {
        fputs("unknown launcher timing scenario\n", stderr);
        return 2;
    }
    return 0;
}
