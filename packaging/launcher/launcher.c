/*
 * Fox Reader launcher
 * ===================
 *
 * A standalone console program. It sits at the root of the distribution, starts
 * bin/fox-reader if it is not already running, opens the UI, and stays in the
 * foreground as the window you close to stop the whole thing.
 *
 * No dependencies beyond libc and the OS: winsock and kernel32 on Windows,
 * nothing extra on Linux. Everything it needs to know it reads from
 * config/fox_config.yaml.
 *
 * Four decisions worth explaining, because none of them is the obvious one.
 *
 * 1. The child is detached from this console (Windows) or put in its own process
 *    group (Linux). It is tempting to let it share the console so its logs
 *    appear for free -- but then clicking the X sends CTRL_CLOSE_EVENT to the
 *    child as well as to us, Python has no handler for it, and the backend dies
 *    on the spot without running its shutdown path. No shutdown path means the
 *    cache is never cleared, which is exactly what we were asked to guarantee.
 *    So the child gets no console, we take its output over a pipe and relay it,
 *    and every signal that reaches it comes from us.
 *
 * 2. Shutdown is an HTTP request, not a signal. POST /api/shutdown flips
 *    uvicorn's should_exit, which unwinds the ASGI lifespan -- releasing the
 *    models and clearing cache/ on the way. A signal cannot do this on Windows:
 *    os.kill there is TerminateProcess, which runs no Python at all.
 *
 * 3. A Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (PR_SET_PDEATHSIG on
 *    Linux) is the backstop under all of it. Windows gives a console program
 *    about five seconds after the X is clicked before killing it; if a slow
 *    shutdown runs past that, the OS closes our last job handle as we die and
 *    the child goes with it. The backend cannot be orphaned even if this program
 *    is killed outright.
 *
 * 4. One TCP connect answers two questions. uvicorn creates its listening socket
 *    only after the lifespan startup finishes, so a successful connect means the
 *    models are loaded and the app is ready -- the same probe is both the
 *    "is it already running" test at startup and the readiness wait afterwards.
 *
 * Build: see packaging/toolchain.py, which finds a compiler and invokes it. By
 * hand it is a single translation unit:
 *
 *     cl /O2 /MT launcher.c /Fe:launcher.exe ws2_32.lib
 *     gcc -O2 -o launcher launcher.c
 */

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef _CRT_SECURE_NO_WARNINGS
#    define _CRT_SECURE_NO_WARNINGS 1
#  endif
   /* Windows 7. Not cosmetic: MinGW headers gate the job-object limit structures
    * behind this, and without it the build fails on symbols that plainly exist. */
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT 0x0601
#  endif
   /* winsock2.h must precede windows.h, which would otherwise pull in the
    * original winsock.h and collide with it. */
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
   /* Explicit, and not redundant. MSVC and mingw-w64 reach winerror.h through
    * windows.h, but MinGW.org's windows.h includes only windef.h and winbase.h,
    * so ERROR_SHARING_VIOLATION and friends are undeclared there. The header is
    * include-guarded everywhere, so asking for it by name costs nothing. */
#  include <winerror.h>
#  include <io.h>
#else
#  define _GNU_SOURCE 1
#  include <arpa/inet.h>
#  include <errno.h>
#  include <fcntl.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <signal.h>
#  include <sys/file.h>
#  include <sys/select.h>
#  include <sys/socket.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <time.h>
#  include <unistd.h>
#  ifdef __linux__
#    include <sys/prctl.h>
#  endif
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FOX_LAUNCHER_VERSION "1.0.0"

#define DEFAULT_HOST "127.0.0.1"
#define DEFAULT_PORT 7954

#define PROBE_TIMEOUT_MS 1500  /* loopback answers instantly or not at all */
#define HTTP_TIMEOUT_MS 5000
#define READY_POLL_MS 400
#define OWNER_POLL_MS 250
#define ATTACH_POLL_MS 2000
#define SHUTDOWN_GRACE_MS 20000 /* releasing models and clearing cache/ */
#define FORCE_GRACE_MS 3000
/* The close button, a logoff and a shutdown all come with a guillotine: Windows
 * terminates this process a few seconds after the handler is entered -- five is
 * the documented figure -- whatever the handler is still doing. So those paths
 * get a ladder sized to fit inside it rather than the leisurely one Ctrl+C can
 * afford, where nothing is going to interrupt us. Overrunning is survivable,
 * because process teardown closes the last job handle and the child dies with
 * it, but it means the backend is killed instead of clearing cache/ -- the one
 * part of a clean shutdown anybody can see.
 *
 * An idle backend answers the shutdown request and exits in about 0.8s, so this
 * leaves roughly three times the headroom it needs, and the whole ladder --
 * PROBE_TIMEOUT_MS asking, then these two -- comes to 4.4s worst case. A backend
 * with models resident and work in flight can take longer; that one gets forced
 * down, which is the same outcome the OS would have imposed anyway, only at a
 * moment of our choosing and with the child reliably dead. */
#define CLOSE_GRACE_MS 2500
#define CLOSE_FORCE_MS 400
#define PEER_NOTE_MS 20000 /* how often to say we are still waiting on a peer */

#define MAX_HOST 256
#define MAX_URL 320
#define TOKEN_CHARS 32 /* 16 random bytes, hex */

/* ------------------------------------------------------------------------- */
/* Paths.                                                                    */
/*                                                                           */
/* Wide on Windows throughout. The narrow API would break on any install path */
/* the active code page cannot spell, which in a non-Latin locale is most of  */
/* them, and "D:\Users\<name>\Fox Reader" is a completely ordinary place to   */
/* put this. Printing goes through to_utf8 so there is only ever one stream   */
/* orientation.                                                              */
/* ------------------------------------------------------------------------- */

#ifdef _WIN32
typedef wchar_t pchar;
#  define PL(s) L##s
#  define PSEP L'\\'
#  define p_strlen wcslen
#  define p_fopen _wfopen
#else
typedef char pchar;
#  define PL(s) s
#  define PSEP '/'
#  define p_strlen strlen
#  define p_fopen fopen
#endif

#define PATH_CAP 4096

#ifndef FOX_BIN_DIR
#  define FOX_BIN_DIR PL("bin")
#endif

/* Candidate names for the compiled backend, in preference order. The build
 * names it fox-reader; the others are here so a hand-made bin/ still works. */
static const pchar *const BACKEND_NAMES[] = {
#ifdef _WIN32
    PL("fox-reader.exe"), PL("fox_reader.exe"), PL("foxreader.exe"),
#else
    PL("fox-reader"), PL("fox_reader"), PL("foxreader"),
#endif
};

#define BACKEND_NAME_COUNT ((int)(sizeof(BACKEND_NAMES) / sizeof(BACKEND_NAMES[0])))

/* ------------------------------------------------------------------------- */
/* Output.                                                                   */
/*                                                                           */
/* Plain text, no escape sequences: a black and white console, as asked. The  */
/* backend's own logs come through the relay thread and land on the same      */
/* stream, so everything is flushed as it is written to keep the two in order.*/
/* ------------------------------------------------------------------------- */

static void out(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
    fflush(stdout);
}

static void errline(const char *fmt, ...)
{
    va_list ap;
    fflush(stdout);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fflush(stderr);
}

static void sleep_ms(int ms)
{
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}

#ifdef _WIN32
/* True when this console exists only for us -- i.e. the user double-clicked.
 * Then an error message has to be held on screen or it is never read. */
static int owns_console(void)
{
    DWORD pids[4];
    DWORD n = GetConsoleProcessList(pids, 4);
    return n == 1;
}
#endif

static void maybe_pause(void)
{
#ifdef _WIN32
    if (owns_console()) {
        out("\nPress Enter to close this window.\n");
        (void)getchar();
    }
#endif
}

static void fatal(const char *fmt, ...)
{
    va_list ap;
    fflush(stdout);
    fputs("error: ", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
    maybe_pause();
    exit(1);
}

#ifdef _WIN32
/* Wide to UTF-8. Caller frees. Used only for printing paths. */
static char *to_utf8(const wchar_t *w)
{
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    char *buf;
    if (n <= 0)
        return NULL;
    buf = (char *)malloc((size_t)n);
    if (!buf)
        return NULL;
    if (WideCharToMultiByte(CP_UTF8, 0, w, -1, buf, n, NULL, NULL) <= 0) {
        free(buf);
        return NULL;
    }
    return buf;
}

static wchar_t *to_wide(const char *s)
{
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    wchar_t *buf;
    if (n <= 0)
        return NULL;
    buf = (wchar_t *)malloc((size_t)n * sizeof(wchar_t));
    if (!buf)
        return NULL;
    if (MultiByteToWideChar(CP_UTF8, 0, s, -1, buf, n) <= 0) {
        free(buf);
        return NULL;
    }
    return buf;
}

#  define PRINTABLE(p) to_utf8(p)
#  define PRINTABLE_FREE(s) free(s)
#else
#  define PRINTABLE(p) ((char *)(p))
#  define PRINTABLE_FREE(s) ((void)(s))
#endif

/* dir + leaf, with the separator sorted out. 0 if it would not fit. */
static int p_join(pchar *dst, size_t cap, const pchar *dir, const pchar *leaf)
{
    size_t dl = p_strlen(dir), ll = p_strlen(leaf), i;
    int sep;

    sep = dl > 0 && dir[dl - 1] != PSEP;
#ifdef _WIN32
    if (sep && dir[dl - 1] == L'/')
        sep = 0;
#endif
    if (dl + (size_t)(sep ? 1 : 0) + ll + 1 > cap)
        return 0;

    memcpy(dst, dir, dl * sizeof(pchar));
    i = dl;
    if (sep)
        dst[i++] = PSEP;
    memcpy(dst + i, leaf, ll * sizeof(pchar));
    dst[i + ll] = 0;
    return 1;
}

static int path_exists(const pchar *path)
{
#ifdef _WIN32
    DWORD attr = GetFileAttributesW(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
#endif
}

/* The directory holding this executable. That is the distribution root: the
 * launcher sits beside bin/, config/, cache/, fonts/ and models/. */
static int exe_dir(pchar *dst, size_t cap)
{
#ifdef _WIN32
    DWORD n = GetModuleFileNameW(NULL, dst, (DWORD)cap);
    size_t i;
    if (n == 0 || n >= cap)
        return 0;
    for (i = n; i > 0; i--) {
        if (dst[i - 1] == L'\\' || dst[i - 1] == L'/') {
            dst[i - 1] = 0;
            return 1;
        }
    }
    return 0;
#else
    ssize_t n = readlink("/proc/self/exe", dst, cap - 1);
    size_t i;
    if (n <= 0) {
        /* No procfs (a container without it, or a non-Linux Unix). argv[0] is
         * set by main() before this runs in that case. */
        return 0;
    }
    dst[n] = 0;
    for (i = (size_t)n; i > 0; i--) {
        if (dst[i - 1] == '/') {
            dst[i - 1] = 0;
            return 1;
        }
    }
    return 0;
#endif
}

/* ------------------------------------------------------------------------- */
/* Config.                                                                   */
/*                                                                           */
/* config/fox_config.yaml is written by ConfigManager as a flat mapping of    */
/* key: value lines -- tests/test_config.py pins that shape precisely so this */
/* parser can stay this small. user_config.yaml, if it exists, wins: that is  */
/* the same precedence the Python side applies, and the two disagreeing about */
/* the port would leave the launcher probing the wrong one forever.           */
/* ------------------------------------------------------------------------- */

static char *read_whole_file(const pchar *path, size_t *out_len)
{
    FILE *f = p_fopen(path, PL("rb"));
    char *buf;
    long size;
    size_t got;

    *out_len = 0;
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    size = ftell(f);
    /* A config file is a few hundred bytes. Anything past 1 MiB is not one. */
    if (size < 0 || size > 1024 * 1024) {
        fclose(f);
        return NULL;
    }
    rewind(f);

    buf = (char *)malloc((size_t)size + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    got = fread(buf, 1, (size_t)size, f);
    fclose(f);
    buf[got] = 0;
    *out_len = got;
    return buf;
}

static void trim_right(char *s)
{
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\r'))
        s[--n] = 0;
}

/* One `key: value` line. Fills key/val, or returns 0 for a line we ignore --
 * blank, a comment, or indented (indentation means a nested mapping, and this
 * parser only claims to understand the flat file the app writes). */
static int split_line(const char *line, size_t len, char *key, size_t keycap, char *val, size_t valcap)
{
    size_t i = 0, k = 0, v = 0;

    if (len == 0)
        return 0;
    if (line[0] == ' ' || line[0] == '\t' || line[0] == '#' || line[0] == '\r')
        return 0;
    if (len >= 3 && memcmp(line, "---", 3) == 0)
        return 0;

    while (i < len && line[i] != ':') {
        if (k + 1 < keycap)
            key[k++] = line[i];
        i++;
    }
    if (i >= len)
        return 0; /* no colon: not a mapping entry */
    key[k] = 0;
    trim_right(key);
    if (key[0] == 0)
        return 0;

    i++; /* the colon */
    while (i < len && (line[i] == ' ' || line[i] == '\t'))
        i++;

    if (i < len && (line[i] == '"' || line[i] == '\'')) {
        char quote = line[i++];
        while (i < len && line[i] != quote) {
            if (v + 1 < valcap)
                val[v++] = line[i];
            i++;
        }
        val[v] = 0;
        return 1;
    }

    while (i < len) {
        /* An unquoted value ends at a comment, which YAML requires be preceded
         * by whitespace -- so "1.2.3.4 # home" is an address, not a mangled one. */
        if (line[i] == '#' && v > 0 && (val[v - 1] == ' ' || val[v - 1] == '\t'))
            break;
        if (v + 1 < valcap)
            val[v++] = line[i];
        i++;
    }
    val[v] = 0;
    trim_right(val);
    return 1;
}

static int valid_host(const char *s)
{
    size_t i;
    if (!s[0] || strlen(s) >= MAX_HOST)
        return 0;
    for (i = 0; s[i]; i++) {
        char c = s[i];
        int ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                 c == '.' || c == '-' || c == ':' || c == '_' || c == '%';
        if (!ok)
            return 0;
    }
    return 1;
}

/* Overlays whatever `path` defines onto host/port. Absent keys, an unreadable
 * file and a malformed value all leave the current values alone -- the Python
 * side falls back to the same defaults when it rejects a file, so a config this
 * cannot make sense of still leaves both sides agreeing. */
static void apply_config_file(const pchar *path, char *host, size_t hostcap, int *port)
{
    size_t len = 0, start = 0, i;
    char *data = read_whole_file(path, &len);
    char key[128], val[MAX_HOST + 32];

    if (!data)
        return;

    for (i = 0; i <= len; i++) {
        if (i < len && data[i] != '\n')
            continue;
        if (split_line(data + start, i - start, key, sizeof key, val, sizeof val)) {
            if (strcmp(key, "host") == 0) {
                if (valid_host(val)) {
                    strncpy(host, val, hostcap - 1);
                    host[hostcap - 1] = 0;
                }
            } else if (strcmp(key, "port") == 0) {
                char *end = NULL;
                long n = strtol(val, &end, 10);
                if (end && end != val && n >= 1 && n <= 65535)
                    *port = (int)n;
            }
        }
        start = i + 1;
    }

    free(data);
}

static void read_config(const pchar *root, char *host, size_t hostcap, int *port)
{
    pchar dir[PATH_CAP], path[PATH_CAP];

    strncpy(host, DEFAULT_HOST, hostcap - 1);
    host[hostcap - 1] = 0;
    *port = DEFAULT_PORT;

    if (!p_join(dir, PATH_CAP, root, PL("config")))
        return;
    if (p_join(path, PATH_CAP, dir, PL("fox_config.yaml")))
        apply_config_file(path, host, hostcap, port);
    if (p_join(path, PATH_CAP, dir, PL("user_config.yaml")))
        apply_config_file(path, host, hostcap, port);
}

/* The address to *connect* to. A backend bound to 0.0.0.0 is listening on
 * loopback as well, and connecting to 0.0.0.0 is not a thing you can do. */
static const char *connect_host(const char *host)
{
    if (strcmp(host, "0.0.0.0") == 0 || host[0] == 0)
        return "127.0.0.1";
    if (strcmp(host, "::") == 0 || strcmp(host, "[::]") == 0)
        return "::1";
    return host;
}

static void build_url(char *url, size_t cap, const char *host, int port)
{
    /* An IPv6 literal needs brackets in a URL; the marker is a colon, which no
     * hostname or IPv4 address contains. */
    if (strchr(host, ':'))
        snprintf(url, cap, "http://[%s]:%d/", host, port);
    else
        snprintf(url, cap, "http://%s:%d/", host, port);
}

/* ------------------------------------------------------------------------- */
/* HTTP over a raw socket. Two requests, both tiny: GET /api/health and       */
/* POST /api/shutdown.                                                       */
/* ------------------------------------------------------------------------- */

#ifdef _WIN32
typedef SOCKET sock_t;
#  define SOCK_BAD INVALID_SOCKET
#  define sock_close closesocket
#else
typedef int sock_t;
#  define SOCK_BAD (-1)
#  define sock_close close
#endif

static int net_start(void)
{
#ifdef _WIN32
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
#else
    return 1;
#endif
}

static void set_timeout(sock_t s, int ms)
{
#ifdef _WIN32
    DWORD tv = (DWORD)ms;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof tv);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tv, sizeof tv);
#else
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
}

static int set_blocking(sock_t s, int blocking)
{
#ifdef _WIN32
    u_long mode = blocking ? 0 : 1;
    return ioctlsocket(s, FIONBIO, &mode) == 0;
#else
    int flags = fcntl(s, F_GETFL, 0);
    if (flags < 0)
        return 0;
    if (blocking)
        flags &= ~O_NONBLOCK;
    else
        flags |= O_NONBLOCK;
    return fcntl(s, F_SETFL, flags) == 0;
#endif
}

/* A connect that gives up on schedule. A blocking connect to an address that
 * silently drops packets waits out the SYN retry -- twenty seconds or so -- and
 * this runs on the path where the answer "nothing is listening" has to be quick. */
static sock_t tcp_connect(const char *host, int port, int timeout_ms)
{
    struct addrinfo hints, *res = NULL, *ai;
    char portstr[16];
    sock_t s = SOCK_BAD;

    snprintf(portstr, sizeof portstr, "%d", port);
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    if (getaddrinfo(host, portstr, &hints, &res) != 0 || !res)
        return SOCK_BAD;

    for (ai = res; ai; ai = ai->ai_next) {
        fd_set wfds, efds;
        struct timeval tv;
        int err = 0, r;
#ifdef _WIN32
        int errlen = (int)sizeof err;
#else
        socklen_t errlen = (socklen_t)sizeof err;
#endif

        s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (s == SOCK_BAD)
            continue;

        if (!set_blocking(s, 0)) {
            sock_close(s);
            s = SOCK_BAD;
            continue;
        }

        r = connect(s, ai->ai_addr, (int)ai->ai_addrlen);
        if (r == 0) {
            set_blocking(s, 1);
            break; /* loopback usually lands here */
        }

#ifdef _WIN32
        if (WSAGetLastError() != WSAEWOULDBLOCK) {
            sock_close(s);
            s = SOCK_BAD;
            continue;
        }
#else
        if (errno != EINPROGRESS) {
            sock_close(s);
            s = SOCK_BAD;
            continue;
        }
#endif

        FD_ZERO(&wfds);
        FD_ZERO(&efds);
        FD_SET(s, &wfds);
        FD_SET(s, &efds);
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;

        r = select((int)s + 1, NULL, &wfds, &efds, &tv);
        if (r <= 0 || FD_ISSET(s, &efds)) {
            sock_close(s);
            s = SOCK_BAD;
            continue;
        }
        /* Writable is not the same as connected: the error has to be collected. */
        if (getsockopt(s, SOL_SOCKET, SO_ERROR, (char *)&err, &errlen) != 0 || err != 0) {
            sock_close(s);
            s = SOCK_BAD;
            continue;
        }

        set_blocking(s, 1);
        break;
    }

    freeaddrinfo(res);
    return s;
}

static int send_all(sock_t s, const char *buf, size_t len)
{
    size_t sent = 0;
    while (sent < len) {
        int n = (int)send(s, buf + sent, (int)(len - sent), 0);
        if (n <= 0)
            return 0;
        sent += (size_t)n;
    }
    return 1;
}

/*
 * One request, one response. `status` gets the HTTP status; `body` gets as much
 * of the response as fits (headers included -- the caller only ever searches it
 * for a marker). Returns 0 when the server could not be reached at all, which
 * is the distinction that matters: not-listening and answered-with-403 are very
 * different situations.
 */
static int http_req(const char *host, int port, const char *method, const char *path,
                    const char *token, int connect_ms, int *status, char *body, size_t bodycap)
{
    sock_t s;
    char req[1024];
    char buf[4096];
    size_t total = 0;
    int n, len;

    if (status)
        *status = 0;
    if (body && bodycap)
        body[0] = 0;

    s = tcp_connect(host, port, connect_ms);
    if (s == SOCK_BAD)
        return 0;

    set_timeout(s, HTTP_TIMEOUT_MS);

    len = snprintf(req, sizeof req,
                   "%s %s HTTP/1.1\r\n"
                   "Host: %s:%d\r\n"
                   "User-Agent: fox-reader-launcher/%s\r\n"
                   "Accept: application/json\r\n"
                   "Content-Length: 0\r\n"
                   "%s%s%s"
                   "Connection: close\r\n"
                   "\r\n",
                   method, path, host, port, FOX_LAUNCHER_VERSION,
                   token && token[0] ? "X-Fox-Reader-Token: " : "",
                   token && token[0] ? token : "",
                   token && token[0] ? "\r\n" : "");

    if (len <= 0 || (size_t)len >= sizeof req || !send_all(s, req, (size_t)len)) {
        sock_close(s);
        return 0;
    }

    /* Connection: close means the read ends at EOF. Anything past the buffer is
     * of no interest -- the status line and a short JSON body are all we read. */
    while (total + 1 < sizeof buf) {
        n = (int)recv(s, buf + total, (int)(sizeof buf - 1 - total), 0);
        if (n <= 0)
            break;
        total += (size_t)n;
    }
    sock_close(s);
    buf[total] = 0;

    if (total == 0)
        return 0;

    if (status && total > 12 && memcmp(buf, "HTTP/", 5) == 0) {
        const char *sp = strchr(buf, ' ');
        if (sp)
            *status = atoi(sp + 1);
    }
    if (body && bodycap) {
        size_t copy = total < bodycap - 1 ? total : bodycap - 1;
        memcpy(body, buf, copy);
        body[copy] = 0;
    }
    return 1;
}

typedef enum {
    PROBE_DOWN = 0,   /* nothing is listening */
    PROBE_FOX = 1,    /* Fox Reader, up and past its startup */
    PROBE_FOREIGN = 2 /* the port is taken by something else */
} probe_t;

static probe_t probe(const char *host, int port)
{
    char body[2048];
    int status = 0;

    if (!http_req(host, port, "GET", "/api/health", NULL, PROBE_TIMEOUT_MS, &status, body, sizeof body))
        return PROBE_DOWN;

    /* The marker exists for exactly this: telling our own backend from whatever
     * else might have claimed the port. */
    if (status == 200 && strstr(body, "fox-reader"))
        return PROBE_FOX;

    return PROBE_FOREIGN;
}

/* ------------------------------------------------------------------------- */
/* Shutdown token.                                                           */
/*                                                                           */
/* /api/shutdown is on loopback, and a page in the user's browser can POST to */
/* loopback cross-origin -- it cannot read the reply, but the request lands.  */
/* A token in a custom header forces a preflight that no such page will pass, */
/* so drive-by shutdowns stop being possible. It is passed to the child in its */
/* environment and never written anywhere: a launcher that did not start the  */
/* backend has no token, which is the point, because it must not be able to   */
/* stop it.                                                                  */
/* ------------------------------------------------------------------------- */

static void make_token(char *out_hex, size_t cap)
{
    static const char hex[] = "0123456789abcdef";
    unsigned char raw[16];
    size_t i, n = sizeof raw;
    int ok = 0;

    if (cap < n * 2 + 1)
        n = (cap - 1) / 2;

#ifdef _WIN32
    {
        /* RtlGenRandom, by its export name. Loaded rather than linked so the
         * build needs no import library beyond ws2_32. */
        typedef BOOLEAN(WINAPI * gen_random_fn)(PVOID, ULONG);
        HMODULE adv = LoadLibraryW(L"advapi32.dll");
        if (adv) {
            gen_random_fn gen = (gen_random_fn)(void *)GetProcAddress(adv, "SystemFunction036");
            if (gen && gen(raw, (ULONG)n))
                ok = 1;
            FreeLibrary(adv);
        }
    }
#else
    {
        FILE *f = fopen("/dev/urandom", "rb");
        if (f) {
            ok = fread(raw, 1, n, f) == n;
            fclose(f);
        }
    }
#endif

    if (!ok) {
        /* Not a good token, but a shutdown request still has to come from
         * loopback, so the fallback is not the only thing standing there. */
        unsigned long seed;
#ifdef _WIN32
        LARGE_INTEGER qpc;

        /* GetTickCount, not GetTickCount64: MinGW.org's w32api never declares
         * the 64-bit form, and a 49-day wrap means nothing to a seed. The
         * performance counter supplies the sub-millisecond bits it lacks. */
        if (!QueryPerformanceCounter(&qpc))
            qpc.QuadPart = 0;
        seed = (unsigned long)GetTickCount() ^ (unsigned long)qpc.LowPart ^
               ((unsigned long)GetCurrentProcessId() << 16);
#else
        seed = (unsigned long)time(NULL) ^ ((unsigned long)getpid() << 16);
#endif
        for (i = 0; i < n; i++) {
            seed = seed * 1103515245u + 12345u;
            raw[i] = (unsigned char)((seed >> 16) & 0xFF);
        }
    }

    for (i = 0; i < n; i++) {
        out_hex[i * 2] = hex[(raw[i] >> 4) & 0xF];
        out_hex[i * 2 + 1] = hex[raw[i] & 0xF];
    }
    out_hex[n * 2] = 0;
}

/* ------------------------------------------------------------------------- */
/* Single-instance lock.                                                     */
/*                                                                           */
/* The port probe already answers "is the backend up". This answers the other */
/* question: is another launcher in the middle of starting one? Two launchers */
/* opened at the same moment would both probe an empty port, both spawn, and  */
/* the loser's backend would die on bind. Held for as long as we are the      */
/* owner, and not held at all in monitor mode.                                */
/*                                                                           */
/* It lives at <root>/.fox-reader.lock. Not in config/, which holds exactly   */
/* one file by design, and emphatically not in cache/, which the backend      */
/* empties at both ends of its life -- a lock the application deletes from    */
/* under itself is not a lock. Both implementations tie the lock to an open    */
/* handle, so a launcher that is killed releases it: if the file is locked,    */
/* the process that locked it is alive.                                       */
/* ------------------------------------------------------------------------- */

typedef struct {
#ifdef _WIN32
    HANDLE h;
#else
    int fd;
#endif
    pchar path[PATH_CAP];
    int held;
} lock_t;

static void lock_init(lock_t *l)
{
    memset(l, 0, sizeof *l);
#ifdef _WIN32
    l->h = INVALID_HANDLE_VALUE;
#else
    l->fd = -1;
#endif
}

/* 1 acquired, 0 held by someone else, -1 could not try (unwritable location). */
static int lock_acquire(lock_t *l, const pchar *root)
{
    lock_init(l);
    if (!p_join(l->path, PATH_CAP, root, PL(".fox-reader.lock")))
        return -1;

#ifdef _WIN32
    /* FILE_FLAG_DELETE_ON_CLOSE rather than a DeleteFileW at release time: the
     * two-step version has a window where the name is free but a third launcher
     * could already have opened the old file, and then two of them hold what
     * they each think is the lock. Needs DELETE in the access mask. */
    l->h = CreateFileW(l->path, GENERIC_READ | GENERIC_WRITE | DELETE,
                       0, /* no sharing: the open itself is the lock */
                       NULL, OPEN_ALWAYS,
                       FILE_ATTRIBUTE_HIDDEN | FILE_FLAG_DELETE_ON_CLOSE, NULL);
    if (l->h == INVALID_HANDLE_VALUE) {
        DWORD e = GetLastError();
        if (e == ERROR_SHARING_VIOLATION || e == ERROR_LOCK_VIOLATION || e == ERROR_ACCESS_DENIED)
            return 0;
        return -1;
    }
    {
        char line[64];
        DWORD written = 0;
        int n = snprintf(line, sizeof line, "%lu\n", (unsigned long)GetCurrentProcessId());
        SetEndOfFile(l->h);
        if (n > 0)
            WriteFile(l->h, line, (DWORD)n, &written, NULL);
        FlushFileBuffers(l->h);
    }
    l->held = 1;
    return 1;
#else
    l->fd = open(l->path, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (l->fd < 0)
        return -1;
    if (flock(l->fd, LOCK_EX | LOCK_NB) != 0) {
        int held_by_other = (errno == EWOULDBLOCK || errno == EAGAIN || errno == EACCES);
        close(l->fd);
        l->fd = -1;
        return held_by_other ? 0 : -1;
    }
    {
        char line[64];
        int n = snprintf(line, sizeof line, "%ld\n", (long)getpid());
        if (ftruncate(l->fd, 0) == 0 && n > 0) {
            if (write(l->fd, line, (size_t)n) < 0) {
                /* The pid is a courtesy; the lock is the contract. */
            }
        }
    }
    l->held = 1;
    return 1;
#endif
}

static void lock_release(lock_t *l)
{
    if (!l->held)
        return;
    l->held = 0;
#ifdef _WIN32
    if (l->h != INVALID_HANDLE_VALUE) {
        CloseHandle(l->h); /* releases the lock and deletes the file */
        l->h = INVALID_HANDLE_VALUE;
    }
#else
    if (l->fd >= 0) {
        /* Left on disk on purpose. Unlinking it first would open the same
         * two-owner window described above, and an empty dotfile holding the
         * last owner's pid costs nothing -- the next run reopens and relocks it. */
        close(l->fd); /* releases the flock */
        l->fd = -1;
    }
#endif
}

/* ------------------------------------------------------------------------- */
/* The child process.                                                        */
/* ------------------------------------------------------------------------- */

typedef struct {
    int started;
    int exited;
    int code;
#ifdef _WIN32
    HANDLE job;
    HANDLE proc;
    HANDLE pipe_read;
    HANDLE relay;
    DWORD pid;
#else
    pid_t pid;
#endif
} child_t;

#ifdef _WIN32
/* Copies the child's stdout/stderr onto ours. The child has no console of its
 * own -- see note 1 at the top -- so this is the only way its logs are seen. */
static DWORD WINAPI relay_thread(LPVOID param)
{
    HANDLE pipe = (HANDLE)param;
    char buf[4096];
    DWORD n = 0;

    for (;;) {
        if (!ReadFile(pipe, buf, sizeof buf, &n, NULL) || n == 0)
            break;
        fwrite(buf, 1, n, stdout);
        fflush(stdout);
    }
    CloseHandle(pipe);
    return 0;
}
#endif

static void set_child_env(const pchar *root, const char *token)
{
    /* Set on ourselves, then inherited: much less error-prone than hand-building
     * an environment block, and nothing here changes how the launcher behaves
     * (the browser flag is read before this runs). */
#ifdef _WIN32
    wchar_t *wtok = to_wide(token);
    SetEnvironmentVariableW(L"FOX_READER_SHUTDOWN_TOKEN", wtok ? wtok : L"");
    free(wtok);
    /* Unbuffered, or the relayed log arrives in 8 KiB lumps and a console that
     * shows nothing for a minute looks like a hang. */
    SetEnvironmentVariableW(L"PYTHONUNBUFFERED", L"1");
    SetEnvironmentVariableW(L"PYTHONIOENCODING", L"utf-8");
    /* The backend derives the root from its own location, which is bin/. This
     * makes it explicit and survives someone moving the binary. */
    SetEnvironmentVariableW(L"FOX_READER_ROOT", root);
#else
    setenv("FOX_READER_SHUTDOWN_TOKEN", token, 1);
    setenv("PYTHONUNBUFFERED", "1", 1);
    setenv("PYTHONIOENCODING", "utf-8", 1);
    setenv("FOX_READER_ROOT", root, 1);
#endif
}

static int child_start(child_t *c, const pchar *exe, const pchar *root, const char *token)
{
    memset(c, 0, sizeof *c);
    set_child_env(root, token);

#ifdef _WIN32
    {
        SECURITY_ATTRIBUTES sa;
        STARTUPINFOW si;
        PROCESS_INFORMATION pi;
        HANDLE rd = NULL, wr = NULL, nul = INVALID_HANDLE_VALUE;
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits;
        wchar_t cmdline[PATH_CAP + 8];
        size_t elen = p_strlen(exe);

        c->job = INVALID_HANDLE_VALUE;
        c->proc = INVALID_HANDLE_VALUE;

        memset(&sa, 0, sizeof sa);
        sa.nLength = sizeof sa;
        sa.bInheritHandle = TRUE;

        if (!CreatePipe(&rd, &wr, &sa, 64 * 1024))
            return 0;
        /* Our end must not reach the child, or the pipe never reports EOF. */
        SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

        nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                          OPEN_EXISTING, 0, NULL);

        memset(&si, 0, sizeof si);
        si.cb = sizeof si;
        si.dwFlags = STARTF_USESTDHANDLES;
        /* NULL rather than INVALID_HANDLE_VALUE if NUL could not be opened:
         * STARTF_USESTDHANDLES with an invalid handle gives the child a stdin it
         * cannot even test for, where NULL is simply "no stdin". Nothing reads
         * it either way -- this only stops a stray read from misbehaving. */
        si.hStdInput = (nul == INVALID_HANDLE_VALUE) ? NULL : nul;
        si.hStdOutput = wr;
        si.hStdError = wr;

        if (elen + 3 > PATH_CAP) {
            CloseHandle(rd);
            CloseHandle(wr);
            if (nul != INVALID_HANDLE_VALUE)
                CloseHandle(nul);
            return 0;
        }
        cmdline[0] = L'"';
        memcpy(cmdline + 1, exe, elen * sizeof(wchar_t));
        cmdline[elen + 1] = L'"';
        cmdline[elen + 2] = 0;

        memset(&pi, 0, sizeof pi);
        /* DETACHED_PROCESS: no console, so no console control event can reach it
         * behind our back. CREATE_SUSPENDED: it joins the job before it runs a
         * single instruction, so a grandchild cannot be spawned outside it. */
        if (!CreateProcessW(exe, cmdline, NULL, NULL, TRUE,
                            DETACHED_PROCESS | CREATE_SUSPENDED, NULL, root, &si, &pi)) {
            CloseHandle(rd);
            CloseHandle(wr);
            if (nul != INVALID_HANDLE_VALUE)
                CloseHandle(nul);
            return 0;
        }

        c->job = CreateJobObjectW(NULL, NULL);
        if (c->job) {
            memset(&limits, 0, sizeof limits);
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (!SetInformationJobObject(c->job, JobObjectExtendedLimitInformation, &limits, sizeof limits) ||
                !AssignProcessToJobObject(c->job, pi.hProcess)) {
                /* Nested jobs need Windows 8. Without one the explicit stop path
                 * still works; only the kill-if-we-are-killed backstop is lost. */
                CloseHandle(c->job);
                c->job = INVALID_HANDLE_VALUE;
            }
        } else {
            c->job = INVALID_HANDLE_VALUE;
        }

        ResumeThread(pi.hThread);
        CloseHandle(pi.hThread);

        CloseHandle(wr); /* the child holds the only writer now */
        if (nul != INVALID_HANDLE_VALUE)
            CloseHandle(nul);

        c->proc = pi.hProcess;
        c->pid = pi.dwProcessId;
        c->pipe_read = rd;
        c->relay = CreateThread(NULL, 0, relay_thread, rd, 0, NULL);
        if (!c->relay) {
            CloseHandle(rd);
            c->pipe_read = NULL;
        }
        c->started = 1;
        return 1;
    }
#else
    {
        pid_t pid = fork();
        if (pid < 0)
            return 0;

        if (pid == 0) {
            /* Its own process group, so Ctrl+C and a terminal hangup -- both of
             * which go to the foreground group only -- never reach it. Every
             * signal it gets is one we sent deliberately. */
            setpgid(0, 0);
#ifdef __linux__
            /* And if this launcher is killed outright, the kernel takes the
             * backend with it. The Windows equivalent is the job object. */
            prctl(PR_SET_PDEATHSIG, SIGKILL);
            if (getppid() == 1)
                _exit(127); /* already orphaned: the signal above will never come */
#endif
            if (chdir(root) != 0) {
                /* Not fatal: the backend locates everything from its own path. */
            }
            {
                char *argv[2];
                argv[0] = (char *)exe;
                argv[1] = NULL;
                execv(exe, argv);
            }
            _exit(127);
        }

        c->pid = pid;
        c->started = 1;
        return 1;
    }
#endif
}

static int child_alive(child_t *c)
{
    if (!c->started || c->exited)
        return 0;
#ifdef _WIN32
    {
        DWORD code = 0;
        if (WaitForSingleObject(c->proc, 0) == WAIT_OBJECT_0) {
            GetExitCodeProcess(c->proc, &code);
            c->exited = 1;
            c->code = (int)code;
            return 0;
        }
        return 1;
    }
#else
    {
        int st = 0;
        pid_t r = waitpid(c->pid, &st, WNOHANG);
        if (r == c->pid) {
            c->exited = 1;
            c->code = WIFEXITED(st) ? WEXITSTATUS(st) : (WIFSIGNALED(st) ? 128 + WTERMSIG(st) : 1);
            return 0;
        }
        if (r < 0 && errno == ECHILD) {
            c->exited = 1;
            c->code = 0;
            return 0;
        }
        return 1;
    }
#endif
}

/* 1 if it is gone by the deadline. */
static int child_wait(child_t *c, int timeout_ms)
{
    int waited = 0;
    while (child_alive(c)) {
        if (waited >= timeout_ms)
            return 0;
        sleep_ms(100);
        waited += 100;
    }
    return 1;
}

static void child_kill(child_t *c)
{
    if (!c->started || c->exited)
        return;
#ifdef _WIN32
    /* The job, so anything the backend spawned goes too -- a torch worker that
     * outlived its parent would keep a GPU pinned. */
    if (c->job != INVALID_HANDLE_VALUE)
        TerminateJobObject(c->job, 1);
    else
        TerminateProcess(c->proc, 1);
#else
    kill(-c->pid, SIGKILL); /* the whole group we put it in */
    kill(c->pid, SIGKILL);
#endif
}

static void child_close(child_t *c)
{
#ifdef _WIN32
    if (c->relay) {
        /* The pipe's last writer is gone once the child exits, so the relay
         * returns on its own; this only bounds how long we wait for it. */
        WaitForSingleObject(c->relay, 2000);
        CloseHandle(c->relay);
        c->relay = NULL;
    }
    if (c->proc != INVALID_HANDLE_VALUE && c->proc != NULL) {
        CloseHandle(c->proc);
        c->proc = INVALID_HANDLE_VALUE;
    }
    if (c->job != INVALID_HANDLE_VALUE && c->job != NULL) {
        CloseHandle(c->job); /* KILL_ON_JOB_CLOSE fires here if anything is left */
        c->job = INVALID_HANDLE_VALUE;
    }
#else
    (void)c;
#endif
}

/* ------------------------------------------------------------------------- */
/* Stopping, and the console handler that has to do it in place.             */
/* ------------------------------------------------------------------------- */

static child_t g_child;
static char g_host[MAX_HOST];
static int g_port;
static char g_token[TOKEN_CHARS + 1];
static int g_owner; /* did we start the backend? */

/* Ctrl+C / close / SIGTERM seen. sig_atomic_t because on POSIX a handler writes
 * it, and that is the only type the standard promises is safe to touch there. */
#ifdef _WIN32
static volatile int g_stop_asked;
static volatile LONG g_stopping;
static HANDLE g_stopped_event;
#else
static volatile sig_atomic_t g_stop_asked;
static volatile sig_atomic_t g_stopping;
#endif

/* Idempotent. Called from the main loop on Linux, and from the console handler
 * thread on Windows -- where returning from the handler ends the process, so the
 * work has to happen before the return rather than being handed to main.
 *
 * The two grace values are arguments rather than constants because the caller
 * knows how long it has: see CLOSE_GRACE_MS. */
static void stop_backend_once(int grace_ms, int force_ms)
{
    int status = 0;

    /* Tested before the flag is latched, deliberately. `g_owner` is set just
     * before the child is created, so a Ctrl+C landing in that window would
     * otherwise latch `g_stopping` with nothing to stop -- and the real teardown
     * a moment later would take the early return and leave the backend to the
     * job-object backstop: killed, but with cache/ never cleared. */
    if (!g_child.started || !child_alive(&g_child))
        return;

#ifdef _WIN32
    if (InterlockedExchange(&g_stopping, 1) != 0) {
        /* A second Ctrl+C, or a close event on top of a Ctrl+C, while the first
         * is still working: wait for it rather than starting a second teardown.
         * Bounded by *this* caller's budget, not the first one's -- if the close
         * button arrives while a 20-second Ctrl+C ladder is running, waiting the
         * full 23 seconds would just hand the guillotine something to cut. */
        if (g_stopped_event)
            WaitForSingleObject(g_stopped_event, grace_ms + force_ms);
        return;
    }
#else
    if (g_stopping)
        return;
    g_stopping = 1;
#endif

    out("\nStopping Fox Reader...\n");

    if (http_req(connect_host(g_host), g_port, "POST", "/api/shutdown", g_token,
                 PROBE_TIMEOUT_MS, &status, NULL, 0) &&
        status == 200) {
        if (child_wait(&g_child, grace_ms)) {
            out("Backend stopped; cache cleared.\n");
#ifdef _WIN32
            if (g_stopped_event)
                SetEvent(g_stopped_event);
#endif
            return;
        }
        errline("warning: the backend did not finish shutting down in time.\n");
    } else if (status != 0) {
        errline("warning: the backend refused the shutdown request (HTTP %d).\n", status);
    } else {
        errline("warning: the backend did not answer; it may not have finished starting.\n");
    }

    out("Forcing it down.\n");
    child_kill(&g_child);
    child_wait(&g_child, force_ms);

#ifdef _WIN32
    if (g_stopped_event)
        SetEvent(g_stopped_event);
#endif
}

#ifdef _WIN32
static BOOL WINAPI console_handler(DWORD type)
{
    switch (type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        g_stop_asked = 1;
        if (g_owner) {
            /* Ctrl+C and Ctrl+Break leave this process alive, so the teardown can
             * take as long as a clean stop needs. The other three are on the
             * clock: the OS kills this process shortly after the handler is
             * entered, and returning from here is what lets it. Either way the
             * work happens now, on this thread -- only the budget differs. */
            int closing = (type != CTRL_C_EVENT && type != CTRL_BREAK_EVENT);
            stop_backend_once(closing ? CLOSE_GRACE_MS : SHUTDOWN_GRACE_MS,
                              closing ? CLOSE_FORCE_MS : FORCE_GRACE_MS);
        } else {
            out("\nDetaching. Fox Reader is still running.\n");
        }
        return TRUE;
    default:
        return FALSE;
    }
}
#else
static void signal_handler(int sig)
{
    (void)sig;
    /* Flag only: nothing here is async-signal-safe enough to open a socket.
     * Unlike Windows there is no grace-period guillotine, so the main loop gets
     * to do the work. */
    g_stop_asked = 1;
}
#endif

static void install_handlers(void)
{
#ifdef _WIN32
    g_stopped_event = CreateEventW(NULL, TRUE, FALSE, NULL);
    SetConsoleCtrlHandler(console_handler, TRUE);
#else
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);
    /* A dead pipe must not take the launcher down mid-teardown. */
    signal(SIGPIPE, SIG_IGN);
#endif
}

/* ------------------------------------------------------------------------- */
/* Browser.                                                                  */
/* ------------------------------------------------------------------------- */

static void open_browser(const char *url)
{
#ifdef _WIN32
    /* ShellExecuteW by pointer, so shell32 need not be on the link line. */
    typedef HINSTANCE(WINAPI * shell_exec_fn)(HWND, LPCWSTR, LPCWSTR, LPCWSTR, LPCWSTR, INT);
    HMODULE shell = LoadLibraryW(L"shell32.dll");
    wchar_t *wurl = to_wide(url);
    int done = 0;

    if (shell && wurl) {
        shell_exec_fn exec = (shell_exec_fn)(void *)GetProcAddress(shell, "ShellExecuteW");
        if (exec) {
            HINSTANCE r = exec(NULL, L"open", wurl, NULL, NULL, SW_SHOWNORMAL);
            done = ((INT_PTR)r > 32);
        }
    }
    free(wurl);
    if (shell)
        FreeLibrary(shell);
    if (!done)
        errline("warning: could not open a browser. Go to %s\n", url);
#else
    pid_t pid = fork();
    if (pid == 0) {
        /* Its own session, output discarded: a browser launcher that logs to the
         * terminal would scribble over the backend's log. */
        int devnull = open("/dev/null", O_RDWR);
        setsid();
        if (devnull >= 0) {
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            if (devnull > STDERR_FILENO)
                close(devnull);
        }
#  ifdef __APPLE__
        execlp("open", "open", url, (char *)NULL);
#  endif
        execlp("xdg-open", "xdg-open", url, (char *)NULL);
        execlp("gio", "gio", "open", url, (char *)NULL);
        execlp("sensible-browser", "sensible-browser", url, (char *)NULL);
        execlp("x-www-browser", "x-www-browser", url, (char *)NULL);
        _exit(127);
    } else if (pid < 0) {
        errline("warning: could not open a browser. Go to %s\n", url);
    }
    /* Deliberately not reaped here: the monitor loop's waitpid is scoped to the
     * backend's pid, and this one exits on its own. */
#endif
}

/* ------------------------------------------------------------------------- */
/* Modes.                                                                    */
/* ------------------------------------------------------------------------- */

/* Someone else's backend. Report it, offer the UI, and watch. Closing this
 * window must not stop an instance this window did not start. */
static int monitor_attached(const char *url, int want_browser)
{
    int misses = 0;

    out("\nFox Reader is already running at %s\n", url);
    if (want_browser) {
        out("Opening it in your browser.\n");
        open_browser(url);
    }
    out("\nThis window is only watching it. Closing this window leaves Fox Reader\n"
        "running -- stop it from the window that started it, or from the UI.\n\n");

    while (!g_stop_asked) {
        sleep_ms(ATTACH_POLL_MS);
        if (g_stop_asked)
            break;
        if (probe(connect_host(g_host), g_port) == PROBE_FOX) {
            misses = 0;
            continue;
        }
        /* One miss can be a request that arrived mid-restart. Two is gone. */
        if (++misses >= 2) {
            out("Fox Reader has stopped.\n");
            return 0;
        }
    }

#ifndef _WIN32
    out("\nDetaching. Fox Reader is still running.\n");
#endif
    return 0;
}

/* Our backend. Wait for it, open the UI, and hold the console until it or the
 * user says otherwise. */
static int monitor_owned(const char *url, int want_browser)
{
    int waited = 0;
    int announced_slow = 0;
    int ready = 0;

    out("\nStarting the backend");
#ifdef _WIN32
    out(" (pid %lu)", (unsigned long)g_child.pid);
#else
    out(" (pid %ld)", (long)g_child.pid);
#endif
    out("...\n");

    /* No deadline on purpose. A first run downloads model weights, which can
     * take a very long time on a slow line, and a launcher that gave up at
     * ninety seconds would look like a broken program. The child's own log is
     * being relayed to this console, so there is always something to look at,
     * and the exit below is what catches a backend that actually failed. */
    while (!g_stop_asked) {
        if (!child_alive(&g_child)) {
            errline("\nerror: the backend exited before it was ready (code %d).\n", g_child.code);
            errline("       The log above should say why.\n");
            return 1;
        }

        if (probe(connect_host(g_host), g_port) == PROBE_FOX) {
            ready = 1;
            break;
        }

        sleep_ms(READY_POLL_MS);
        waited += READY_POLL_MS;
        if (!announced_slow && waited > 30000) {
            announced_slow = 1;
            out("\nStill starting. A first run downloads model weights, which can take a while.\n");
        }
    }

    if (!ready)
        return 0; /* interrupted; main stops the child */

    out("\nFox Reader is ready at %s\n", url);
    if (want_browser) {
        out("Opening it in your browser.\n");
        open_browser(url);
    }
    out("\nLeave this window open. Close it, or press Ctrl+C, to stop Fox Reader.\n\n");

    while (!g_stop_asked) {
        if (!child_alive(&g_child)) {
            if (g_child.code == 0)
                out("\nThe backend has shut down.\n");
            else
                errline("\nerror: the backend stopped unexpectedly (code %d).\n", g_child.code);
            return g_child.code == 0 ? 0 : 1;
        }
        sleep_ms(OWNER_POLL_MS);
    }

    return 0;
}

/* ------------------------------------------------------------------------- */
/* Entry point.                                                              */
/* ------------------------------------------------------------------------- */

static void usage(void)
{
    out("Fox Reader launcher %s\n"
        "\n"
        "  launcher [options]\n"
        "\n"
        "  --no-browser   start without opening the UI\n"
        "  --version      print the launcher version\n"
        "  --help, -h     this message\n"
        "\n"
        "Starts bin/fox-reader if it is not already running, using the address in\n"
        "config/fox_config.yaml (config/user_config.yaml wins if it is there).\n"
        "If Fox Reader is already up, this window watches it instead of starting a\n"
        "second copy, and closing this window leaves it running.\n"
        "\n"
        "FOX_READER_NO_BROWSER=1 has the same effect as --no-browser.\n",
        FOX_LAUNCHER_VERSION);
}

static int env_truthy(const char *name)
{
    const char *v = getenv(name);
    if (!v || !v[0])
        return 0;
    return !(strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "FALSE") == 0 ||
             strcmp(v, "no") == 0 || strcmp(v, "off") == 0);
}

int main(int argc, char **argv)
{
    pchar root[PATH_CAP], backend[PATH_CAP], bindir[PATH_CAP];
    char url[MAX_URL];
    char *printable;
    lock_t lock;
    int want_browser = 1;
    int i, locked, rc = 0;
    probe_t state;

#ifdef _WIN32
    /* UTF-8 out, so a path with non-ASCII in it prints as itself. Set before
     * anything is written. */
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleTitleW(L"Fox Reader");
#endif

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-browser") == 0) {
            want_browser = 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage();
            return 0;
        } else if (strcmp(argv[i], "--version") == 0) {
            out("%s\n", FOX_LAUNCHER_VERSION);
            return 0;
        } else {
            errline("error: unknown option %s\n\n", argv[i]);
            usage();
            maybe_pause();
            return 2;
        }
    }

    if (env_truthy("FOX_READER_NO_BROWSER"))
        want_browser = 0;

    if (!exe_dir(root, PATH_CAP)) {
#ifdef _WIN32
        fatal("cannot work out where this program is installed.");
#else
        /* No procfs. argv[0] with a slash in it is the next best thing. */
        char *slash;
        if (argc < 1 || !argv[0] || !strchr(argv[0], '/'))
            fatal("cannot work out where this program is installed; run it by path.");
        strncpy(root, argv[0], PATH_CAP - 1);
        root[PATH_CAP - 1] = 0;
        slash = strrchr(root, '/');
        if (!slash)
            fatal("cannot work out where this program is installed; run it by path.");
        *slash = 0;
#endif
    }

    if (!p_join(bindir, PATH_CAP, root, FOX_BIN_DIR))
        fatal("the install path is too long to work with.");

    backend[0] = 0;
    for (i = 0; i < BACKEND_NAME_COUNT; i++) {
        if (p_join(backend, PATH_CAP, bindir, BACKEND_NAMES[i]) && path_exists(backend))
            break;
        backend[0] = 0;
    }
    if (!backend[0]) {
        printable = PRINTABLE(bindir);
        errline("error: no Fox Reader binary in %s\n", printable ? printable : "bin");
        PRINTABLE_FREE(printable);
        errline("       This launcher has to sit next to the bin folder it came with.\n");
        maybe_pause();
        return 1;
    }

    if (!net_start())
        fatal("could not initialise networking.");

    read_config(root, g_host, sizeof g_host, &g_port);
    build_url(url, sizeof url, connect_host(g_host), g_port);

    out("Fox Reader\n");
    printable = PRINTABLE(root);
    out("  folder : %s\n", printable ? printable : "?");
    PRINTABLE_FREE(printable);
    out("  address: %s\n", url);

    install_handlers();

    /*
     * Who is in charge here?
     *
     *   lock held by a peer -> another launcher is starting one; wait for it and
     *                          then watch, rather than racing it to the port.
     *   port answers as Fox  -> already up; watch it, never stop it.
     *   port answers as else -> something unrelated has the port; say so.
     *   nothing              -> ours to start.
     */
    locked = lock_acquire(&lock, root);

    if (locked == 0) {
        /* A held lock means a live launcher, since the lock dies with the process
         * that took it -- so this waits rather than giving up on a deadline. It
         * has to: a first run downloads model weights, and any timeout short
         * enough to be useful is far too short for that. */
        int waited = 0;
        int took_over = 0;
        out("\nAnother launcher is already starting Fox Reader. Waiting for it.\n");
        out("Press Ctrl+C to stop waiting.\n");
        while (!g_stop_asked) {
            if (probe(connect_host(g_host), g_port) == PROBE_FOX)
                break;
            sleep_ms(ATTACH_POLL_MS);
            waited += ATTACH_POLL_MS;
            /* The peer can lose, too: its backend can fail to start, or it can be
             * killed outright. Then the port will never answer and the lock is
             * free again, so retaking it is how this loop learns to stop waiting
             * for something that is not coming and start Fox Reader itself. With
             * only the probe to go on, that wait really is forever. */
            if (lock_acquire(&lock, root) == 1) {
                out("  the other launcher stopped; starting Fox Reader.\n");
                took_over = 1;
                break;
            }
            if (waited % PEER_NOTE_MS < ATTACH_POLL_MS)
                out("  still waiting...\n");
        }
        if (g_stop_asked) {
            if (took_over)
                lock_release(&lock);
            return 0;
        }
        if (!took_over)
            return monitor_attached(url, want_browser);
        /* Holding the lock now, so fall through to the ordinary start path -- which
         * probes once more before committing to anything. */
    }

    state = probe(connect_host(g_host), g_port);

    if (state == PROBE_FOX) {
        /* Up without a launcher holding the lock -- started from a terminal, or
         * by a launcher that was killed. Either way it is not ours to stop. */
        lock_release(&lock);
        return monitor_attached(url, want_browser);
    }

    if (state == PROBE_FOREIGN) {
        errline("\nerror: something else is already using %s\n", url);
        errline("       Change the port in config/fox_config.yaml, or stop whatever has it.\n");
        lock_release(&lock);
        maybe_pause();
        return 1;
    }

    if (g_stop_asked) {
        lock_release(&lock);
        return 0;
    }

    make_token(g_token, sizeof g_token);
    g_owner = 1;

    if (!child_start(&g_child, backend, root, g_token)) {
        printable = PRINTABLE(backend);
        errline("\nerror: could not start %s\n", printable ? printable : "the backend");
        PRINTABLE_FREE(printable);
        lock_release(&lock);
        maybe_pause();
        return 1;
    }

    rc = monitor_owned(url, want_browser);

    /* On Windows the console handler has usually done this already; it is
     * idempotent, so calling it here covers Ctrl+C on Linux and the plain
     * fall-through when the backend exited on its own. Nothing is racing a
     * deadline on this path, so it gets the unhurried ladder. */
    stop_backend_once(SHUTDOWN_GRACE_MS, FORCE_GRACE_MS);
    child_close(&g_child);
    lock_release(&lock);

#ifdef _WIN32
    if (g_stopped_event)
        CloseHandle(g_stopped_event);
    WSACleanup();
#endif

    return rc;
}
