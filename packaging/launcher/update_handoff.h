/* Internal updater gate, included after the launcher's path and lock helpers. */
#ifndef FOX_UPDATE_HANDOFF_H
#define FOX_UPDATE_HANDOFF_H

#ifndef _WIN32
#include <dirent.h>
#endif

#define UPDATE_TXN_PREFIX ".foxreader-update-"
#define UPDATE_READY_TIMEOUT_MS 15000
#define UPDATE_CLEANUP_TIMEOUT_MS 30000

static int update_before_deadline(long long deadline)
{
    long long current;
    if (deadline < 0) return 0;
    current = monotonic_ms();
    return current >= 0 && current < deadline;
}

typedef struct update_terminal_t {
    pchar name[51];
    struct update_terminal_t *next;
} update_terminal_t;

static void update_free_terminals(update_terminal_t *entry)
{
    while (entry) {
        update_terminal_t *next = entry->next;
        free(entry);
        entry = next;
    }
}

/* 0 absent, 1 regular file, 2 directory, -1 unsafe or inaccessible. */
static int update_path_kind(const pchar *path)
{
#ifdef _WIN32
    DWORD attributes = GetFileAttributesW(path);
    if (attributes == INVALID_FILE_ATTRIBUTES) {
        DWORD error = GetLastError();
        return error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND ? 0 : -1;
    }
    if (attributes & FILE_ATTRIBUTE_REPARSE_POINT) return -1;
    return attributes & FILE_ATTRIBUTE_DIRECTORY ? 2 : 1;
#else
    struct stat info;
    if (lstat(path, &info)) return errno == ENOENT ? 0 : -1;
    if (S_ISREG(info.st_mode)) return 1;
    if (S_ISDIR(info.st_mode)) return 2;
    return -1;
#endif
}

/* Inspect the same first-byte/flock lock held by the native updater. A stale
 * unlocked file is harmless; neither process unlinks the shared lock inode. */
static int update_lock_busy(const pchar *root)
{
    pchar path[PATH_CAP];
    int kind;
    if (!p_join(path, PATH_CAP, root, PL(".fox-reader.update.lock"))) return -1;
    kind = update_path_kind(path);
    if (kind == 0) return 0;
    if (kind != 1) return -1;
#ifdef _WIN32
    {
        OVERLAPPED offset;
        HANDLE handle = CreateFileW(path, GENERIC_READ | GENERIC_WRITE,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        int result;
        if (handle == INVALID_HANDLE_VALUE) {
            DWORD error = GetLastError();
            return error == ERROR_FILE_NOT_FOUND ? 0 : -1;
        }
        memset(&offset, 0, sizeof offset);
        if (LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                       0, 1, 0, &offset)) {
            UnlockFileEx(handle, 0, 1, 0, &offset);
            result = 0;
        } else {
            DWORD error = GetLastError();
            result = error == ERROR_LOCK_VIOLATION || error == ERROR_SHARING_VIOLATION ? 1 : -1;
        }
        CloseHandle(handle);
        return result;
    }
#else
    {
        int result, fd = open(path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) return errno == ENOENT ? 0 : -1;
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
            flock(fd, LOCK_UN);
            result = 0;
        } else result = errno == EWOULDBLOCK || errno == EAGAIN ? 1 : -1;
        close(fd);
        return result;
    }
#endif
}

static int update_hex(const char *text, size_t size)
{
    size_t i;
    if (strlen(text) != size) return 0;
    for (i = 0; i < size; ++i)
        if (!((text[i] >= '0' && text[i] <= '9') || (text[i] >= 'a' && text[i] <= 'f'))) return 0;
    return 1;
}

static int update_transaction_name(const pchar *name)
{
    char *text = PRINTABLE(name);
    int result = text && strncmp(text, UPDATE_TXN_PREFIX, 18) == 0 && update_hex(text + 18, 32);
    PRINTABLE_FREE(text);
    return result;
}

/* An interrupted journal must be recovered before even looking for bin/. */
static int update_inspect_transaction(const pchar *root, const pchar *name, pchar *found,
                                      update_terminal_t **finished)
{
    pchar transaction[PATH_CAP], path[PATH_CAP];
    char *state;
    size_t size;
    int kind, journal, terminal = 0;
    if (!p_join(transaction, PATH_CAP, root, name) || update_path_kind(transaction) != 2 ||
        !p_join(path, PATH_CAP, transaction, PL("journal.bin"))) return -1;
    kind = update_path_kind(path);
    if (kind < 0 || kind == 2 || !p_join(path, PATH_CAP, transaction, PL("state"))) return -1;
    journal = kind == 1;
    kind = update_path_kind(path);
    if (kind < 0 || kind == 2) return journal ? -1 : 0;
    if (kind == 1) {
        state = read_whole_file(path, &size);
        if (!state) return journal ? -1 : 0;
        terminal = (size == 10 && memcmp(state, "COMMITTED\n", 10) == 0) ||
                   (size == 12 && memcmp(state, "ROLLED_BACK\n", 12) == 0);
        free(state);
    }
    if (terminal) {
        update_terminal_t *entry = (update_terminal_t *)malloc(sizeof *entry);
        if (entry) {
            memcpy(entry->name, name, (p_strlen(name) + 1) * sizeof(pchar));
            entry->next = *finished;
            *finished = entry;
        }
        return 0; /* Cleanup is optional; allocation failure cannot block startup. */
    }
    if (!journal) return 0;
    if (found[0]) return -1; /* Multiple incomplete journals have no safe ordering. */
    memcpy(found, name, (p_strlen(name) + 1) * sizeof(pchar));
    return 0;
}

static int update_find_recovery(const pchar *root, pchar *found, update_terminal_t **finished)
{
    int result = 0;
    found[0] = 0;
    *finished = NULL;
#ifdef _WIN32
    {
        pchar pattern[PATH_CAP];
        WIN32_FIND_DATAW entry;
        HANDLE search;
        if (!p_join(pattern, PATH_CAP, root, PL("*"))) return -1;
        search = FindFirstFileW(pattern, &entry);
        if (search == INVALID_HANDLE_VALUE) return -1;
        do {
            if (update_transaction_name(entry.cFileName) &&
                update_inspect_transaction(root, entry.cFileName, found, finished) < 0) {
                result = -1;
                break;
            }
        } while (FindNextFileW(search, &entry));
        if (result == 0 && GetLastError() != ERROR_NO_MORE_FILES) result = -1;
        FindClose(search);
    }
#else
    {
        DIR *directory = opendir(root);
        struct dirent *entry;
        if (!directory) return -1;
        errno = 0;
        while ((entry = readdir(directory)) != NULL) {
            if (update_transaction_name(entry->d_name) &&
                update_inspect_transaction(root, entry->d_name, found, finished) < 0) {
                result = -1;
                break;
            }
            errno = 0;
        }
        if (result == 0 && errno) result = -1;
        closedir(directory);
    }
#endif
    return result;
}

static int update_nonce(char nonce[65])
{
    unsigned char bytes[32];
    static const char digits[] = "0123456789abcdef";
    size_t i;
    int ok = 0;
#ifdef _WIN32
    typedef BOOLEAN(WINAPI *random_function)(PVOID, ULONG);
    HMODULE library = LoadLibraryW(L"advapi32.dll");
    if (library) {
        random_function generate = (random_function)(void *)GetProcAddress(library, "SystemFunction036");
        ok = generate && generate(bytes, sizeof bytes);
        FreeLibrary(library);
    }
#else
    FILE *source = fopen("/dev/urandom", "rb");
    if (source) {
        ok = fread(bytes, 1, sizeof bytes, source) == sizeof bytes;
        fclose(source);
    }
#endif
    if (!ok) return 0; /* Recovery authorization never uses a weak fallback. */
    for (i = 0; i < sizeof bytes; ++i) {
        nonce[i * 2] = digits[bytes[i] >> 4];
        nonce[i * 2 + 1] = digits[bytes[i] & 15];
    }
    nonce[64] = 0;
    return 1;
}

static int update_write_ticket(const pchar *transaction, const char *nonce)
{
    pchar ticket[PATH_CAP], temporary[PATH_CAP], name[29];
    size_t i;
    int kind, ok = 0;
    /* A crashed writer must never block later recovery with a fixed name.
     * The filename uses 96 bits of the fresh nonce; authentication still uses
     * all 256 bits in the file. Keep this path no longer than the Windows
     * runner/foxreader-updater.exe path so otherwise usable installations do
     * not hit MAX_PATH solely while preparing recovery authorization. */
    for (i = 0; i < 24; ++i) name[i] = (pchar)nonce[i];
    memcpy(name + 24, PL(".new"), 5 * sizeof(pchar));
    if (!p_join(ticket, PATH_CAP, transaction, PL("ticket.token")) ||
        !p_join(temporary, PATH_CAP, transaction, name)) return 0;
    kind = update_path_kind(ticket);
    if ((kind != 0 && kind != 1) || update_path_kind(temporary) != 0) return 0;
#ifdef _WIN32
    {
        DWORD count;
        HANDLE file = CreateFileW(temporary, GENERIC_WRITE, 0, NULL, CREATE_NEW,
                                  FILE_ATTRIBUTE_NORMAL, NULL);
        if (file == INVALID_HANDLE_VALUE) return 0;
        ok = WriteFile(file, nonce, 64, &count, NULL) && count == 64 && FlushFileBuffers(file);
        CloseHandle(file);
        if (ok) ok = MoveFileExW(temporary, ticket, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
        if (!ok) DeleteFileW(temporary);
    }
#else
    {
        int file = open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (file < 0) return 0;
        ok = write(file, nonce, 64) == 64 && fsync(file) == 0;
        if (close(file) != 0) ok = 0;
        if (ok) ok = rename(temporary, ticket) == 0;
        if (!ok) unlink(temporary);
    }
#endif
    return ok;
}

/* Only the inherited pipes carry authorization. The child is independent of
 * the backend's job/process group. Recovery waits for this launcher to exit;
 * terminal cleanup can finish while the launcher waits for the helper. */
static int update_start_helper(const pchar *root, const pchar *runner, const char *payload, int cleanup)
{
    char response[16];
    size_t used = 0, size = strlen(payload);
    int ready = 0;
    long long deadline = monotonic_ms();
    if (deadline < 0) return 0;
    /* Polling sleeps can overrun on macOS or a busy host. The payload and READY
     * response share one real deadline, independent of how often we wake. */
    deadline += UPDATE_READY_TIMEOUT_MS;
#ifdef _WIN32
    SECURITY_ATTRIBUTES security;
    STARTUPINFOEXW startup;
    PROCESS_INFORMATION process;
    HANDLE input_read = NULL, input_write = NULL, output_read = NULL, output_write = NULL;
    HANDLE errors = INVALID_HANDLE_VALUE;
    pchar command[PATH_CAP + 3];
    DWORD count;
    SIZE_T attributes_size = 0;
    HANDLE inherited[3];
    int attributes_ready = 0;
    int started = 0;
    memset(&security, 0, sizeof security);
    security.nLength = sizeof security;
    security.bInheritHandle = TRUE;
    memset(&startup, 0, sizeof startup);
    memset(&process, 0, sizeof process);
    if (!CreatePipe(&input_read, &input_write, &security, 65536) ||
        !CreatePipe(&output_read, &output_write, &security, 65536)) goto cleanup;
    if (!SetHandleInformation(input_write, HANDLE_FLAG_INHERIT, 0) ||
        !SetHandleInformation(output_read, HANDLE_FLAG_INHERIT, 0)) goto cleanup;
    errors = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
                         OPEN_EXISTING, 0, NULL);
    if (errors == INVALID_HANDLE_VALUE) goto cleanup;
    startup.StartupInfo.cb = sizeof startup;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = input_read;
    startup.StartupInfo.hStdOutput = output_write;
    startup.StartupInfo.hStdError = errors;
    InitializeProcThreadAttributeList(NULL, 1, 0, &attributes_size);
    if (!attributes_size) goto cleanup;
    startup.lpAttributeList = (LPPROC_THREAD_ATTRIBUTE_LIST)malloc(attributes_size);
    if (!startup.lpAttributeList ||
        !InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &attributes_size)) goto cleanup;
    attributes_ready = 1;
    inherited[0] = input_read; inherited[1] = output_write; inherited[2] = errors;
    if (!UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                    inherited, sizeof inherited, NULL, NULL)) goto cleanup;
    command[0] = L'"';
    memcpy(command + 1, runner, p_strlen(runner) * sizeof(pchar));
    command[p_strlen(runner) + 1] = L'"';
    command[p_strlen(runner) + 2] = 0;
    if (!CreateProcessW(runner, command, NULL, NULL, TRUE,
                        DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | EXTENDED_STARTUPINFO_PRESENT,
                        NULL, root, &startup.StartupInfo, &process)) goto cleanup;
    started = 1;
    CloseHandle(input_read); input_read = NULL;
    CloseHandle(output_write); output_write = NULL;
    if (!WriteFile(input_write, payload, (DWORD)size, &count, NULL) || count != size) goto cleanup;
    CloseHandle(input_write); input_write = NULL;
    while (used < sizeof response - 1 && update_before_deadline(deadline)) {
        DWORD available = 0;
        if (!PeekNamedPipe(output_read, NULL, 0, NULL, &available, NULL)) break;
        if (available) {
            if (!ReadFile(output_read, response + used, 1, &count, NULL) || count != 1) break;
            if (response[used++] == '\n') break;
        } else if (WaitForSingleObject(process.hProcess, 0) != WAIT_TIMEOUT) break;
        Sleep(25);
    }
    response[used] = 0;
    ready = strcmp(response, "READY\n") == 0 && update_before_deadline(deadline);
    if (ready && cleanup) {
        DWORD exit_code = 1;
        ready = WaitForSingleObject(process.hProcess, UPDATE_CLEANUP_TIMEOUT_MS) == WAIT_OBJECT_0 &&
                GetExitCodeProcess(process.hProcess, &exit_code) && exit_code == 0;
    }
cleanup:
    if (started && !ready) {
        /* Recovery cannot mutate before our exit. Cleanup may only remove a
         * terminal transaction. Reap failures before returning either way. */
        TerminateProcess(process.hProcess, 1);
        WaitForSingleObject(process.hProcess, 5000);
    }
    if (process.hThread) CloseHandle(process.hThread);
    if (process.hProcess) CloseHandle(process.hProcess);
    if (input_read) CloseHandle(input_read);
    if (input_write) CloseHandle(input_write);
    if (output_read) CloseHandle(output_read);
    if (output_write) CloseHandle(output_write);
    if (errors != INVALID_HANDLE_VALUE) CloseHandle(errors);
    if (attributes_ready) DeleteProcThreadAttributeList(startup.lpAttributeList);
    free(startup.lpAttributeList);
#else
    int input[2] = {-1, -1}, output[2] = {-1, -1};
    pid_t child;
    ssize_t count;
    size_t sent = 0;
    int reaped = 0;
    if (pipe(input) || pipe(output)) {
        if (input[0] >= 0) { close(input[0]); close(input[1]); }
        return 0;
    }
    if (fcntl(input[0], F_SETFD, FD_CLOEXEC) < 0 || fcntl(input[1], F_SETFD, FD_CLOEXEC) < 0 ||
        fcntl(output[0], F_SETFD, FD_CLOEXEC) < 0 || fcntl(output[1], F_SETFD, FD_CLOEXEC) < 0 ||
        fcntl(input[1], F_SETFL, O_NONBLOCK) < 0 || fcntl(output[0], F_SETFL, O_NONBLOCK) < 0) {
        close(input[0]); close(input[1]); close(output[0]); close(output[1]);
        return 0;
    }
    child = fork();
    if (child == 0) {
        char *arguments[2];
        int errors = open("/dev/null", O_WRONLY);
        if (setsid() < 0 || dup2(input[0], STDIN_FILENO) < 0 ||
            dup2(output[1], STDOUT_FILENO) < 0 || errors < 0 || dup2(errors, STDERR_FILENO) < 0 ||
            fcntl(STDIN_FILENO, F_SETFD, 0) < 0 || fcntl(STDOUT_FILENO, F_SETFD, 0) < 0 ||
            fcntl(STDERR_FILENO, F_SETFD, 0) < 0 || chdir(root)) _exit(127);
        if (errors > STDERR_FILENO) close(errors);
        if (input[0] > STDERR_FILENO) close(input[0]);
        if (input[1] > STDERR_FILENO) close(input[1]);
        if (output[0] > STDERR_FILENO) close(output[0]);
        if (output[1] > STDERR_FILENO) close(output[1]);
        arguments[0] = (char *)runner;
        arguments[1] = NULL;
        execv(runner, arguments);
        _exit(127);
    }
    close(input[0]); close(output[1]);
    if (child < 0) { close(input[1]); close(output[0]); return 0; }
    while (sent < size && update_before_deadline(deadline)) {
        count = write(input[1], payload + sent, size - sent);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            sleep_ms(25);
            continue;
        }
        if (count <= 0) break;
        sent += (size_t)count;
    }
    close(input[1]);
    while (sent == size && used < sizeof response - 1 && update_before_deadline(deadline)) {
        count = read(output[0], response + used, 1);
        if (count == 1) {
            if (response[used++] == '\n') break;
        } else if (count == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) break;
        sleep_ms(25);
    }
    close(output[0]);
    response[used] = 0;
    ready = strcmp(response, "READY\n") == 0 && update_before_deadline(deadline);
    if (ready && cleanup) {
        int status = 0;
        deadline = monotonic_ms();
        if (deadline >= 0) deadline += UPDATE_CLEANUP_TIMEOUT_MS;
        while (update_before_deadline(deadline)) {
            pid_t stopped = waitpid(child, &status, WNOHANG);
            if (stopped == child) {
                reaped = 1;
                ready = WIFEXITED(status) && WEXITSTATUS(status) == 0 && update_before_deadline(deadline);
                break;
            }
            if (stopped < 0 && errno != EINTR) { reaped = 1; ready = 0; break; }
            sleep_ms(25);
        }
        if (!reaped) ready = 0;
    }
    if (!ready && !reaped) {
        kill(child, SIGKILL);
        while (waitpid(child, NULL, 0) < 0 && errno == EINTR) { }
    }
#endif
    return ready;
}

/* Terminal cleanup uses the installed helper, never the copy it is removing.
 * Its authenticated cleanup action does not wait for this launcher to exit. */
static int update_cleanup_finished(const pchar *root, update_terminal_t *finished)
{
    pchar runner[PATH_CAP], transaction[PATH_CAP];
    char nonce[65], *root_utf8, *name_utf8, *payload;
    size_t capacity;
    lock_t cleanup_lock;
    unsigned long parent;
    int busy, failed = 0;
    if (!finished) return 0;
#ifdef _WIN32
    if (!p_join(runner, PATH_CAP, root, PL("foxreader-updater.exe")) || update_path_kind(runner) != 1)
#else
    if (!p_join(runner, PATH_CAP, root, PL("foxreader-updater")) || update_path_kind(runner) != 1)
#endif
        return 0;
    /* A second launcher or active application can retry cleanup next launch. */
    if (lock_acquire(&cleanup_lock, root) != 1) return 0;
    busy = update_lock_busy(root);
    if (busy != 0) { lock_release(&cleanup_lock); return busy > 0 ? 1 : -1; }
    root_utf8 = PRINTABLE(root);
    if (!root_utf8 || strpbrk(root_utf8, "\r\n")) {
        PRINTABLE_FREE(root_utf8);
        lock_release(&cleanup_lock);
        return 0;
    }
#ifdef _WIN32
    parent = (unsigned long)GetCurrentProcessId();
#else
    parent = (unsigned long)getpid();
#endif
    capacity = strlen(root_utf8) + 384;
    payload = (char *)malloc(capacity);
    if (payload) for (; finished; finished = finished->next) {
        name_utf8 = PRINTABLE(finished->name);
        if (!name_utf8 || !p_join(transaction, PATH_CAP, root, finished->name) ||
            !update_nonce(nonce) || !update_write_ticket(transaction, nonce)) {
            failed = 1;
            PRINTABLE_FREE(name_utf8);
            continue;
        }
        snprintf(payload, capacity, "FOXREADER-UPDATER-1\ncleanup\n%s\n%s\n%s\n%lu\n0\n"
                 "0000000000000000000000000000000000000000000000000000000000000000\n",
                 root_utf8, name_utf8, nonce, parent);
        PRINTABLE_FREE(name_utf8);
        if (!update_start_helper(root, runner, payload, 1)) {
            failed = 1;
            break; /* A broken or unresponsive helper need only fail once per launch. */
        }
    }
    free(payload);
    PRINTABLE_FREE(root_utf8);
    lock_release(&cleanup_lock);
    if (failed) out("Some finished update files could not be removed. Cleanup will be retried next launch.\n");
    return 0; /* Failure to remove terminal artifacts cannot stop the application. */
}

/* 0 normal startup, 1 updater owns the installation, -1 unsafe recovery. */
static int update_guard(const pchar *root)
{
    pchar name[PATH_CAP], transaction[PATH_CAP], path[PATH_CAP], runner[PATH_CAP];
    char nonce[65], digest[65], *source, *root_utf8, *name_utf8, *payload;
    size_t size, capacity;
    unsigned long parent;
    int busy = update_lock_busy(root), result = -1;
    lock_t recovery_lock;
    update_terminal_t *finished = NULL;
    if (busy == 1) {
        out("Fox Reader is being updated. Wait for the updater to finish before opening it.\n");
#ifdef FOX_GUI
        if (g_gui) fox_window_status("Fox Reader is being updated. The updater will report when it is finished.");
#endif
        return 1;
    }
    if (busy < 0 || update_find_recovery(root, name, &finished) < 0) {
        update_free_terminals(finished);
        errline("Cannot safely check the installation's update state. Keep its update folders and check folder permissions.\n");
        return -1;
    }
    if (!name[0]) {
        result = update_cleanup_finished(root, finished);
        update_free_terminals(finished);
        return result;
    }
    update_free_terminals(finished);
    if (!p_join(transaction, PATH_CAP, root, name) ||
        !p_join(path, PATH_CAP, transaction, PL("runner")) || update_path_kind(path) != 2 ||
#ifdef _WIN32
        !p_join(runner, PATH_CAP, path, PL("foxreader-updater.exe")) ||
#else
        !p_join(runner, PATH_CAP, path, PL("foxreader-updater")) ||
#endif
        update_path_kind(runner) != 1 ||
        !p_join(path, PATH_CAP, transaction, PL("inventory.sha256")) || update_path_kind(path) != 1) {
        errline("An interrupted Fox Reader update needs recovery, but its updater or inventory digest is missing. "
                "Keep the .foxreader-update-* folder and restore the updater files before trying again.\n");
        return -1;
    }
    source = read_whole_file(path, &size);
    if (!source) {
        errline("Cannot read the interrupted update's inventory digest. Keep the update folder and check permissions.\n");
        return -1;
    }
    if (size == 65 && source[64] == '\n') source[64] = 0;
    else if (size != 64) {
        free(source);
        errline("The interrupted update's inventory digest is invalid. Keep the update folder for recovery.\n");
        return -1;
    }
    if (!update_hex(source, 64)) {
        free(source);
        errline("The interrupted update's inventory digest is invalid. Keep the update folder for recovery.\n");
        return -1;
    }
    memcpy(digest, source, 65);
    free(source);
    /* Serialize two recovery launch attempts without holding the updater's own
     * lock, which the helper must acquire before acknowledging the handoff. */
    busy = lock_acquire(&recovery_lock, root);
    if (busy != 1) {
        errline("An interrupted update needs recovery. Close the other Fox Reader windows, then launch Fox Reader again.\n");
        return busy == 0 ? 1 : -1;
    }
    root_utf8 = PRINTABLE(root);
    name_utf8 = PRINTABLE(name);
    if (!root_utf8 || !name_utf8 || strpbrk(root_utf8, "\r\n") || !update_nonce(nonce) ||
        !update_write_ticket(transaction, nonce)) goto release;
#ifdef _WIN32
    parent = (unsigned long)GetCurrentProcessId();
#else
    parent = (unsigned long)getpid();
#endif
    capacity = strlen(root_utf8) + strlen(name_utf8) + 256;
    payload = (char *)malloc(capacity);
    if (!payload) goto release;
    snprintf(payload, capacity, "FOXREADER-UPDATER-1\nrecover\n%s\n%s\n%s\n%lu\n0\n%s\n",
             root_utf8, name_utf8, nonce, parent, digest);
    if (update_start_helper(root, runner, payload, 0)) {
        out("The Fox Reader updater is recovering the interrupted installation. Fox Reader will remain closed.\n");
        result = 1;
    }
    free(payload);
release:
    PRINTABLE_FREE(root_utf8);
    PRINTABLE_FREE(name_utf8);
    lock_release(&recovery_lock);
    if (result < 0)
        errline("Could not start update recovery. Keep the .foxreader-update-* folder, "
                "check logs and folder permissions, then launch Fox Reader again. No backend was started.\n");
    return result;
}

#endif
