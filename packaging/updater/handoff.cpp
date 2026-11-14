#include "handoff.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <tlhelp32.h>
#include <fcntl.h>
#include <io.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __APPLE__
#include <libproc.h>
#include <mach-o/dyld.h>
#else
#include <sys/syscall.h>
#endif
#endif

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace fox_updater {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr const char* updater_name =
#ifdef _WIN32
    "foxreader-updater.exe";
#else
    "foxreader-updater";
#endif

bool hex_string(const std::string& value, std::size_t size) {
    return value.size() == size && value.find_first_not_of("0123456789abcdef") == std::string::npos;
}

std::uint32_t process_id(const std::string& value, bool allow_zero = false) {
    if (value.empty() || value.size() > 10 || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("The update ticket has an invalid process identifier.");
    const auto id = std::stoull(value);
    if (id > static_cast<unsigned long long>(std::numeric_limits<int>::max()) || (!id && !allow_zero))
        throw std::runtime_error("The update ticket has an invalid process identifier.");
    return static_cast<std::uint32_t>(id);
}

std::uint32_t self_pid() {
#ifdef _WIN32
    return GetCurrentProcessId();
#else
    return static_cast<std::uint32_t>(getpid());
#endif
}

std::uint32_t parent_pid() {
#ifdef _WIN32
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot verify the updater's parent process.");
    PROCESSENTRY32W entry{};
    entry.dwSize = sizeof entry;
    std::uint32_t parent = 0;
    if (Process32FirstW(snapshot, &entry)) do {
        if (entry.th32ProcessID == GetCurrentProcessId()) { parent = entry.th32ParentProcessID; break; }
    } while (Process32NextW(snapshot, &entry));
    CloseHandle(snapshot);
    return parent;
#else
    return static_cast<std::uint32_t>(getppid());
#endif
}

fs::path own_executable() {
#ifdef _WIN32
    std::wstring value(32768, L'\0');
    DWORD count = GetModuleFileNameW(nullptr, value.data(), static_cast<DWORD>(value.size()));
    if (!count || count >= value.size()) throw std::runtime_error("Cannot locate the updater executable.");
    value.resize(count);
    return fs::canonical(value);
#elif defined(__APPLE__)
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> value(size);
    if (_NSGetExecutablePath(value.data(), &size)) throw std::runtime_error("Cannot locate the updater executable.");
    return fs::canonical(value.data());
#else
    return fs::canonical("/proc/self/exe");
#endif
}

struct Process {
    std::uint32_t pid = 0;
#ifdef _WIN32
    HANDLE handle = nullptr;
    ~Process() { if (handle) CloseHandle(handle); }
#else
    std::string identity;
#ifndef __APPLE__
    int pidfd = -1;
    ~Process() { if (pidfd >= 0) close(pidfd); }
#endif
#endif
    Process() = default;
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;

#ifndef _WIN32
    std::string current_identity() const {
#ifdef __APPLE__
        proc_bsdinfo info{};
        if (proc_pidinfo(static_cast<int>(pid), PROC_PIDTBSDINFO, 0, &info, sizeof info) != sizeof info) return {};
        return std::to_string(info.pbi_start_tvsec) + ":" + std::to_string(info.pbi_start_tvusec);
#else
        std::ifstream stream("/proc/" + std::to_string(pid) + "/stat");
        std::string line;
        std::getline(stream, line);
        auto end = line.rfind(')');
        if (end == std::string::npos) return {};
        std::istringstream fields(line.substr(end + 2));
        std::string field;
        for (int index = 3; index <= 22; ++index) if (!(fields >> field)) return {};
        return field;
#endif
    }
#endif

    fs::path capture(std::uint32_t value) {
        pid = value;
#ifdef _WIN32
        handle = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
        if (!handle) throw std::runtime_error("Cannot verify a Fox Reader process. Keep Fox Reader open and retry.");
        std::wstring name(32768, L'\0');
        DWORD size = static_cast<DWORD>(name.size());
        if (!QueryFullProcessImageNameW(handle, 0, name.data(), &size))
            throw std::runtime_error("Cannot identify the Fox Reader executable.");
        name.resize(size);
        if (WaitForSingleObject(handle, 0) != WAIT_TIMEOUT)
            throw std::runtime_error("Fox Reader exited before the update handoff was authenticated.");
        return fs::canonical(name);
#else
        identity = current_identity();
        if (identity.empty()) throw std::runtime_error("Cannot verify the Fox Reader process.");
#ifdef __APPLE__
        std::array<char, PROC_PIDPATHINFO_MAXSIZE> name{};
        if (proc_pidpath(static_cast<int>(pid), name.data(), static_cast<std::uint32_t>(name.size())) <= 0)
            throw std::runtime_error("Cannot identify the Fox Reader executable.");
        fs::path executable = fs::canonical(name.data());
#else
#ifdef SYS_pidfd_open
        pidfd = static_cast<int>(syscall(SYS_pidfd_open, pid, 0));
#endif
        fs::path executable = fs::canonical("/proc/" + std::to_string(pid) + "/exe");
#endif
        if (current_identity() != identity) throw std::runtime_error("The Fox Reader process changed during handoff.");
        return executable;
#endif
    }

    bool exited() const {
        if (!pid) return true;
#ifdef _WIN32
        return WaitForSingleObject(handle, 0) == WAIT_OBJECT_0;
#else
#ifndef __APPLE__
        if (pidfd >= 0) {
            pollfd state{pidfd, POLLIN, 0};
            return poll(&state, 1, 0) > 0;
        }
#endif
        return current_identity() != identity;
#endif
    }
};

bool same_path(const fs::path& first, const fs::path& second) {
#ifdef _WIN32
    return CompareStringOrdinal(first.c_str(), -1, second.c_str(), -1, TRUE) == CSTR_EQUAL;
#else
    return first == second;
#endif
}

bool installation_image(const fs::path& root, const fs::path& image) {
    const auto bin = root / "bin";
    for (auto parent = image.parent_path(); !parent.empty(); parent = parent.parent_path()) {
        if (same_path(parent, bin)) return true;
        if (parent == parent.parent_path()) break;
    }
#ifdef _WIN32
    return same_path(image, root / "launcher.exe");
#else
    if (image == root / "launcher") return true;
#ifdef __APPLE__
    auto relative = image.lexically_relative(root);
    auto part = relative.begin();
    if (part == relative.end() || part->extension() != ".app") return false;
    fs::path tail;
    for (++part; part != relative.end(); ++part) tail /= *part;
    return tail == fs::path("Contents/MacOS/launcher");
#else
    return false;
#endif
#endif
}

std::vector<std::unique_ptr<Process>> installation_processes(const fs::path& root) {
    std::vector<std::uint32_t> ids;
#ifdef _WIN32
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot check whether Fox Reader has fully stopped.");
    PROCESSENTRY32W entry{};
    entry.dwSize = sizeof entry;
    if (Process32FirstW(snapshot, &entry)) do { ids.push_back(entry.th32ProcessID); } while (Process32NextW(snapshot, &entry));
    CloseHandle(snapshot);
#elif defined(__APPLE__)
    const int bytes = proc_listpids(PROC_ALL_PIDS, 0, nullptr, 0);
    if (bytes <= 0) throw std::runtime_error("Cannot check whether Fox Reader has fully stopped.");
    std::vector<int> processes(static_cast<std::size_t>(bytes) / sizeof(int) + 256);
    const int actual = proc_listpids(PROC_ALL_PIDS, 0, processes.data(), static_cast<int>(processes.size() * sizeof(int)));
    if (actual <= 0) throw std::runtime_error("Cannot enumerate Fox Reader processes.");
    for (int index = 0; index < actual / static_cast<int>(sizeof(int)); ++index)
        if (processes[index] > 0) ids.push_back(static_cast<std::uint32_t>(processes[index]));
#else
    for (const auto& entry : fs::directory_iterator("/proc")) {
        const auto name = entry.path().filename().string();
        if (!name.empty() && name.find_first_not_of("0123456789") == std::string::npos)
            ids.push_back(process_id(name, true));
    }
#endif
    std::vector<std::unique_ptr<Process>> found;
    for (auto id : ids) {
        if (!id || id == self_pid()) continue;
        auto process = std::make_unique<Process>();
        try {
            if (installation_image(root, process->capture(id))) found.push_back(std::move(process));
        } catch (const std::exception&) {
            // Unrelated system processes may deny access; processes which exit
            // during enumeration also no longer prevent the update.
        }
    }
    return found;
}

struct UpdateLock {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
    ~UpdateLock() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
#else
    int fd = -1;
    ~UpdateLock() { if (fd >= 0) close(fd); }
#endif
    bool acquire(const fs::path& root) {
        const auto path = root / ".fox-reader.update.lock";
        if (fs::is_symlink(fs::symlink_status(path))) throw std::runtime_error("The installation update lock is unsafe.");
#ifdef _WIN32
        handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                             nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open the installation update lock.");
        OVERLAPPED offset{};
        if (LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &offset)) return true;
        CloseHandle(handle);
        handle = INVALID_HANDLE_VALUE;
        return false;
#else
        fd = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) throw std::runtime_error("Cannot open the installation update lock.");
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) return true;
        close(fd); fd = -1;
        return false;
#endif
    }
};

std::string read_ticket() {
#ifdef _WIN32
    HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
    if (!input || input == INVALID_HANDLE_VALUE || GetFileType(input) != FILE_TYPE_PIPE)
        throw std::runtime_error("Start updates from Settings in Fox Reader.");
#else
    struct stat info{};
    if (fstat(STDIN_FILENO, &info) || !S_ISFIFO(info.st_mode))
        throw std::runtime_error("Start updates from Settings in Fox Reader.");
#endif
    std::string text;
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    unsigned lines = 0;
    while (lines < 8 && text.size() < 65536 && Clock::now() < deadline) {
        char byte = 0;
#ifdef _WIN32
        DWORD available = 0, count = 0;
        if (!PeekNamedPipe(input, nullptr, 0, nullptr, &available, nullptr)) break;
        if (!available) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); continue; }
        if (!ReadFile(input, &byte, 1, &count, nullptr) || !count) break;
#else
        pollfd descriptor{STDIN_FILENO, POLLIN, 0};
        if (poll(&descriptor, 1, 20) <= 0) continue;
        if (read(STDIN_FILENO, &byte, 1) != 1) break;
#endif
        if (byte == '\0' || byte == '\r') throw std::runtime_error("The update ticket is malformed.");
        text += byte;
        if (byte == '\n') ++lines;
    }
    if (lines != 8) throw std::runtime_error("Fox Reader did not send a complete update ticket.");
    return text;
}

bool launcher_image(const fs::path& root, const fs::path& image) {
#ifdef _WIN32
    std::error_code error;
    return fs::equivalent(image, root / "launcher.exe", error);
#else
    if (image == (root / "launcher")) return true;
#ifdef __APPLE__
    const auto relative = image.lexically_relative(root);
    auto part = relative.begin();
    if (part == relative.end() || part->extension() != ".app") return false;
    ++part;
    fs::path tail;
    for (; part != relative.end(); ++part) tail /= *part;
    return tail == fs::path("Contents/MacOS/launcher");
#else
    return false;
#endif
#endif
}

std::string random_token() {
    std::array<unsigned char, 32> bytes{};
#ifdef _WIN32
    if (BCryptGenRandom(nullptr, bytes.data(), static_cast<ULONG>(bytes.size()), BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0)
        throw std::runtime_error("Cannot create a cleanup handoff token.");
#else
    int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("Cannot create a cleanup handoff token.");
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto count = read(fd, bytes.data() + offset, bytes.size() - offset);
        if (count <= 0) { close(fd); throw std::runtime_error("Cannot create a cleanup handoff token."); }
        offset += static_cast<std::size_t>(count);
    }
    close(fd);
#endif
    constexpr char digits[] = "0123456789abcdef";
    std::string token;
    for (auto byte : bytes) { token += digits[byte >> 4]; token += digits[byte & 15]; }
    return token;
}

void write_token(const fs::path& path, const std::string& token) {
#ifdef _WIN32
    HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                             FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot prepare cleanup authentication.");
    DWORD written = 0;
    bool ok = WriteFile(file, token.data(), static_cast<DWORD>(token.size()), &written, nullptr) &&
              written == token.size() && FlushFileBuffers(file);
    CloseHandle(file);
#else
    int file = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (file < 0) throw std::runtime_error("Cannot prepare cleanup authentication.");
    bool ok = write(file, token.data(), token.size()) == static_cast<ssize_t>(token.size()) && fsync(file) == 0;
    close(file);
#endif
    if (!ok) { std::error_code ignored; fs::remove(path, ignored); throw std::runtime_error("Cannot save cleanup authentication."); }
}

bool spawn_cleanup(const fs::path& executable, const std::string& ticket) {
#ifdef _WIN32
    SECURITY_ATTRIBUTES security{sizeof security, nullptr, TRUE};
    HANDLE child_input = nullptr, writer = nullptr, reader = nullptr, child_output = nullptr;
    if (!CreatePipe(&child_input, &writer, &security, 0)) return false;
    if (!CreatePipe(&reader, &child_output, &security, 0)) { CloseHandle(child_input); CloseHandle(writer); return false; }
    SetHandleInformation(writer, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(reader, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof startup;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    startup.StartupInfo.wShowWindow = SW_HIDE;
    startup.StartupInfo.hStdInput = child_input;
    startup.StartupInfo.hStdOutput = child_output;
    startup.StartupInfo.hStdError = child_output;
    SIZE_T attribute_size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_size);
    std::vector<unsigned char> attributes(attribute_size);
    startup.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
    HANDLE inherited[] = {child_input, child_output};
    if (!InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &attribute_size) ||
        !UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                   inherited, sizeof inherited, nullptr, nullptr)) {
        CloseHandle(child_input); CloseHandle(child_output); CloseHandle(reader); CloseHandle(writer);
        return false;
    }
    PROCESS_INFORMATION process{};
    std::wstring command = L"\"" + executable.native() + L"\"";
    const bool created = CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | EXTENDED_STARTUPINFO_PRESENT,
        nullptr, nullptr, &startup.StartupInfo, &process) != 0;
    DeleteProcThreadAttributeList(startup.lpAttributeList);
    CloseHandle(child_input); CloseHandle(child_output);
    if (!created) { CloseHandle(reader); CloseHandle(writer); return false; }
    DWORD sent = 0;
    const bool wrote = WriteFile(writer, ticket.data(), static_cast<DWORD>(ticket.size()), &sent, nullptr) && sent == ticket.size();
    CloseHandle(writer);
    std::string response;
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    while (wrote && response.size() < 6 && Clock::now() < deadline) {
        DWORD available = 0, count = 0; char byte = 0;
        if (!PeekNamedPipe(reader, nullptr, 0, nullptr, &available, nullptr)) break;
        if (available && ReadFile(reader, &byte, 1, &count, nullptr) && count) response += byte;
        else if (WaitForSingleObject(process.hProcess, 20) == WAIT_OBJECT_0) break;
    }
    CloseHandle(reader); CloseHandle(process.hThread); CloseHandle(process.hProcess);
    return response == "READY\n";
#else
    int input[2], output[2];
    if (pipe(input)) return false;
    if (pipe(output)) { close(input[0]); close(input[1]); return false; }
    const std::string program = executable.string();
    const pid_t child = fork();
    if (child == 0) {
        setsid();
        dup2(input[0], STDIN_FILENO); dup2(output[1], STDOUT_FILENO); dup2(output[1], STDERR_FILENO);
        close(input[0]); close(input[1]); close(output[0]); close(output[1]);
        execl(program.c_str(), program.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    close(input[0]); close(output[1]);
    if (child < 0) { close(input[1]); close(output[0]); return false; }
    const bool wrote = write(input[1], ticket.data(), ticket.size()) == static_cast<ssize_t>(ticket.size());
    close(input[1]);
    std::string response;
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    while (wrote && response.size() < 6 && Clock::now() < deadline) {
        pollfd state{output[0], POLLIN, 0}; char byte = 0;
        if (poll(&state, 1, 20) <= 0) continue;
        if (read(output[0], &byte, 1) != 1) break;
        response += byte;
    }
    close(output[0]);
    return response == "READY\n";
#endif
}
}  // namespace

struct Handoff::Impl {
    Options options;
    std::string action;
    Process parent, owner;
    UpdateLock lock;
    bool wait_cleanup_parent = true;
    bool cleanup_allowed = false;
};

Handoff::Handoff(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Handoff::~Handoff() = default;

std::shared_ptr<Handoff> Handoff::receive(std::string& error) {
#ifdef _WIN32
    // The pipe protocol is UTF-8 with LF on every OS; CRT text mode would
    // otherwise silently change READY\n into READY\r\n.
    if (_fileno(stdout) >= 0) _setmode(_fileno(stdout), _O_BINARY);
#endif
    try {
        std::istringstream input(read_ticket());
        std::array<std::string, 8> fields;
        for (auto& field : fields) std::getline(input, field);
        if (fields[0] != "FOXREADER-UPDATER-1" ||
            (fields[1] != "install" && fields[1] != "recover" && fields[1] != "cleanup"))
            throw std::runtime_error("This updater requires a valid Fox Reader update handoff.");
        constexpr const char* prefix = ".foxreader-update-";
        if (fields[3].compare(0, 18, prefix) != 0 || !hex_string(fields[3].substr(18), 32) ||
            !hex_string(fields[4], 64) || !hex_string(fields[7], 64))
            throw std::runtime_error("The update transaction identifier is invalid.");
        auto state = std::make_unique<Impl>();
        state->action = fields[1];
        auto root = fs::u8path(fields[2]);
        if (!root.is_absolute() || fs::is_symlink(fs::symlink_status(root)))
            throw std::runtime_error("The installation folder is unsafe.");
        state->options.root = fs::canonical(root);
        state->options.transaction = state->options.root / fields[3];
        state->options.inventory_sha256 = fields[7];
        const auto& transaction = state->options.transaction;
        if (!fs::is_directory(transaction) || fs::is_symlink(fs::symlink_status(transaction)) ||
            fs::canonical(transaction).parent_path() != state->options.root)
            throw std::runtime_error("The update transaction is outside this installation.");
        const auto runner = transaction / "runner" / updater_name;
        const auto installed = state->options.root / updater_name;
        if (own_executable() != fs::canonical(state->action == "cleanup" ? installed : runner))
            throw std::runtime_error("This updater was not started from its prepared transaction.");
        auto parent = process_id(fields[5]);
        if (parent_pid() != parent) throw std::runtime_error("The update was not started by the expected Fox Reader process.");
        const auto parent_image = state->parent.capture(parent);
        if (state->action == "cleanup") {
            if (launcher_image(state->options.root, parent_image)) state->wait_cleanup_parent = false;
            else if (parent_image != fs::canonical(runner))
                throw std::runtime_error("Cleanup must be started by the active updater or launcher.");
        } else {
            const auto backend = state->options.root / "bin" /
#ifdef _WIN32
                "fox-reader.exe";
#else
                "fox-reader";
#endif
            std::error_code image_error;
            if (!fs::equivalent(parent_image, backend, image_error) && !launcher_image(state->options.root, parent_image))
                throw std::runtime_error("Updates must be started by this installation of Fox Reader.");
        }
        const auto owner = process_id(fields[6], true);
        if (owner && owner != parent && !launcher_image(state->options.root, state->owner.capture(owner)))
            throw std::runtime_error("The launcher process does not belong to this installation.");
        const auto token_path = transaction / "ticket.token";
        if (!fs::is_regular_file(fs::symlink_status(token_path)) || fs::file_size(token_path) > 65)
            throw std::runtime_error("The one-use update ticket is missing or unsafe.");
        std::ifstream token_stream(token_path, std::ios::binary);
        std::string token((std::istreambuf_iterator<char>(token_stream)), {});
        token_stream.close();  // Windows cannot rename a file still opened by this reader.
        if (!token.empty() && token.back() == '\n') token.pop_back();
        if (token != fields[4]) throw std::runtime_error("The update ticket did not match its transaction.");
        if (state->action != "cleanup" && !state->lock.acquire(state->options.root))
            throw std::runtime_error("Another update is already running for this installation.");
#ifdef _WIN32
        if (!MoveFileExW(token_path.c_str(), (transaction / "ticket.claimed").c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("The update ticket has already been claimed.");
#else
        fs::rename(token_path, transaction / "ticket.claimed");
#endif
        return std::shared_ptr<Handoff>(new Handoff(std::move(state)));
    } catch (const std::exception& exc) {
        error = exc.what();
        std::cout << "ERROR\n" << std::flush;
        return {};
    }
}

bool Handoff::silent() const { return impl_->action == "cleanup"; }

Result Handoff::run(ProgressCallback progress, Cancellation cancelled) {
    try {
        std::error_code ignored;
        fs::create_directory(impl_->options.root / "logs", ignored);
        std::ofstream log(impl_->options.root / "logs" / "updater.log", std::ios::app);
        std::string last_stage;
        const auto display = [&](const Progress& update) {
            if (update.stage != last_stage) {
                log << update.stage << ": " << update.message << '\n' << std::flush;
                last_stage = update.stage;
            }
            if (progress) progress(update);
        };
        const auto report = [&](const std::string& stage, const std::string& message, bool can_cancel) {
            display({stage, message, 0, 0, can_cancel});
        };
        const auto finish = [&](Result result) {
            // Cancellation and verification failures can leave a full archive
            // and extracted payload. Remove those while holding the update
            // lock, but preserve every artifact if recovery is still needed.
            if (!result.recovery_required) {
                try {
                    if (inspect(impl_->options) == TransactionState::interrupted) {
                        result.recovery_required = true;
                        result.message += " An interrupted update still needs recovery.";
                    } else {
                        report("cleanup", "Removing update files...", false);
                        const auto cleaned = cleanup(impl_->options);
                        result.cleanup_pending = !cleaned.success || cleaned.cleanup_pending;
                        result.recovery_required = cleaned.recovery_required;
                        impl_->cleanup_allowed = !cleaned.recovery_required;
                    }
                } catch (const std::exception& error) {
                    result.recovery_required = true;
                    result.message += " Update files were preserved because their state could not be checked. " +
                                      std::string(error.what());
                }
            }
            if (result.recovery_required) {
                result.success = false;
                result.cancelled = false;
                result.rolled_back = false;
                impl_->cleanup_allowed = false;
            }
            log << (result.success ? "success: " : result.cancelled ? "cancelled: " : "error: ") << result.message << '\n';
            return result;
        };
        std::cout << "READY\n" << std::flush;
        report("waiting", "Waiting for Fox Reader to close safely...", !silent());
        const auto wait_deadline = Clock::now() + std::chrono::seconds(60);
        while ((!silent() || impl_->wait_cleanup_parent) && (!impl_->parent.exited() || !impl_->owner.exited())) {
            if (!silent() && cancelled && cancelled())
                return finish({false, true, false, false, true, "Update cancelled before any files changed."});
            if (Clock::now() > wait_deadline) {
                const std::string message = "Fox Reader is still running. Close its remaining windows and retry the update.";
                const Result result{false, false, false, false, true, message};
                return silent() ? result : finish(result);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (silent()) {
            const auto lock_deadline = Clock::now() + std::chrono::seconds(5);
            while (!impl_->lock.acquire(impl_->options.root)) {
                if (Clock::now() > lock_deadline) return {false, false, false, false, true, "Cleanup will be retried next launch."};
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            return cleanup(impl_->options);
        }
        // Attached controllers and backend workers can outlive the original
        // parent. Pin and wait for exact executable identities under this root,
        // then rescan before mutation; never terminate someone else's process.
        for (;;) {
            auto remaining = installation_processes(impl_->options.root);
            if (remaining.empty()) break;
            report("waiting", "Waiting for the remaining Fox Reader windows and workers to close...", true);
            for (;;) {
                if (cancelled && cancelled())
                    return finish({false, true, false, false, true, "Update cancelled before any files changed."});
                bool alive = false;
                for (const auto& process : remaining) if (!process->exited()) { alive = true; break; }
                if (!alive) break;
                if (Clock::now() > wait_deadline) {
                    const std::string message = "Some Fox Reader windows or workers are still running. Close them and retry the update.";
                    return finish({false, false, false, false, true, message});
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        return finish(impl_->action == "recover" ? recover(impl_->options, display) :
                                                  apply(impl_->options, display, cancelled));
    } catch (const std::exception& exc) {
        return {false, false, false, true, true, exc.what()};
    }
}

bool Handoff::start_cleanup() {
    try {
        if (!impl_->cleanup_allowed) return false;
        if (!fs::exists(impl_->options.transaction)) return true;
        if (inspect(impl_->options) == TransactionState::interrupted) return false;
        const auto token = random_token();
        write_token(impl_->options.transaction / "ticket.token", token);
        const std::string ticket = "FOXREADER-UPDATER-1\ncleanup\n" + impl_->options.root.u8string() + "\n" +
            impl_->options.transaction.filename().u8string() + "\n" + token + "\n" + std::to_string(self_pid()) +
            "\n0\n" + impl_->options.inventory_sha256 + "\n";
        return spawn_cleanup(impl_->options.root / updater_name, ticket);
    } catch (...) { return false; }
}
}  // namespace fox_updater
