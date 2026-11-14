#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

int run(const std::vector<std::string>& arguments) {
    if (arguments.size() == 3 && arguments[1] == "--hold") {
        while (!fs::exists(fs::u8path(arguments[2]))) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return 0;
    }
    if (arguments.size() != 5) return 64;
    const auto root = fs::u8path(arguments[1]);
    const auto transaction = root / arguments[2];
    auto executable = transaction / "runner" /
#ifdef _WIN32
        "foxreader-updater.exe";
    const auto pid = GetCurrentProcessId();
#else
        "foxreader-updater";
    const auto pid = getpid();
#endif
    if (arguments[4] == "cleanup") executable = root / executable.filename();
    const std::string token(64, 'a');
    std::ofstream(transaction / "ticket.token", std::ios::binary) << token;
    const std::string nonce = arguments[4] == "bad-token" ? std::string(64, 'b') : token;
    const auto sender = arguments[4] == "bad-parent" ? 1 : pid;
    const std::string action = arguments[4] == "cleanup" ? "cleanup" : "install";
    const std::string ticket = "FOXREADER-UPDATER-1\n" + action + '\n' + root.u8string() + '\n' + arguments[2] + '\n' +
        nonce + '\n' + std::to_string(sender) + "\n0\n" + arguments[3] + '\n';
    std::string response;
#ifdef _WIN32
    SECURITY_ATTRIBUTES security{sizeof security, nullptr, TRUE};
    HANDLE child_input, writer, reader, child_output;
    if (!CreatePipe(&child_input, &writer, &security, 0) || !CreatePipe(&reader, &child_output, &security, 0)) return 3;
    SetHandleInformation(writer, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(reader, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW startup{};
    startup.cb = sizeof startup;
    startup.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    startup.wShowWindow = SW_HIDE;
    startup.hStdInput = child_input; startup.hStdOutput = child_output; startup.hStdError = child_output;
    PROCESS_INFORMATION process{};
    auto command = L"\"" + executable.native() + L"\"";
    if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                        nullptr, root.c_str(), &startup, &process)) return 4;
    CloseHandle(child_input); CloseHandle(child_output);
    std::ofstream(root / "updater-pid") << process.dwProcessId;
    DWORD sent = 0;
    if (!WriteFile(writer, ticket.data(), static_cast<DWORD>(ticket.size()), &sent, nullptr)) return 5;
    CloseHandle(writer);
    char byte; DWORD count;
    while (response.size() < 128 && ReadFile(reader, &byte, 1, &count, nullptr) && count) {
        response += byte;
        if (byte == '\n') break;
    }
    CloseHandle(reader); CloseHandle(process.hThread); CloseHandle(process.hProcess);
#else
    std::signal(SIGPIPE, SIG_IGN);
    int input[2], output[2];
    if (pipe(input) || pipe(output)) return 3;
    const auto program = executable.string();
    auto child = fork();
    if (child == 0) {
        chdir(root.c_str()); setsid();
        dup2(input[0], STDIN_FILENO); dup2(output[1], STDOUT_FILENO); dup2(output[1], STDERR_FILENO);
        close(input[0]); close(input[1]); close(output[0]); close(output[1]);
        execl(program.c_str(), program.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    if (child < 0) return 4;
    close(input[0]); close(output[1]);
    std::ofstream(root / "updater-pid") << child;
    if (write(input[1], ticket.data(), ticket.size()) != static_cast<ssize_t>(ticket.size())) return 5;
    close(input[1]);
    char byte;
    while (response.size() < 128 && read(output[0], &byte, 1) == 1) {
        response += byte;
        if (byte == '\n') break;
    }
    close(output[0]);
#endif
    std::ofstream(root / "handoff-response") << response;
    if (response != "READY\n") return 6;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!fs::exists(root / "exit-parent") && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> arguments;
    for (int i = 0; i < argc; ++i) {
        int size = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string argument(static_cast<std::size_t>(size), '\0');
        WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, argument.data(), size, nullptr, nullptr);
        argument.pop_back(); arguments.push_back(argument);
    }
    return run(arguments);
}
#else
int main(int argc, char** argv) { return run(std::vector<std::string>(argv, argv + argc)); }
#endif
