#include "../packaging/updater/handoff.h"
#include <csignal>
#include <fstream>
#include <iostream>

int main() {
#ifndef _WIN32
    std::signal(SIGPIPE, SIG_IGN);
#endif
    std::string error;
    auto handoff = fox_updater::Handoff::receive(error);
    if (!handoff) { std::ofstream("updater-error") << error; return 2; }
    const bool silent = handoff->silent();
    auto result = handoff->run({}, [] { return std::filesystem::exists("cancel-request"); });
    if (!silent) {
        std::ofstream("updater-result") << result.success << ' ' << result.cancelled << ' ' << result.rolled_back
                                       << ' ' << result.recovery_required << '\n' << result.message;
        if (result.cleanup_pending && !result.recovery_required) handoff->start_cleanup();
        if (std::filesystem::exists("probe-cleanup"))
            std::ofstream("cleanup-started") << handoff->start_cleanup();
    }
    return result.success || result.cancelled || result.rolled_back ? 0 : 1;
}
