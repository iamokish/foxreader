#pragma once

#include "handoff.h"
#include <atomic>
#include <mutex>
#include <thread>

namespace fox_updater {

struct ViewState {
    Progress progress{"preparing", "Preparing the update...", 0, 0, false};
    bool finished = false;
    bool successful = false;
    int exit_code = 1;
};

class UiState {
public:
    UiState(std::shared_ptr<Handoff> handoff, const std::string& error);
    ~UiState();
    void start();
    ViewState snapshot() const;
    // True means it is safe for the native window/application to close.
    bool request_close();
private:
    mutable std::mutex mutex_;
    ViewState view_;
    std::shared_ptr<Handoff> handoff_;
    std::atomic<bool> cancelled_{false};
    std::thread worker_;
    bool cleanup_ = false;
    bool closing_ = false;
};

std::string stage_title(const std::string& stage);
}  // namespace fox_updater
