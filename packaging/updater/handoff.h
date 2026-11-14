#pragma once

#include "core.h"
#include <memory>

namespace fox_updater {

// Only an authenticated, one-use anonymous-pipe ticket can construct a handoff.
class Handoff {
public:
    ~Handoff();
    Handoff(const Handoff&) = delete;
    Handoff& operator=(const Handoff&) = delete;
    static std::shared_ptr<Handoff> receive(std::string& error);
    bool silent() const;
    Result run(ProgressCallback progress, Cancellation cancelled);
    bool start_cleanup();
private:
    struct Impl;
    explicit Handoff(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace fox_updater
