#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace fox_updater {

struct Options {
    std::filesystem::path root;
    std::filesystem::path transaction;
    std::string inventory_sha256;
};

struct Progress {
    std::string stage;
    std::string message;
    std::uint64_t done = 0;
    std::uint64_t total = 0;
    bool cancellable = false;
};

using ProgressCallback = std::function<void(const Progress&)>;
using Cancellation = std::function<bool()>;

struct Result {
    bool success = false;
    bool cancelled = false;
    bool rolled_back = false;
    bool recovery_required = false;
    bool cleanup_pending = false;
    std::string message;
};

enum class TransactionState { prepared, interrupted, committed, rolled_back };

// The caller must authenticate the one-use inherited-pipe handoff and hold the
// installation's update lock after all backend/launcher processes have exited.
// Cancellation is observed only before journal.bin is published. Once mutation
// begins, the operation completes or rolls back without accepting cancellation.
Result apply(const Options&, ProgressCallback = {}, Cancellation = {});
Result recover(const Options&, ProgressCallback = {});
Result cleanup(const Options&);
TransactionState inspect(const Options&);  // Throws on unsafe/corrupt state.

// Cross-process recovery contract: journal.bin means installation mutation may
// have begun; state contains COMMITTED\n or ROLLED_BACK\n only once safe to start.
// Keep both files until cleanup has removed all payload/old/runner artifacts.

#ifdef FOX_UPDATER_TESTING
// Only native test harnesses expose deterministic failure/crash injection.
using FaultHook = std::function<void(const std::string&)>;
void set_fault_hook(FaultHook);
#endif

}  // namespace fox_updater
