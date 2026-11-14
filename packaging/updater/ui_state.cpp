#include "ui_state.h"

namespace fox_updater {
UiState::UiState(std::shared_ptr<Handoff> handoff, const std::string& error) : handoff_(std::move(handoff)) {
    if (!handoff_) {
        view_.progress = {"error", error, 0, 0, false};
        view_.finished = true;
    }
}

UiState::~UiState() { if (worker_.joinable()) worker_.join(); }

void UiState::start() {
    if (!handoff_) return;
    worker_ = std::thread([this] {
        auto result = handoff_->run([this](const Progress& progress) {
            std::lock_guard<std::mutex> guard(mutex_);
            view_.progress = progress;
        }, [this] { return cancelled_.load(); });
        std::lock_guard<std::mutex> guard(mutex_);
        cleanup_ = result.cleanup_pending && !result.recovery_required;
        view_.finished = true;
        view_.successful = result.success;
        view_.exit_code = result.success || result.cancelled || result.rolled_back ? 0 : 1;
        view_.progress.cancellable = false;
        if (result.cancelled) {
            view_.progress.stage = "cancelled";
            view_.progress.message = "Update cancelled. Your installation was not changed.";
        } else if (result.rolled_back) {
            view_.progress.stage = "restored";
            view_.progress.message = "The previous version was restored. Fox Reader remains closed.\n\n" + result.message;
        } else if (result.success) {
            view_.progress.stage = "complete";
            view_.progress.message = "Update complete. You can start Fox Reader when you are ready.";
        } else {
            view_.progress.stage = "error";
            view_.progress.message = result.message;
            if (result.recovery_required)
                view_.progress.message += "\n\nRecovery is required. Keep the update files in this installation.";
        }
    });
}

ViewState UiState::snapshot() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return view_;
}

bool UiState::request_close() {
    bool cleanup = false;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (closing_) return false;
        if (!view_.finished) {
            if (view_.progress.cancellable) {
                cancelled_.store(true);
                view_.progress.cancellable = false;
                view_.progress.message = "Cancelling the update safely...";
            }
            return false;
        }
        closing_ = true;
        cleanup = cleanup_;
    }
    if (cleanup && handoff_) handoff_->start_cleanup();
    return true;
}

std::string stage_title(const std::string& stage) {
    if (stage == "waiting") return "Waiting for Fox Reader";
    if (stage == "verify" || stage == "verifying") return "Checking the update";
    if (stage == "apply" || stage == "applying" || stage == "install") return "Installing update";
    if (stage == "rollback" || stage == "rolling_back") return "Restoring the previous version";
    if (stage == "cleanup") return "Cleaning up";
    if (stage == "complete" || stage == "success") return "Update complete";
    if (stage == "restored") return "Previous version restored";
    if (stage == "cancelled") return "Update cancelled";
    if (stage == "error") return "Update could not finish";
    return "Preparing update";
}
}  // namespace fox_updater
