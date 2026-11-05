#ifndef FOX_READER_WINDOW_RECOVERY_H
#define FOX_READER_WINDOW_RECOVERY_H

#include <stdint.h>

/* Main-thread-only policy shared by the three native hosts. WAITING has no
 * deadline: backend readiness and model loading belong to the launcher. Each
 * native startup phase gets its own budget; later page loads are not timed. */
typedef enum {
    FOX_RECOVERY_WAITING = 0,
    FOX_RECOVERY_INITIALIZING,
    FOX_RECOVERY_NAVIGATING,
    FOX_RECOVERY_READY,
    FOX_RECOVERY_FALLBACK,
    FOX_RECOVERY_STOPPED
} FoxRecoveryPhase;

typedef struct {
    FoxRecoveryPhase phase;
    uint64_t started_ms;
} FoxWindowRecovery;

#define FOX_RECOVERY_TIMEOUT_MS UINT64_C(30000)

static inline int fox_recovery_begin_initialization(FoxWindowRecovery *state, uint64_t now_ms)
{
    if (state->phase != FOX_RECOVERY_WAITING) return 0;
    state->phase = FOX_RECOVERY_INITIALIZING;
    state->started_ms = now_ms;
    return 1;
}

static inline void fox_recovery_engine_ready(FoxWindowRecovery *state)
{
    if (state->phase == FOX_RECOVERY_INITIALIZING) state->phase = FOX_RECOVERY_WAITING;
}

static inline int fox_recovery_begin_navigation(FoxWindowRecovery *state, uint64_t now_ms)
{
    if (state->phase != FOX_RECOVERY_WAITING) return 0;
    state->phase = FOX_RECOVERY_NAVIGATING;
    state->started_ms = now_ms;
    return 1;
}

static inline void fox_recovery_loaded(FoxWindowRecovery *state)
{
    if (state->phase == FOX_RECOVERY_NAVIGATING) state->phase = FOX_RECOVERY_READY;
}

static inline int fox_recovery_timed_out(const FoxWindowRecovery *state, uint64_t now_ms)
{
    return (state->phase == FOX_RECOVERY_INITIALIZING || state->phase == FOX_RECOVERY_NAVIGATING)
        && now_ms >= state->started_ms && now_ms - state->started_ms >= FOX_RECOVERY_TIMEOUT_MS;
}

/* Claim recovery before stopping the view: teardown can call back into the
 * host synchronously. Only the first failure may log/open browser fallback. */
static inline int fox_recovery_fallback(FoxWindowRecovery *state)
{
    if (state->phase == FOX_RECOVERY_FALLBACK || state->phase == FOX_RECOVERY_STOPPED) return 0;
    state->phase = FOX_RECOVERY_FALLBACK;
    return 1;
}

static inline void fox_recovery_stop(FoxWindowRecovery *state)
{
    state->phase = FOX_RECOVERY_STOPPED;
}

#endif
