/* Exercise the actual policy used by all native hosts, with a controlled clock. */
#include "../packaging/launcher/window_recovery.h"
#include <stdio.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { fprintf(stderr, "line %d: %s\n", __LINE__, #expression); return 1; } \
} while (0)

int main(int argc, char **argv)
{
    FoxWindowRecovery state = {0};
    const uint64_t much_later = UINT64_C(86400000);
    if (argc != 2) return 2;
    if (!strcmp(argv[1], "backend_wait")) {
        CHECK(!fox_recovery_timed_out(&state, much_later));
        CHECK(fox_recovery_begin_initialization(&state, much_later));
        fox_recovery_engine_ready(&state);
        CHECK(!fox_recovery_timed_out(&state, much_later * 2));
        CHECK(fox_recovery_begin_navigation(&state, much_later * 2));
        CHECK(!fox_recovery_timed_out(&state, much_later * 2 + 29999));
    } else if (!strcmp(argv[1], "initialization_deadline")) {
        CHECK(fox_recovery_begin_initialization(&state, 500));
        CHECK(!fox_recovery_timed_out(&state, 499));
        CHECK(!fox_recovery_timed_out(&state, 30499));
        CHECK(fox_recovery_timed_out(&state, 30500));
        CHECK(fox_recovery_fallback(&state));
        CHECK(!fox_recovery_timed_out(&state, much_later));
    } else if (!strcmp(argv[1], "separate_navigation_budget")) {
        CHECK(fox_recovery_begin_initialization(&state, 1));
        fox_recovery_engine_ready(&state);
        CHECK(fox_recovery_begin_navigation(&state, 29000));
        CHECK(!fox_recovery_timed_out(&state, 58999));
        CHECK(fox_recovery_timed_out(&state, 59000));
    } else if (!strcmp(argv[1], "ready_has_no_deadline")) {
        CHECK(fox_recovery_begin_navigation(&state, 50));
        fox_recovery_loaded(&state);
        CHECK(!fox_recovery_timed_out(&state, much_later));
        CHECK(!fox_recovery_begin_navigation(&state, much_later));
        fox_recovery_engine_ready(&state);
        CHECK(!fox_recovery_begin_navigation(&state, much_later));
    } else if (!strcmp(argv[1], "repeated_events_keep_deadline")) {
        CHECK(fox_recovery_begin_initialization(&state, 1));
        CHECK(!fox_recovery_begin_initialization(&state, 10000));
        CHECK(!fox_recovery_begin_navigation(&state, 10000));
        CHECK(fox_recovery_timed_out(&state, 30001));
        fox_recovery_engine_ready(&state);
        CHECK(fox_recovery_begin_navigation(&state, 40000));
        CHECK(!fox_recovery_begin_navigation(&state, 60000));
        CHECK(fox_recovery_timed_out(&state, 70000));
    } else if (!strcmp(argv[1], "fallback_before_backend_ready")) {
        CHECK(fox_recovery_fallback(&state));
        fox_recovery_engine_ready(&state);
        fox_recovery_loaded(&state);
        CHECK(!fox_recovery_begin_navigation(&state, much_later));
        CHECK(!fox_recovery_begin_initialization(&state, much_later));
        CHECK(!fox_recovery_fallback(&state));
    } else if (!strcmp(argv[1], "renderer_failure_after_ready")) {
        CHECK(fox_recovery_begin_navigation(&state, 0));
        fox_recovery_loaded(&state);
        CHECK(fox_recovery_fallback(&state));
        CHECK(!fox_recovery_fallback(&state));
        fox_recovery_loaded(&state);
        CHECK(!fox_recovery_fallback(&state));
        CHECK(!fox_recovery_timed_out(&state, much_later));
    } else if (!strcmp(argv[1], "shutdown_suppresses_recovery")) {
        int phase;
        for (phase = FOX_RECOVERY_WAITING; phase <= FOX_RECOVERY_STOPPED; ++phase) {
            state.phase = (FoxRecoveryPhase)phase;
            fox_recovery_stop(&state);
            fox_recovery_engine_ready(&state);
            fox_recovery_loaded(&state);
            CHECK(!fox_recovery_timed_out(&state, much_later));
            CHECK(!fox_recovery_fallback(&state));
            CHECK(!fox_recovery_begin_initialization(&state, much_later));
            CHECK(!fox_recovery_begin_navigation(&state, much_later));
        }
    } else return 2;
    return 0;
}
