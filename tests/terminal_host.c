/* Exercise the terminal handoff without launching an application backend. */
#include "../packaging/launcher/terminal_linux.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile sig_atomic_t stopping;
static void stop_requested(int signal_number) { (void)signal_number; stopping = 1; }
static int cancelled(void) { return stopping != 0; }

int main(int argc, char **argv)
{
    const char *marker;
    FILE *file;
    signal(SIGTERM, stop_requested);
    signal(SIGINT, stop_requested);
    if (argc == 2 && strcmp(argv[1], "--available") == 0)
        return fox_terminal_available() ? 0 : 3;
    if (argc == 2 && strcmp(argv[1], "--handoff") == 0)
        return fox_terminal_handoff(argv[0], cancelled) == 0 ? 0 : 3;
    if (argc != 3 || strcmp(argv[1], "--terminal-manager") != 0) return 2;
    if (fox_terminal_connect(argv[2]) != 0 || cancelled()) return 3;
    marker = getenv("FOX_TEST_STARTED");
    if (!marker || !(file = fopen(marker, "w"))) return 4;
    fputs("started only after visible-terminal handshake\n", file);
    fclose(file);
    return 0;
}
