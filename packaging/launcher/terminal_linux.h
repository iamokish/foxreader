#ifndef FOX_READER_TERMINAL_LINUX_H
#define FOX_READER_TERMINAL_LINUX_H

/* These helpers never start a backend. Only a successor with an interactive
 * controlling terminal may complete the handoff and continue launcher startup. */
int fox_terminal_available(void);
int fox_terminal_handoff(const char *executable, int (*cancelled)(void));
int fox_terminal_connect(const char *socket_path);
#ifndef FOX_NO_WINDOW
void fox_terminal_notify(const char *message);
#endif

#endif
