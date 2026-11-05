#ifndef FOX_READER_WINDOW_H
#define FOX_READER_WINDOW_H

#ifdef __cplusplus
extern "C" {
#endif

/* create/run/destroy belong to the main thread. The remaining calls may come
 * from the backend lifecycle worker. Strings are UTF-8 and copied on entry.
 * Callbacks must return promptly: request_stop only signals the worker.
 * Browser mode keeps a small manager window so its close button still stops
 * the owned backend. A missing system WebView uses the same browser fallback.
 */
int fox_window_create(const char *root_utf8, void (*request_stop)(void),
                      void (*open_browser)(const char *url), void (*log)(const char *message),
                      int browser_mode);
void fox_window_status(const char *message);
/* Ownership is explicit so a browser controller can label its close action
 * accurately; an attached window must not promise to stop another launcher. */
void fox_window_ready(const char *url, int owns_backend);
/* Success closes the window; a failure remains visible until dismissed.
 * A close requested by the user always waits for the worker to finish. */
void fox_window_finish(int exit_code);
int fox_window_run(void);
void fox_window_destroy(void);

#ifdef __cplusplus
}
#endif

#endif
