/* The launcher itself requires neither GTK nor WebKit. Browser control is a
 * separate GTK-only plugin, so missing WebKit does not remove the close UI. */
#include "window.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    void *library;
    int (*create)(const char *, void (*)(void), void (*)(const char *), void (*)(const char *), int);
    void (*status)(const char *);
    void (*ready)(const char *, int);
    void (*finish)(int);
    int (*run)(void);
    void (*destroy)(void);
} WindowAPI;

static WindowAPI selected;

static int load_window(WindowAPI *api, const char *root, const char *name,
                       void (*stop)(void), void (*browser)(const char *),
                       void (*log)(const char *), int browser_mode,
                       char *reason, size_t reason_size)
{
    char path[4096];
    int length = snprintf(path, sizeof path, "%s/%s", root, name);
    memset(api, 0, sizeof *api);
    if (length < 0 || length >= (int)sizeof path) {
        snprintf(reason, reason_size, "%s: installation path is too long", name);
        return 0;
    }
    api->library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!api->library) {
        snprintf(reason, reason_size, "%s: %s", name, dlerror());
        return 0;
    }
    api->create = dlsym(api->library, "fox_window_create");
    api->status = dlsym(api->library, "fox_window_status");
    api->ready = dlsym(api->library, "fox_window_ready");
    api->finish = dlsym(api->library, "fox_window_finish");
    api->run = dlsym(api->library, "fox_window_run");
    api->destroy = dlsym(api->library, "fox_window_destroy");
    if (!api->create || !api->status || !api->ready || !api->finish || !api->run || !api->destroy) {
        snprintf(reason, reason_size, "%s: required window functions are missing", name);
        /* No create call has run, so no GTK callbacks/types belong to us yet. */
        dlclose(api->library);
        memset(api, 0, sizeof *api);
        return 0;
    }
    if (!api->create(root, stop, browser, log, browser_mode)) {
        snprintf(reason, reason_size, "%s: a graphical window could not be opened", name);
        /* Creation can initialize process-wide GTK types. Leave this handle
         * loaded even on failure; unloading could invalidate those types. */
        memset(api, 0, sizeof *api);
        return 0;
    }
    return 1;
}

static void report(void (*log)(const char *), const char *prefix, const char *reason)
{
    char message[4608];
    snprintf(message, sizeof message, "%s: %s", prefix, reason);
    if (log) log(message);
    else fprintf(stderr, "Fox Reader: %s\n", message);
}

int fox_window_create(const char *root, void (*stop)(void),
                      void (*browser)(const char *), void (*log)(const char *), int browser_mode)
{
    WindowAPI candidate;
    char reason[4096];
    if (selected.library) return 0;
    if (!browser_mode) {
        if (load_window(&candidate, root, "libfox-window.so", stop, browser, log, 0, reason, sizeof reason)) {
            selected = candidate;
            return 1;
        }
        report(log, "Native reader unavailable; trying the browser controller", reason);
    }
    if (load_window(&candidate, root, "libfox-browser.so", stop, browser, log, 1, reason, sizeof reason)) {
        selected = candidate;
        return 1;
    }
    report(log, "Browser controller unavailable", reason);
    return 0;
}

void fox_window_status(const char *value) { if (selected.status) selected.status(value); }
void fox_window_ready(const char *value, int owned) { if (selected.ready) selected.ready(value, owned); }
void fox_window_finish(int value) { if (selected.finish) selected.finish(value); }
int fox_window_run(void) { return selected.run ? selected.run() : 1; }
void fox_window_destroy(void)
{
    if (selected.destroy) selected.destroy();
    /* GTK/WebKit keep process-wide types and callbacks. Do not dlclose them. */
    memset(&selected, 0, sizeof selected);
}
