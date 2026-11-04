/* Keep --browser usable even on machines without WebKitGTK installed. */
#include "window.h"
#include <dlfcn.h>
#include <stdio.h>

static void *library;
static void (*status_fn)(const char *);
static void (*ready_fn)(const char *);
static void (*finish_fn)(int);
static int (*run_fn)(void);
static void (*destroy_fn)(void);

int fox_window_create(const char *root, void (*stop)(void),
                      void (*browser)(const char *), void (*log)(const char *), int browser_mode)
{
    char path[4096];
    int (*create_fn)(const char *, void (*)(void), void (*)(const char *), void (*)(const char *), int);
    if (snprintf(path, sizeof path, "%s/libfox-window.so", root) >= (int)sizeof path)
        return 0;
    library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        fprintf(stderr, "Fox Reader: WebKitGTK unavailable: %s\n"
                "Install libwebkit2gtk-4.1-0. Opening the browser instead.\n", dlerror());
        return 0;
    }
    create_fn = dlsym(library, "fox_window_create");
    status_fn = dlsym(library, "fox_window_status");
    ready_fn = dlsym(library, "fox_window_ready");
    finish_fn = dlsym(library, "fox_window_finish");
    run_fn = dlsym(library, "fox_window_run");
    destroy_fn = dlsym(library, "fox_window_destroy");
    if (!create_fn || !status_fn || !ready_fn || !finish_fn || !run_fn || !destroy_fn)
        return 0;
    return create_fn(root, stop, browser, log, browser_mode);
}
void fox_window_status(const char *value) { status_fn(value); }
void fox_window_ready(const char *value) { ready_fn(value); }
void fox_window_finish(int value) { finish_fn(value); }
int fox_window_run(void) { return run_fn(); }
void fox_window_destroy(void)
{
    destroy_fn();
    /* GTK/WebKit keep process-wide types and callbacks. Do not dlclose them. */
}
