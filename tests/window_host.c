/* Real native viewer fixture. Browser/log callbacks only write test evidence;
 * there is no backend process, model load, or external browser launch. */
#include "../packaging/launcher/window.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif
#ifdef __linux__
#include <gtk/gtk.h>
#include <webkit2/webkit2.h>
#endif

static void record(const char *name, const char *value)
{
    FILE *file = fopen(name, "ab");
    if (!file) return;
    fputs(value, file);
    fputc('\n', file);
    fclose(file);
}

static void stop_requested(void)
{
    record("stop-requested", "stop");
    fox_window_finish(0);
}

static void browser_requested(const char *url)
{
    record("browser-open", url);
}

static void log_message(const char *message)
{
    record("log-calls", "log");
    record("window.log", message);
}

#if defined(_WIN32) || defined(__linux__)
static int command_pending(const char *name)
{
    FILE *file = fopen(name, "rb");
    if (!file) return 0;
    fclose(file);
    remove(name);
    return 1;
}
#endif

#ifdef _WIN32
static BOOL CALLBACK close_window(HWND hwnd, LPARAM unused)
{
    DWORD process;
    wchar_t name[128];
    (void)unused;
    GetWindowThreadProcessId(hwnd, &process);
    GetClassNameW(hwnd, name, 128);
    if (process == GetCurrentProcessId() && wcscmp(name, L"FoxReaderWindow") == 0) {
        PostMessageW(hwnd, WM_CLOSE, 0, 0);
        return FALSE;
    }
    return TRUE;
}

static void CALLBACK control_tick(HWND hwnd, UINT message, UINT_PTR timer, DWORD time)
{
    (void)hwnd; (void)message; (void)timer; (void)time;
    if (command_pending("close-view")) EnumWindows(close_window, 0);
}
#endif

#ifdef __linux__
static GtkWidget *find_webview(GtkWidget *widget)
{
    GList *children, *entry;
    GtkWidget *result = NULL;
    if (WEBKIT_IS_WEB_VIEW(widget)) return widget;
    if (!GTK_IS_CONTAINER(widget)) return NULL;
    children = gtk_container_get_children(GTK_CONTAINER(widget));
    for (entry = children; entry && !result; entry = entry->next)
        result = find_webview(GTK_WIDGET(entry->data));
    g_list_free(children);
    return result;
}

static gboolean control_tick(gpointer unused)
{
    int close = command_pending("close-view");
    int crash = command_pending("crash-view");
    GList *windows, *entry;
    (void)unused;
    if (!close && !crash) return G_SOURCE_CONTINUE;
    windows = gtk_window_list_toplevels();
    for (entry = windows; entry; entry = entry->next) {
        GtkWindow *window = GTK_WINDOW(entry->data);
        if (g_strcmp0(gtk_window_get_title(window), "Fox Reader") != 0) continue;
        if (close) gtk_window_close(window);
        else {
            GtkWidget *view = find_webview(GTK_WIDGET(window));
            if (view) {
                record("crash-requested", "crash");
                webkit_web_view_terminate_web_process(WEBKIT_WEB_VIEW(view));
            }
        }
        break;
    }
    g_list_free(windows);
    return G_SOURCE_CONTINUE;
}
#endif

static int run_host(void)
{
    char root[4096];
    const char *url = getenv("FOX_TEST_URL");
    int result;
#ifdef _WIN32
    wchar_t wide_root[4096];
    UINT_PTR timer;
    DWORD length = GetCurrentDirectoryW(4096, wide_root);
    if (!length || length >= 4096 ||
        !WideCharToMultiByte(CP_UTF8, 0, wide_root, -1, root, sizeof root, NULL, NULL)) return 2;
#else
#ifdef __linux__
    guint timer;
#endif
    if (!getcwd(root, sizeof root)) return 2;
#endif
    if (!url) return 2;
    if (!fox_window_create(root, stop_requested, browser_requested, log_message, 0)) {
        record("unavailable", "No graphical native viewer is available");
        return 77;
    }
#ifdef _WIN32
    timer = SetTimer(NULL, 0, 20, control_tick);
#elif defined(__linux__)
    timer = g_timeout_add(20, control_tick, NULL);
#endif
    fox_window_ready(url);
    result = fox_window_run();
#ifdef _WIN32
    if (timer) KillTimer(NULL, timer);
#elif defined(__linux__)
    if (timer) g_source_remove(timer);
#endif
    fox_window_destroy();
    record("window-exited", "exited");
    return result;
}

#ifdef _WIN32
int WINAPI WinMain(HINSTANCE instance, HINSTANCE previous, LPSTR command, int show)
{
    (void)instance; (void)previous; (void)command; (void)show;
    return run_host();
}
#else
int main(void) { return run_host(); }
#endif
