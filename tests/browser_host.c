/* GTK-only controller fixture: inspect actual widgets and invoke their normal
 * signals. Browser callbacks record evidence instead of opening a real browser. */
#include "../packaging/launcher/window.h"
#include <gtk/gtk.h>
#include <glib/gstdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *app_url;

static void record(const char *name, const char *value)
{
    FILE *file = fopen(name, "ab");
    if (!file) return;
    fprintf(file, "%s\n", value);
    fclose(file);
}

static void stop_requested(void)
{
    record("stop-requested", "stop");
    /* Defer completion to let tests verify the stopping state and late events. */
}

static void browser_requested(const char *url) { record("browser-open", url); }
static void log_message(const char *message) { record("window.log", message); }

static int command_pending(const char *name)
{
    FILE *file = fopen(name, "rb");
    if (!file) return 0;
    fclose(file);
    remove(name);
    return 1;
}

#include "close_confirmation.h"

static GtkWidget *find_named(GtkWidget *widget, const char *name)
{
    GList *children, *item;
    GtkWidget *found = NULL;
    if (g_strcmp0(gtk_widget_get_name(widget), name) == 0) return widget;
    if (!GTK_IS_CONTAINER(widget)) return NULL;
    children = gtk_container_get_children(GTK_CONTAINER(widget));
    for (item = children; item && !found; item = item->next)
        found = find_named(GTK_WIDGET(item->data), name);
    g_list_free(children);
    return found;
}

static GtkWidget *controller(void)
{
    GList *windows = gtk_window_list_toplevels(), *item;
    GtkWidget *found = NULL;
    for (item = windows; item && !found; item = item->next)
        found = find_named(GTK_WIDGET(item->data), "fox-browser-controller");
    g_list_free(windows);
    return found;
}

static void snapshot(GtkWidget *window)
{
    static const char *names[] = {
        "fox-browser-state", "fox-browser-message", "fox-browser-url",
        "fox-browser-open", "fox-browser-close"
    };
    FILE *file = fopen("snapshot.tmp", "wb");
    size_t i;
    if (!file) return;
    fprintf(file, "title=%s\nvisible=%d\n", gtk_window_get_title(GTK_WINDOW(window)),
            gtk_widget_get_visible(window));
    for (i = 0; i < sizeof names / sizeof names[0]; ++i) {
        GtkWidget *widget = find_named(window, names[i]);
        const char *text = NULL;
        if (!widget) {
            fprintf(file, "%s=missing\n", names[i]);
            continue;
        }
        if (GTK_IS_LABEL(widget)) text = gtk_label_get_text(GTK_LABEL(widget));
        else if (GTK_IS_BUTTON(widget)) text = gtk_button_get_label(GTK_BUTTON(widget));
        fprintf(file, "%s=%s\n%s-sensitive=%d\n", names[i], text ? text : "",
                names[i], gtk_widget_get_sensitive(widget));
    }
    fclose(file);
    g_rename("snapshot.tmp", "snapshot.txt");
}

static void click(GtkWidget *window, const char *name)
{
    GtkWidget *button = find_named(window, name);
    if (button && GTK_IS_BUTTON(button) && gtk_widget_get_sensitive(button))
        gtk_button_clicked(GTK_BUTTON(button));
}

static gboolean control_tick(gpointer unused)
{
    GtkWidget *window = controller();
    (void)unused;
    if (command_pending("ready-owned")) fox_window_ready(app_url, 1);
    if (command_pending("ready-attached")) fox_window_ready(app_url, 0);
    if (command_pending("failure")) {
        fox_window_status("Fixture backend failed.");
        fox_window_finish(1);
    }
    if (command_pending("finish")) fox_window_finish(0);
    if (!window) return G_SOURCE_CONTINUE;
    if (command_pending("open-reader")) click(window, "fox-browser-open");
    if (command_pending("quit-view")) click(window, "fox-browser-close");
    if (command_pending("close-view")) gtk_window_close(GTK_WINDOW(window));
    if (command_pending("inspect")) snapshot(window);
    confirmation_tick();
    return G_SOURCE_CONTINUE;
}

int main(void)
{
    char root[4096];
    const char *mode = getenv("FOX_TEST_BROWSER_MODE");
    guint timer;
    int result;
    if (!getcwd(root, sizeof root)) return 2;
    app_url = getenv("FOX_TEST_URL");
    if (!app_url) return 2;
    if (!fox_window_create(root, stop_requested, browser_requested, log_message,
                           mode && strcmp(mode, "1") == 0)) {
        record("unavailable", "No controller available");
        return 77;
    }
    record("created", "created");
    timer = g_timeout_add(20, control_tick, NULL);
    result = fox_window_run();
    g_source_remove(timer);
    /* Deliberately queue a late message. Destruction must safely discard it. */
    fox_window_status("Late message after the event loop stopped.");
    fox_window_destroy();
    while (g_main_context_iteration(NULL, FALSE)) { }
    record("window-exited", "exited");
    return result;
}
