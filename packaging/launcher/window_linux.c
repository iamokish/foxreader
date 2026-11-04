#include "window.h"
#include "window_recovery.h"
#include <gtk/gtk.h>
#include <webkit2/webkit2.h>
#include <glib/gstdio.h>
#include <string.h>
#include <unistd.h>

static GtkWidget *window, *label, *stack, *view;
static void (*stop_callback)(void);
static void (*browser_callback)(const char *);
static void (*log_callback)(const char *);
static gboolean browser_mode, closing, finished, fullscreen, navigation_failed;
static gboolean fallback_active, browser_opened;
static guint fullscreen_key;
static guint startup_timer;
static int result_code;
static char *app_url;
static const char *fallback_reason;
static GList *downloads;
static FoxWindowRecovery recovery;

typedef struct {
    char *target, *temporary;
    GtkWidget *chooser;
    gboolean failed, done;
} ExportDownload;

static void cancel_downloads(void);

static void cancel_startup_timer(void)
{
    if (startup_timer) {
        g_source_remove(startup_timer);
        startup_timer = 0;
    }
}
static void show_browser_status(const char *backend_message)
{
    const char *status = backend_message ? backend_message :
        (app_url ? "Fox Reader is running in your browser.\nKeep this window open while reading." :
                   "Waiting for the backend. Your browser will open when it is ready.");
    char *message = fallback_reason ? g_strdup_printf("%s\n\n%s", fallback_reason, status) : g_strdup(status);
    gtk_label_set_text(GTK_LABEL(label), message);
    g_free(message);
    gtk_stack_set_visible_child(GTK_STACK(stack), label);
    if (app_url && !browser_opened && browser_callback) {
        browser_opened = TRUE;
        browser_callback(app_url);
    }
}
static void recover_to_browser(const char *reason)
{
    if (closing || finished || !fox_recovery_fallback(&recovery)) return;
    /* Latch before stopping WebKit: cancellation can deliver nested callbacks. */
    fallback_active = TRUE;
    navigation_failed = TRUE;
    fallback_reason = reason;
    cancel_startup_timer();
    if (log_callback) log_callback(reason);
    if (view) webkit_web_view_stop_loading(WEBKIT_WEB_VIEW(view));
    cancel_downloads();
    show_browser_status(NULL);
}
static gboolean startup_timed_out(gpointer data)
{
    (void)data;
    startup_timer = 0;
    if (!closing && !finished && fox_recovery_timed_out(&recovery,
            (uint64_t)(g_get_monotonic_time() / 1000)))
        recover_to_browser("The app window did not finish loading within 30 seconds. Continuing in your browser.");
    return G_SOURCE_REMOVE;
}
static void web_process_terminated(WebKitWebView *web, WebKitWebProcessTerminationReason reason, gpointer data)
{
    (void)web; (void)data;
    const char *message = "The app's rendering process stopped unexpectedly. Continuing in your browser.";
    if (reason == WEBKIT_WEB_PROCESS_CRASHED)
        message = "The app's rendering process crashed. Continuing in your browser.";
    else if (reason == WEBKIT_WEB_PROCESS_EXCEEDED_MEMORY_LIMIT)
        message = "The app's rendering process exceeded its memory limit. Continuing in your browser.";
    recover_to_browser(message);
}

static void toggle_fullscreen(void)
{
    if (closing) return;
    if (fullscreen) gtk_window_unfullscreen(GTK_WINDOW(window));
    else gtk_window_fullscreen(GTK_WINDOW(window));
}
static gboolean window_state(GtkWidget *widget, GdkEventWindowState *event, gpointer data)
{
    (void)widget; (void)data;
    fullscreen = (event->new_window_state & GDK_WINDOW_STATE_FULLSCREEN) != 0;
    return FALSE;
}
static gboolean key_pressed(GtkWidget *widget, GdkEventKey *event, gpointer data)
{
    guint forbidden = GDK_CONTROL_MASK | GDK_SHIFT_MASK | GDK_SUPER_MASK | GDK_META_MASK;
    (void)widget; (void)data;
    if (!(event->state & forbidden) &&
        (event->keyval == GDK_KEY_F11 ||
         ((event->state & GDK_MOD1_MASK) &&
          (event->keyval == GDK_KEY_Return || event->keyval == GDK_KEY_KP_Enter)))) {
        if (!fullscreen_key) toggle_fullscreen();
        fullscreen_key = event->keyval;
        return TRUE;
    }
    if ((event->state & GDK_CONTROL_MASK) &&
        (event->keyval == GDK_KEY_plus || event->keyval == GDK_KEY_minus ||
         event->keyval == GDK_KEY_equal || event->keyval == GDK_KEY_0)) return TRUE;
    return FALSE;
}
static gboolean key_released(GtkWidget *widget, GdkEventKey *event, gpointer data)
{
    (void)widget; (void)data;
    if (event->keyval != fullscreen_key) return FALSE;
    fullscreen_key = 0;
    return TRUE;
}
static gboolean focus_left(GtkWidget *widget, GdkEventFocus *event, gpointer data)
{
    (void)widget; (void)event; (void)data;
    fullscreen_key = 0;
    return FALSE;
}
static void zoom_changed(WebKitWebView *web, GParamSpec *property, gpointer data)
{
    (void)property; (void)data;
    if (webkit_web_view_get_zoom_level(web) != 1.0)
        webkit_web_view_set_zoom_level(web, 1.0);
}
static void fullscreen_message(WebKitUserContentManager *manager, WebKitJavascriptResult *value, gpointer data)
{
    (void)manager; (void)value; (void)data;
    if (!fallback_active && !closing && !finished) toggle_fullscreen();
}
static gboolean close_requested(GtkWidget *widget, GdkEvent *event, gpointer data)
{
    (void)widget; (void)event; (void)data;
    if (finished) { gtk_main_quit(); return TRUE; }
    if (!closing) {
        closing = TRUE;
        fox_recovery_stop(&recovery);
        cancel_startup_timer();
        gtk_label_set_text(GTK_LABEL(label), "Closing Fox Reader...\nWaiting for the backend to finish safely.");
        gtk_stack_set_visible_child(GTK_STACK(stack), label);
        cancel_downloads();
        stop_callback();
    }
    return TRUE;
}
static gboolean context_menu(WebKitWebView *web, WebKitContextMenu *menu, GdkEvent *event,
                             WebKitHitTestResult *hit, gpointer data)
{
    (void)web; (void)menu; (void)event; (void)hit; (void)data;
    return TRUE;
}
static gboolean same_origin(const char *uri)
{
    GUri *base, *target;
    gboolean same = FALSE;
    int base_port, target_port;
    if (!app_url || !uri) return FALSE;
    base = g_uri_parse(app_url, G_URI_FLAGS_NONE, NULL);
    target = g_uri_parse(uri, G_URI_FLAGS_NONE, NULL);
    if (base && target && g_uri_get_host(base) && g_uri_get_host(target) &&
        g_uri_get_scheme(target) && !g_uri_get_userinfo(target)) {
        base_port = g_uri_get_port(base);
        target_port = g_uri_get_port(target);
        if (base_port < 0) base_port = 80;
        if (target_port < 0) target_port = 80;
        same = g_ascii_strcasecmp(g_uri_get_scheme(base), g_uri_get_scheme(target)) == 0 &&
               g_ascii_strcasecmp(g_uri_get_host(base), g_uri_get_host(target)) == 0 &&
               base_port == target_port;
    }
    if (base) g_uri_unref(base);
    if (target) g_uri_unref(target);
    return same;
}
static gboolean local_blob(const char *uri)
{
    return uri && g_str_has_prefix(uri, "blob:") && same_origin(uri + 5);
}
static void export_error(const char *message)
{
    GtkWidget *dialog;
    if (closing || finished || fallback_active) return;
    dialog = gtk_message_dialog_new(GTK_WINDOW(window), GTK_DIALOG_DESTROY_WITH_PARENT,
        GTK_MESSAGE_ERROR, GTK_BUTTONS_CLOSE, "%s", message);
    gtk_window_set_title(GTK_WINDOW(dialog), "Fox Reader export");
    g_signal_connect_swapped(dialog, "response", G_CALLBACK(gtk_widget_destroy), dialog);
    gtk_widget_show(dialog);
}
static void export_free(gpointer value)
{
    ExportDownload *item = value;
    if (item->temporary) g_unlink(item->temporary);
    g_free(item->temporary);
    g_free(item->target);
    g_free(item);
}
static gboolean export_destination(WebKitDownload *download, const char *suggested, gpointer value)
{
    ExportDownload *item = value;
    GtkWidget *chooser;
    char *name, *directory, *uri;
    int response, fd;
    /* The modal dialog runs a GTK loop, so shutdown/download callbacks remain
     * live. Keep its signal data alive if cancellation finishes in that loop. */
    g_object_ref(download);
    chooser = gtk_file_chooser_dialog_new("Save Fox Reader export", GTK_WINDOW(window),
        GTK_FILE_CHOOSER_ACTION_SAVE, "Cancel", GTK_RESPONSE_CANCEL, "Save", GTK_RESPONSE_ACCEPT, NULL);
    item->chooser = chooser;
    gtk_window_set_destroy_with_parent(GTK_WINDOW(chooser), TRUE);
    gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(chooser), TRUE);
    gtk_file_chooser_set_local_only(GTK_FILE_CHOOSER(chooser), TRUE);
    name = g_path_get_basename(suggested && *suggested ? suggested : "fox-reader-export");
    gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(chooser),
        strcmp(name, ".") && strcmp(name, "..") ? name : "fox-reader-export");
    g_free(name);
    response = gtk_dialog_run(GTK_DIALOG(chooser));
    if (response == GTK_RESPONSE_ACCEPT && !closing && !finished && !fallback_active && !item->failed && !item->done)
        item->target = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(chooser));
    item->chooser = NULL;
    gtk_widget_destroy(chooser);
    if (!item->target) {
        webkit_download_cancel(download);
        g_object_unref(download);
        return TRUE;
    }
    /* Do not write over an existing export until all bytes arrive successfully.
     * The unique temporary file is beside the selected destination, so rename
     * publishes it atomically on the same filesystem. */
    directory = g_path_get_dirname(item->target);
    item->temporary = g_build_filename(directory, ".fox-reader-XXXXXX", NULL);
    g_free(directory);
    fd = g_mkstemp(item->temporary);
    if (fd < 0) {
        g_clear_pointer(&item->temporary, g_free);
        export_error("The export folder is not writable. Choose another location.");
        webkit_download_cancel(download);
        g_object_unref(download);
        return TRUE;
    }
    close(fd);
    uri = g_filename_to_uri(item->temporary, NULL, NULL);
    if (!uri) {
        export_error("The selected export path could not be used.");
        webkit_download_cancel(download);
    } else {
        webkit_download_set_allow_overwrite(download, TRUE);
        webkit_download_set_destination(download, uri);
        g_free(uri);
    }
    g_object_unref(download);
    return TRUE;
}
static void export_failed(WebKitDownload *download, GError *error, gpointer value)
{
    ExportDownload *item = value;
    (void)download;
    item->failed = TRUE;
    if (item->chooser) gtk_dialog_response(GTK_DIALOG(item->chooser), GTK_RESPONSE_CANCEL);
    if (!g_error_matches(error, WEBKIT_DOWNLOAD_ERROR, WEBKIT_DOWNLOAD_ERROR_CANCELLED_BY_USER))
        export_error("The export could not be downloaded. Your previous file has been kept.");
}
static void export_finished(WebKitDownload *download, gpointer value)
{
    ExportDownload *item = value;
    item->done = TRUE;
    if (item->chooser) gtk_dialog_response(GTK_DIALOG(item->chooser), GTK_RESPONSE_CANCEL);
    if (!item->failed && item->temporary && item->target &&
        g_rename(item->temporary, item->target) != 0)
        export_error("The export could not be saved. Your previous file has been kept.");
    if (item->temporary) g_unlink(item->temporary);
    g_clear_pointer(&item->temporary, g_free);
    downloads = g_list_remove(downloads, download);
    g_object_unref(download);
}
static void download_started(WebKitWebContext *context, WebKitDownload *download, gpointer data)
{
    ExportDownload *item;
    const char *uri = webkit_uri_request_get_uri(webkit_download_get_request(download));
    (void)context; (void)data;
    if (closing || finished || fallback_active || (!same_origin(uri) && !local_blob(uri))) {
        webkit_download_cancel(download);
        return;
    }
    item = g_new0(ExportDownload, 1);
    g_object_set_data_full(G_OBJECT(download), "fox-reader-export", item, export_free);
    downloads = g_list_prepend(downloads, g_object_ref(download));
    g_signal_connect(download, "decide-destination", G_CALLBACK(export_destination), item);
    g_signal_connect(download, "failed", G_CALLBACK(export_failed), item);
    g_signal_connect(download, "finished", G_CALLBACK(export_finished), item);
}
static void cancel_downloads(void)
{
    GList *pending = g_list_copy(downloads), *entry;
    for (entry = pending; entry; entry = entry->next) g_object_ref(entry->data);
    for (entry = pending; entry; entry = entry->next) {
        ExportDownload *item = g_object_get_data(G_OBJECT(entry->data), "fox-reader-export");
        item->failed = TRUE;
        if (item->chooser) gtk_dialog_response(GTK_DIALOG(item->chooser), GTK_RESPONSE_CANCEL);
        if (item->temporary) g_unlink(item->temporary);
        webkit_download_cancel(WEBKIT_DOWNLOAD(entry->data));
        g_object_unref(entry->data);
    }
    g_list_free(pending);
}
static void external_url(const char *uri)
{
    if (uri && (g_str_has_prefix(uri, "https://") || g_str_has_prefix(uri, "http://")))
        browser_callback(uri);
}
static gboolean decide_policy(WebKitWebView *web, WebKitPolicyDecision *decision,
                               WebKitPolicyDecisionType type, gpointer data)
{
    (void)web; (void)data;
    if (closing || finished || fallback_active) {
        webkit_policy_decision_ignore(decision);
        return TRUE;
    }
    if (type == WEBKIT_POLICY_DECISION_TYPE_NAVIGATION_ACTION ||
        type == WEBKIT_POLICY_DECISION_TYPE_NEW_WINDOW_ACTION) {
        WebKitNavigationAction *action = webkit_navigation_policy_decision_get_navigation_action(
            WEBKIT_NAVIGATION_POLICY_DECISION(decision));
        const char *uri = webkit_uri_request_get_uri(webkit_navigation_action_get_request(action));
        if (local_blob(uri)) {
            webkit_policy_decision_download(decision);
            return TRUE;
        }
        if (!same_origin(uri) || type == WEBKIT_POLICY_DECISION_TYPE_NEW_WINDOW_ACTION) {
            webkit_policy_decision_ignore(decision);
            if (webkit_navigation_action_is_user_gesture(action)) external_url(uri);
            return TRUE;
        }
    }
    return FALSE;
}
static void load_changed(WebKitWebView *web, WebKitLoadEvent event, gpointer data)
{
    (void)data;
    if (closing || finished || fallback_active) return;
    if (event == WEBKIT_LOAD_STARTED) navigation_failed = FALSE;
    if (event == WEBKIT_LOAD_FINISHED && !navigation_failed && same_origin(webkit_web_view_get_uri(web))) {
        fox_recovery_loaded(&recovery);
        cancel_startup_timer();
        gtk_stack_set_visible_child(GTK_STACK(stack), view);
    }
}
static gboolean load_failed(WebKitWebView *web, WebKitLoadEvent event, const char *uri,
                            GError *error, gpointer data)
{
    (void)web; (void)event; (void)uri; (void)data;
    /* External links deliberately interrupt navigation; an overlapping local
     * navigation is also cancelled by WebKit. Keep the current reader visible. */
    if (closing || finished || fallback_active ||
        g_error_matches(error, WEBKIT_NETWORK_ERROR, WEBKIT_NETWORK_ERROR_CANCELLED) ||
        g_error_matches(error, WEBKIT_POLICY_ERROR, WEBKIT_POLICY_ERROR_FRAME_LOAD_INTERRUPTED_BY_POLICY_CHANGE))
        return TRUE;
    navigation_failed = TRUE;
    cancel_startup_timer();
    gtk_label_set_text(GTK_LABEL(label), "The reader could not load. Close and relaunch with --browser.\nSee logs/backend.log for details.");
    gtk_stack_set_visible_child(GTK_STACK(stack), label);
    return TRUE;
}
int fox_window_create(const char *root, void (*stop)(void), void (*browser)(const char *),
                      void (*log)(const char *), int use_browser)
{
    char *icon;
    WebKitUserContentManager *manager;
    WebKitSettings *settings;
    WebKitUserScript *script;
    g_set_prgname("fox-reader");
    g_set_application_name("Fox Reader");
    gdk_set_program_class("FoxReader");
    if (!gtk_init_check(NULL, NULL)) return 0;
    stop_callback = stop;
    browser_callback = browser;
    log_callback = log;
    browser_mode = use_browser != 0;
    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Fox Reader");
    gtk_window_set_default_size(GTK_WINDOW(window), 1180, 800);
    gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
    icon = g_build_filename(root, "icons", "fox-reader.png", NULL);
    gtk_window_set_icon_from_file(GTK_WINDOW(window), icon, NULL);
    g_free(icon);
    g_signal_connect(window, "delete-event", G_CALLBACK(close_requested), NULL);
    g_signal_connect(window, "key-press-event", G_CALLBACK(key_pressed), NULL);
    g_signal_connect(window, "key-release-event", G_CALLBACK(key_released), NULL);
    g_signal_connect(window, "focus-out-event", G_CALLBACK(focus_left), NULL);
    g_signal_connect(window, "window-state-event", G_CALLBACK(window_state), NULL);
    stack = gtk_stack_new();
    gtk_container_add(GTK_CONTAINER(window), stack);
    label = gtk_label_new("Starting Fox Reader...\nThe backend usually takes 10-15 seconds to become ready.");
    gtk_label_set_line_wrap(GTK_LABEL(label), TRUE);
    gtk_label_set_justify(GTK_LABEL(label), GTK_JUSTIFY_CENTER);
    gtk_widget_set_margin_start(label, 32);
    gtk_widget_set_margin_end(label, 32);
    gtk_stack_add_named(GTK_STACK(stack), label, "status");
    if (!browser_mode) {
        manager = webkit_user_content_manager_new();
        webkit_user_content_manager_register_script_message_handler(manager, "foxFullscreen");
        g_signal_connect(manager, "script-message-received::foxFullscreen", G_CALLBACK(fullscreen_message), NULL);
        script = webkit_user_script_new(
            "window.foxReaderNativeFullscreen=()=>window.webkit.messageHandlers.foxFullscreen.postMessage('toggle');",
            WEBKIT_USER_CONTENT_INJECT_TOP_FRAME, WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_START, NULL, NULL);
        webkit_user_content_manager_add_script(manager, script);
        webkit_user_script_unref(script);
        view = webkit_web_view_new_with_user_content_manager(manager);
        g_object_unref(manager);
        g_signal_connect(webkit_web_view_get_context(WEBKIT_WEB_VIEW(view)), "download-started",
            G_CALLBACK(download_started), NULL);
        settings = webkit_web_view_get_settings(WEBKIT_WEB_VIEW(view));
        webkit_settings_set_enable_developer_extras(settings, FALSE);
        webkit_settings_set_enable_write_console_messages_to_stdout(settings, FALSE);
        webkit_settings_set_enable_page_cache(settings, FALSE);
        g_signal_connect(view, "key-press-event", G_CALLBACK(key_pressed), NULL);
        g_signal_connect(view, "key-release-event", G_CALLBACK(key_released), NULL);
        g_signal_connect(view, "notify::zoom-level", G_CALLBACK(zoom_changed), NULL);
        g_signal_connect(view, "context-menu", G_CALLBACK(context_menu), NULL);
        g_signal_connect(view, "decide-policy", G_CALLBACK(decide_policy), NULL);
        g_signal_connect(view, "load-changed", G_CALLBACK(load_changed), NULL);
        g_signal_connect(view, "load-failed", G_CALLBACK(load_failed), NULL);
        g_signal_connect(view, "web-process-terminated", G_CALLBACK(web_process_terminated), NULL);
        gtk_stack_add_named(GTK_STACK(stack), view, "reader");
    }
    gtk_stack_set_visible_child(GTK_STACK(stack), label);
    gtk_widget_show_all(window);
    return 1;
}
static gboolean status_on_main(gpointer text)
{
    if (!closing && !finished) {
        if (fallback_active) show_browser_status(text);
        else {
            gtk_label_set_text(GTK_LABEL(label), text);
            gtk_stack_set_visible_child(GTK_STACK(stack), label);
        }
    }
    return G_SOURCE_REMOVE;
}
void fox_window_status(const char *message)
{
    g_idle_add_full(G_PRIORITY_DEFAULT, status_on_main, g_strdup(message), g_free);
}
static gboolean ready_on_main(gpointer text)
{
    if (closing || finished) return G_SOURCE_REMOVE;
    g_free(app_url);
    app_url = g_strdup(text);
    if (browser_mode || fallback_active) show_browser_status(NULL);
    else if (fox_recovery_begin_navigation(&recovery, (uint64_t)(g_get_monotonic_time() / 1000))) {
        /* WebKit starts its rendering process on this first load. Backend
         * readiness and later page/model operations have no viewer deadline. */
        startup_timer = g_timeout_add((guint)FOX_RECOVERY_TIMEOUT_MS, startup_timed_out, NULL);
        webkit_web_view_load_uri(WEBKIT_WEB_VIEW(view), app_url);
    }
    return G_SOURCE_REMOVE;
}
void fox_window_ready(const char *url)
{
    g_idle_add_full(G_PRIORITY_DEFAULT, ready_on_main, g_strdup(url), g_free);
}
static gboolean finish_on_main(gpointer code)
{
    finished = TRUE;
    fox_recovery_stop(&recovery);
    cancel_startup_timer();
    result_code = GPOINTER_TO_INT(code);
    cancel_downloads();
    if (closing || !result_code) gtk_main_quit();
    else gtk_stack_set_visible_child(GTK_STACK(stack), label);
    return G_SOURCE_REMOVE;
}
void fox_window_finish(int code) { g_idle_add(finish_on_main, GINT_TO_POINTER(code)); }
int fox_window_run(void) { gtk_main(); return result_code; }
void fox_window_destroy(void)
{
    finished = TRUE;
    fox_recovery_stop(&recovery);
    cancel_startup_timer();
    cancel_downloads();
    if (view)
        g_signal_handlers_disconnect_by_func(webkit_web_view_get_context(WEBKIT_WEB_VIEW(view)),
                                             G_CALLBACK(download_started), NULL);
    gtk_widget_destroy(window);
    g_free(app_url);
    app_url = NULL;
}
