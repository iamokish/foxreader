/* Browser controller: GTK3 only, with no dependency on WebKit/JavaScriptCore. */
#include "window.h"
#include "window_gtk_close.h"
#include <gtk/gtk.h>

static GtkWidget *window, *state_label, *message_label, *url_label;
static GtkWidget *open_button, *close_button;
static void (*stop_callback)(void);
static void (*browser_callback)(const char *);
static char *app_url, *last_status;
static gboolean closing, finished, owns_backend, browser_opened;
static int result_code;
static FoxGtkCloseConfirmation close_confirmation;

/* The lifecycle worker publishes to one idle source. Only the main thread
 * touches GTK widgets. Removing the source on destroy also clears late work. */
static GMutex pending_lock;
static gboolean active, has_ready, has_finish;
static guint dispatch_id;
static char *pending_message, *pending_url;
static int pending_owner, pending_code;

static void begin_close(void)
{
    if (closing || finished) return;
    closing = TRUE;
    gtk_widget_set_sensitive(open_button, FALSE);
    gtk_widget_set_sensitive(close_button, FALSE);
    gtk_label_set_text(GTK_LABEL(state_label), app_url && owns_backend ? "Stopping" : "Closing");
    gtk_label_set_text(GTK_LABEL(message_label), !app_url ?
        "Cancelling this launch. Waiting for it to finish safely." : owns_backend ?
        "Stopping Fox Reader. Waiting for the backend to finish safely." :
        "Closing this controller. The existing Fox Reader session will keep running.");
    if (stop_callback) stop_callback();
}

static void request_close(void)
{
    if (finished) {
        gtk_main_quit();
        return;
    }
    if (!closing)
        fox_gtk_close_show(&close_confirmation, GTK_WINDOW(window), app_url != NULL, owns_backend, begin_close);
}

static gboolean close_requested(GtkWidget *widget, GdkEvent *event, gpointer data)
{
    (void)widget; (void)event; (void)data;
    request_close();
    return TRUE;
}

static void close_clicked(GtkButton *button, gpointer data)
{
    (void)button; (void)data;
    request_close();
}

static void open_clicked(GtkButton *button, gpointer data)
{
    (void)button; (void)data;
    if (!closing && !finished && app_url && browser_callback) browser_callback(app_url);
}

static gboolean dispatch_on_main(gpointer unused)
{
    char *message, *url;
    gboolean ready, finish;
    int owned, code;
    (void)unused;
    g_mutex_lock(&pending_lock);
    dispatch_id = 0;
    message = pending_message;
    url = pending_url;
    ready = has_ready;
    finish = has_finish;
    owned = pending_owner;
    code = pending_code;
    pending_message = pending_url = NULL;
    has_ready = has_finish = FALSE;
    g_mutex_unlock(&pending_lock);

    if (!closing && !finished) {
        if (ready && !finish) {
            g_clear_pointer(&last_status, g_free);
            g_free(app_url);
            app_url = g_strdup(url);
            owns_backend = owned != 0;
            fox_gtk_close_update(&close_confirmation, TRUE, owns_backend);
            gtk_label_set_text(GTK_LABEL(state_label), "Running");
            gtk_label_set_text(GTK_LABEL(message_label), owns_backend ?
                "Fox Reader is running in your browser. Closing the browser leaves this controller open.\n"
                "Quit Fox Reader or close this window when you finish reading." :
                "Connected to an existing Fox Reader session.\n"
                "Closing this controller leaves the original session running.");
            gtk_label_set_text(GTK_LABEL(url_label), app_url);
            gtk_widget_show(url_label);
            gtk_widget_set_sensitive(open_button, TRUE);
            gtk_button_set_label(GTK_BUTTON(close_button), owns_backend ? "Quit Fox Reader" : "Close This Window");
            if (!browser_opened && browser_callback) {
                browser_opened = TRUE;
                browser_callback(app_url);
            }
        }
        if (message) {
            g_free(last_status);
            last_status = g_strdup(message);
            gtk_label_set_text(GTK_LABEL(message_label), message);
        }
    }
    if (finish && !finished) {
        finished = TRUE;
        fox_gtk_close_dismiss(&close_confirmation);
        result_code = code;
        gtk_widget_set_sensitive(open_button, FALSE);
        if (closing || code == 0) gtk_main_quit();
        else {
            gtk_label_set_text(GTK_LABEL(state_label), "Error");
            if (!last_status) gtk_label_set_text(GTK_LABEL(message_label),
                "Fox Reader could not continue. See logs/launcher.log and logs/backend.log for details.");
            gtk_button_set_label(GTK_BUTTON(close_button), "Close");
            gtk_widget_set_sensitive(close_button, TRUE);
        }
    }
    g_free(message);
    g_free(url);
    return G_SOURCE_REMOVE;
}

static void schedule_dispatch(void)
{
    /* Called with pending_lock held, including while the source ID is set. */
    if (!dispatch_id) dispatch_id = g_idle_add(dispatch_on_main, NULL);
}

int fox_window_create(const char *root, void (*stop)(void),
                      void (*browser)(const char *), void (*log)(const char *), int browser_mode)
{
    GtkWidget *box, *buttons;
    char *icon;
    (void)log; (void)browser_mode;
    g_set_prgname("fox-reader");
    g_set_application_name("Fox Reader");
    gdk_set_program_class("FoxReader");
    if (!gtk_init_check(NULL, NULL)) return 0;
    stop_callback = stop;
    browser_callback = browser;
    closing = finished = owns_backend = browser_opened = FALSE;
    result_code = 1;
    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_widget_set_name(window, "fox-browser-controller");
    gtk_window_set_title(GTK_WINDOW(window), "Fox Reader Control");
    gtk_window_set_default_size(GTK_WINDOW(window), 480, 230);
    gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
    icon = g_build_filename(root, "icons", "fox-reader.png", NULL);
    gtk_window_set_icon_from_file(GTK_WINDOW(window), icon, NULL);
    g_free(icon);
    g_signal_connect(window, "delete-event", G_CALLBACK(close_requested), NULL);

    box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 14);
    gtk_container_set_border_width(GTK_CONTAINER(box), 24);
    gtk_container_add(GTK_CONTAINER(window), box);
    state_label = gtk_label_new("Starting");
    gtk_widget_set_name(state_label, "fox-browser-state");
    gtk_label_set_xalign(GTK_LABEL(state_label), 0);
    gtk_box_pack_start(GTK_BOX(box), state_label, FALSE, FALSE, 0);
    message_label = gtk_label_new("Starting Fox Reader. Your browser will open when it is ready.");
    gtk_widget_set_name(message_label, "fox-browser-message");
    gtk_label_set_xalign(GTK_LABEL(message_label), 0);
    gtk_label_set_line_wrap(GTK_LABEL(message_label), TRUE);
    gtk_label_set_max_width_chars(GTK_LABEL(message_label), 64);
    gtk_box_pack_start(GTK_BOX(box), message_label, TRUE, TRUE, 0);
    url_label = gtk_label_new("");
    gtk_widget_set_name(url_label, "fox-browser-url");
    gtk_label_set_xalign(GTK_LABEL(url_label), 0);
    gtk_label_set_selectable(GTK_LABEL(url_label), TRUE);
    gtk_label_set_line_wrap(GTK_LABEL(url_label), TRUE);
    gtk_label_set_line_wrap_mode(GTK_LABEL(url_label), PANGO_WRAP_CHAR);
    gtk_label_set_max_width_chars(GTK_LABEL(url_label), 64);
    gtk_widget_set_no_show_all(url_label, TRUE);
    gtk_box_pack_start(GTK_BOX(box), url_label, FALSE, FALSE, 0);
    buttons = gtk_button_box_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_button_box_set_layout(GTK_BUTTON_BOX(buttons), GTK_BUTTONBOX_END);
    gtk_box_set_spacing(GTK_BOX(buttons), 8);
    gtk_box_pack_end(GTK_BOX(box), buttons, FALSE, FALSE, 0);
    open_button = gtk_button_new_with_label("Open Reader");
    gtk_widget_set_name(open_button, "fox-browser-open");
    gtk_widget_set_sensitive(open_button, FALSE);
    g_signal_connect(open_button, "clicked", G_CALLBACK(open_clicked), NULL);
    gtk_container_add(GTK_CONTAINER(buttons), open_button);
    close_button = gtk_button_new_with_label("Cancel Startup");
    gtk_widget_set_name(close_button, "fox-browser-close");
    g_signal_connect(close_button, "clicked", G_CALLBACK(close_clicked), NULL);
    gtk_container_add(GTK_CONTAINER(buttons), close_button);
    g_mutex_lock(&pending_lock);
    active = TRUE;
    g_mutex_unlock(&pending_lock);
    gtk_widget_show_all(window);
    return 1;
}

void fox_window_status(const char *message)
{
    g_mutex_lock(&pending_lock);
    if (active) {
        g_free(pending_message);
        pending_message = g_strdup(message ? message : "");
        schedule_dispatch();
    }
    g_mutex_unlock(&pending_lock);
}

void fox_window_ready(const char *url, int owned)
{
    g_mutex_lock(&pending_lock);
    if (active) {
        /* Readiness replaces earlier startup text. Messages arriving after
         * this update (for example a backend error) must still be displayed. */
        g_clear_pointer(&pending_message, g_free);
        g_free(pending_url);
        pending_url = g_strdup(url ? url : "");
        pending_owner = owned;
        has_ready = TRUE;
        schedule_dispatch();
    }
    g_mutex_unlock(&pending_lock);
}

void fox_window_finish(int code)
{
    g_mutex_lock(&pending_lock);
    if (active) {
        pending_code = code;
        has_finish = TRUE;
        schedule_dispatch();
    }
    g_mutex_unlock(&pending_lock);
}

int fox_window_run(void) { gtk_main(); return result_code; }

void fox_window_destroy(void)
{
    guint source;
    g_mutex_lock(&pending_lock);
    active = FALSE;
    source = dispatch_id;
    dispatch_id = 0;
    g_clear_pointer(&pending_message, g_free);
    g_clear_pointer(&pending_url, g_free);
    has_ready = has_finish = FALSE;
    g_mutex_unlock(&pending_lock);
    if (source) g_source_remove(source);
    finished = TRUE;
    fox_gtk_close_dismiss(&close_confirmation);
    if (window) gtk_widget_destroy(window);
    window = NULL;
    g_clear_pointer(&app_url, g_free);
    g_clear_pointer(&last_status, g_free);
}
