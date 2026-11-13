#ifndef FOX_READER_WINDOW_GTK_CLOSE_H
#define FOX_READER_WINDOW_GTK_CLOSE_H

#include <gtk/gtk.h>

/* GTK-only so the browser controller keeps working without WebKit. */
typedef struct {
    GtkWidget *dialog;
    void (*accept)(void);
} FoxGtkCloseConfirmation;

static inline void fox_gtk_close_dismiss(FoxGtkCloseConfirmation *confirmation)
{
    GtkWidget *dialog = confirmation->dialog;
    confirmation->dialog = NULL;
    if (dialog) gtk_widget_destroy(dialog);
}

static void fox_gtk_close_destroyed(GtkWidget *dialog, gpointer data)
{
    FoxGtkCloseConfirmation *confirmation = data;
    if (confirmation->dialog == dialog) confirmation->dialog = NULL;
}

static void fox_gtk_close_response(GtkDialog *dialog, gint response, gpointer data)
{
    FoxGtkCloseConfirmation *confirmation = data;
    void (*accept)(void) = confirmation->accept;
    if (confirmation->dialog != GTK_WIDGET(dialog)) return;
    fox_gtk_close_dismiss(confirmation);
    if (response == GTK_RESPONSE_YES && accept) accept();
}

static inline void fox_gtk_close_update(FoxGtkCloseConfirmation *confirmation,
                                       gboolean ready, gboolean owns_backend)
{
    if (!confirmation->dialog) return;
    gtk_message_dialog_format_secondary_text(GTK_MESSAGE_DIALOG(confirmation->dialog), "%s",
        !ready ? "Startup will be cancelled and this window will close." : owns_backend ?
        "Fox Reader will stop and this window will close." :
        "Only this window will close. The existing Fox Reader session will keep running.");
}

static inline void fox_gtk_close_show(FoxGtkCloseConfirmation *confirmation, GtkWindow *parent,
                                     gboolean ready, gboolean owns_backend, void (*accept)(void))
{
    if (confirmation->dialog) {
        gtk_window_present(GTK_WINDOW(confirmation->dialog));
        return;
    }
    confirmation->accept = accept;
    confirmation->dialog = gtk_message_dialog_new(parent,
        GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
        GTK_MESSAGE_QUESTION, GTK_BUTTONS_YES_NO, "%s", "Close Fox Reader?");
    gtk_widget_set_name(confirmation->dialog, "fox-close-confirmation");
    gtk_window_set_title(GTK_WINDOW(confirmation->dialog), "Close Fox Reader?");
    gtk_window_set_position(GTK_WINDOW(confirmation->dialog), GTK_WIN_POS_CENTER_ON_PARENT);
    fox_gtk_close_update(confirmation, ready, owns_backend);
    gtk_dialog_set_default_response(GTK_DIALOG(confirmation->dialog), GTK_RESPONSE_NO);
    g_signal_connect(confirmation->dialog, "response", G_CALLBACK(fox_gtk_close_response), confirmation);
    g_signal_connect(confirmation->dialog, "destroy", G_CALLBACK(fox_gtk_close_destroyed), confirmation);
    /* GtkDialog maps Escape/window dismissal to a non-Yes response. No nested loop. */
    gtk_widget_show(confirmation->dialog);
    gtk_widget_grab_focus(gtk_dialog_get_widget_for_response(GTK_DIALOG(confirmation->dialog), GTK_RESPONSE_NO));
    gtk_window_present(GTK_WINDOW(confirmation->dialog));
}

#endif
