#include "ui_state.h"
#include <gtk/gtk.h>
#include <csignal>
#include <memory>
#include <iostream>

namespace {
std::unique_ptr<fox_updater::UiState> state;
GtkWidget *title_label, *message_label, *progress_bar, *close_button;
int exit_code = 1;

gboolean refresh(gpointer) {
    const auto view = state->snapshot();
    gtk_label_set_text(GTK_LABEL(title_label), fox_updater::stage_title(view.progress.stage).c_str());
    gtk_label_set_text(GTK_LABEL(message_label), view.progress.message.c_str());
    gtk_widget_set_sensitive(close_button, view.finished || view.progress.cancellable);
    gtk_button_set_label(GTK_BUTTON(close_button), view.finished ? "Close" : "Cancel");
    if (!view.finished && !view.progress.total) gtk_progress_bar_pulse(GTK_PROGRESS_BAR(progress_bar));
    else {
        const double fraction = view.finished && view.successful ? 1.0 : view.progress.total ?
            static_cast<double>(view.progress.done) / static_cast<double>(view.progress.total) : 0.0;
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(progress_bar), fraction);
    }
    exit_code = view.exit_code;
    return G_SOURCE_CONTINUE;
}

gboolean close_requested(GtkWidget*, GdkEvent*, gpointer) {
    if (state->request_close()) gtk_main_quit();
    else refresh(nullptr);
    return TRUE;
}

void close_clicked(GtkButton*, gpointer) { close_requested(nullptr, nullptr, nullptr); }
}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    std::string error;
    auto handoff = fox_updater::Handoff::receive(error);
    if (!handoff) { std::cerr << error << '\n'; return 1; }
    if (handoff && handoff->silent()) return handoff->run({}, {}).success ? 0 : 1;
    if (!gtk_init_check(&argc, &argv)) return 1;
    state = std::make_unique<fox_updater::UiState>(handoff, error);
    GtkWidget* window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Fox Reader Update");
    gtk_window_set_default_size(GTK_WINDOW(window), 560, 290);
    gtk_window_set_resizable(GTK_WINDOW(window), FALSE);
    gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
    gtk_container_set_border_width(GTK_CONTAINER(window), 28);
    GtkWidget* column = gtk_box_new(GTK_ORIENTATION_VERTICAL, 18);
    gtk_container_add(GTK_CONTAINER(window), column);
    title_label = gtk_label_new("Preparing update");
    gtk_label_set_xalign(GTK_LABEL(title_label), 0.0F);
    auto attributes = pango_attr_list_new();
    pango_attr_list_insert(attributes, pango_attr_weight_new(PANGO_WEIGHT_BOLD));
    pango_attr_list_insert(attributes, pango_attr_scale_new(1.4));
    gtk_label_set_attributes(GTK_LABEL(title_label), attributes);
    pango_attr_list_unref(attributes);
    gtk_box_pack_start(GTK_BOX(column), title_label, FALSE, FALSE, 0);
    message_label = gtk_label_new("");
    gtk_label_set_xalign(GTK_LABEL(message_label), 0.0F);
    gtk_label_set_yalign(GTK_LABEL(message_label), 0.0F);
    gtk_label_set_line_wrap(GTK_LABEL(message_label), TRUE);
    gtk_label_set_line_wrap_mode(GTK_LABEL(message_label), PANGO_WRAP_WORD_CHAR);
    gtk_label_set_max_width_chars(GTK_LABEL(message_label), 65);
    gtk_label_set_selectable(GTK_LABEL(message_label), TRUE);
    gtk_box_pack_start(GTK_BOX(column), message_label, TRUE, TRUE, 0);
    progress_bar = gtk_progress_bar_new();
    gtk_box_pack_start(GTK_BOX(column), progress_bar, FALSE, FALSE, 0);
    GtkWidget* buttons = gtk_button_box_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_button_box_set_layout(GTK_BUTTON_BOX(buttons), GTK_BUTTONBOX_END);
    close_button = gtk_button_new_with_label("Cancel");
    gtk_container_add(GTK_CONTAINER(buttons), close_button);
    gtk_box_pack_start(GTK_BOX(column), buttons, FALSE, FALSE, 0);
    g_signal_connect(window, "delete-event", G_CALLBACK(close_requested), nullptr);
    g_signal_connect(close_button, "clicked", G_CALLBACK(close_clicked), nullptr);
    gtk_widget_show_all(window);
    refresh(nullptr);
    const guint timer = g_timeout_add(100, refresh, nullptr);
    state->start();
    gtk_main();
    g_source_remove(timer);
    gtk_widget_destroy(window);
    state.reset();
    return exit_code;
}
