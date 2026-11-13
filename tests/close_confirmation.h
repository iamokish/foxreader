/* Test-only automation of native dialog widgets, shared by both host fixtures.
 * Commands run on the host's UI thread; no production testing hooks are used. */
#ifdef _WIN32
static HWND confirmation_main, confirmation_dialog;
static int confirmation_count;

static BOOL CALLBACK find_confirmation(HWND hwnd, LPARAM unused)
{
    DWORD process;
    wchar_t name[128], title[128];
    (void)unused;
    GetWindowThreadProcessId(hwnd, &process);
    if (process != GetCurrentProcessId()) return TRUE;
    GetClassNameW(hwnd, name, 128);
    GetWindowTextW(hwnd, title, 128);
    if (wcscmp(name, L"FoxReaderWindow") == 0) confirmation_main = hwnd;
    if (wcscmp(name, L"#32770") == 0 && wcscmp(title, L"Close Fox Reader?") == 0 && IsWindowVisible(hwnd)) {
        confirmation_dialog = hwnd;
        ++confirmation_count;
    }
    return TRUE;
}

static void confirmation_tick(void)
{
    HWND dialog;
    confirmation_main = confirmation_dialog = NULL;
    confirmation_count = 0;
    EnumWindows(find_confirmation, 0);
    dialog = confirmation_dialog;
    if (command_pending("inspect-confirmation")) {
        FILE *file = fopen("confirmation.tmp", "wb");
        if (file) {
            fprintf(file, "count=%d\nid=%p\nparent=%d\nmodal=%d\ndefault-no=%d\nyes=%d\nno=%d\nmain-enabled=%d\n",
                confirmation_count, (void *)dialog,
                dialog && GetWindow(dialog, GW_OWNER) == confirmation_main,
                dialog && !IsWindowEnabled(confirmation_main),
                dialog && LOWORD(SendMessageW(dialog, DM_GETDEFID, 0, 0)) == IDNO,
                dialog && IsWindowVisible(GetDlgItem(dialog, IDYES)),
                dialog && IsWindowVisible(GetDlgItem(dialog, IDNO)),
                IsWindowEnabled(confirmation_main));
            fprintf(file, "yes-ownerdraw=%d\nno-ownerdraw=%d\n",
                dialog && (GetWindowLongPtrW(GetDlgItem(dialog, IDYES), GWL_STYLE) & BS_TYPEMASK) == BS_OWNERDRAW,
                dialog && (GetWindowLongPtrW(GetDlgItem(dialog, IDNO), GWL_STYLE) & BS_TYPEMASK) == BS_OWNERDRAW);
            fclose(file);
            remove("confirmation.txt");
            rename("confirmation.tmp", "confirmation.txt");
        }
    }
    if (!dialog) return; /* Retain queued answers until a prompt exists. */
    if (command_pending("confirm-no")) PostMessageW(dialog, WM_COMMAND, IDNO, 0);
    if (command_pending("confirm-yes")) PostMessageW(dialog, WM_COMMAND, IDYES, 0);
    if (command_pending("confirm-escape")) PostMessageW(dialog, WM_KEYDOWN, VK_ESCAPE, 0);
    if (command_pending("confirm-default")) PostMessageW(dialog, WM_KEYDOWN, VK_RETURN, 0);
    if (command_pending("confirm-dismiss")) PostMessageW(dialog, WM_CLOSE, 0, 0);
    if (command_pending("confirm-focus-yes"))
        SendMessageW(dialog, WM_NEXTDLGCTL, (WPARAM)GetDlgItem(dialog, IDYES), TRUE);
    if (command_pending("confirm-yes-default"))
        PostMessageW(GetDlgItem(dialog, IDYES), WM_KEYDOWN, VK_RETURN, 0);
}
#elif defined(__linux__)
static void confirmation_tick(void)
{
    GList *windows = gtk_window_list_toplevels(), *entry;
    GtkWidget *dialog = NULL;
    int count = 0;
    for (entry = windows; entry; entry = entry->next) {
        GtkWidget *widget = GTK_WIDGET(entry->data);
        if (GTK_IS_DIALOG(widget) && gtk_widget_get_visible(widget) &&
            g_strcmp0(gtk_widget_get_name(widget), "fox-close-confirmation") == 0) {
            dialog = widget;
            ++count;
        }
    }
    g_list_free(windows);
    if (command_pending("inspect-confirmation")) {
        FILE *file = fopen("confirmation.tmp", "wb");
        if (file) {
            GtkWindow *parent = dialog ? gtk_window_get_transient_for(GTK_WINDOW(dialog)) : NULL;
            GtkWidget *yes = dialog ? gtk_dialog_get_widget_for_response(GTK_DIALOG(dialog), GTK_RESPONSE_YES) : NULL;
            GtkWidget *no = dialog ? gtk_dialog_get_widget_for_response(GTK_DIALOG(dialog), GTK_RESPONSE_NO) : NULL;
            const char *title = parent ? gtk_window_get_title(parent) : NULL;
            fprintf(file, "count=%d\nid=%p\nparent=%d\nmodal=%d\ndefault-no=%d\nyes=%d\nno=%d\n",
                count, (void *)dialog,
                parent && gtk_widget_get_visible(GTK_WIDGET(parent)) &&
                    (g_strcmp0(title, "Fox Reader") == 0 || g_strcmp0(title, "Fox Reader Control") == 0),
                dialog && gtk_window_get_modal(GTK_WINDOW(dialog)),
                no && gtk_window_get_default_widget(GTK_WINDOW(dialog)) == no,
                yes && gtk_widget_get_visible(yes), no && gtk_widget_get_visible(no));
            fclose(file);
            rename("confirmation.tmp", "confirmation.txt");
        }
    }
    if (!dialog) return;
    /* A response can destroy the dialog synchronously, so dispatch one action. */
    if (command_pending("confirm-no")) gtk_dialog_response(GTK_DIALOG(dialog), GTK_RESPONSE_NO);
    else if (command_pending("confirm-yes")) gtk_dialog_response(GTK_DIALOG(dialog), GTK_RESPONSE_YES);
    else if (command_pending("confirm-escape")) gtk_bindings_activate(G_OBJECT(dialog), GDK_KEY_Escape, 0);
    else if (command_pending("confirm-default")) gtk_window_activate_default(GTK_WINDOW(dialog));
    else if (command_pending("confirm-dismiss")) gtk_window_close(GTK_WINDOW(dialog));
}
#elif defined(__APPLE__)
#import <Cocoa/Cocoa.h>

static NSWindow *fixture_main_window(void)
{
    for (NSWindow *window in NSApp.windows)
        if ([window.title isEqualToString:@"Fox Reader"]) return window;
    return nil;
}

static NSButton *confirmation_button(NSView *view, NSString *title)
{
    if ([view isKindOfClass:[NSButton class]] && [((NSButton *)view).title isEqualToString:title])
        return (NSButton *)view;
    for (NSView *child in view.subviews) {
        NSButton *button = confirmation_button(child, title);
        if (button) return button;
    }
    return nil;
}

static void confirmation_tick(void)
{
    NSWindow *parent = fixture_main_window();
    NSWindow *dialog = parent.attachedSheet;
    NSButton *yes = confirmation_button(dialog.contentView, @"Yes");
    NSButton *no = confirmation_button(dialog.contentView, @"No");
    if (command_pending("inspect-confirmation")) {
        FILE *file = fopen("confirmation.tmp", "wb");
        if (file) {
            fprintf(file, "count=%d\nid=%ld\nparent=%d\nmodal=%d\ndefault-no=%d\nyes=%d\nno=%d\n",
                (int)parent.sheets.count, (long)dialog.windowNumber,
                dialog && dialog.sheetParent == parent && parent.visible,
                dialog && dialog.isSheet,
                no && dialog.defaultButtonCell == no.cell,
                yes != nil && !yes.hidden, no != nil && !no.hidden);
            fclose(file);
            rename("confirmation.tmp", "confirmation.txt");
        }
    }
    if (!dialog || !yes || !no) return;
    if (command_pending("confirm-no")) [no performClick:nil];
    else if (command_pending("confirm-yes")) [yes performClick:nil];
    else {
        BOOL escape = command_pending("confirm-escape");
        BOOL enter = command_pending("confirm-default");
        if (escape || enter) {
            NSString *key = escape ? @"\033" : @"\r";
            NSEvent *event = [NSEvent keyEventWithType:NSEventTypeKeyDown location:NSZeroPoint
                modifierFlags:0 timestamp:NSProcessInfo.processInfo.systemUptime
                windowNumber:dialog.windowNumber context:nil characters:key
                charactersIgnoringModifiers:key isARepeat:NO keyCode:escape ? 53 : 36];
            [NSApp postEvent:event atStart:NO];
        }
        if (command_pending("confirm-dismiss")) [dialog performClose:nil];
    }
}
#endif
