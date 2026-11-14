#define NOMINMAX
#include <windows.h>
#include <commctrl.h>
#include "ui_state.h"

#include <memory>
#include <iostream>

namespace {
std::unique_ptr<fox_updater::UiState> state;
HWND title_label, message_label, progress_bar, close_button;
HFONT title_font, body_font;
bool marquee = false;
int exit_code = 1;

std::wstring wide(const std::string& value) {
    const int count = MultiByteToWideChar(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0);
    std::wstring result(static_cast<std::size_t>(count), L'\0');
    if (count) MultiByteToWideChar(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), result.data(), count);
    return result;
}

void refresh(HWND window) {
    const auto view = state->snapshot();
    SetWindowTextW(title_label, wide(fox_updater::stage_title(view.progress.stage)).c_str());
    SetWindowTextW(message_label, wide(view.progress.message).c_str());
    EnableWindow(close_button, view.finished || view.progress.cancellable);
    SetWindowTextW(close_button, view.finished ? L"Close" : L"Cancel");
    const bool pulse = !view.finished && !view.progress.total;
    if (marquee != pulse) {
        marquee = pulse;
        auto style = GetWindowLongPtrW(progress_bar, GWL_STYLE);
        SetWindowLongPtrW(progress_bar, GWL_STYLE, pulse ? style | PBS_MARQUEE : style & ~PBS_MARQUEE);
        SendMessageW(progress_bar, PBM_SETMARQUEE, pulse, 30);
    }
    if (!pulse) {
        int value = view.finished && view.successful ? 1000 : view.progress.total ?
            static_cast<int>((static_cast<long double>(view.progress.done) / view.progress.total) * 1000) : 0;
        SendMessageW(progress_bar, PBM_SETPOS, value, 0);
    }
    auto menu = GetSystemMenu(window, FALSE);
    EnableMenuItem(menu, SC_CLOSE, MF_BYCOMMAND | (view.finished || view.progress.cancellable ? MF_ENABLED : MF_GRAYED));
    exit_code = view.exit_code;
}

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    switch (message) {
    case WM_CREATE: {
        title_font = CreateFontW(-23, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                                OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
        body_font = CreateFontW(-16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                               OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
        title_label = CreateWindowW(L"STATIC", L"Preparing update", WS_CHILD | WS_VISIBLE,
                                    28, 24, 500, 32, window, nullptr, nullptr, nullptr);
        message_label = CreateWindowW(L"STATIC", L"", WS_CHILD | WS_VISIBLE | SS_LEFT,
                                      28, 75, 500, 100, window, nullptr, nullptr, nullptr);
        progress_bar = CreateWindowW(PROGRESS_CLASSW, L"", WS_CHILD | WS_VISIBLE | PBS_SMOOTH,
                                     28, 189, 500, 18, window, nullptr, nullptr, nullptr);
        close_button = CreateWindowW(L"BUTTON", L"Cancel", WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_DEFPUSHBUTTON,
                                     418, 229, 110, 34, window, reinterpret_cast<HMENU>(1), nullptr, nullptr);
        SendMessageW(title_label, WM_SETFONT, reinterpret_cast<WPARAM>(title_font), TRUE);
        SendMessageW(message_label, WM_SETFONT, reinterpret_cast<WPARAM>(body_font), TRUE);
        SendMessageW(close_button, WM_SETFONT, reinterpret_cast<WPARAM>(body_font), TRUE);
        SendMessageW(progress_bar, PBM_SETRANGE32, 0, 1000);
        SetTimer(window, 1, 100, nullptr);
        refresh(window);
        return 0;
    }
    case WM_TIMER: refresh(window); return 0;
    case WM_COMMAND:
        if (LOWORD(wparam) == 1) SendMessageW(window, WM_CLOSE, 0, 0);
        return 0;
    case WM_CLOSE:
        if (state->request_close()) DestroyWindow(window);
        else refresh(window);
        return 0;
    case WM_DESTROY:
        KillTimer(window, 1);
        if (title_font) DeleteObject(title_font);
        if (body_font) DeleteObject(body_font);
        PostQuitMessage(exit_code);
        return 0;
    }
    return DefWindowProcW(window, message, wparam, lparam);
}
}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show) {
    std::string error;
    auto handoff = fox_updater::Handoff::receive(error);
    if (!handoff) { std::cerr << error << '\n'; return 1; }
    if (handoff && handoff->silent()) return handoff->run({}, {}).success ? 0 : 1;
    state = std::make_unique<fox_updater::UiState>(handoff, error);
    SetProcessDPIAware();
    INITCOMMONCONTROLSEX controls{sizeof controls, ICC_PROGRESS_CLASS};
    InitCommonControlsEx(&controls);
    WNDCLASSW klass{};
    klass.hInstance = instance;
    klass.lpszClassName = L"FoxReaderUpdater";
    klass.lpfnWndProc = window_proc;
    klass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    klass.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    klass.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    if (!RegisterClassW(&klass)) return 1;
    RECT bounds{0, 0, 558, 289};
    AdjustWindowRect(&bounds, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX, FALSE);
    HWND window = CreateWindowW(klass.lpszClassName, L"Fox Reader Update",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        (GetSystemMetrics(SM_CXSCREEN) - (bounds.right - bounds.left)) / 2,
        (GetSystemMetrics(SM_CYSCREEN) - (bounds.bottom - bounds.top)) / 2,
        bounds.right - bounds.left, bounds.bottom - bounds.top, nullptr, nullptr, instance, nullptr);
    if (!window) return 1;
    ShowWindow(window, show == SW_HIDE ? SW_SHOWNORMAL : show);
    UpdateWindow(window);
    state->start();
    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        if (!IsDialogMessageW(window, &message)) { TranslateMessage(&message); DispatchMessageW(&message); }
    }
    state.reset();
    return exit_code;
}
