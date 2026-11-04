/* System WebView2 host. The backend and its lifetime remain in launcher.c. */
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#ifndef UNICODE
#define UNICODE
#endif
#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>
#include <wrl.h>
#include <WebView2.h>
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cwctype>
#include <memory>
#include <mutex>
#include <string>
#include "window.h"
#include "window_recovery.h"

using Microsoft::WRL::Callback;
using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT kDispatch = WM_APP + 17;
constexpr UINT kFullscreen = WM_APP + 18;
constexpr UINT_PTR kSpinner = 1;
constexpr UINT_PTR kWatchdog = 2;
constexpr int kInstallRuntime = 102;
constexpr wchar_t kClassName[] = L"FoxReaderWindow";
constexpr char kRuntimeUrl[] = "https://go.microsoft.com/fwlink/p/?LinkId=2124703";

struct Window {
    std::atomic<HWND> hwnd{nullptr};
    void (*request_stop)(void) = nullptr;
    void (*open_browser)(const char *) = nullptr;
    void (*log)(const char *) = nullptr;
    FoxWindowRecovery recovery{};
    bool browser_mode = false, fallback = false, runtime_missing = false;
    bool browser_opened = false, stopping = false, finished = false;
    bool page_visible = false, navigated = false, script_ready = false, fullscreen = false, dark = true;
    bool com_initialized = false;
    int exit_code = 0, spinner = 0;
    DWORD saved_style = 0;
    WINDOWPLACEMENT placement{sizeof(WINDOWPLACEMENT)};
    std::wstring root, profile, message = L"Starting Fox Reader…";
    std::wstring detail = L"Loading the backend. This can take 10–15 seconds.";
    std::wstring url, origin;
    std::wstring fallback_reason;
    std::string url_utf8;
    HFONT title_font = nullptr, body_font = nullptr;
    HICON icon = nullptr;
    HWND install_button = nullptr;
    ComPtr<ICoreWebView2Environment> environment;
    ComPtr<ICoreWebView2Controller> controller;
    ComPtr<ICoreWebView2> webview;
    std::mutex pending_mutex;
    std::string pending_message, pending_url;
    bool has_message = false, has_url = false, has_finish = false;
    int pending_exit_code = 0;
};

std::shared_ptr<Window> active;

std::wstring wide(const char *text)
{
    if (!text || !*text) return {};
    int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, -1, nullptr, 0);
    if (!count) return {};
    std::wstring result(static_cast<size_t>(count), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, -1, result.data(), count);
    result.pop_back();
    return result;
}

std::string utf8(const wchar_t *text)
{
    if (!text || !*text) return {};
    int count = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (!count) return {};
    std::string result(static_cast<size_t>(count), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), count, nullptr, nullptr);
    result.pop_back();
    return result;
}

std::wstring url_origin(const std::wstring &url)
{
    size_t scheme = url.find(L"://");
    if (scheme == std::wstring::npos) return {};
    size_t end = url.find_first_of(L"/?#", scheme + 3);
    return url.substr(0, end);
}

bool same_origin(const Window &window, const wchar_t *url)
{
    auto origin = url_origin(url ? url : L"");
    return !window.origin.empty() && !origin.empty() &&
           _wcsicmp(origin.c_str(), window.origin.c_str()) == 0;
}

bool local_blob(const Window &window, const wchar_t *url)
{
    return url && _wcsnicmp(url, L"blob:", 5) == 0 && same_origin(window, url + 5);
}

bool external_http(const wchar_t *url)
{
    return url && (_wcsnicmp(url, L"https://", 8) == 0 ||
                   _wcsnicmp(url, L"http://", 7) == 0);
}

void external(Window &window, const wchar_t *url)
{
    if (external_http(url) && window.open_browser) {
        auto value = utf8(url);
        window.open_browser(value.c_str());
    }
}

void resize_browser(Window &window)
{
    if (!window.controller) return;
    RECT bounds{};
    GetClientRect(window.hwnd.load(), &bounds);
    window.controller->put_Bounds(bounds);
}

void set_fullscreen(Window &window, bool enabled)
{
    if (enabled == window.fullscreen) return;
    HWND hwnd = window.hwnd.load();
    if (enabled) {
        window.saved_style = static_cast<DWORD>(GetWindowLongPtrW(hwnd, GWL_STYLE));
        window.placement.length = sizeof(WINDOWPLACEMENT);
        GetWindowPlacement(hwnd, &window.placement);
        MONITORINFO monitor{sizeof(MONITORINFO)};
        if (!GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &monitor)) return;
        SetWindowLongPtrW(hwnd, GWL_STYLE, window.saved_style & ~WS_OVERLAPPEDWINDOW);
        SetWindowPos(hwnd, HWND_TOP, monitor.rcMonitor.left, monitor.rcMonitor.top,
                     monitor.rcMonitor.right - monitor.rcMonitor.left,
                     monitor.rcMonitor.bottom - monitor.rcMonitor.top,
                     SWP_FRAMECHANGED | SWP_NOOWNERZORDER);
    } else {
        SetWindowLongPtrW(hwnd, GWL_STYLE, window.saved_style);
        SetWindowPlacement(hwnd, &window.placement);
        SetWindowPos(hwnd, nullptr, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
    }
    window.fullscreen = enabled;
    resize_browser(window);
}

bool fullscreen_key(Window &window, UINT key, bool repeat)
{
    if (key == VK_F11 || (key == VK_RETURN && (GetKeyState(VK_MENU) & 0x8000))) {
        if (!repeat) PostMessageW(window.hwnd.load(), kFullscreen, 1, 0);
        return true;
    }
    if (key == VK_ESCAPE && window.fullscreen) {
        if (!repeat) PostMessageW(window.hwnd.load(), kFullscreen, 0, 0);
        return true;
    }
    return false;
}

void apply_theme(Window &window)
{
    DWORD light = 0, size = sizeof(light);
    window.dark = RegGetValueW(HKEY_CURRENT_USER,
        L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize",
        L"AppsUseLightTheme", RRF_RT_REG_DWORD, nullptr, &light, &size) != ERROR_SUCCESS || !light;
    HMODULE dwm = LoadLibraryW(L"dwmapi.dll");
    if (dwm) {
        using SetAttribute = HRESULT(WINAPI *)(HWND, DWORD, LPCVOID, DWORD);
        auto set_attribute = reinterpret_cast<SetAttribute>(GetProcAddress(dwm, "DwmSetWindowAttribute"));
        BOOL dark = window.dark;
        if (set_attribute && FAILED(set_attribute(window.hwnd.load(), 20, &dark, sizeof(dark))))
            set_attribute(window.hwnd.load(), 19, &dark, sizeof(dark));
        FreeLibrary(dwm);
    }
    if (window.controller) {
        ComPtr<ICoreWebView2Controller2> controller2;
        if (SUCCEEDED(window.controller.As(&controller2))) {
            BYTE color = window.dark ? 24 : 248;
            controller2->put_DefaultBackgroundColor({255, color, color, color});
        }
    }
    InvalidateRect(window.hwnd.load(), nullptr, TRUE);
}

void fallback(Window &window, HRESULT error, const wchar_t *reason = nullptr)
{
    if (window.stopping || window.finished) return;
    if (fox_recovery_fallback(&window.recovery)) {
        window.fallback = true;
        window.fallback_reason = reason ? reason : window.runtime_missing
            ? L"Microsoft WebView2 is not installed. The browser will be used.\nInstall the runtime to use the app window next time."
            : L"The app window could not start, so the browser will be used.\nKeep this window open. Close it to stop Fox Reader.";
        char message[512];
        std::snprintf(message, sizeof message, "WebView2 fallback (HRESULT 0x%08lx): %s",
                      static_cast<unsigned long>(error), utf8(window.fallback_reason.c_str()).c_str());
        if (window.log) window.log(message);
        KillTimer(window.hwnd.load(), kWatchdog);
        // Stop a stalled page, but leave backend ownership with launcher.c.
        if (window.webview) window.webview->Stop();
    }
    window.page_visible = false;
    if (window.controller) window.controller->put_IsVisible(FALSE);
    if (window.runtime_missing && window.install_button)
        ShowWindow(window.install_button, SW_SHOW);
    if (!window.url_utf8.empty() && !window.browser_opened && window.open_browser) {
        window.browser_opened = true;
        window.open_browser(window.url_utf8.c_str());
    }
    window.message = window.url_utf8.empty() ? L"Starting Fox Reader…" : L"Fox Reader is open in your browser";
    window.detail = window.fallback_reason;
    InvalidateRect(window.hwnd.load(), nullptr, TRUE);
}

bool start_watchdog(Window &window)
{
    // Only initialization/first navigation arm this timer. The backend wait
    // between those phases and later model transfers remain untimed.
    if (SetTimer(window.hwnd.load(), kWatchdog, 1000, nullptr)) return true;
    fallback(window, E_FAIL);
    return false;
}

void navigate(Window &window)
{
    if (window.stopping || window.finished || window.url.empty()) return;
    if (window.browser_mode || window.fallback) {
        if (!window.browser_opened && window.open_browser) {
            window.browser_opened = true;
            window.open_browser(window.url_utf8.c_str());
        }
        if (window.browser_mode) {
            window.message = L"Fox Reader is open in your browser";
            window.detail = L"Keep this window open while reading.\nClose it to stop Fox Reader.";
        } else fallback(window, S_OK);
        InvalidateRect(window.hwnd.load(), nullptr, TRUE);
        KillTimer(window.hwnd.load(), kSpinner);
    } else if (window.webview && window.script_ready && !window.navigated) {
        window.navigated = true;
        fox_recovery_begin_navigation(&window.recovery, GetTickCount64());
        if (!start_watchdog(window)) return;
        if (FAILED(window.webview->Navigate(window.url.c_str()))) fallback(window, E_FAIL);
    }
}

void configure_browser(const std::shared_ptr<Window> &window)
{
    auto weak = std::weak_ptr<Window>(window);
    if (FAILED(window->controller->get_CoreWebView2(&window->webview))) {
        fallback(*window, E_NOINTERFACE);
        return;
    }
    window->controller->put_IsVisible(FALSE);
    resize_browser(*window);
    apply_theme(*window);
    ComPtr<ICoreWebView2Settings> settings;
    if (SUCCEEDED(window->webview->get_Settings(&settings))) {
        settings->put_AreDevToolsEnabled(FALSE);
        settings->put_AreDefaultContextMenusEnabled(FALSE);
        settings->put_IsZoomControlEnabled(FALSE);
        settings->put_IsStatusBarEnabled(FALSE);
        settings->put_IsBuiltInErrorPageEnabled(FALSE);
        ComPtr<ICoreWebView2Settings3> settings3;
        if (SUCCEEDED(settings.As(&settings3))) settings3->put_AreBrowserAcceleratorKeysEnabled(FALSE);
    }
    EventRegistrationToken token{};
    window->controller->add_AcceleratorKeyPressed(
        Callback<ICoreWebView2AcceleratorKeyPressedEventHandler>([weak](ICoreWebView2Controller *,
            ICoreWebView2AcceleratorKeyPressedEventArgs *args) -> HRESULT {
            auto state = weak.lock();
            if (!state || state->stopping || state->finished || state->fallback) return S_OK;
            COREWEBVIEW2_KEY_EVENT_KIND kind{};
            args->get_KeyEventKind(&kind);
            if (kind == COREWEBVIEW2_KEY_EVENT_KIND_KEY_DOWN || kind == COREWEBVIEW2_KEY_EVENT_KIND_SYSTEM_KEY_DOWN) {
                UINT key = 0;
                COREWEBVIEW2_PHYSICAL_KEY_STATUS physical{};
                args->get_VirtualKey(&key);
                args->get_PhysicalKeyStatus(&physical);
                if (fullscreen_key(*state, key, physical.WasKeyDown != FALSE)) args->put_Handled(TRUE);
            }
            return S_OK;
        }).Get(), &token);
    window->webview->add_NavigationStarting(
        Callback<ICoreWebView2NavigationStartingEventHandler>([weak](ICoreWebView2 *,
            ICoreWebView2NavigationStartingEventArgs *args) -> HRESULT {
            auto state = weak.lock();
            LPWSTR uri = nullptr;
            args->get_Uri(&uri);
            if (!state || state->stopping || state->finished || state->fallback ||
                (!same_origin(*state, uri) && !local_blob(*state, uri))) {
                args->put_Cancel(TRUE);
                BOOL user = FALSE;
                args->get_IsUserInitiated(&user);
                if (state && user && !state->stopping && !state->finished && !state->fallback)
                    external(*state, uri);
            }
            CoTaskMemFree(uri);
            return S_OK;
        }).Get(), &token);
    window->webview->add_NewWindowRequested(
        Callback<ICoreWebView2NewWindowRequestedEventHandler>([weak](ICoreWebView2 *,
            ICoreWebView2NewWindowRequestedEventArgs *args) -> HRESULT {
            args->put_Handled(TRUE);
            auto state = weak.lock();
            BOOL user = FALSE;
            args->get_IsUserInitiated(&user);
            LPWSTR uri = nullptr;
            args->get_Uri(&uri);
            if (state && user && !state->stopping && !state->finished && !state->fallback)
                external(*state, uri);
            CoTaskMemFree(uri);
            return S_OK;
        }).Get(), &token);
    window->webview->add_NavigationCompleted(
        Callback<ICoreWebView2NavigationCompletedEventHandler>([weak](ICoreWebView2 *,
            ICoreWebView2NavigationCompletedEventArgs *args) -> HRESULT {
            auto state = weak.lock();
            if (!state || state->stopping || state->finished || state->fallback) return S_OK;
            // WebView2 may complete its initial about:blank page before the
            // backend URL is supplied. That is not our first navigation.
            if (!state->navigated) return S_OK;
            BOOL success = FALSE;
            args->get_IsSuccess(&success);
            if (!success) {
                COREWEBVIEW2_WEB_ERROR_STATUS error{};
                args->get_WebErrorStatus(&error);
                // Cancelled external navigation leaves the reader intact.
                if (error != COREWEBVIEW2_WEB_ERROR_STATUS_OPERATION_CANCELED) fallback(*state, E_FAIL);
                return S_OK;
            }
            LPWSTR source = nullptr;
            state->webview->get_Source(&source);
            bool local = same_origin(*state, source);
            CoTaskMemFree(source);
            if (!local) return S_OK;
            fox_recovery_loaded(&state->recovery);
            KillTimer(state->hwnd.load(), kWatchdog);
            state->page_visible = true;
            state->controller->put_IsVisible(TRUE);
            state->controller->MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC);
            KillTimer(state->hwnd.load(), kSpinner);
            return S_OK;
        }).Get(), &token);
    window->webview->add_ProcessFailed(
        Callback<ICoreWebView2ProcessFailedEventHandler>([weak](ICoreWebView2 *,
            ICoreWebView2ProcessFailedEventArgs *) -> HRESULT {
            if (auto state = weak.lock()) fallback(*state, E_FAIL);
            return S_OK;
        }).Get(), &token);
    window->webview->add_ContainsFullScreenElementChanged(
        Callback<ICoreWebView2ContainsFullScreenElementChangedEventHandler>([weak](ICoreWebView2 *view,
            IUnknown *) -> HRESULT {
            BOOL fullscreen = FALSE;
            view->get_ContainsFullScreenElement(&fullscreen);
            if (auto state = weak.lock()) {
                if (!state->stopping && !state->finished && !state->fallback)
                    set_fullscreen(*state, fullscreen != FALSE);
            }
            return S_OK;
        }).Get(), &token);
    window->webview->add_WebMessageReceived(
        Callback<ICoreWebView2WebMessageReceivedEventHandler>([weak](ICoreWebView2 *,
            ICoreWebView2WebMessageReceivedEventArgs *args) -> HRESULT {
            auto state = weak.lock();
            LPWSTR source = nullptr, message = nullptr;
            args->get_Source(&source);
            if (state && !state->stopping && !state->finished && !state->fallback && same_origin(*state, source) &&
                SUCCEEDED(args->TryGetWebMessageAsString(&message)) && message &&
                wcscmp(message, L"fox-reader:fullscreen") == 0)
                set_fullscreen(*state, !state->fullscreen);
            CoTaskMemFree(source);
            CoTaskMemFree(message);
            return S_OK;
        }).Get(), &token);
    // A narrow native capability; no filesystem or process access is exposed.
    HRESULT result = window->webview->AddScriptToExecuteOnDocumentCreated(
        LR"JS(Object.defineProperty(window, 'foxReaderNativeFullscreen', { value: function () {
            window.chrome.webview.postMessage('fox-reader:fullscreen');
        }});)JS",
        Callback<ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler>(
            [weak](HRESULT error, LPCWSTR) -> HRESULT {
                if (auto state = weak.lock()) {
                    if (state->stopping || state->finished || state->fallback) return S_OK;
                    if (FAILED(error)) fallback(*state, error);
                    else {
                        state->script_ready = true;
                        fox_recovery_engine_ready(&state->recovery);
                        KillTimer(state->hwnd.load(), kWatchdog);
                        navigate(*state);
                    }
                }
                return S_OK;
            }).Get());
    if (FAILED(result)) fallback(*window, result);
}

std::wstring profile_path(const std::wstring &root)
{
    wchar_t local[MAX_PATH]{};
    if (FAILED(SHGetFolderPathW(nullptr, CSIDL_LOCAL_APPDATA | CSIDL_FLAG_CREATE,
                              nullptr, SHGFP_TYPE_CURRENT, local))) return {};
    // Separate installs should not fight for WebView2's profile lock.
    unsigned long long hash = 14695981039346656037ULL;
    for (wchar_t c : root) { hash ^= static_cast<unsigned short>(towlower(c)); hash *= 1099511628211ULL; }
    wchar_t suffix[24]{};
    swprintf_s(suffix, L"%016llx", hash);
    std::wstring path = std::wstring(local) + L"\\Fox Reader\\WebView2\\" + suffix;
    int result = SHCreateDirectoryExW(nullptr, path.c_str(), nullptr);
    if (result != ERROR_SUCCESS && result != ERROR_ALREADY_EXISTS && result != ERROR_FILE_EXISTS) return {};
    return path;
}

void start_webview(const std::shared_ptr<Window> &window)
{
    fox_recovery_begin_initialization(&window->recovery, GetTickCount64());
    if (!start_watchdog(*window)) return;
    LPWSTR version = nullptr;
    HRESULT result = GetAvailableCoreWebView2BrowserVersionString(nullptr, &version);
    CoTaskMemFree(version);
    if (FAILED(result)) {
        window->runtime_missing = true;
        fallback(*window, result);
        return;
    }
    window->profile = profile_path(window->root);
    if (window->profile.empty()) { fallback(*window, E_ACCESSDENIED); return; }
    auto weak = std::weak_ptr<Window>(window);
    result = CreateCoreWebView2EnvironmentWithOptions(nullptr, window->profile.c_str(), nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>([weak](HRESULT error,
            ICoreWebView2Environment *environment) -> HRESULT {
            auto state = weak.lock();
            if (!state || state->stopping || state->finished || state->fallback) return S_OK;
            if (FAILED(error) || !environment) { fallback(*state, error); return S_OK; }
            state->environment = environment;
            HRESULT result = environment->CreateCoreWebView2Controller(state->hwnd.load(),
                Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>([weak](HRESULT status,
                    ICoreWebView2Controller *controller) -> HRESULT {
                    auto current = weak.lock();
                    if (!current || current->stopping || current->finished || current->fallback) {
                        if (controller) controller->Close();
                        return S_OK;
                    }
                    if (FAILED(status) || !controller) { fallback(*current, status); return S_OK; }
                    current->controller = controller;
                    configure_browser(current);
                    return S_OK;
                }).Get());
            if (FAILED(result)) fallback(*state, result);
            return S_OK;
        }).Get());
    if (FAILED(result)) fallback(*window, result);
}

void stop_view(Window &window)
{
    fox_recovery_stop(&window.recovery);
    KillTimer(window.hwnd.load(), kWatchdog);
    window.page_visible = false;
    if (window.controller) window.controller->Close();
    window.webview.Reset();
    window.controller.Reset();
    window.environment.Reset();
}

void dispatch(Window &window)
{
    bool finish = false;
    {
        std::lock_guard<std::mutex> guard(window.pending_mutex);
        if (window.has_message && !window.stopping) {
            window.message = wide(window.pending_message.c_str());
            window.has_message = false;
        }
        if (window.has_url && !window.stopping) {
            window.url_utf8 = window.pending_url;
            window.url = wide(window.pending_url.c_str());
            window.origin = url_origin(window.url);
            window.has_url = false;
        }
        if (window.has_finish) {
            window.exit_code = window.pending_exit_code;
            window.has_finish = false;
            finish = true;
        }
    }
    if (finish) {
        window.finished = true;
        stop_view(window);
        if (window.exit_code == 0 || window.stopping) {
            DestroyWindow(window.hwnd.load());
            return;
        }
        window.detail = window.message + L"\nSee logs/launcher.log and logs/backend.log for details.";
        window.message = L"Fox Reader could not continue";
        if (window.install_button) ShowWindow(window.install_button, SW_HIDE);
        KillTimer(window.hwnd.load(), kSpinner);
    } else navigate(window);
    InvalidateRect(window.hwnd.load(), nullptr, TRUE);
}

void paint(Window &window, HWND hwnd)
{
    PAINTSTRUCT paint{};
    HDC dc = BeginPaint(hwnd, &paint);
    RECT bounds{};
    GetClientRect(hwnd, &bounds);
    HBRUSH background = CreateSolidBrush(window.dark ? RGB(24, 24, 24) : RGB(248, 248, 248));
    FillRect(dc, &bounds, background);
    DeleteObject(background);
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, window.dark ? RGB(238, 238, 238) : RGB(25, 25, 25));
    int center = (bounds.right - bounds.left) / 2;
    int top = std::max(32L, (bounds.bottom - bounds.top) / 2 - 110);
    if (window.icon) DrawIconEx(dc, center - 28, top, window.icon, 56, 56, 0, nullptr, DI_NORMAL);
    RECT title{24, top + 82, bounds.right - 24, top + 135};
    HGDIOBJ previous = SelectObject(dc, window.title_font);
    DrawTextW(dc, window.message.c_str(), -1, &title, DT_CENTER | DT_WORDBREAK | DT_NOPREFIX);
    RECT body{32, top + 142, bounds.right - 32, bounds.bottom - 64};
    SelectObject(dc, window.body_font);
    SetTextColor(dc, window.dark ? RGB(184, 184, 184) : RGB(80, 80, 80));
    DrawTextW(dc, window.detail.c_str(), -1, &body, DT_CENTER | DT_WORDBREAK | DT_NOPREFIX);
    if (!window.finished && (window.url.empty() || window.stopping)) {
        for (int i = 0; i < 3; ++i) {
            COLORREF color = i == window.spinner % 3 ? RGB(236, 146, 68) :
                (window.dark ? RGB(76, 76, 76) : RGB(204, 204, 204));
            HBRUSH dot = CreateSolidBrush(color);
            RECT rectangle{center - 19 + i * 14, top + 64, center - 13 + i * 14, top + 70};
            FillRect(dc, &rectangle, dot);
            DeleteObject(dot);
        }
    }
    SelectObject(dc, previous);
    EndPaint(hwnd, &paint);
}

LRESULT CALLBACK window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
{
    auto window = reinterpret_cast<Window *>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (message == WM_NCCREATE) {
        window = static_cast<Window *>(reinterpret_cast<CREATESTRUCTW *>(lparam)->lpCreateParams);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(window));
        window->hwnd.store(hwnd);
    }
    if (!window) return DefWindowProcW(hwnd, message, wparam, lparam);
    switch (message) {
    case kDispatch: dispatch(*window); return 0;
    case kFullscreen: set_fullscreen(*window, wparam ? !window->fullscreen : false); return 0;
    case WM_CLOSE:
        if (window->finished) { DestroyWindow(hwnd); return 0; }
        if (!window->stopping) {
            window->stopping = true;
            stop_view(*window);
            window->message = L"Closing Fox Reader…";
            window->detail = L"Saving and releasing models. Please wait.";
            SetTimer(hwnd, kSpinner, 500, nullptr);
            if (window->install_button) ShowWindow(window->install_button, SW_HIDE);
            InvalidateRect(hwnd, nullptr, TRUE);
            if (window->request_stop) window->request_stop();
        }
        return 0;
    case WM_QUERYENDSESSION:
        if (!window->stopping) {
            window->stopping = true;
            fox_recovery_stop(&window->recovery);
            KillTimer(hwnd, kWatchdog);
            if (window->request_stop) window->request_stop();
        }
        return TRUE;
    case WM_DESTROY:
        window->hwnd.store(nullptr);
        KillTimer(hwnd, kSpinner);
        KillTimer(hwnd, kWatchdog);
        PostQuitMessage(window->exit_code);
        return 0;
    case WM_SIZE:
        resize_browser(*window);
        if (window->install_button)
            MoveWindow(window->install_button, std::max(12, (LOWORD(lparam) - 220) / 2),
                       std::max(12, HIWORD(lparam) - 50), 220, 30, TRUE);
        return 0;
    case WM_GETMINMAXINFO:
        reinterpret_cast<MINMAXINFO *>(lparam)->ptMinTrackSize = {560, 360};
        return 0;
    case WM_SETFOCUS:
        if (window->controller && window->page_visible)
            window->controller->MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC);
        return 0;
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
        if (fullscreen_key(*window, static_cast<UINT>(wparam), (lparam & (1L << 30)) != 0)) return 0;
        break;
    case WM_COMMAND:
        if (LOWORD(wparam) == kInstallRuntime && window->open_browser) {
            window->open_browser(kRuntimeUrl);
            return 0;
        }
        break;
    case WM_SETTINGCHANGE:
    case WM_THEMECHANGED: apply_theme(*window); return 0;
    case WM_DPICHANGED: {
        auto bounds = reinterpret_cast<RECT *>(lparam);
        SetWindowPos(hwnd, nullptr, bounds->left, bounds->top, bounds->right - bounds->left,
                     bounds->bottom - bounds->top, SWP_NOZORDER | SWP_NOACTIVATE);
        return 0;
    }
    case WM_ERASEBKGND: return 1;
    case WM_PAINT: paint(*window, hwnd); return 0;
    case WM_TIMER:
        if (wparam == kWatchdog) {
            if (fox_recovery_timed_out(&window->recovery, GetTickCount64()))
                fallback(*window, HRESULT_FROM_WIN32(WAIT_TIMEOUT), window->script_ready
                    ? L"The reader page did not finish loading within 30 seconds.\nFox Reader will continue in your browser. Keep this window open."
                    : L"The app window did not initialize within 30 seconds.\nFox Reader will open in your browser when the backend is ready.");
            return 0;
        }
        ++window->spinner;
        if (!window->page_visible && !window->finished) InvalidateRect(hwnd, nullptr, FALSE);
        return 0;
    }
    return DefWindowProcW(hwnd, message, wparam, lparam);
}
} // namespace

extern "C" int fox_window_create(const char *root_utf8, void (*request_stop)(void),
                                  void (*open_browser)(const char *), void (*log)(const char *), int browser_mode)
{
    if (std::atomic_load(&active)) return 0;
    auto window = std::make_shared<Window>();
    window->request_stop = request_stop;
    window->open_browser = open_browser;
    window->log = log;
    window->browser_mode = browser_mode != 0;
    window->root = wide(root_utf8);
    HRESULT result = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(result)) return 0;
    window->com_initialized = true;
    using SetDpiAwareness = BOOL(WINAPI *)(HANDLE);
    auto set_dpi = reinterpret_cast<SetDpiAwareness>(GetProcAddress(GetModuleHandleW(L"user32.dll"),
                                                                   "SetProcessDpiAwarenessContext"));
    if (set_dpi) set_dpi(reinterpret_cast<HANDLE>(-4));
    HINSTANCE instance = GetModuleHandleW(nullptr);
    window->icon = static_cast<HICON>(LoadImageW(instance, MAKEINTRESOURCEW(101),
                                               IMAGE_ICON, 64, 64, LR_SHARED));
    if (!window->icon) window->icon = LoadIconW(nullptr, IDI_APPLICATION);
    WNDCLASSEXW cls{sizeof(WNDCLASSEXW)};
    cls.lpfnWndProc = window_proc;
    cls.hInstance = instance;
    cls.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    cls.hIcon = window->icon;
    cls.hIconSm = window->icon;
    cls.lpszClassName = kClassName;
    if (!RegisterClassExW(&cls) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        CoUninitialize();
        return 0;
    }
    window->title_font = CreateFontW(-26, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
    window->body_font = CreateFontW(-16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
    std::atomic_store(&active, window);
    HWND hwnd = CreateWindowExW(0, kClassName, L"Fox Reader", WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
        CW_USEDEFAULT, CW_USEDEFAULT, 1100, 760, nullptr, nullptr, instance, window.get());
    if (!hwnd) { fox_window_destroy(); return 0; }
    window->install_button = CreateWindowExW(0, L"BUTTON", L"Install Microsoft WebView2", WS_CHILD | BS_PUSHBUTTON,
        430, 650, 220, 30, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kInstallRuntime)), instance, nullptr);
    SendMessageW(window->install_button, WM_SETFONT, reinterpret_cast<WPARAM>(window->body_font), TRUE);
    apply_theme(*window);
    ShowWindow(hwnd, SW_SHOWNORMAL);
    UpdateWindow(hwnd);
    SetTimer(hwnd, kSpinner, 500, nullptr);
    if (!window->browser_mode) start_webview(window);
    return 1;
}

extern "C" void fox_window_status(const char *message)
{
    if (auto window = std::atomic_load(&active)) {
        { std::lock_guard<std::mutex> guard(window->pending_mutex);
          window->pending_message = message ? message : ""; window->has_message = true; }
        if (HWND hwnd = window->hwnd.load()) PostMessageW(hwnd, kDispatch, 0, 0);
    }
}

extern "C" void fox_window_ready(const char *url)
{
    if (auto window = std::atomic_load(&active)) {
        { std::lock_guard<std::mutex> guard(window->pending_mutex);
          window->pending_url = url ? url : ""; window->has_url = true; }
        if (HWND hwnd = window->hwnd.load()) PostMessageW(hwnd, kDispatch, 0, 0);
    }
}

extern "C" void fox_window_finish(int exit_code)
{
    if (auto window = std::atomic_load(&active)) {
        { std::lock_guard<std::mutex> guard(window->pending_mutex);
          window->pending_exit_code = exit_code; window->has_finish = true; }
        if (HWND hwnd = window->hwnd.load()) PostMessageW(hwnd, kDispatch, 0, 0);
    }
}

extern "C" int fox_window_run(void)
{
    MSG message{};
    int status = 0;
    while ((status = GetMessageW(&message, nullptr, 0, 0)) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    return status == -1 ? 1 : static_cast<int>(message.wParam);
}

extern "C" void fox_window_destroy(void)
{
    auto window = std::atomic_exchange(&active, std::shared_ptr<Window>{});
    if (!window) return;
    window->stopping = true;
    stop_view(*window);
    if (HWND hwnd = window->hwnd.load()) DestroyWindow(hwnd);
    if (window->title_font) DeleteObject(window->title_font);
    if (window->body_font) DeleteObject(window->body_font);
    if (window->com_initialized) CoUninitialize();
}
