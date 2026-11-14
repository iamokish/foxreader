# Desktop launcher

The launcher starts the existing detached backend and displays the same FastAPI
frontend in the system WebView. It does not ship Chromium or duplicate the reader
UI. The startup window appears before the backend begins loading. The reader is
revealed after the marked health check and successful page navigation.

| Platform | Viewer | Runtime requirement |
| --- | --- | --- |
| Windows x64 | WebView2 | Microsoft Evergreen runtime; falls back to the default browser and offers Microsoft's installer link when missing |
| Linux x86_64 | WebKitGTK 4.1 / GTK 3 | Debian/Ubuntu: `sudo apt install libwebkit2gtk-4.1-0` for the reader window; the browser controller needs only GTK 3. `libwebkit2gtk-4.1-dev pkg-config` are needed to build |
| macOS arm64, 14+ | WKWebView | Built into macOS |

Run `launcher --browser` to open the system browser with a controller window.
On Linux the controller is loaded independently of WebKitGTK, so it remains
available when the reader's web engine is missing. It shows startup and running
status, offers **Open Reader** to reopen the page, and keeps **Quit Fox Reader**
available after the browser closes. Closing the browser does not stop Fox Reader.

While the backend starts, the Linux controller offers **Cancel Startup**.
A controller that started the backend offers **Quit Fox Reader**; an attached
controller offers **Close This Window** and explains that Fox Reader will remain
running. Its title-bar close button follows the same ownership rules.

If GTK or the display is unavailable, Linux uses an attached terminal or opens
an available terminal manager before starting the backend. If no controller or
terminal can open, it reports an error and does not start a new invisible backend.
This also applies when launching from the applications menu with `Terminal=false`.

`launcher --cli` and `FOX_READER_CLI=1` explicitly select terminal/headless mode:
the launcher stays in the foreground without opening a browser or window. Ctrl+C
stops its owned backend; interrupting an attached manager only detaches it. This
explicit mode remains available for scripts and environments without a display.

## Optional build without Fox Reader windows

Build locally with `python packaging/build.py --no-window` to remove all Fox
Reader graphical windows, including the reader, loading banner, browser
controller, and error dialogs. The browser opens by default once the backend
is ready; `--browser` selects the same behavior. Status and errors appear in
the terminal. `--cli` or `FOX_READER_CLI=1` starts the foreground manager
without opening a browser. This option does not change the regular CI builds.

Open `launcher.exe` on Windows. On Linux, `./install-desktop.sh` installs a
`Terminal=true` menu entry; direct browser-mode launches reuse a terminal or
open one before starting the backend. On macOS, open `Fox Reader.command`,
which starts the launcher in Terminal; this variant has no `.app` bundle.
A bare Mac launcher or a detached Windows launcher without a usable terminal
reports an error before starting a new backend in browser mode. Explicit
`--cli` remains available for scripts.

Keep the terminal manager open while reading. Closing the browser does not
stop Fox Reader; Ctrl+C in the manager stops its owned backend. Interrupting
an attached manager leaves the existing backend running. Linux browser-mode
startup fails before launching a new backend if no terminal can be opened.

The variant omits the WebView2 SDK/loader and both Linux window plugins, so
its launcher does not require WebView2 or GTK/WebKitGTK. Backend and model
requirements are unchanged. Archives have a `-no-window` suffix and record
`"no_window": true` in `build-info.json`. Update checks still report new
releases, but require rebuilding this variant with `--no-window` instead of
downloading a regular native-window build.

## Native window behavior

Native-engine initialization and the first page navigation each have a
30-second deadline. Waiting for backend readiness has no viewer deadline, and
later page navigation and model transfers are not timed by this watchdog.
If native startup stalls, the manager explains the failure and opens the same
running backend in the browser once. Linux also uses this recovery path if the
WebKit rendering process crashes or exceeds its memory limit. Recovery is
recorded in `logs/launcher.log`; it does not restart the backend. Closing the
manager still follows the normal owned/attached shutdown rules below.

F11 and Alt+Enter toggle fullscreen, including on the setup and settings pages.
The reader's existing F shortcut and fullscreen button use the same native
window action. Window chrome follows the system theme. Icons come from the
existing Fox Reader artwork, with multiple Windows icon sizes.

The reader, setup, settings, and custom-endpoint pages monitor backend health in
both native and browser windows. An offline notice offers **Retry** and clears
when the same backend recovers. A restart requires an explicit **Refresh**; it
does not automatically reload the page or discard unsaved forms. These themed
notices stay above open editor dialogs.

During an update handoff, Settings pauses its health warnings because the
backend is closing intentionally. Follow the separate updater window instead.

On macOS, open `Fox Reader.app` in the extracted distribution; keep `bin`,
`config`, `fonts`, `models`, and `cache` beside it. The root `launcher` also
remains available. After choosing a permanent location on Linux, run
`./install-desktop.sh` to install a per-user menu entry with `Terminal=false`.

The backend creates `config/fox_config.yaml` on first launch; configuration is
not bundled in the archive. Existing configuration is checked and updated on
launch without automatic configuration backups. See
[Configuration](configuration.md) for custom host/port values and automatic
repairs, including replacement of malformed documents with defaults.

Project documents and bundled font/icon licences are in `legal/` beside the
launcher. Native Windows builds place their SDK licence at
`legal/WebView2-LICENSE.txt`; the WebView2 browser runtime remains separate.

Native-window distributions also include `foxreader-updater(.exe)`. Choose
**Download and install** in Settings after saving your work. Fox Reader verifies
and prepares the release, opens the updater, then closes. The updater waits for
shutdown before replacing managed application files; configuration and models
stay in place and fonts are merged. It displays success without restarting the
reader. Close it, then start Fox Reader yourself.

The updater uses Win32 on Windows, GTK 3 on Linux, and Cocoa on macOS. It opens
only with the app's inherited-pipe handoff; double-clicking the executable does
not start an installation. Failed replacements roll back automatically. The
launcher blocks normal startup during an active update or unfinished recovery,
and uses the transaction's updater to recover an interrupted replacement. See
[Updating Fox Reader](updating.md) for cancellation, cleanup, and manual fallback.

The title-bar close button asks **Close Fox Reader?** in a native **Yes / No**
dialog on Windows, Linux, and macOS. The dialog follows the system theme and
stays above its parent window, including during startup and browser fallback.
**No** is selected by default; **No** or Escape returns to the current session.
Repeated close requests reuse the same dialog. The browser controller's close
action and macOS **Quit Fox Reader** use the same confirmation.

Choosing **Yes** in a window that started the backend sends its shutdown token,
waits up to 20 seconds for cleanup, then uses the existing process/job backstop. Closing a
second, attached window leaves the owner running. Backend completion and OS
shutdown signals do not wait for confirmation. Errors remain visible until
dismissed. An unwritable installation lock is an error, not permission to start
another backend.

Diagnostics are stored under the installation's `logs/` directory:

- `backend.log`: Python and server diagnostics, rotating at 5 MiB with three backups.
- `frontend.log`: bounded frontend errors/warnings, rotating at 1 MiB with two backups.
- `backend-output.log`: early/native library stdout and stderr, rotating at 8 MiB with one backup.
- `launcher.log`: startup, ownership, and shutdown diagnostics, rolled on the next launch after 8 MiB.

Browser diagnostics accept only local same-origin requests, with payload and
rate limits. Credential-like strings and URL query strings are redacted from
Python/frontend diagnostics. Native library output is kept as received.
Runtime profiles stay outside the backend's disposable `cache/` directory.

See [updating.md](updating.md) for explicit update checks, verified downloads,
the release manifest, and preserving data during installation.

## Image rendering

`FOX_IMAGE_RENDER_QUALITY` controls the main-image and gallery improvements:

| Value | Native window (Windows, Linux, macOS) | Regular browser |
| --- | --- | --- |
| `1` (default) | Enabled | Existing rendering |
| `2` | Enabled | Enabled |
| `3` | Existing rendering | Existing rendering |

Set the environment variable before starting Fox Reader's backend, then reopen
the reader page. An already running backend keeps its inherited environment;
restart it when changing the setting. Invalid values disable the improvements
and produce a backend warning. The native viewer's injected capability identifies
each window, so a browser fallback or a browser beside the native app retains
the browser policy.

For example, in PowerShell, `$env:FOX_IMAGE_RENDER_QUALITY = '2'` followed by
`.\launcher.exe` enables both viewers. On Linux or macOS,
`FOX_IMAGE_RENDER_QUALITY=2 ./launcher` does the same.

When enabled, the main image keeps GPU transforms during movement and releases
the forced layer after movement settles. Gallery previews use lossless Lanczos
resizing at the current display density. Previews are generated on demand with
one worker and a bounded memory cache; changes to the source page invalidate
them. Full-resolution pages remain the source for reading, captures and OCR.
Unsupported or failed previews fall back to the original image.

## Validation and rollout

`.github/workflows/launcher.yml` compiles all native hosts on Windows,
Ubuntu 22.04/24.04 and Apple Silicon. Its isolated backend fixture checks
ownership, authenticated shutdown, and file logging; Windows additionally
exercises the window close button and fullscreen restoration. The regular
release builds compile the same hosts before compiling the heavy backend.
The shared recovery-policy tests exercise deadlines and late events with a
controlled clock. Native host tests hold a local page response open to exercise
the real watchdog without opening a browser or loading models.

Before shipping a release, run this interactive pass against a staged build on
each OS. The automated tests do not replace these runtime checks:

1. Start from a path containing spaces and non-ASCII characters. Confirm the
   banner appears immediately, then the setup wizard or reader loads without a
   blank page. Close during startup, choose No, and confirm loading continues.
   Close again, choose Yes, then relaunch.
2. Complete setup, open settings, load a folder, import/export characters, open
   an image and run OCR/translation. Confirm websocket progress and image URLs
   work. Confirm export cancellation does not overwrite an existing file.
3. Toggle F11 and Alt+Enter in the reader, settings, and text inputs. Restore the
   window's size and position. Change the system theme, move between monitors,
   and check title bars and icons.
4. In each system theme, open the close dialog and confirm it stays above its
   window, including in fullscreen. Repeat the close request; only one dialog
   should appear. Test No, Return on the default No, and Escape; reading and
   any running work should continue. Open a second launcher and confirm its
   closure; reading should continue in the first. Confirm closing the first
   during a backend operation; confirm its processes exit and
   its cache cleanup runs. Test an occupied port and a backend startup failure.
5. On Linux, repeat file pickers, drag/drop, and fullscreen on both Wayland and
   X11. Launch from the applications menu without WebKitGTK and with `--browser`.
   Confirm the GTK-only controller stays visible after closing the browser,
   **Open Reader** reopens the page, and quitting stops only the owned backend.
   Test missing GTK with and without an available terminal; a failed terminal
   launch must not leave a new backend running. Check `--cli` opens no browser
   or GUI. On Windows, test missing WebView2 and its browser fallback.
6. Check and cancel an update before handoff, then retry. Test a corrupted or
   missing part and insufficient disk space; these must leave the installation
   unchanged. Complete an update and confirm the app closes, the separate updater
   reports success, and the reader does not restart automatically. Check custom
   fonts, configuration, and model weights afterwards. On an isolated test copy,
   interrupt replacement and confirm the next launcher recovers before starting
   the backend.

macOS distributions remain ad-hoc signed. Public Gatekeeper-friendly releases
still require the maintainer's Developer ID signing and notarization credentials;
these are not substituted with a bundled runtime or a security-setting change.
