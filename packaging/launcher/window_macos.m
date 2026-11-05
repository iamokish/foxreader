/* Cocoa/WKWebView viewer. All backend work stays in the launcher worker. */
#import <Cocoa/Cocoa.h>
#import <WebKit/WebKit.h>
#include <stdio.h>
#include "window.h"
#include "window_recovery.h"

@interface FoxWebView : WKWebView
@end
@implementation FoxWebView
- (NSMenu *)menuForEvent:(NSEvent *)event { (void)event; return nil; }
- (void)rightMouseDown:(NSEvent *)event { (void)event; }
- (void)magnifyWithEvent:(NSEvent *)event { (void)event; }
@end

@interface FoxWindowHost : NSObject <NSApplicationDelegate, NSWindowDelegate,
    WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler, WKDownloadDelegate>
{
    FoxWindowRecovery _recovery;
}
@property(nonatomic, strong) NSWindow *window;
@property(nonatomic, strong) NSView *loading;
@property(nonatomic, strong) NSTextField *title;
@property(nonatomic, strong) NSTextField *detail;
@property(nonatomic, strong) NSProgressIndicator *progress;
@property(nonatomic, strong) FoxWebView *webview;
@property(nonatomic, strong) NSURL *readyURL;
@property(nonatomic, strong) id keyMonitor;
@property(nonatomic, strong) NSTimer *navigationDeadline;
@property(nonatomic, strong) NSMutableSet<WKDownload *> *downloads;
@property(nonatomic, strong) NSMapTable<WKDownload *, NSDictionary<NSString *, NSURL *> *> *downloadTargets;
@property(nonatomic) void (*requestStop)(void);
@property(nonatomic) void (*openBrowser)(const char *);
@property(nonatomic) void (*logMessage)(const char *);
@property(nonatomic) BOOL browserMode, browserOpened, stopping, finished, fullscreenTransition;
@property(nonatomic) int exitCode;
- (void)requestClose;
- (void)showReadyURL:(NSString *)url;
- (void)finish:(int)code;
- (void)stopView;
- (void)stopRecovery;
- (void)cancelNavigationDeadline;
- (void)armNavigationDeadline;
- (void)fallbackToBrowser:(NSString *)reason log:(NSString *)message;
- (void)toggleFullscreen:(id)sender;
@end

static FoxWindowHost *activeHost;

static uint64_t monotonicMilliseconds(void)
{
    return (uint64_t)([NSProcessInfo processInfo].systemUptime * 1000.0);
}

static BOOL isHTTP(NSURL *url)
{
    NSString *scheme = url.scheme.lowercaseString;
    return [scheme isEqualToString:@"http"] || [scheme isEqualToString:@"https"];
}

static BOOL sameOrigin(NSURL *left, NSURL *right)
{
    if (!left || !right || !isHTTP(left) || !isHTTP(right)) return NO;
    NSNumber *leftPort = left.port ?: ([left.scheme.lowercaseString isEqualToString:@"https"] ? @443 : @80);
    NSNumber *rightPort = right.port ?: ([right.scheme.lowercaseString isEqualToString:@"https"] ? @443 : @80);
    return [left.scheme caseInsensitiveCompare:right.scheme] == NSOrderedSame &&
           [left.host caseInsensitiveCompare:right.host] == NSOrderedSame && [leftPort isEqual:rightPort];
}

static BOOL localDownload(NSURL *base, NSURL *url)
{
    if (sameOrigin(base, url)) return YES;
    if ([url.scheme.lowercaseString isEqualToString:@"blob"])
        return sameOrigin(base, [NSURL URLWithString:[url.absoluteString substringFromIndex:5]]);
    return NO;
}

static void wakeApplication(void)
{
    [NSApp stop:nil];
    NSEvent *event = [NSEvent otherEventWithType:NSEventTypeApplicationDefined location:NSZeroPoint
        modifierFlags:0 timestamp:0 windowNumber:0 context:nil subtype:0 data1:0 data2:0];
    [NSApp postEvent:event atStart:NO];
}

@implementation FoxWindowHost
- (void)cancelNavigationDeadline
{
    [self.navigationDeadline invalidate];
    self.navigationDeadline = nil;
}

- (void)stopRecovery
{
    fox_recovery_stop(&_recovery);
    [self cancelNavigationDeadline];
}

- (void)armNavigationDeadline
{
    [self cancelNavigationDeadline];
    if (self.stopping || self.finished || _recovery.phase != FOX_RECOVERY_NAVIGATING) return;
    uint64_t now = monotonicMilliseconds();
    if (fox_recovery_timed_out(&_recovery, now)) {
        [self fallbackToBrowser:@"The app window did not finish loading within 30 seconds."
            log:@"WKWebView initial navigation timed out after 30 seconds; using the system browser."];
        return;
    }
    uint64_t elapsed = now >= _recovery.started_ms ? now - _recovery.started_ms : 0;
    NSTimeInterval remaining = (FOX_RECOVERY_TIMEOUT_MS - elapsed) / 1000.0;
    __weak FoxWindowHost *weakHost = self;
    self.navigationDeadline = [NSTimer timerWithTimeInterval:remaining repeats:NO block:^(NSTimer *timer) {
        FoxWindowHost *host = weakHost;
        if (!host || host.navigationDeadline != timer) return;
        // Recheck monotonic elapsed time; an early/obsolete callback must not
        // shorten the budget or revive a cancelled watchdog.
        [host armNavigationDeadline];
    }];
    [[NSRunLoop mainRunLoop] addTimer:self.navigationDeadline forMode:NSRunLoopCommonModes];
}

- (void)stopView
{
    [self cancelNavigationDeadline];
    self.webview.navigationDelegate = nil;
    self.webview.UIDelegate = nil;
    [self.webview stopLoading];
    [self.webview.configuration.userContentController removeScriptMessageHandlerForName:@"foxReaderFullscreen"];
    [self.webview removeFromSuperview];
    self.webview = nil;
    self.loading.hidden = NO;
}

- (void)requestClose
{
    if (self.finished) { [self.window close]; return; }
    if (self.stopping) return;
    self.stopping = YES;
    [self stopRecovery];
    if (self.window.attachedSheet)
        [self.window endSheet:self.window.attachedSheet returnCode:NSModalResponseCancel];
    for (WKDownload *download in self.downloads.allObjects) [download cancel:nil];
    [self.downloads removeAllObjects];
    [self stopView];
    self.title.stringValue = @"Closing Fox Reader…";
    self.detail.stringValue = @"Saving and releasing models. Please wait.";
    [self.progress startAnimation:nil];
    if (self.requestStop) self.requestStop();
}

- (BOOL)windowShouldClose:(NSWindow *)sender
{
    (void)sender;
    if (self.finished) return YES;
    [self requestClose];
    return NO;
}

- (void)windowWillClose:(NSNotification *)notification
{
    (void)notification;
    wakeApplication();
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender
{
    (void)sender;
    [self requestClose];
    return NSTerminateCancel;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender
{
    (void)sender;
    return NO;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification
{
    (void)notification;
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (void)toggleFullscreen:(id)sender
{
    (void)sender;
    if (!self.stopping && !self.finished && !self.fullscreenTransition)
        [self.window toggleFullScreen:nil];
}

- (void)windowWillEnterFullScreen:(NSNotification *)notification
{ (void)notification; self.fullscreenTransition = YES; }
- (void)windowWillExitFullScreen:(NSNotification *)notification
{ (void)notification; self.fullscreenTransition = YES; }
- (void)windowDidEnterFullScreen:(NSNotification *)notification
{ (void)notification; self.fullscreenTransition = NO; }
- (void)windowDidExitFullScreen:(NSNotification *)notification
{ (void)notification; self.fullscreenTransition = NO; }
- (void)windowDidFailToEnterFullScreen:(NSWindow *)window
{ (void)window; self.fullscreenTransition = NO; }
- (void)windowDidFailToExitFullScreen:(NSWindow *)window
{ (void)window; self.fullscreenTransition = NO; }

- (void)showReadyURL:(NSString *)url
{
    if (self.stopping || self.finished || self.readyURL) return;
    NSURL *address = [NSURL URLWithString:url];
    if (!isHTTP(address)) {
        self.title.stringValue = @"The reader address is invalid";
        return;
    }
    self.readyURL = address;
    if (self.browserMode) {
        if (!self.browserOpened && self.openBrowser) {
            self.browserOpened = YES;
            self.openBrowser(url.UTF8String);
        }
        self.title.stringValue = @"Fox Reader is open in your browser";
        self.detail.stringValue = @"Keep this window open while reading.\nClose it to stop Fox Reader.";
        [self.progress stopAnimation:nil];
        return;
    }
    WKWebViewConfiguration *configuration = [[WKWebViewConfiguration alloc] init];
    // Built-in WebKit owns the runtime. No web engine or Python UI is bundled.
    configuration.websiteDataStore = [WKWebsiteDataStore defaultDataStore];
    [configuration.userContentController addScriptMessageHandler:self name:@"foxReaderFullscreen"];
    WKUserScript *bridge = [[WKUserScript alloc] initWithSource:
        @"Object.defineProperty(window, 'foxReaderNativeFullscreen', { value: function () {"
         "window.webkit.messageHandlers.foxReaderFullscreen.postMessage('toggle'); }});"
        injectionTime:WKUserScriptInjectionTimeAtDocumentStart forMainFrameOnly:YES];
    [configuration.userContentController addUserScript:bridge];
    self.webview = [[FoxWebView alloc] initWithFrame:self.window.contentView.bounds configuration:configuration];
    self.webview.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.webview.navigationDelegate = self;
    self.webview.UIDelegate = self;
    self.webview.allowsBackForwardNavigationGestures = NO;
    self.webview.allowsMagnification = NO;
    self.webview.hidden = YES;
    if (@available(macOS 13.3, *)) self.webview.inspectable = NO;
    if (@available(macOS 12.0, *)) self.webview.underPageBackgroundColor = [NSColor windowBackgroundColor];
    [self.window.contentView addSubview:self.webview];
    // The backend supplied a healthy URL before this phase begins. No timer
    // runs while waiting for backend readiness or after the reader has loaded.
    if (fox_recovery_begin_navigation(&_recovery, monotonicMilliseconds()))
        [self armNavigationDeadline];
    [self.webview loadRequest:[NSURLRequest requestWithURL:address]];
}

- (void)finish:(int)code
{
    if (self.finished) return;
    self.finished = YES;
    [self stopRecovery];
    self.exitCode = code;
    [self stopView];
    [self.progress stopAnimation:nil];
    if (code == 0 || self.stopping) {
        [self.window close];
    } else {
        NSString *reason = self.title.stringValue;
        self.title.stringValue = @"Fox Reader could not continue";
        self.detail.stringValue = [NSString stringWithFormat:@"%@\nSee logs/launcher.log and logs/backend.log for details.", reason];
    }
}

- (void)openExternal:(NSURL *)url
{
    if (!self.stopping && !self.finished && self.webview && isHTTP(url) && self.openBrowser)
        self.openBrowser(url.absoluteString.UTF8String);
}

- (void)webView:(WKWebView *)webView decidePolicyForNavigationAction:(WKNavigationAction *)action
    decisionHandler:(void (^)(WKNavigationActionPolicy))decisionHandler
{
    NSURL *url = action.request.URL;
    if (self.stopping || self.finished || webView != self.webview) {
        decisionHandler(WKNavigationActionPolicyCancel);
        return;
    }
    if (!self.stopping && action.shouldPerformDownload && localDownload(self.readyURL, url)) {
        decisionHandler(WKNavigationActionPolicyDownload);
        return;
    }
    if (!self.stopping && sameOrigin(self.readyURL, url) && action.targetFrame != nil) {
        decisionHandler(WKNavigationActionPolicyAllow);
        return;
    }
    if (action.navigationType == WKNavigationTypeLinkActivated) [self openExternal:url];
    decisionHandler(WKNavigationActionPolicyCancel);
}

- (void)webView:(WKWebView *)webView decidePolicyForNavigationResponse:(WKNavigationResponse *)response
    decisionHandler:(void (^)(WKNavigationResponsePolicy))decisionHandler
{
    if (!self.stopping && !self.finished && webView == self.webview &&
        localDownload(self.readyURL, response.response.URL)) {
        decisionHandler(response.canShowMIMEType ? WKNavigationResponsePolicyAllow : WKNavigationResponsePolicyDownload);
    } else decisionHandler(WKNavigationResponsePolicyCancel);
}

- (void)webView:(WKWebView *)webView navigationAction:(WKNavigationAction *)action didBecomeDownload:(WKDownload *)download
{
    (void)action;
    if (self.stopping || self.finished || webView != self.webview) { [download cancel:nil]; return; }
    download.delegate = self;
    [self.downloads addObject:download];
}

- (void)webView:(WKWebView *)webView navigationResponse:(WKNavigationResponse *)response didBecomeDownload:(WKDownload *)download
{
    (void)response;
    if (self.stopping || self.finished || webView != self.webview) { [download cancel:nil]; return; }
    download.delegate = self;
    [self.downloads addObject:download];
}

- (void)download:(WKDownload *)download decideDestinationUsingResponse:(NSURLResponse *)response
    suggestedFilename:(NSString *)filename completionHandler:(void (^)(NSURL *))completionHandler
{
    (void)response;
    if (self.stopping || self.finished || !self.webview) { completionHandler(nil); return; }
    NSSavePanel *panel = [NSSavePanel savePanel];
    panel.nameFieldStringValue = filename.lastPathComponent.length ? filename.lastPathComponent : @"download";
    [panel beginSheetModalForWindow:self.window completionHandler:^(NSModalResponse result) {
        if (result != NSModalResponseOK || self.stopping || self.finished || !self.webview) {
            completionHandler(nil);
            return;
        }
        NSURL *destination = panel.URL;
        // Download beside the selected file and replace only after success, so
        // cancellation or a lost connection cannot destroy an earlier export.
        NSString *temporaryName = [NSString stringWithFormat:@".fox-reader-%@.part", NSUUID.UUID.UUIDString];
        NSURL *temporary = [destination.URLByDeletingLastPathComponent URLByAppendingPathComponent:temporaryName];
        [self.downloadTargets setObject:@{@"temporary": temporary, @"destination": destination} forKey:download];
        completionHandler(temporary);
    }];
}

- (void)downloadDidFinish:(WKDownload *)download
{
    NSDictionary<NSString *, NSURL *> *target = [self.downloadTargets objectForKey:download];
    [self.downloadTargets removeObjectForKey:download];
    [self.downloads removeObject:download];
    if (!target) return;
    NSError *error = nil;
    NSFileManager *files = [NSFileManager defaultManager];
    if (self.stopping || self.finished || !self.webview) {
        [files removeItemAtURL:target[@"temporary"] error:nil];
        return;
    }
    BOOL saved = [files fileExistsAtPath:target[@"destination"].path]
        ? [files replaceItemAtURL:target[@"destination"] withItemAtURL:target[@"temporary"]
            backupItemName:nil options:0 resultingItemURL:nil error:&error]
        : [files moveItemAtURL:target[@"temporary"] toURL:target[@"destination"] error:&error];
    if (!saved) {
        fprintf(stderr, "Could not publish download: %s\n", error.localizedDescription.UTF8String);
        if (!self.stopping && !self.finished) {
            NSAlert *alert = [[NSAlert alloc] init];
            alert.messageText = @"The download could not be saved to that filename";
            alert.informativeText = [NSString stringWithFormat:@"%@\nThe downloaded file remains at %@.",
                error.localizedDescription, target[@"temporary"].path];
            [alert beginSheetModalForWindow:self.window completionHandler:nil];
        }
    }
}
- (void)download:(WKDownload *)download didFailWithError:(NSError *)error resumeData:(NSData *)resumeData
{
    (void)resumeData;
    [self.downloads removeObject:download];
    NSURL *temporary = [self.downloadTargets objectForKey:download][@"temporary"];
    if (temporary) [[NSFileManager defaultManager] removeItemAtURL:temporary error:nil];
    [self.downloadTargets removeObjectForKey:download];
    if (self.stopping || self.finished || !self.webview || error.code == NSURLErrorCancelled) return;
    fprintf(stderr, "Download failed: %s\n", error.localizedDescription.UTF8String);
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"The download could not be saved";
    alert.informativeText = error.localizedDescription;
    [alert beginSheetModalForWindow:self.window completionHandler:nil];
}

- (WKWebView *)webView:(WKWebView *)webView createWebViewWithConfiguration:(WKWebViewConfiguration *)configuration
    forNavigationAction:(WKNavigationAction *)action windowFeatures:(WKWindowFeatures *)features
{
    (void)configuration; (void)features;
    if (webView == self.webview && action.navigationType == WKNavigationTypeLinkActivated)
        [self openExternal:action.request.URL];
    return nil;
}

- (void)webView:(WKWebView *)webView didFinishNavigation:(WKNavigation *)navigation
{
    (void)navigation;
    if (self.stopping || self.finished || webView != self.webview) return;
    if (!sameOrigin(self.readyURL, webView.URL)) return;
    fox_recovery_loaded(&_recovery);
    [self cancelNavigationDeadline];
    self.loading.hidden = YES;
    webView.hidden = NO;
    [self.progress stopAnimation:nil];
    [self.window makeFirstResponder:webView];
}

- (void)fallbackToBrowser:(NSString *)reason log:(NSString *)message
{
    if (self.stopping || self.finished || !fox_recovery_fallback(&_recovery)) return;
    if (self.logMessage) self.logMessage(message.UTF8String);
    // Claim fallback before teardown: WebKit can deliver cancellations while
    // stopping. A late success or failure cannot bring back the hidden view.
    [self stopView];
    if (self.window.attachedSheet)
        [self.window endSheet:self.window.attachedSheet returnCode:NSModalResponseCancel];
    for (WKDownload *download in self.downloads.allObjects) [download cancel:nil];
    [self.downloads removeAllObjects];
    self.title.stringValue = @"The app window could not load";
    self.detail.stringValue = [NSString stringWithFormat:
        @"%@\nFox Reader is opening in your browser.\nKeep this window open; close it to stop Fox Reader.", reason];
    [self.progress stopAnimation:nil];
    if (!self.browserOpened && self.openBrowser && self.readyURL) {
        self.browserOpened = YES;
        self.openBrowser(self.readyURL.absoluteString.UTF8String);
    }
}

- (void)navigationFailed:(NSError *)error
{
    if (error.code == NSURLErrorCancelled) return;
    [self fallbackToBrowser:@"The native reader encountered a loading error."
        log:[NSString stringWithFormat:@"WKWebView failed: %@; using the system browser.", error.localizedDescription]];
}

- (void)webView:(WKWebView *)webView didFailNavigation:(WKNavigation *)navigation withError:(NSError *)error
{ (void)navigation; if (webView == self.webview) [self navigationFailed:error]; }
- (void)webView:(WKWebView *)webView didFailProvisionalNavigation:(WKNavigation *)navigation withError:(NSError *)error
{ (void)navigation; if (webView == self.webview) [self navigationFailed:error]; }
- (void)webViewWebContentProcessDidTerminate:(WKWebView *)webView
{
    if (webView != self.webview) return;
    [self navigationFailed:[NSError errorWithDomain:@"FoxReader.WebView" code:1
        userInfo:@{NSLocalizedDescriptionKey: @"The WebKit content process stopped."}]];
}

- (void)userContentController:(WKUserContentController *)controller didReceiveScriptMessage:(WKScriptMessage *)message
{
    (void)controller;
    if (self.webview && !self.stopping && !self.finished && message.frameInfo.mainFrame &&
        sameOrigin(self.readyURL, message.frameInfo.request.URL) &&
        [message.name isEqualToString:@"foxReaderFullscreen"] && [message.body isEqual:@"toggle"])
        [self toggleFullscreen:nil];
}

- (void)webView:(WKWebView *)webView runOpenPanelWithParameters:(WKOpenPanelParameters *)parameters
    initiatedByFrame:(WKFrameInfo *)frame completionHandler:(void (^)(NSArray<NSURL *> *))completionHandler
{
    if (self.stopping || self.finished || webView != self.webview ||
        !sameOrigin(self.readyURL, frame.request.URL)) { completionHandler(nil); return; }
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.allowsMultipleSelection = parameters.allowsMultipleSelection;
    panel.canChooseFiles = YES;
    panel.canChooseDirectories = parameters.allowsDirectories;
    [panel beginSheetModalForWindow:self.window completionHandler:^(NSModalResponse response) {
        completionHandler(response == NSModalResponseOK && !self.stopping && !self.finished &&
                          webView == self.webview ? panel.URLs : nil);
    }];
}
@end

static NSTextField *label(NSString *text, CGFloat size, NSFontWeight weight)
{
    NSTextField *field = [NSTextField wrappingLabelWithString:text];
    field.font = [NSFont systemFontOfSize:size weight:weight];
    field.alignment = NSTextAlignmentCenter;
    field.translatesAutoresizingMaskIntoConstraints = NO;
    return field;
}

static void installMenu(FoxWindowHost *host)
{
    NSMenu *bar = [[NSMenu alloc] init];
    NSMenuItem *appItem = [[NSMenuItem alloc] init];
    [bar addItem:appItem];
    NSMenu *appMenu = [[NSMenu alloc] initWithTitle:@"Fox Reader"];
    [appMenu addItemWithTitle:@"Hide Fox Reader" action:@selector(hide:) keyEquivalent:@"h"];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Quit Fox Reader" action:@selector(terminate:) keyEquivalent:@"q"];
    appItem.submenu = appMenu;
    NSMenuItem *editItem = [[NSMenuItem alloc] init];
    [bar addItem:editItem];
    NSMenu *edit = [[NSMenu alloc] initWithTitle:@"Edit"];
    [edit addItemWithTitle:@"Undo" action:@selector(undo:) keyEquivalent:@"z"];
    [edit addItemWithTitle:@"Redo" action:@selector(redo:) keyEquivalent:@"Z"];
    [edit addItem:[NSMenuItem separatorItem]];
    [edit addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
    [edit addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
    [edit addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
    [edit addItemWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"];
    editItem.submenu = edit;
    NSMenuItem *viewItem = [[NSMenuItem alloc] init];
    [bar addItem:viewItem];
    NSMenu *view = [[NSMenu alloc] initWithTitle:@"View"];
    NSMenuItem *fullscreen = [view addItemWithTitle:@"Toggle Full Screen" action:@selector(toggleFullscreen:) keyEquivalent:@"f"];
    fullscreen.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagControl;
    fullscreen.target = host;
    viewItem.submenu = view;
    NSApp.mainMenu = bar;
}

int fox_window_create(const char *root_utf8, void (*request_stop)(void),
                      void (*open_browser)(const char *), void (*log_message)(const char *), int browser_mode)
{
    @autoreleasepool {
        if (activeHost || ![NSThread isMainThread]) return 0;
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        FoxWindowHost *host = [[FoxWindowHost alloc] init];
        activeHost = host;
        host.requestStop = request_stop;
        host.openBrowser = open_browser;
        host.logMessage = log_message;
        host.browserMode = browser_mode != 0;
        host.downloads = [NSMutableSet set];
        host.downloadTargets = [NSMapTable strongToStrongObjectsMapTable];
        NSApp.delegate = host;
        host.window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 1100, 760)
            styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable |
                      NSWindowStyleMaskResizable backing:NSBackingStoreBuffered defer:NO];
        host.window.title = @"Fox Reader";
        host.window.minSize = NSMakeSize(560, 360);
        host.window.releasedWhenClosed = NO;
        host.window.collectionBehavior = NSWindowCollectionBehaviorFullScreenPrimary;
        host.window.delegate = host;
        host.window.backgroundColor = [NSColor windowBackgroundColor];
        NSString *root = root_utf8 ? [NSString stringWithUTF8String:root_utf8] : nil;
        NSImage *icon = root ? [[NSImage alloc] initWithContentsOfFile:[root stringByAppendingPathComponent:@"icons/fox-reader.png"]] : nil;
        if (icon) NSApp.applicationIconImage = icon;
        host.loading = [[NSView alloc] initWithFrame:host.window.contentView.bounds];
        host.loading.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        [host.window.contentView addSubview:host.loading];
        NSImageView *image = [[NSImageView alloc] init];
        image.image = icon ?: NSApp.applicationIconImage;
        image.translatesAutoresizingMaskIntoConstraints = NO;
        host.title = label(@"Starting Fox Reader…", 25, NSFontWeightSemibold);
        host.detail = label(@"Loading the backend. This can take 10–15 seconds.", 14, NSFontWeightRegular);
        host.detail.textColor = [NSColor secondaryLabelColor];
        host.progress = [[NSProgressIndicator alloc] init];
        host.progress.style = NSProgressIndicatorStyleSpinning;
        host.progress.indeterminate = YES;
        host.progress.displayedWhenStopped = NO;
        host.progress.translatesAutoresizingMaskIntoConstraints = NO;
        NSStackView *stack = [NSStackView stackViewWithViews:@[image, host.title, host.detail, host.progress]];
        stack.orientation = NSUserInterfaceLayoutOrientationVertical;
        stack.alignment = NSLayoutAttributeCenterX;
        stack.spacing = 18;
        stack.translatesAutoresizingMaskIntoConstraints = NO;
        [host.loading addSubview:stack];
        [NSLayoutConstraint activateConstraints:@[
            [stack.centerXAnchor constraintEqualToAnchor:host.loading.centerXAnchor],
            [stack.centerYAnchor constraintEqualToAnchor:host.loading.centerYAnchor],
            [stack.widthAnchor constraintEqualToAnchor:host.loading.widthAnchor constant:-64],
            [image.widthAnchor constraintEqualToConstant:64], [image.heightAnchor constraintEqualToConstant:64],
            [host.title.widthAnchor constraintEqualToAnchor:stack.widthAnchor],
            [host.detail.widthAnchor constraintEqualToAnchor:stack.widthAnchor]
        ]];
        [host.progress startAnimation:nil];
        __weak FoxWindowHost *weakHost = host;
        host.keyMonitor = [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskKeyDown handler:^NSEvent *(NSEvent *event) {
            FoxWindowHost *current = weakHost;
            if (!current || current.stopping || current.finished) return event;
            BOOL toggle = event.keyCode == 103 || (event.keyCode == 36 &&
                          (event.modifierFlags & NSEventModifierFlagOption));
            BOOL leave = event.keyCode == 53 && (current.window.styleMask & NSWindowStyleMaskFullScreen);
            if (toggle || leave) {
                if (!event.isARepeat) [current toggleFullscreen:nil];
                return nil;
            }
            return event;
        }];
        installMenu(host);
        [host.window center];
        [host.window makeKeyAndOrderFront:nil];
        [host.window displayIfNeeded];
        [NSApp activateIgnoringOtherApps:YES];
        return 1;
    }
}

void fox_window_status(const char *message)
{
    @autoreleasepool {
        NSString *text = message ? [NSString stringWithUTF8String:message] : @"";
        dispatch_async(dispatch_get_main_queue(), ^{
            if (activeHost && !activeHost.stopping && !activeHost.finished)
                activeHost.title.stringValue = text ?: @"Fox Reader";
        });
    }
}

void fox_window_ready(const char *url, int owns_backend)
{
    (void)owns_backend;
    @autoreleasepool {
        NSString *text = url ? [NSString stringWithUTF8String:url] : @"";
        dispatch_async(dispatch_get_main_queue(), ^{ [activeHost showReadyURL:text]; });
    }
}

void fox_window_finish(int exit_code)
{
    dispatch_async(dispatch_get_main_queue(), ^{ [activeHost finish:exit_code]; });
}

int fox_window_run(void)
{
    @autoreleasepool { [NSApp run]; return activeHost.exitCode; }
}

void fox_window_destroy(void)
{
    @autoreleasepool {
        FoxWindowHost *host = activeHost;
        activeHost = nil;
        if (!host) return;
        host.finished = YES;
        [host stopRecovery];
        if (host.keyMonitor) [NSEvent removeMonitor:host.keyMonitor];
        for (WKDownload *download in host.downloads.allObjects) [download cancel:nil];
        [host.downloads removeAllObjects];
        [host stopView];
        host.window.delegate = nil;
        [host.window orderOut:nil];
        [host.window close];
        NSApp.delegate = nil;
        NSApp.mainMenu = nil;
    }
}
