#import <Cocoa/Cocoa.h>
#include "ui_state.h"
#include <csignal>
#include <memory>
#include <iostream>

namespace {
std::unique_ptr<fox_updater::UiState> state;
int exit_code = 1;
NSString* native(const std::string& value) {
    return [[NSString alloc] initWithBytes:value.data() length:value.size() encoding:NSUTF8StringEncoding] ?: @"";
}
}

@interface FoxUpdaterHost : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property(nonatomic, strong) NSWindow* window;
@property(nonatomic, strong) NSTextField* heading;
@property(nonatomic, strong) NSTextField* detail;
@property(nonatomic, strong) NSProgressIndicator* progress;
@property(nonatomic, strong) NSButton* closeButton;
@property(nonatomic, strong) NSTimer* timer;
@end

@implementation FoxUpdaterHost
- (void)applicationDidFinishLaunching:(NSNotification*)notification {
    (void)notification;
    self.window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 560, 290)
        styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
        backing:NSBackingStoreBuffered defer:NO];
    self.window.title = @"Fox Reader Update";
    self.window.delegate = self;
    self.window.releasedWhenClosed = NO;
    self.heading = [NSTextField labelWithString:@"Preparing update"];
    self.heading.font = [NSFont systemFontOfSize:23 weight:NSFontWeightSemibold];
    self.heading.frame = NSMakeRect(28, 232, 504, 34);
    self.detail = [NSTextField wrappingLabelWithString:@""];
    self.detail.font = [NSFont systemFontOfSize:14];
    self.detail.selectable = YES;
    self.detail.frame = NSMakeRect(28, 111, 504, 104);
    self.progress = [[NSProgressIndicator alloc] initWithFrame:NSMakeRect(28, 77, 504, 18)];
    self.progress.style = NSProgressIndicatorStyleBar;
    self.progress.minValue = 0;
    self.progress.maxValue = 1;
    self.closeButton = [NSButton buttonWithTitle:@"Cancel" target:self action:@selector(requestClose:)];
    self.closeButton.frame = NSMakeRect(420, 22, 112, 34);
    self.closeButton.keyEquivalent = @"\r";
    for (NSView* view in @[self.heading, self.detail, self.progress, self.closeButton])
        [self.window.contentView addSubview:view];
    [self.window center];
    [self refresh:nil];
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    self.timer = [NSTimer scheduledTimerWithTimeInterval:0.1 target:self selector:@selector(refresh:)
        userInfo:nil repeats:YES];
    state->start();
}

- (void)refresh:(NSTimer*)timer {
    (void)timer;
    const auto view = state->snapshot();
    self.heading.stringValue = native(fox_updater::stage_title(view.progress.stage));
    self.detail.stringValue = native(view.progress.message);
    self.closeButton.enabled = view.finished || view.progress.cancellable;
    self.closeButton.title = view.finished ? @"Close" : @"Cancel";
    [[self.window standardWindowButton:NSWindowCloseButton] setEnabled:self.closeButton.enabled];
    self.progress.indeterminate = !view.finished && !view.progress.total;
    if (self.progress.indeterminate) [self.progress startAnimation:nil];
    else {
        [self.progress stopAnimation:nil];
        self.progress.doubleValue = view.finished && view.successful ? 1.0 : view.progress.total ?
            static_cast<double>(view.progress.done) / static_cast<double>(view.progress.total) : 0;
    }
    exit_code = view.exit_code;
}

- (void)requestClose:(id)sender { (void)sender; [self.window performClose:nil]; }
- (BOOL)windowShouldClose:(NSWindow*)sender {
    (void)sender;
    if (!state->request_close()) { [self refresh:nil]; return NO; }
    [self.timer invalidate];
    [NSApp stop:nil];
    return YES;
}
- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication*)sender {
    (void)sender;
    if (!state->request_close()) { [self refresh:nil]; return NSTerminateCancel; }
    [self.timer invalidate];
    return NSTerminateNow;
}
@end

int main() {
    std::signal(SIGPIPE, SIG_IGN);
    @autoreleasepool {
        std::string error;
        auto handoff = fox_updater::Handoff::receive(error);
        if (!handoff) { std::cerr << error << '\n'; return 1; }
        if (handoff && handoff->silent()) return handoff->run({}, {}).success ? 0 : 1;
        state = std::make_unique<fox_updater::UiState>(handoff, error);
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        FoxUpdaterHost* host = [FoxUpdaterHost new];
        [NSApp setDelegate:host];
        NSMenu* menu = [NSMenu new];
        NSMenuItem* app = [NSMenuItem new];
        NSMenu* actions = [NSMenu new];
        [actions addItemWithTitle:@"Quit Fox Reader Update" action:@selector(terminate:) keyEquivalent:@"q"];
        app.submenu = actions;
        [menu addItem:app];
        [NSApp setMainMenu:menu];
        [NSApp run];
        state.reset();
        return exit_code;
    }
}
