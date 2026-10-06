/* A read-only, stay-open JXA applet, built using macOS /usr/bin/osacompile.
 * Deliberately does not capture the screen, post input, invoke other apps,
 * request permissions, change TCC, or install developer tools.
 */
ObjC.import('AppKit');
ObjC.import('ApplicationServices');
ObjC.import('CoreGraphics');
ObjC.bindFunction('CGPreflightScreenCaptureAccess', ['bool', []]);

var expectedBundleId = 'org.qw3.computeruse.mac-test';
var runtime = ObjC.unwrap($.NSHomeDirectory()) +
    '/Library/Application Support/QW3ComputerUseTest/permissions';

function writeStatus(state) {
    var bundle = $.NSBundle.mainBundle;
    var bundleId = ObjC.unwrap(bundle.bundleIdentifier);
    if (bundleId !== expectedBundleId) {
        throw new Error('Run the packaged QW3 Mac Test.app, not this source in osascript.');
    }
    var screen = $.NSScreen.mainScreen;
    var status = {
        protocol: 'qw3-macos-permissions/1',
        timestamp: new Date().toISOString(),
        state: state,
        paused: true,
        read_only: true,
        bundle_id: bundleId,
        bundle_path: ObjC.unwrap(bundle.bundlePath),
        pid: Number($.NSProcessInfo.processInfo.processIdentifier),
        macos_version: ObjC.unwrap($.NSProcessInfo.processInfo.operatingSystemVersionString),
        ax_trusted: !!$.AXIsProcessTrusted(),
        screen_capture_allowed: !!$.CGPreflightScreenCaptureAccess(),
        main_screen: screen ? {
            points: screen.frame,
            backing_scale_factor: Number(screen.backingScaleFactor)
        } : null,
        captures: 0,
        inputs: 0,
        model_calls: 0,
        scope: 'Permission and display metadata only; not a desktop executor qualification.'
    };
    var error = Ref();
    if (!$.NSFileManager.defaultManager.createDirectoryAtPathWithIntermediateDirectoriesAttributesError(
            $(runtime), true, null, error)) {
        throw new Error('Cannot prepare the private permission-probe directory.');
    }
    var text = $(JSON.stringify(status, null, 2) + '\n');
    if (!text.writeToFileAtomicallyEncodingError($(runtime + '/status.json'), true,
            $.NSUTF8StringEncoding, error)) {
        throw new Error('Cannot write the permission status.');
    }
}

function run() {
    writeStatus('running');
}

function idle() {
    writeStatus('running');
    return 5;
}
