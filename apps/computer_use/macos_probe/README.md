# Mac permissions probe

This is an opt-in, read-only preparation app for a physical Mac. It is not
the Mac desktop executor and is not evidence of model task success. The
Windows, Linux, and Android drivers and frozen test sources are unchanged.

It uses macOS's built-in JXA runtime and `osacompile`; Command Line Tools,
Python, Homebrew, and a local model are unnecessary for this probe. Build it
as a stay-open application (`osacompile -s -l JavaScript`) with bundle ID
`org.qw3.computeruse.mac-test`, then finish modifying/signing the bundle before
requesting that the user grant permissions. Launch the app in the logged-in
GUI session, not its script through SSH's `osascript`. This gives the user an
identifiable app in Privacy & Security.

The app writes a status snapshot every five seconds in
`~/Library/Application Support/QW3ComputerUseTest/permissions/status.json`.
Create the test runtime with mode 0700 before launch. Snapshots include the
actual bundle path, PID, permission flags, and main display size in points
with its backing scale. `CGPreflightScreenCaptureAccess` checks screen
permission without requesting it. `AXIsProcessTrusted` checks accessibility
permission. Both checks are performed inside the actual GUI application.

No screenshots, app contents, synthetic input, model requests, permission
requests, TCC changes, or system installations occur. Quit from the Dock to
stop it; a stopped app's last snapshot remains on disk, so check its timestamp
and process identity before using it. Relaunch after granting permissions if necessary: permission checks
in a running process may not immediately reflect a change.

Grant Screen Recording and Accessibility to **QW3 Mac Test.app** manually in
System Settings → Privacy & Security. The applet may not automatically appear
because it never requests permission; use the `+` button to add its actual
bundle. A future desktop executor needs its own implementation, permission
and coordinate qualification, durable action receipts, and explicit paused
state. Passing this probe does not enable desktop input.

Primary references:

- [Apple: screen recording permissions](https://support.apple.com/guide/mac-help/mchld6aa7d23/13.0/mac/13.0)
- [Apple: accessibility permissions](https://support.apple.com/guide/mac-help/mh43185/13.0/mac/13.0)
- [Apple: automation scripting](https://developer.apple.com/library/archive/documentation/LanguagesUtilities/Conceptual/MacAutomationScriptingGuide/HowMacScriptingWorks.html)
- The destination Mac's `man osacompile` is the packaging reference.
