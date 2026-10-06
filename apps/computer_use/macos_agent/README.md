# Native Mac candidate

Opt-in physical-Mac desktop adapter. The initial build is qualified on Intel
macOS 13.7.7. It is separate from the frozen Windows/Linux/Android comparison
sources and from the earlier read-only `QW3 Mac Test` permissions probe.

`Main.swift` provides CoreGraphics screenshot/input and bounded AXUIElement
reads. Its authenticated HTTP listener binds **127.0.0.1 only**; use SSH
forwarding to connect from the QW3 host. `macos_agent_driver.py` plugs into the
existing canonical Gateway, while `macos_gateway_entry.py` forwards pause and
resume to both the host and the native executor. Existing `Desktop` supplies
normalized coordinates, observation-bound control IDs, and durable receipts.

Important boundaries:

- Startup and a new owner claim are paused. A concrete Mac gets one native
  executor lock and one input owner. Heartbeat loss expires the owner after
  120 seconds and pauses input. The Mac menu-bar control also pauses input.
- Input requires the current native session/capture, unchanged display mode
  and foreground window, and a configured application plus window-title
  prefix. Main-display screenshots use their real pixel dimensions; the
  native adapter converts to Quartz points. Rotated displays are rejected.
- Native receipts are reserved before input. Host and native input calls
  have no automatic retry. Partial or uncertain delivery stays reserved;
  recovery reads the receipt, rather than resending the action.
- `/frame` records native screenshots without replacing `/observe`'s action
  capture. Recording therefore cannot continually invalidate the model's
  observation IDs. Secure AX fields and their children are omitted.
- The initial named-app action activates prepared TextEdit using the public
  `notepad` alias. Public `CTRL` shortcuts map to Mac Command; ALT maps to
  Option. Drag, mobile operation, general software installation, and a
  complete Mac model task are outside this first qualification.

Build on the target Mac using `build.sh SOURCE.swift NEW_APPLICATION.app`.
The script uses installed Command Line Tools, signs the candidate locally,
verifies its signature, and runs eight checks without posting input. The app
reads its private configuration and token from
`~/Library/Application Support/QW3ComputerUseTest/native/`. Configuration has
`port`, `allowed_apps`, and `window_prefixes`; keep that directory mode 0700
and the authentication token mode 0600. Finish building/signing before the
user grants Screen Recording and Accessibility to the actual final app.

Start the host gateway with `python -m macos_gateway_entry` and its documented
CLI arguments. Run `python -m qualify_macos_native --connection PRIVATE_JSON
--output NEW_DIRECTORY` for physical tool qualification. The private
connection file specifies SSH credentials **by file path**, the forwarded
endpoint, token path, device ID, and a dedicated `QW3ComputerUseTest/tasks/`
directory. The qualifier refuses to overwrite prior output or task files.

The first real qualification passes 15 checks using two new TextEdit files:
Retina coordinates, native control IDs, old capture/control rejection,
foreground/task scope, duplicate delivery, literal Unicode input, actual
disk save/reopen, pause, and recording. Independent audit compares all six
native receipts to both host databases and verifies the paused/released
final state. Nine separate HTTP boundary tests include uncertain delivery,
native restart, failed resume, and unauthorized control.

The first video is timestamped native screenshot sampling (19 frames,
median interval 2.20 seconds), rather than a fixed-rate continuous screen
recording. Neither this scripted tool qualification nor its CPU checks prove
autonomous QW3 task success. Mac device selection and its MCP tool schema
still need a separate opt-in integration before model business evaluation;
do not claim the default application already supports Mac end to end.

Primary references: [Apple Command Line Tools](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools/),
[Quartz Event Services](https://developer.apple.com/documentation/coregraphics/quartz-event-services),
[screen recording](https://support.apple.com/guide/mac-help/mchld6aa7d23/13.0/mac/13.0),
[accessibility permissions](https://support.apple.com/guide/mac-help/mh43185/13.0/mac/13.0).
