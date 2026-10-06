// Opt-in native macOS test executor. No shell or script action is exposed.
import AppKit
import ApplicationServices
import CoreGraphics
import ImageIO
import Network
import Foundation
import Darwin

typealias Object = [String: Any]
struct Rejection: Error { let message: String }
func reject(_ message: String) throws -> Never { throw Rejection(message: message) }
func jsonData(_ value: Object) throws -> Data {
    try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
}
func atomicJSON(_ value: Object, _ path: URL) throws {
    try jsonData(value).write(to: path, options: .atomic)
    try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: path.path)
    let fd = Darwin.open(path.path, O_RDONLY)
    guard fd >= 0 else { throw POSIXError(.EIO) }
    defer { Darwin.close(fd) }
    guard fsync(fd) == 0 else { throw POSIXError(.EIO) }
}
func integer(_ value: Any?) -> Int? {
    guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
          n.doubleValue.isFinite, n.doubleValue.rounded() == n.doubleValue else { return nil }
    return n.intValue
}
func pixelPoint(_ x: Int, _ y: Int, _ width: Int, _ height: Int, _ bounds: CGRect) throws -> CGPoint {
    guard width > 0, height > 0, bounds.width > 0, bounds.height > 0,
          x >= 0, y >= 0, x < width, y < height else { try reject("Coordinates outside captured display") }
    return CGPoint(x: bounds.minX + CGFloat(x) * bounds.width / CGFloat(width),
                   y: bounds.minY + CGFloat(y) * bounds.height / CGFloat(height))
}
func attribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
    var result: CFTypeRef?
    return AXUIElementCopyAttributeValue(element, name as CFString, &result) == .success ? result : nil
}
func axRect(_ element: AXUIElement) -> CGRect? {
    guard let rawPosition = attribute(element, kAXPositionAttribute),
          let rawSize = attribute(element, kAXSizeAttribute),
          CFGetTypeID(rawPosition) == AXValueGetTypeID(), CFGetTypeID(rawSize) == AXValueGetTypeID() else { return nil }
    var position = CGPoint.zero; var size = CGSize.zero
    guard AXValueGetValue(unsafeBitCast(rawPosition, to: AXValue.self), .cgPoint, &position),
          AXValueGetValue(unsafeBitCast(rawSize, to: AXValue.self), .cgSize, &size) else { return nil }
    return CGRect(origin: position, size: size)
}

struct Display {
    let id: CGDirectDisplayID
    let bounds: CGRect
    let width: Int
    let height: Int
    let modeWidth: Int
    let modeHeight: Int
    var metadata: Object { ["display_id": Int(id), "point_x": bounds.minX, "point_y": bounds.minY,
        "point_width": bounds.width, "point_height": bounds.height, "width": width, "height": height,
        "mode_pixel_width": modeWidth, "mode_pixel_height": modeHeight, "rotation": CGDisplayRotation(id)] }
}
struct Capture {
    let id: String
    let display: Display
    let foreground: Object
    let timestamp: Date
}

final class Agent: NSObject, NSApplicationDelegate {
    let root: URL; let token: String; let port: UInt16
    let session = UUID().uuidString
    let permittedApps: Set<String>; let windowPrefixes: [String]
    var paused = true
    var owner: String?; var heartbeat = Date.distantPast
    var latest: Capture?
    var inputs = 0; var captures = 0
    var listener: NWListener?
    var statusItem: NSStatusItem?
    var lockFD: Int32 = -1
    var timer: Timer?
    var listenerReady = false
    var inputInFlight = false

    init(root: URL, token: String, port: UInt16, permittedApps: [String], prefixes: [String]) {
        self.root = root; self.token = token; self.port = port
        self.permittedApps = Set(permittedApps); self.windowPrefixes = prefixes
    }
    func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            let lease = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support/QW3ComputerUseTest/native-device.lock")
            try FileManager.default.createDirectory(at: lease.deletingLastPathComponent(), withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
            lockFD = Darwin.open(lease.path, O_CREAT | O_RDWR, 0o600)
            guard lockFD >= 0, flock(lockFD, LOCK_EX | LOCK_NB) == 0 else { try reject("Mac already has a native executor") }
            try FileManager.default.createDirectory(at: root.appendingPathComponent("receipts"),
                withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
            let menu = NSMenu()
            let pause = NSMenuItem(title: "Pause QW3 desktop input", action: #selector(pauseInput), keyEquivalent: "")
            pause.target = self; menu.addItem(pause)
            let quit = NSMenuItem(title: "Quit QW3 Mac Test", action: #selector(quitAgent), keyEquivalent: "")
            quit.target = self; menu.addItem(quit); statusItem?.menu = menu
            let parameters = NWParameters.tcp
            parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!)
            listener = try NWListener(using: parameters)
            listener?.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
            listener?.stateUpdateHandler = { [weak self] state in
                DispatchQueue.main.async {
                    if case .ready = state { self?.listenerReady = true }
                    if case .failed(let error) = state {
                        self?.paused = true; self?.failure(error); NSApplication.shared.terminate(nil)
                    }
                }
            }
            listener?.start(queue: DispatchQueue(label: "org.qw3.mac-test.http"))
            timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
                self?.expireOwner(); try? self?.writeStatus()
            }
            try writeStatus()
        } catch { failure(error); NSApplication.shared.terminate(nil) }
    }
    func failure(_ error: Error) {
        try? atomicJSON(["status": "failed", "error": String(describing: error), "session_id": session],
                        root.appendingPathComponent("startup-error.json"))
    }
    @objc func pauseInput() { paused = true; try? writeStatus() }
    @objc func quitAgent() { paused = true; NSApplication.shared.terminate(nil) }
    func applicationWillTerminate(_ notification: Notification) {
        paused = true; listener?.cancel(); timer?.invalidate()
        try? atomicJSON(status().merging(["state": "stopped"]) { _, v in v }, root.appendingPathComponent("status.json"))
        if lockFD >= 0 { Darwin.close(lockFD) }
    }
    func expireOwner() {
        if owner != nil && Date().timeIntervalSince(heartbeat) > 120 {
            paused = true; owner = nil; latest = nil
        }
    }
    func status() -> Object {
        let point = CGEvent(source: nil)?.location ?? .zero
        return ["protocol": "qw3-macos-native/1", "session_id": session, "pid": ProcessInfo.processInfo.processIdentifier,
            "bundle_id": Bundle.main.bundleIdentifier ?? "", "bundle_path": Bundle.main.bundlePath,
            "state": "running", "paused": paused, "owner_present": owner != nil,
            "listener_ready": listenerReady,
            "ax_trusted": AXIsProcessTrusted(), "screen_capture_allowed": CGPreflightScreenCaptureAccess(),
            "synthetic_actions": inputs, "captures": captures, "timestamp": Date().timeIntervalSince1970,
            "pointer": ["x": point.x, "y": point.y], "supported_actions": ["click", "type", "key", "scroll", "open_app"],
            "allowed_apps": Array(permittedApps).sorted(), "allowed_window_prefixes": windowPrefixes]
    }
    func writeStatus() throws {
        statusItem?.button?.title = paused ? "QW3 ∥" : "QW3 ▶"
        try atomicJSON(status(), root.appendingPathComponent("status.json"))
    }
    func foreground() -> Object {
        guard let app = NSWorkspace.shared.frontmostApplication else { return ["process": "", "title": "", "pid": 0] }
        let element = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(element, 0.2)
        let raw = attribute(element, kAXFocusedWindowAttribute)
        let window = raw.flatMap { CFGetTypeID($0) == AXUIElementGetTypeID() ? unsafeBitCast($0, to: AXUIElement.self) : nil }
        let title = window.flatMap { attribute($0, kAXTitleAttribute) as? String } ?? ""
        var result: Object = ["process": app.bundleIdentifier ?? "", "title": title, "pid": Int(app.processIdentifier)]
        if let rect = window.flatMap(axRect) {
            result["bounds"] = ["x": rect.minX, "y": rect.minY, "width": rect.width, "height": rect.height]
            if let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [Object] {
                let match = windows.first { info in
                    guard integer(info[kCGWindowOwnerPID as String]) == Int(app.processIdentifier),
                          integer(info[kCGWindowLayer as String]) == 0,
                          let raw = info[kCGWindowBounds as String] as? [String: Any],
                          let bounds = CGRect(dictionaryRepresentation: raw as CFDictionary) else { return false }
                    return abs(bounds.minX - rect.minX) < 1 && abs(bounds.minY - rect.minY) < 1
                        && abs(bounds.width - rect.width) < 1 && abs(bounds.height - rect.height) < 1
                }
                result["handle"] = match?[kCGWindowNumber as String]
            }
        }
        return result
    }
    func inScope(_ fg: Object) -> Bool {
        guard let app = fg["process"] as? String, let title = fg["title"] as? String else { return false }
        return permittedApps.contains(app) && windowPrefixes.contains { title.hasPrefix($0) }
    }
    func accessibility(_ fg: Object, _ display: Display) -> Object {
        guard AXIsProcessTrusted(), inScope(fg), let pid = integer(fg["pid"]) else {
            return ["status": "outside_task_scope", "controls": [Object](), "scope": "foreground_window"]
        }
        let app = AXUIElementCreateApplication(pid_t(pid)); AXUIElementSetMessagingTimeout(app, 0.2)
        guard let raw = attribute(app, kAXFocusedWindowAttribute), CFGetTypeID(raw) == AXUIElementGetTypeID() else {
            return ["status": "foreground_unmatched", "controls": [Object](), "scope": "foreground_window"]
        }
        let window = unsafeBitCast(raw, to: AXUIElement.self)
        var pending: [(AXUIElement, Int)] = [(window, 0)]
        var controls: [Object] = []; var visited = 0
        let deadline = Date().addingTimeInterval(0.8)
        while !pending.isEmpty && controls.count < 80 && visited < 250 && Date() < deadline {
            let (element, depth) = pending.removeLast(); visited += 1
            let role = attribute(element, kAXRoleAttribute) as? String ?? ""
            let subrole = attribute(element, kAXSubroleAttribute) as? String ?? ""
            // Never traverse secure fields, even if they contain accessible children.
            if role.lowercased().contains("secure") || subrole.lowercased().contains("secure") { continue }
            if let rect = axRect(element), rect.width > 0, rect.height > 0 {
                let visible = rect.intersection(display.bounds)
                let x = (rect.midX - display.bounds.minX) * CGFloat(display.width) / display.bounds.width
                let y = (rect.midY - display.bounds.minY) * CGFloat(display.height) / display.bounds.height
                let enabled = attribute(element, kAXEnabledAttribute) as? Bool ?? true
                if !visible.isNull && enabled && x >= 0 && y >= 0 && x < CGFloat(display.width) && y < CGFloat(display.height) {
                    let name = (attribute(element, kAXTitleAttribute) as? String)
                        ?? (attribute(element, kAXDescriptionAttribute) as? String) ?? role
                    let value = attribute(element, kAXValueAttribute) as? String ?? ""
                    controls.append(["kind": role, "name": String(name.prefix(1024)), "value": String(value.prefix(4096)),
                        "x": Int((rect.minX - display.bounds.minX) * CGFloat(display.width) / display.bounds.width),
                        "y": Int((rect.minY - display.bounds.minY) * CGFloat(display.height) / display.bounds.height),
                        "width": Int(rect.width * CGFloat(display.width) / display.bounds.width),
                        "height": Int(rect.height * CGFloat(display.height) / display.bounds.height),
                        "center_x": Int(x), "center_y": Int(y)])
                }
            }
            if depth < 12, let children = attribute(element, kAXChildrenAttribute) as? [AXUIElement] {
                pending.append(contentsOf: children.prefix(100).reversed().map { ($0, depth + 1) })
            }
        }
        return ["status": "ok", "controls": controls, "scope": "foreground_window", "provider": "native-macos-AXUIElement",
            "truncated": !pending.isEmpty, "visited": visited, "read_deadline_seconds": 0.8]
    }
    func captureImage() throws -> (Data, Display) {
        guard CGPreflightScreenCaptureAccess() else { try reject("Mac screen permission is not granted to the executor") }
        let id = CGMainDisplayID(); let bounds = CGDisplayBounds(id)
        guard CGDisplayRotation(id) == 0, let image = CGDisplayCreateImage(id) else { try reject("Main display capture unavailable or rotated") }
        guard let mode = CGDisplayCopyDisplayMode(id) else { try reject("Display mode unavailable") }
        let display = Display(id: id, bounds: bounds, width: image.width, height: image.height,
                              modeWidth: mode.pixelWidth, modeHeight: mode.pixelHeight)
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil) else { try reject("PNG encoder unavailable") }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { try reject("PNG encoding failed") }
        captures += 1
        return (data as Data, display)
    }
    func frame() throws -> Object {
        // Recording is read-only and must never replace the action's current capture.
        let (data, display) = try captureImage()
        return ["png": data.base64EncodedString(), "geometry": display.metadata,
                "frame_only": true, "native_session_id": session, "timestamp": Date().timeIntervalSince1970]
    }
    func observe() throws -> Object {
        guard !inputInFlight else { try reject("Input is in progress; observe after its receipt") }
        guard AXIsProcessTrusted() else { try reject("Mac accessibility permission is not granted to the executor") }
        let before = foreground(); let (data, display) = try captureImage()
        let ax = accessibility(before, display); let after = foreground()
        let idCapture = UUID().uuidString
        latest = Capture(id: idCapture, display: display, foreground: after, timestamp: Date())
        return ["png": data.base64EncodedString(), "foreground": after, "geometry": display.metadata,
            "native_capture_id": idCapture, "native_session_id": session, "device_paused": paused,
            "accessibility": NSDictionary(dictionary: before).isEqual(to: after) ? ax : ["status": "foreground_changed", "controls": [Object]()]]
    }
    func assertOwner(_ body: Object) throws {
        expireOwner()
        guard let client = body["owner"] as? String, client == owner, owner != nil else { try reject("Mac input lease is not owned by this client") }
        heartbeat = Date()
    }
    func checkCapture(_ body: Object, _ action: Object) throws -> Capture {
        guard !paused else { try reject("Mac is paused") }
        guard AXIsProcessTrusted(), CGPreflightScreenCaptureAccess() else { try reject("Mac permissions changed") }
        guard let capture = latest, body["capture_id"] as? String == capture.id,
              body["session_id"] as? String == session, Date().timeIntervalSince(capture.timestamp) < 60 else {
            try reject("Native capture identity is stale")
        }
        guard CGMainDisplayID() == capture.display.id,
              CGDisplayBounds(capture.display.id) == capture.display.bounds,
              CGDisplayCopyDisplayMode(capture.display.id)?.pixelWidth == capture.display.modeWidth,
              CGDisplayCopyDisplayMode(capture.display.id)?.pixelHeight == capture.display.modeHeight,
              CGDisplayRotation(capture.display.id) == 0 else { try reject("Display geometry changed") }
        if action["kind"] as? String != "open_app" {
            let now = foreground()
            guard NSDictionary(dictionary: now).isEqual(to: capture.foreground) else { try reject("Foreground changed; no input sent") }
            guard inScope(now) else { try reject("Foreground application or window is outside the allowed task scope") }
            guard integer(now["handle"]) != nil else { try reject("Focused window identity is unavailable") }
        }
        return capture
    }
    func events(_ action: Object, _ capture: Capture) throws -> [CGEvent] {
        let source = CGEventSource(stateID: .hidSystemState)
        func point() throws -> CGPoint {
            guard let x = integer(action["x"]), let y = integer(action["y"]) else { try reject("Pixel coordinates must be integers") }
            return try pixelPoint(x, y, capture.display.width, capture.display.height, capture.display.bounds)
        }
        func keyboard(_ code: CGKeyCode, _ flags: CGEventFlags = []) throws -> [CGEvent] {
            guard let down = CGEvent(keyboardEventSource: source, virtualKey: code, keyDown: true),
                  let up = CGEvent(keyboardEventSource: source, virtualKey: code, keyDown: false) else { try reject("Cannot create keyboard event") }
            down.flags = flags; up.flags = flags; return [down, up]
        }
        switch action["kind"] as? String {
        case "click":
            let p = try point(); let right = action["button"] as? String == "right"
            guard ["left", "right"].contains(action["button"] as? String ?? ""),
                  let count = integer(action["clicks"]), [1, 2].contains(count) else { try reject("Invalid mouse click") }
            var result: [CGEvent] = []
            for i in 1...count {
                guard let down = CGEvent(mouseEventSource: source, mouseType: right ? .rightMouseDown : .leftMouseDown, mouseCursorPosition: p, mouseButton: right ? .right : .left),
                      let up = CGEvent(mouseEventSource: source, mouseType: right ? .rightMouseUp : .leftMouseUp, mouseCursorPosition: p, mouseButton: right ? .right : .left) else { try reject("Cannot create mouse event") }
                down.setIntegerValueField(.mouseEventClickState, value: Int64(i)); up.setIntegerValueField(.mouseEventClickState, value: Int64(i))
                result += [down, up]
            }
            return result
        case "key":
            let (code, flags) = try keySpec(action["key"] as? String ?? "")
            return try keyboard(code, flags)
        case "type":
            guard let text = action["text"] as? String, !text.isEmpty, text.count <= 4096 else { try reject("Invalid text length") }
            var chunks: [[UniChar]] = []; var chunk: [UniChar] = []
            for scalar in text.unicodeScalars {
                let units = Array(String(scalar).utf16)
                if chunk.count + units.count > 20 { chunks.append(chunk); chunk = [] }
                chunk += units
            }
            if !chunk.isEmpty { chunks.append(chunk) }
            var result: [CGEvent] = []
            for units in chunks {
                let pair = try keyboard(0)
                units.withUnsafeBufferPointer { pointer in
                    pair[0].keyboardSetUnicodeString(stringLength: units.count, unicodeString: pointer.baseAddress!)
                    pair[1].keyboardSetUnicodeString(stringLength: units.count, unicodeString: pointer.baseAddress!)
                }
                result += pair
            }
            return result
        case "scroll":
            let p = try point(); let direction = action["direction"] as? String ?? ""
            guard let amount = integer(action["steps"]), amount >= 1, amount <= 8,
                  ["up", "down"].contains(direction) else { try reject("Invalid scroll") }
            let horizontal = direction == "left" || direction == "right"
            let signed = Int32(amount * (["down", "right"].contains(direction) ? -1 : 1))
            guard let move = CGEvent(mouseEventSource: source, mouseType: .mouseMoved, mouseCursorPosition: p, mouseButton: .left),
                  let scroll = CGEvent(scrollWheelEvent2Source: source, units: .line, wheelCount: 2,
                    wheel1: horizontal ? 0 : signed, wheel2: horizontal ? signed : 0, wheel3: 0) else { try reject("Cannot create scroll event") }
            return [move, scroll]
        default: try reject("Unsupported native action")
        }
    }
    func act(_ body: Object) throws -> Object {
        try assertOwner(body)
        guard !inputInFlight else { try reject("An input request is already in progress") }
        guard let id = body["request_id"] as? String, UUID(uuidString: id) != nil,
              let action = body["action"] as? Object else { try reject("Malformed native request") }
        let receipt = root.appendingPathComponent("receipts/" + id + ".json")
        guard !FileManager.default.fileExists(atPath: receipt.path) else { try reject("Native request identity already reserved; never resend") }
        let capture = try checkCapture(body, action)
        let kind = action["kind"] as? String
        var app: NSRunningApplication?
        var plan: [CGEvent] = []
        if kind == "open_app" {
            guard action["app"] as? String == "notepad", permittedApps.contains("com.apple.TextEdit"),
                  let running = NSWorkspace.shared.runningApplications.first(where: { $0.bundleIdentifier == "com.apple.TextEdit" }) else {
                try reject("Only the prepared TextEdit test application can be activated")
            }
            app = running
        } else { plan = try events(action, capture) }
        // All checks and event allocation finish before a durable reservation.
        try atomicJSON(["request_id": id, "status": "dispatching", "session_id": session, "action": action,
            "capture_id": capture.id, "reserved_at": Date().timeIntervalSince1970], receipt)
        inputInFlight = true
        defer { inputInFlight = false }
        do { _ = try checkCapture(body, action) } catch {
            try atomicJSON(["request_id": id, "status": "rejected", "delivery": "rejected", "reason": String(describing: error)], receipt)
            throw error
        }
        if let application = app {
            guard application.activate(options: [.activateIgnoringOtherApps]) else {
                let result: Object = ["request_id": id, "status": "uncertain", "delivery": "uncertain", "reason": "App activation result unknown"]
                paused = true; try atomicJSON(result, receipt); return result
            }
        } else {
            for (index, event) in plan.enumerated() {
                if index % 2 == 0 {
                    // Service local pause/menu and foreground changes between complete event pairs.
                    _ = RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.001))
                    let now = foreground()
                    if paused || integer(now["handle"]) != integer(capture.foreground["handle"])
                        || integer(now["pid"]) != integer(capture.foreground["pid"]) || !inScope(now)
                        || CGDisplayBounds(capture.display.id) != capture.display.bounds {
                        if index == 0 {
                            try atomicJSON(["request_id": id, "status": "rejected", "delivery": "rejected"], receipt)
                            try reject("Paused or focused window changed immediately before input")
                        }
                        paused = true; latest = nil
                        let result: Object = ["request_id": id, "status": "uncertain", "delivery": "uncertain", "posted_events": index]
                        try atomicJSON(result, receipt); return result
                    }
                }
                event.post(tap: .cghidEventTap); usleep(8_000)
            }
        }
        inputs += 1; latest = nil
        let result: Object = ["request_id": id, "status": "completed", "delivery": "delivered",
            "verification": "Input delivery only; independently check the resulting state", "native_session_id": session,
            "posted_events": plan.count, "action": action, "completed_at": Date().timeIntervalSince1970]
        try atomicJSON(result, receipt); return result
    }
    func route(_ method: String, _ path: String, _ body: Object) throws -> Object {
        switch (method, path) {
        case ("GET", "/status"): expireOwner(); return status()
        case ("GET", "/observe"): return try observe()
        case ("GET", "/frame"): return try frame()
        case ("POST", "/request_permissions"):
            _ = CGRequestScreenCaptureAccess()
            let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
            _ = AXIsProcessTrustedWithOptions(options)
            return status()
        case ("POST", "/claim"):
            expireOwner()
            guard let client = body["owner"] as? String, UUID(uuidString: client) != nil,
                  owner == nil || owner == client else { try reject("Mac input owner already exists") }
            owner = client; heartbeat = Date(); paused = true; latest = nil; return status()
        case ("POST", "/heartbeat"): try assertOwner(body); return status()
        case ("POST", "/release"): try assertOwner(body); paused = true; owner = nil; latest = nil; return status()
        case ("POST", "/control"):
            try assertOwner(body)
            guard let command = body["command"] as? String, ["pause", "resume", "stop"].contains(command) else { try reject("Invalid control") }
            paused = command != "resume"; return status()
        case ("POST", "/preflight"):
            guard !inputInFlight else { try reject("Input is in progress") }
            try assertOwner(body); guard let action = body["action"] as? Object else { try reject("Missing action") }
            let capture = try checkCapture(body, action)
            if action["kind"] as? String != "open_app" { _ = try events(action, capture) }
            return ["delivery": "not_sent", "ready": true]
        case ("POST", "/act"): return try act(body)
        default:
            if method == "GET", path.hasPrefix("/receipts/"), let id = path.split(separator: "/").last,
               UUID(uuidString: String(id)) != nil {
                let receipt = root.appendingPathComponent("receipts/" + String(id) + ".json")
                if let data = try? Data(contentsOf: receipt), let result = try JSONSerialization.jsonObject(with: data) as? Object { return result }
                return ["status": "unknown"]
            }
            try reject("Unsupported endpoint")
        }
    }
    func accept(_ connection: NWConnection) {
        connection.start(queue: DispatchQueue(label: "org.qw3.mac-test.connection"))
        var input = Data(); var handled = false
        let deadline = DispatchWorkItem { connection.cancel() }
        DispatchQueue.global().asyncAfter(deadline: .now() + 30, execute: deadline)
        func receive() {
            connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, complete, error in
                if handled { return }
                if let bytes = data { input.append(bytes) }
                guard error == nil, input.count <= 131072, let agent = self else { handled = true; deadline.cancel(); connection.cancel(); return }
                guard let range = input.range(of: Data("\r\n\r\n".utf8)) else {
                    if complete { connection.cancel() } else { receive() }; return
                }
                guard let header = String(data: input[..<range.lowerBound], encoding: .utf8) else { connection.cancel(); return }
                let lines = header.components(separatedBy: "\r\n")
                let parts = lines[0].split(separator: " ")
                guard parts.count == 3 else { connection.cancel(); return }
                var headers: [String: String] = [:]
                for line in lines.dropFirst() {
                    guard let colon = line.firstIndex(of: ":") else { continue }
                    let key = line[..<colon].lowercased()
                    guard headers[key] == nil else { connection.cancel(); return }
                    headers[key] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
                }
                guard headers["transfer-encoding"] == nil else { connection.cancel(); return }
                let length = Int(headers["content-length"] ?? "0") ?? -1
                guard length >= 0, length <= 128 * 1024 else { connection.cancel(); return }
                let tail = input[range.upperBound...]
                if tail.count < length { if complete { connection.cancel() } else { receive() }; return }
                handled = true; deadline.cancel()
                DispatchQueue.main.async {
                    var result: Object; var code = 200
                    if headers["authorization"] != "Bearer " + agent.token {
                        result = ["error": "Unauthorized"]; code = 401
                    } else {
                        do {
                            let body = length == 0 ? [:] : try JSONSerialization.jsonObject(with: Data(tail.prefix(length))) as? Object
                            guard let value = body else { try reject("JSON object required") }
                            result = try agent.route(String(parts[0]), String(parts[1]), value)
                        } catch let error as Rejection {
                            result = ["delivery": "rejected", "error": error.message]; code = 409
                        } catch {
                            agent.paused = true; result = ["delivery": "uncertain", "error": String(describing: error)]; code = 503
                        }
                    }
                    let data = (try? jsonData(result)) ?? Data("{}".utf8)
                    let header = "HTTP/1.1 \(code) Response\r\nContent-Type: application/json\r\nContent-Length: \(data.count)\r\nConnection: close\r\n\r\n"
                    connection.send(content: Data(header.utf8) + data, completion: .contentProcessed { _ in connection.cancel() })
                }
            }
        }
        receive()
    }
}

func keySpec(_ key: String) throws -> (CGKeyCode, CGEventFlags) {
    let codes: [String: CGKeyCode] = ["A": 0, "S": 1, "D": 2, "F": 3, "H": 4, "G": 5, "Z": 6, "X": 7,
        "C": 8, "V": 9, "B": 11, "Q": 12, "W": 13, "E": 14, "R": 15, "Y": 16, "T": 17,
        "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26,
        "8": 28, "0": 29, "O": 31, "U": 32, "I": 34, "P": 35, "L": 37, "J": 38, "K": 40,
        "N": 45, "M": 46, "ENTER": 36, "TAB": 48, "SPACE": 49, "BACKSPACE": 51, "ESC": 53,
        "HOME": 115, "PAGEUP": 116, "DELETE": 117, "END": 119, "PAGEDOWN": 121,
        "LEFT": 123, "RIGHT": 124, "DOWN": 125, "UP": 126, "F2": 120, "F4": 118, "F5": 96,
        "F6": 97, "F11": 103, "F12": 111]
    let components = key.split(separator: "+").map(String.init)
    guard let last = components.last, let code = codes[last] else { try reject("Unsupported Mac key") }
    var flags: CGEventFlags = []
    for modifier in components.dropLast() {
        switch modifier { case "CTRL": flags.insert(.maskCommand)
        case "SHIFT": flags.insert(.maskShift)
        case "ALT": flags.insert(.maskAlternate)
        default: try reject("Unsupported Mac modifier") }
    }
    return (code, flags)
}

if CommandLine.arguments.contains("--self-test") {
    var checks: [String: Bool] = [:]
    checks["retina-two"] = (try? pixelPoint(1600, 800, 3360, 2100, CGRect(x: 0, y: 0, width: 1680, height: 1050))) == CGPoint(x: 800, y: 400)
    checks["offset-display"] = (try? pixelPoint(200, 100, 800, 600, CGRect(x: -400, y: 50, width: 400, height: 300))) == CGPoint(x: -300, y: 100)
    checks["right-edge-rejected"] = (try? pixelPoint(3360, 0, 3360, 2100, CGRect(x: 0, y: 0, width: 1680, height: 1050))) == nil
    checks["negative-rejected"] = (try? pixelPoint(-1, 0, 3360, 2100, CGRect(x: 0, y: 0, width: 1680, height: 1050))) == nil
    checks["boolean-coordinate-rejected"] = integer(true) == nil
    checks["fractional-coordinate-rejected"] = integer(1.5) == nil
    checks["ctrl-is-command"] = (try? keySpec("CTRL+S").1) == .maskCommand
    checks["android-key-rejected"] = (try? keySpec("ANDROID_BACK")) == nil
    print(String(data: try jsonData(["checks": checks, "passed": checks.values.allSatisfy { $0 }, "posted_events": 0]), encoding: .utf8)!)
    exit(checks.values.allSatisfy { $0 } ? 0 : 1)
}
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let configURL = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library/Application Support/QW3ComputerUseTest/native/config.json")
do {
    let config = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as! Object
    let root = configURL.deletingLastPathComponent()
    let token = try String(contentsOf: root.appendingPathComponent("token.txt"), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
    guard token.count >= 32, let port = integer(config["port"]), let value = UInt16(exactly: port),
          let allowed = config["allowed_apps"] as? [String], let prefixes = config["window_prefixes"] as? [String],
          !allowed.isEmpty, !prefixes.isEmpty, prefixes.allSatisfy({ !$0.isEmpty }) else { try reject("Invalid Mac test configuration") }
    let delegate = Agent(root: root, token: token, port: value, permittedApps: allowed, prefixes: prefixes)
    app.delegate = delegate
    withExtendedLifetime(delegate) { app.run() }
} catch { fputs("QW3 native configuration unavailable: \(error)\n", stderr); exit(1) }
