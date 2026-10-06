#!/bin/bash
# Run on the target Mac. Uses Command Line Tools, not a full Xcode installation.
set -euo pipefail
if [ "$(uname -s)" != Darwin ]; then
    echo 'Build this native application on the target Mac.' >&2
    exit 2
fi
if [ "$#" -ne 2 ]; then
    echo 'Usage: build.sh SOURCE.swift NEW_APPLICATION.app' >&2
    exit 2
fi
qw3_source="$1"
qw3_app="$2"
if [ -e "$qw3_app" ]; then
    echo 'Application already exists; use a new candidate path.' >&2
    exit 2
fi
umask 077
mkdir -p "$qw3_app/Contents/MacOS"
/usr/bin/xcrun swiftc -swift-version 5 -O -framework AppKit -framework ApplicationServices \
    -framework CoreGraphics -framework ImageIO -framework Network \
    "$qw3_source" -o "$qw3_app/Contents/MacOS/qw3-mac-agent"
cat > "$qw3_app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>org.qw3.computeruse.mac-agent</string>
<key>CFBundleName</key><string>QW3 Mac Agent</string>
<key>CFBundleDisplayName</key><string>QW3 Mac Agent</string>
<key>CFBundleExecutable</key><string>qw3-mac-agent</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleVersion</key><string>1</string>
<key>CFBundleShortVersionString</key><string>0.1</string>
<key>LSMinimumSystemVersion</key><string>13.0</string>
<key>LSUIElement</key><true/>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
/usr/bin/plutil -lint "$qw3_app/Contents/Info.plist"
/usr/bin/codesign --force --sign - --identifier org.qw3.computeruse.mac-agent "$qw3_app"
/usr/bin/codesign --verify --deep --strict "$qw3_app"
"$qw3_app/Contents/MacOS/qw3-mac-agent" --self-test
