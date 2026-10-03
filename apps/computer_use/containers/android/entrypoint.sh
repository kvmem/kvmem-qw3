#!/bin/sh
set -eu
# Project dependencies are mounted read-only. Only /avd and /emulator-user persist.
export ANDROID_AVD_HOME=/avd
export ANDROID_USER_HOME=/emulator-user
export APPIUM_HOME=/emulator-user/appium-home
if [ ! -d "$APPIUM_HOME" ]; then
  cp -R "${QW3_ANDROID_BASE}/appium-home" "$APPIUM_HOME"
fi
export ANDROID_SDK_ROOT="${QW3_ANDROID_BASE}/sdk"
export ANDROID_HOME="$ANDROID_SDK_ROOT"
export JAVA_HOME="${QW3_ANDROID_BASE}/jdk/usr/lib/jvm/java-17-openjdk-amd64"
export PATH="/node/bin:$ANDROID_SDK_ROOT/platform-tools:$PATH"
adb start-server
"$ANDROID_SDK_ROOT/emulator/emulator" -avd qw3-local-api35 -port 5556 \
  -no-window -no-audio -no-boot-anim -gpu swiftshader_indirect \
  -no-snapshot -no-metrics -memory 4096 -cores 4 > /emulator-user/emulator.log 2>&1 &
emulator_pid=$!
"${QW3_ANDROID_BASE}/appium/node_modules/.bin/appium" --address 0.0.0.0 \
  --port 18200 --log /emulator-user/appium.log --log-no-colors --log-level info &
appium_pid=$!
shutdown() {
  trap - EXIT INT TERM
  kill "$appium_pid" "$emulator_pid" 2>/dev/null || true
  wait "$appium_pid" 2>/dev/null || true
  wait "$emulator_pid" 2>/dev/null || true
}
trap 'shutdown' EXIT
trap 'shutdown; exit 0' INT TERM
# Stop the service if either dependency dies; do not advertise a working
# Appium endpoint while the emulator failed to start.
while kill -0 "$emulator_pid" 2>/dev/null && kill -0 "$appium_pid" 2>/dev/null; do
  sleep 1 &
  wait "$!" || true
done
exit 1
