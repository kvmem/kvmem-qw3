#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/chaidi/qw3
eval_root=$repo_root/benchmark/claude_qwen38_eval
task_id=${1:?usage: run_deepswe_cell.sh TASK_ID CELL_DIR}
cell_dir=${2:?usage: run_deepswe_cell.sh TASK_ID CELL_DIR}
task_dir=$repo_root/benchmark/deep-swe/tasks/$task_id
agent_runner=${CLAUDE_AGENT_RUNNER:-$eval_root/container_claude_agent.sh}
endpoint=${QWEN38_CLAUDE_API_BASE:-http://127.0.0.1:18314}
model_id=${QWEN38_CLAUDE_MODEL_ID:-Qwen3.8-27B}
precision_label=${QWEN38_PRECISION_LABEL:-BF16}
agent_timeout=${QWEN38_CLAUDE_AGENT_TIMEOUT:-0}
agent_memory=${QWEN38_CLAUDE_AGENT_MEMORY:-16g}
claude_context_tokens=${QWEN38_CLAUDE_CONTEXT_TOKENS:-1048576}
disable_compact=${QWEN38_CLAUDE_DISABLE_COMPACT:-1}
auto_compact_window=${QWEN38_CLAUDE_AUTO_COMPACT_WINDOW:-}
verifier_timeout=${QWEN38_CLAUDE_VERIFIER_TIMEOUT:-30m}
shared_vllm_log=${QWEN38_SHARED_VLLM_LOG:-$repo_root/benchmark/opencode_qwen38_vllm_screen/recordings/qwen38_bf16_screen_20260815_174943/server/vllm.log}
condition=${QWEN38_CLAUDE_CONDITION:-claude_code_256k}
agent_network=${QWEN38_AGENT_NETWORK:?QWEN38_AGENT_NETWORK is required}
agent_network_policy=${QWEN38_AGENT_NETWORK_POLICY:-docker_internal_bridge_no_wan}

network_internal=$(docker network inspect "$agent_network" --format '{{.Internal}}' 2>/dev/null || true)
case "$agent_network_policy" in
  docker_internal_bridge_no_wan)
    expected_internal=true
    ;;
  docker_bridge_wan_audited)
    expected_internal=false
    ;;
  *)
    printf 'Unknown Agent network policy %q\n' "$agent_network_policy" >&2
    exit 78
    ;;
esac
if [[ "$network_internal" != "$expected_internal" ]]; then
  printf 'Refusing Agent network %q: policy=%q requires Internal=%q, got %q\n' \
    "$agent_network" "$agent_network_policy" "$expected_internal" "$network_internal" >&2
  exit 78
fi

readarray -t metadata < <(python3 - "$task_dir/task.toml" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
def value(key):
    match = re.search(rf'^\s*{re.escape(key)}\s*=\s*"([^"]*)"\s*$', text, re.MULTILINE)
    if not match:
        raise SystemExit(f"missing {key}")
    return match.group(1)
for key in ("docker_image", "base_commit_hash", "display_title", "language"):
    print(value(key))
PY
)
image=${metadata[0]}
base_commit=${metadata[1]}
display_title=${metadata[2]}
language=${metadata[3]}
safe_name=$(printf '%s' "$task_id" | tr -c 'a-zA-Z0-9_.-' '-')
# Several seed/GPU workers share the host Docker daemon.  A task-only name lets
# two workers running the same task collide, and (more seriously) lets one
# worker's EXIT trap kill the other worker's container.  Keep the readable task
# prefix, but namespace both containers by the immutable arm artifact path.
run_owner=$(printf '%s' "$cell_dir" | sha256sum | cut -c1-16)
container_task=${safe_name:0:64}
container_name=qwen38-claude-${container_task}-${run_owner}
verifier_name=${container_name}-verifier
verifier_image=${QWEN38_VERIFIER_IMAGE:-qwen38-vllm-screen-${task_id}-verifier:v1.1}
container_owner_label=com.qw3.deepswe.run-owner
container_role_label=com.qw3.deepswe.role
agent_cidfile=$cell_dir/agent.container.cid
verifier_cidfile=$cell_dir/verifier.container.cid

mkdir -p "$cell_dir/verifier" "$cell_dir/server"
cp "$task_dir/instruction.md" "$cell_dir/prompt.md"
ln -sfn claude.stream.jsonl "$cell_dir/opencode.log"
ln -sfn "$shared_vllm_log" "$cell_dir/server/qw3_server.log"

write_state() {
  local state=$1 detail=${2:-}
  python3 - "$cell_dir/status.json" "$task_id" "$display_title" "$language" \
    "$condition" "$state" "$detail" <<'PY'
import json, os, sys, time
p=sys.argv[1]
x={"task_id":sys.argv[2],"display_title":sys.argv[3],"language":sys.argv[4],
   "condition":sys.argv[5],"state":sys.argv[6],"detail":sys.argv[7],
   "updated_at":time.time()}
t=p+".tmp"; json.dump(x,open(t,"w"),indent=2); os.replace(t,p)
PY
}

cleanup_owned_container() {
  local role=$1 cidfile=$2 cid actual_owner actual_role
  [[ -s "$cidfile" ]] || return 0
  cid=$(tr -d '[:space:]' <"$cidfile")
  if [[ ! "$cid" =~ ^[0-9a-f]{12,64}$ ]]; then
    printf 'Refusing to clean malformed %s container id %q\n' "$role" "$cid" >&2
    return 0
  fi
  actual_owner=$(docker inspect --format \
    "{{ index .Config.Labels \"$container_owner_label\" }}" \
    "$cid" 2>/dev/null || true)
  actual_role=$(docker inspect --format \
    "{{ index .Config.Labels \"$container_role_label\" }}" \
    "$cid" 2>/dev/null || true)
  if [[ "$actual_owner" == "$run_owner" && "$actual_role" == "$role" ]]; then
    docker kill "$cid" >/dev/null 2>&1 || true
  elif [[ -n "$actual_owner" || -n "$actual_role" ]]; then
    printf 'Refusing to clean foreign container %s: owner=%q role=%q\n' \
      "$cid" "$actual_owner" "$actual_role" >&2
  fi
  rm -f "$cidfile"
}

cleanup() {
  cleanup_owned_container agent "$agent_cidfile"
  cleanup_owned_container verifier "$verifier_cidfile"
}

# A signal trap that only runs cleanup and then returns lets Bash resume after
# the interrupted `docker run`; it may proceed into verification or leave the
# container detached from its dead supervisor.  Exit explicitly after cleaning
# the containers owned by this arm.  The EXIT trap is retained as the ordinary
# completion backstop.
interrupt() {
  local status=$1
  trap - INT TERM
  cleanup
  exit "$status"
}
trap cleanup EXIT
trap 'interrupt 130' INT
trap 'interrupt 143' TERM

cell_started=$(date --iso-8601=seconds)
cell_start_ns=$(date +%s%N)
server_start_offset=$(stat -c '%s' "$shared_vllm_log" 2>/dev/null || printf 0)
printf '%s\n' "$server_start_offset" >"$cell_dir/timed_server_start.offset"
if [[ -z "$agent_timeout" || "$agent_timeout" == 0 ||
      "$agent_timeout" == none || "$agent_timeout" == unlimited ]]; then
  agent_deadline_detail='no agent wall-time deadline; waiting for Claude Code to finish naturally'
else
  agent_deadline_detail="agent wall-time deadline: $agent_timeout"
fi
write_state running_agent "Claude Code is using Qwen3.8-27B ${precision_label}; ${agent_deadline_detail}"
rm -f "$agent_cidfile"
set +e
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy \
  docker run --rm \
    --name "$container_name" \
    --cidfile "$agent_cidfile" \
    --label "$container_owner_label=$run_owner" \
    --label "$container_role_label=agent" \
    --network "$agent_network" \
    --cpus 2 \
    --memory "$agent_memory" \
    -e BASE_COMMIT="$base_commit" \
    -e CLAUDE_API_BASE="$endpoint" \
    -e CLAUDE_MODEL_ID="$model_id" \
    -e CLAUDE_BRANCH_NAME="qwen38-claude-$safe_name" \
    -e CLAUDE_MAX_WALL_TIME="$agent_timeout" \
    -e CLAUDE_CODE_MAX_CONTEXT_TOKENS="$claude_context_tokens" \
    -e CLAUDE_DISABLE_COMPACT="$disable_compact" \
    -e CLAUDE_AUTO_COMPACT_WINDOW="$auto_compact_window" \
    -e CLAUDE_AGENT_NETWORK_POLICY="$agent_network_policy" \
    -v /home/chaidi/.local/share/claude/versions/2.1.148:/usr/local/bin/claude:ro \
    -v "$agent_runner:/runner.sh:ro" \
    -v "$task_dir/instruction.md:/prompt.md:ro" \
    -v "$cell_dir:/artifacts" \
    "$image" /bin/bash -lc \
      'preflight=/artifacts/network_preflight.log
       : >"$preflight"
       network_policy=${CLAUDE_AGENT_NETWORK_POLICY:-docker_internal_bridge_no_wan}
       printf "policy=%s\n" "$network_policy" >>"$preflight"
       if curl --noproxy "*" -fsS --connect-timeout 2 --max-time 5 "$CLAUDE_API_BASE/health" >/dev/null; then
         printf "local_model_endpoint=reachable\n" >>"$preflight"
       else
         printf "local_model_endpoint=unreachable\n" >>"$preflight"
         exit 79
       fi
       if [[ "$network_policy" == docker_bridge_wan_audited ]]; then
         if getent ahostsv4 proxy.golang.org >/dev/null 2>&1; then
           printf "external_dns=reachable\n" >>"$preflight"
         else
           printf "external_dns=blocked\n" >>"$preflight"
           exit 81
         fi
         if curl --noproxy "*" -fsS --connect-timeout 5 --max-time 15 https://proxy.golang.org/ >/dev/null 2>&1; then
           printf "external_https=reachable\n" >>"$preflight"
         else
           printf "external_https=blocked\n" >>"$preflight"
           exit 82
         fi
       else
         if curl --noproxy "*" -sS --connect-timeout 2 --max-time 3 http://1.1.1.1/ >/dev/null 2>&1; then
           printf "external_ip=reachable\n" >>"$preflight"
           exit 80
         else
           printf "external_ip=blocked\n" >>"$preflight"
         fi
         if getent ahostsv4 registry.npmjs.org >/dev/null 2>&1; then
           printf "external_dns=reachable\n" >>"$preflight"
           exit 81
         else
           printf "external_dns=blocked\n" >>"$preflight"
         fi
       fi
       dependency_preflight=/artifacts/dependency_preflight.log
       : >"$dependency_preflight"
       chmod -R a+rwX /artifacts
       mkdir -p /artifacts/home/.cache/go-build
       chown -R nobody:nogroup /artifacts/home

       # The image worktree belongs to root.  Walking it and changing every
       # owner forces a large overlayfs metadata copy-up on repositories with
       # many files.  Run the Agent as nobody with only DAC override instead:
       # Claude still observes a non-root uid, while its subprocesses can edit
       # the disposable container worktree and dependency tools can create
       # transient files such as Vitest tsbuildinfo artifacts.
       if ! command -v setpriv >/dev/null 2>&1; then
         printf "status=failed\nreason=missing_setpriv\n" >>"$dependency_preflight"
         exit 84
       fi
       printf "agent_fs_mode=nobody_cap_dac_override\n" >>"$dependency_preflight"

       # Official DeepSWE images install dependencies as root at image-build
       # time, then run the task air-gapped.  Claude Code itself cannot run
       # --dangerously-skip-permissions as root, so explicitly expose those
       # immutable caches to the nobody Agent instead of silently changing
       # HOME and losing them (the Termenv image keeps uniseg under /root/go).
       if [[ -f /app/go.mod ]]; then
         root_gopath=$(HOME=/root go env GOPATH)
         root_gomodcache=$(HOME=/root go env GOMODCACHE)
         if [[ ! -d "$root_gomodcache" ]]; then
           printf "type=go\nstatus=failed\nreason=missing_preinstalled_gomodcache\n" >>"$dependency_preflight"
           exit 83
         fi
         chmod o+x /root
         export GOPATH="$root_gopath"
         export GOMODCACHE="$root_gomodcache"
         export GOCACHE=/artifacts/home/.cache/go-build
         export GOPROXY=off
         # Some pinned task images use an older bootstrap `go` binary but
         # preinstall the newer toolchain required by go.mod in GOMODCACHE.
         # `local` incorrectly rejects those valid air-gapped images before
         # Claude starts.  `auto` may use only that cached toolchain here:
         # the Agent container is network-isolated and GOPROXY remains off,
         # so a missing toolchain still fails this authoritative preflight.
         export GOTOOLCHAIN=auto
         # The disposable DeepSWE worktree may expose repository metadata that
         # the unprivileged Agent cannot use for Go VCS stamping.  Dependency
         # resolution does not require build provenance, and treating a Git
         # status failure as a missing offline module cache aborts an otherwise
         # valid arm before Claude starts.  Disable only VCS stamping while
         # retaining the authoritative readonly/offline dependency check.
         export GOFLAGS="${GOFLAGS:+$GOFLAGS }-buildvcs=false"
         printf "type=go\ngopath=%s\ngomodcache=%s\ngotoolchain=%s\n" \
           "$GOPATH" "$GOMODCACHE" "$GOTOOLCHAIN" >>"$dependency_preflight"
         # Validate dependencies without requiring every repository package to
         # build for the host GOOS.  Valid DeepSWE repositories may contain
         # browser-only packages importing syscall/js; plain `go list` reports
         # those as package errors even though all offline modules are present.
         # Keep the strict list first, but fail only for dependency/cache
         # diagnostics.  Other package/build-tag errors belong to the task and
         # must not abort Claude before it can edit the repository.
         go_list_log=/artifacts/dependency_preflight.go-list.log
         if su -m -s /bin/bash nobody -c \
              "cd /app && go list -mod=readonly -deps ./... >/dev/null" \
              >"$go_list_log" 2>&1; then
           printf "package_scan=clean\nstatus=passed\n" >>"$dependency_preflight"
         elif grep -Eiq \
              "module lookup disabled|missing go.sum entry|no required module provides package|cannot find module providing package|go: downloading|unrecognized import path|module .*: not a known dependency" \
              "$go_list_log"; then
           cat "$go_list_log" >&2
           printf "status=failed\nreason=offline_go_dependency_resolution\n" >>"$dependency_preflight"
           exit 83
         else
           cat "$go_list_log" >&2
           printf "package_scan=non_dependency_errors_ignored\nstatus=passed\n" \
             >>"$dependency_preflight"
         fi
       elif [[ -f /app/Cargo.toml ]]; then
         chmod o+x /root
         if [[ -d /root/.cargo ]]; then
           export CARGO_HOME=/root/.cargo
         fi
         if [[ -d /root/.rustup ]]; then
           export RUSTUP_HOME=/root/.rustup
         fi
         # These caches are immutable image layers and are already installed
         # with world-readable files/directories.  Recursively changing them
         # here forces tens of thousands of overlayfs copy-ups on every Rust
         # sample and can stall the container before Claude starts.  The
         # metadata command below is the authoritative access check: fail the
         # preflight if an image ships an unreadable cache instead of rewriting
         # the entire cache at runtime.
         printf "type=rust\n" >>"$dependency_preflight"
         if su -m -s /bin/bash nobody -c "cd /app && cargo metadata --offline --no-deps --format-version 1 >/dev/null"; then
           printf "status=passed\n" >>"$dependency_preflight"
         else
           printf "status=failed\nreason=offline_cargo_dependency_resolution\n" >>"$dependency_preflight"
           exit 83
         fi
       elif [[ -f /app/package.json ]]; then
         printf "type=node\n" >>"$dependency_preflight"
         if [[ -d /app/node_modules ]]; then
           printf "node_modules=present\nstatus=passed\n" >>"$dependency_preflight"
         else
           printf "node_modules=absent\nstatus=unchecked\n" >>"$dependency_preflight"
         fi
       else
         printf "type=other\nstatus=unchecked\n" >>"$dependency_preflight"
       fi
       exec setpriv --reuid nobody --regid nogroup --clear-groups \
         --inh-caps +dac_override --ambient-caps +dac_override \
         /bin/bash /runner.sh' \
    2>&1 | tee "$cell_dir/agent.console.log"
agent_status=${PIPESTATUS[0]}
set -e

# The service is persistent across cells.  Freeze only the byte range produced
# by this Agent arm so prefill/decode/cache/reselection metrics are not mixed
# with model startup or adjacent samples.
previous_size=-1
stable_reads=0
while [[ "$stable_reads" -lt 10 ]]; do
  current_size=$(stat -c '%s' "$shared_vllm_log" 2>/dev/null || printf 0)
  if [[ "$current_size" == "$previous_size" ]]; then
    stable_reads=$((stable_reads + 1))
  else
    stable_reads=0
  fi
  previous_size=$current_size
  sleep 0.1
done
printf '%s\n' "$current_size" >"$cell_dir/timed_server_end.offset"
python3 - "$shared_vllm_log" "$server_start_offset" "$current_size" \
  "$cell_dir/qw3_server.timed.log" <<'PY'
from pathlib import Path
import sys

source, lo, hi, output = sys.argv[1:]
with Path(source).open("rb") as stream:
    stream.seek(int(lo))
    Path(output).write_bytes(stream.read(max(0, int(hi) - int(lo))))
PY

if [[ ! -f "$cell_dir/model.patch" ]]; then
  touch "$cell_dir/model.patch"
fi

write_state verifying 'running hidden DeepSWE verifier against the committed patch'
rm -f "$verifier_cidfile"
set +e
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy \
  timeout --foreground "$verifier_timeout" docker run --rm \
    --name "$verifier_name" \
    --cidfile "$verifier_cidfile" \
    --label "$container_owner_label=$run_owner" \
    --label "$container_role_label=verifier" \
    --network none \
    --cpus 2 \
    --memory 8g \
    -v "$cell_dir/model.patch:/logs/artifacts/model.patch:ro" \
    -v "$cell_dir/verifier:/logs/verifier" \
    "$verifier_image" /tests/test.sh 2>&1 | tee "$cell_dir/verifier.console.log"
verifier_status=${PIPESTATUS[0]}
set -e

cell_end_ns=$(date +%s%N)
cell_finished=$(date --iso-8601=seconds)
cell_elapsed=$(awk -v s="$cell_start_ns" -v e="$cell_end_ns" \
  'BEGIN { printf "%.3f", (e-s)/1000000000 }')
python3 - "$cell_dir/cell_summary.json" "$task_id" "$display_title" "$language" \
  "$condition" "$agent_status" "$verifier_status" "$cell_elapsed" \
  "$cell_started" "$cell_finished" <<'PY'
import json, sys
x={"task_id":sys.argv[2],"display_title":sys.argv[3],"language":sys.argv[4],
   "condition":sys.argv[5],"agent_exit":int(sys.argv[6]),
   "verifier_exit":int(sys.argv[7]),"elapsed_sec":float(sys.argv[8]),
   "started_at":sys.argv[9],"finished_at":sys.argv[10]}
try: x["reward"]=json.load(open(sys.argv[1].rsplit('/',1)[0]+'/verifier/reward.json'))
except Exception: x["reward"]=None
json.dump(x,open(sys.argv[1],'w'),indent=2)
PY
reward=$(python3 - "$cell_dir/verifier/reward.json" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get('reward','missing'))
except Exception: print('missing')
PY
)
write_state finished "agent=$agent_status verifier=$verifier_status reward=$reward"
printf 'CELL FINISHED: %s agent=%s verifier=%s reward=%s elapsed=%ss\n' \
  "$task_id" "$agent_status" "$verifier_status" "$reward" "$cell_elapsed"
