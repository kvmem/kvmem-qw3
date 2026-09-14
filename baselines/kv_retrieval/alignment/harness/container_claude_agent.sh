#!/usr/bin/env bash
set -euo pipefail

# Run Claude Code with no inherited per-user settings.  This is important on the
# benchmark host because the normal account points at a different Anthropic API.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

: "${BASE_COMMIT:?BASE_COMMIT is required}"
: "${CLAUDE_API_BASE:?CLAUDE_API_BASE is required}"
: "${CLAUDE_MODEL_ID:?CLAUDE_MODEL_ID is required}"
: "${CLAUDE_BRANCH_NAME:?CLAUDE_BRANCH_NAME is required}"

artifact_dir=/artifacts
export HOME=$artifact_dir/home
export CLAUDE_CONFIG_DIR=$artifact_dir/claude-config
mkdir -p "$HOME" "$CLAUDE_CONFIG_DIR"

export ANTHROPIC_BASE_URL=$CLAUDE_API_BASE
export ANTHROPIC_API_KEY=local-vllm
export ANTHROPIC_AUTH_TOKEN=local-vllm
export ANTHROPIC_DEFAULT_OPUS_MODEL=$CLAUDE_MODEL_ID
export ANTHROPIC_DEFAULT_SONNET_MODEL=$CLAUDE_MODEL_ID
export ANTHROPIC_DEFAULT_HAIKU_MODEL=$CLAUDE_MODEL_ID
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
# Keep client-side compaction out of controlled long-context/KVMem runs.  The
# server still enforces its own physical --ctx value; these variables only make
# Claude Code advertise the matching logical window and retain the full wire
# history.  They are supported by the pinned Claude Code 2.1.148 binary.
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=${CLAUDE_CODE_MAX_CONTEXT_TOKENS:-1048576}
# Claude Code 2.1.148 treats unknown/custom model names as 200K while native
# compaction is enabled.  A caller can pair a model id carrying the supported
# `[1m]` capability marker with this explicit window so a 256K dense backend
# compacts at its real limit rather than at the custom-model 180K fallback.
claude_auto_compact_window=${CLAUDE_AUTO_COMPACT_WINDOW:-}
if [[ -n "$claude_auto_compact_window" ]]; then
  export CLAUDE_CODE_AUTO_COMPACT_WINDOW=$claude_auto_compact_window
else
  unset CLAUDE_CODE_AUTO_COMPACT_WINDOW
fi
# DISABLE_COMPACT is a presence-sensitive Claude Code control.  Do not export
# the string "0" for the dense control arm: unset it so Claude Code can use its
# native compaction policy.  KVMem arms explicitly export 1.
claude_disable_compact=${CLAUDE_DISABLE_COMPACT:-${DISABLE_COMPACT:-1}}
if [[ "$claude_disable_compact" == 1 ]]; then
  export DISABLE_COMPACT=1
else
  unset DISABLE_COMPACT
fi
export TERM=xterm-256color
export COLORTERM=truecolor
export FORCE_COLOR=3

cd /app
git config --global --add safe.directory /app
git config --global user.name 'Qwen3.8 Claude Code Eval'
git config --global user.email 'qwen38-claude-eval@localhost'
git checkout -B "$CLAUDE_BRANCH_NAME" "$BASE_COMMIT"

git status --short >"$artifact_dir/git_status.before.txt"

prompt=$(< "$artifact_dir/prompt.md")
started_at=$(date --iso-8601=seconds)
start_ns=$(date +%s%N)
claude_max_wall_time=${CLAUDE_MAX_WALL_TIME:-0}
claude_cmd=(
  /usr/local/bin/claude
  -p
  --output-format stream-json
  --verbose
  --include-partial-messages
  --dangerously-skip-permissions
  # KVMem currently owns one warm trajectory.  Disable Claude Code's
  # sub-Agent fan-out so interleaved sessions cannot overwrite that slot.
  --disallowedTools Agent
  --setting-sources project,local
  --settings '{}'
  --debug-file "$artifact_dir/claude.debug.log"
  --model "$CLAUDE_MODEL_ID"
  --effort xhigh
  --
  "$prompt"
)
set +e
if [[ -z "$claude_max_wall_time" || "$claude_max_wall_time" == 0 ||
      "$claude_max_wall_time" == none || "$claude_max_wall_time" == unlimited ]]; then
  "${claude_cmd[@]}" 2>&1 | tee "$artifact_dir/claude.stream.jsonl"
  claude_status=${PIPESTATUS[0]}
else
  timeout --foreground "$claude_max_wall_time" "${claude_cmd[@]}" \
    2>&1 | tee "$artifact_dir/claude.stream.jsonl"
  claude_status=${PIPESTATUS[0]}
fi
set -e

end_ns=$(date +%s%N)
finished_at=$(date --iso-8601=seconds)
elapsed_sec=$(awk -v s="$start_ns" -v e="$end_ns" \
  'BEGIN { printf "%.3f", (e-s)/1000000000 }')

# DeepSWE collects only committed changes.  Also retain the whole working tree
# diff so a missing model commit can be diagnosed without changing the score.
git diff --binary "$BASE_COMMIT" HEAD >"$artifact_dir/model.patch"
git diff --binary "$BASE_COMMIT" >"$artifact_dir/working_tree.patch"
git status --short >"$artifact_dir/git_status.txt"
git log --oneline --decorate "$BASE_COMMIT"..HEAD >"$artifact_dir/commits.txt"

python3 - "$artifact_dir/agent_summary.json" "$claude_status" "$elapsed_sec" \
  "$started_at" "$finished_at" "$CLAUDE_CODE_MAX_CONTEXT_TOKENS" \
  "$claude_disable_compact" "$claude_auto_compact_window" "$CLAUDE_MODEL_ID" <<'PY'
import json, sys
json.dump({
    "agent_exit": int(sys.argv[2]),
    "elapsed_sec": float(sys.argv[3]),
    "started_at": sys.argv[4],
    "finished_at": sys.argv[5],
    "claude_code_max_context_tokens": int(sys.argv[6]),
    "disable_compact": sys.argv[7] == "1",
    "auto_compact_window": int(sys.argv[8]) if sys.argv[8] else None,
    "model_id": sys.argv[9],
}, open(sys.argv[1], "w"), indent=2)
PY
printf '%s\n' "$claude_status" >"$artifact_dir/agent.exit_status"
# The container Agent runs as nobody while the recovery/verifier supervisor
# runs as the host user.  Publish every hand-off artifact with group/world
# write permission so recovery can create/replace terminal state without
# depending on the container UID mapping.
chmod a+rw \
  "$artifact_dir/model.patch" \
  "$artifact_dir/working_tree.patch" \
  "$artifact_dir/git_status.before.txt" \
  "$artifact_dir/git_status.txt" \
  "$artifact_dir/commits.txt" \
  "$artifact_dir/agent_summary.json" \
  "$artifact_dir/agent.exit_status"
exit "$claude_status"
