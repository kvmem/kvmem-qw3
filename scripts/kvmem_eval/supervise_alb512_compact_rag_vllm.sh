#!/usr/bin/env bash
set -uo pipefail

SERVER_SESSION=alb512_compact_rag64_vllm_fp8_mtp4_resume1_20260811
RUNNER_SESSION=alb512_compact_rag64_vllm_fp8_mtp4_resume1_runner_20260811
ROOT=/data/chaidi/kvmem_eval/results/agentlongbench_512k_compact_rag_notail_cap64k_vllm_fp8_mtp4_20260811
SERVER_LOG=/data/chaidi/kvmem_eval/logs/agentlongbench_512k_compact_rag_notail_cap64k_20260811_vllm_fp8_mtp4_resume1_server.log
RUN_LOG=/data/chaidi/kvmem_eval/logs/agentlongbench_512k_compact_rag_notail_cap64k_vllm_fp8_mtp4_resume1_full.log
WATCH_LOG=/data/chaidi/kvmem_eval/logs/agentlongbench_512k_compact_rag_notail_cap64k_vllm_fp8_mtp4_resume1_watch.log
MODEL=/data/huggingface/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/e89b16ebf1988b3d6befa7de50abc2d76f26eb09

log() {
    printf '%s %s\n' "$(date '+%F %T %z')" "$*" | tee -a "$WATCH_LOG"
}

completed() {
    find "$ROOT/answers" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l
}

healthy() {
    curl -fsS --max-time 3 http://127.0.0.1:18121/health >/dev/null 2>&1
}

session_alive() {
    tmux has-session -t "$1" 2>/dev/null
}

start_server() {
    tmux kill-session -t "$SERVER_SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SERVER_SESSION" \
        "cd /home/chaidi/qw3 && env PATH=/home/chaidi/vllm/.venv/bin:\$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 /home/chaidi/vllm/.venv/bin/vllm serve '$MODEL' --served-model-name Qwen3.6-27B-FP8 --host 127.0.0.1 --port 18121 --language-model-only --safetensors-load-strategy prefetch --dtype bfloat16 --kv-cache-dtype fp8 --max-model-len 262144 --gpu-memory-utilization 0.92 --max-num-seqs 2 --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching --async-scheduling --reasoning-parser qwen3 --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}' --compilation-config '{\"mode\":3}' >> '$SERVER_LOG' 2>&1"
    log "server start requested"
    for _ in $(seq 1 90); do
        if healthy; then
            log "server healthy"
            return 0
        fi
        if ! session_alive "$SERVER_SESSION"; then
            log "server exited during startup"
            return 1
        fi
        sleep 5
    done
    log "server startup timeout"
    return 1
}

start_runner() {
    tmux kill-session -t "$RUNNER_SESSION" 2>/dev/null || true
    tmux new-session -d -s "$RUNNER_SESSION" \
        "cd /home/chaidi/AgentLongBench-Long && set -o pipefail; env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 /usr/bin/time -v /home/chaidi/kvmem-efficiency-bench/.venv-rag/bin/python script/compactRag/run_no_tail_cap64k.py --dataset /data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl --source-compact-root /home/chaidi/AgentLongBench-Long/results/agentlongbench_512k_normal100/compact_only_normal100/output --output-root '$ROOT' --api-base http://127.0.0.1:18121/v1 --api-backend vllm --model Qwen3.6-27B-FP8 --tokenizer-path /home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8 --embedding-model-path /data/chaidi/kvmem_eval/models/jinaai/jina-embeddings-v2-small-en --rag-chunk-tokenizer-path /home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8 --embedding-device cuda --embedding-batch-size 256 --workers 2 2>&1 | tee -a '$RUN_LOG'; status=\${PIPESTATUS[0]}; echo RUNNER_EXIT_STATUS=\$status | tee -a '$RUN_LOG'; exit \$status"
    log "runner start requested completed=$(completed)/100"
}

restart_count=0
health_failures=0
log "watch started completed=$(completed)/100"

while (( $(completed) < 100 )); do
    if healthy; then
        health_failures=0
    else
        health_failures=$((health_failures + 1))
    fi

    if (( health_failures >= 3 )); then
        restart_count=$((restart_count + 1))
        log "server unhealthy; recovery=$restart_count completed=$(completed)/100"
        tmux kill-session -t "$RUNNER_SESSION" 2>/dev/null || true
        tmux kill-session -t "$SERVER_SESSION" 2>/dev/null || true
        sleep 8
        if (( restart_count > 4 )); then
            log "too many server recoveries; stopping supervisor"
            exit 2
        fi
        if ! start_server; then
            sleep 10
            health_failures=3
            continue
        fi
        start_runner
        health_failures=0
    elif ! session_alive "$RUNNER_SESSION"; then
        log "runner absent with healthy server; restarting completed=$(completed)/100"
        start_runner
    fi
    sleep 10
done

log "experiment complete completed=$(completed)/100"
