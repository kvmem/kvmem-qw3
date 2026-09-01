#!/usr/bin/env bash
set -euo pipefail

# Pier and the DeepSWE task containers run on this host. The wrapper replaces
# only the local model process with an SSH tunnel to a foreground QW3 KVMem
# service on the A40 worker.
export QW3_DEEPSWE_BINARY="${QW3_DEEPSWE_BINARY:-/home/chaidi/qw3/benchmark/mini_swe_deepswe/remote_qw3_ssh_wrapper.py}"
export QW3_REMOTE_SSH_HOST="${QW3_REMOTE_SSH_HOST:-root@ssh-cn-huabei1.ebcloud.com}"
export QW3_REMOTE_SSH_PORT="${QW3_REMOTE_SSH_PORT:-31623}"
export QW3_REMOTE_BINARY="${QW3_REMOTE_BINARY:-/root/data/qw3/build/sm86_fp8_batch/qw3}"
export QW3_DEEPSWE_REMOTE_BINARY_SHA256="${QW3_DEEPSWE_REMOTE_BINARY_SHA256:-e62b7f15b7583563ee0c6b3aad9e95cc96d584abfbe9ef5ebb14ae4cfb94dea0}"
export QW3_REMOTE_WORKDIR="${QW3_REMOTE_WORKDIR:-/home/chaidi/qw3}"
export QW3_REMOTE_LISTEN_HOST="${QW3_REMOTE_LISTEN_HOST:-127.0.0.1}"
export QW3_REMOTE_LISTEN_PORT="${QW3_REMOTE_LISTEN_PORT:-8000}"

exec /home/chaidi/qw3/.venv-deepswe-pier/bin/python \
  /home/chaidi/qw3/benchmark/mini_swe_deepswe/run_requestplan10.py \
  --run-name "${RUN_NAME:-stratified20_kvmem_k64g32_official_r1_s1000_a40remote_20260901}" \
  --tasks-file /home/chaidi/qw3/benchmark/mini_swe_deepswe/stratified20_seed20260831.json \
  --mode kvmem \
  --seed 1000 \
  --guided-query-tokens 4096 \
  --kvmem-budget 65536 \
  --kvmem-prefill-budget 65536 \
  --kvmem-gen-budget 32768 \
  --host 172.17.0.1 \
  --port 8010 \
  --api-host 172.17.0.1.nip.io \
  --api-port 443 \
  "$@"
