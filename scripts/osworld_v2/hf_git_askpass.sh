#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  *Username*)
    printf '%s\n' hf_user
    ;;
  *Password*)
    "${HF_PYTHON:-python3}" -c \
      'from huggingface_hub import get_token; token = get_token(); assert token; print(token)'
    ;;
  *)
    exit 1
    ;;
esac
