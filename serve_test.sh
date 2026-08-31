#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8200}"
MODEL="${MODEL:-glm5.3-flash}"

curl -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"반드시 SM80-OK 라고만 대답해\"}],\"temperature\":0,\"max_tokens\":16}" \
  | python -m json.tool
