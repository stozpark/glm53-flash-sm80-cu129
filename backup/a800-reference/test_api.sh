#!/usr/bin/env bash
# Smoke tests for the GLM-5.3-Flash A800 deployment.
# Usage: ./test_api.sh [PORT]   (default 8008)
set -euo pipefail

PORT="${1:-8008}"
BASE="http://localhost:${PORT}"
MODEL="glm5.3-flash"

echo "== 1) GET ${BASE}/v1/models =="
curl -sf "${BASE}/v1/models" | python3 -m json.tool | head -20 || {
  echo "FAIL: server not reachable / not ready"; exit 1; }

echo ""
echo "== 2) chat completion (reasoning_effort=low, math sanity) =="
curl -s "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is 13*17? Answer with just the number.\"}],
    \"temperature\": 0,
    \"max_tokens\": 512,
    \"chat_template_kwargs\": {\"reasoning_effort\": \"low\"}
  }" | python3 -c "
import sys, json
r = json.load(sys.stdin)
if 'error' in r:
    print('ERROR:', r['error']); sys.exit(1)
m = r['choices'][0]['message']
print('content          :', (m.get('content') or '')[:200])
rt = m.get('reasoning') or m.get('reasoning_content')
print('reasoning(head)  :', (rt or '')[:120] if rt else None)
u = r.get('usage', {})
print('usage            : prompt=%s completion=%s reasoning_tokens=%s' % (
    u.get('prompt_tokens'), u.get('completion_tokens'), u.get('reasoning_tokens')))
print('finish_reason    :', r['choices'][0].get('finish_reason'))
"
echo ""
echo "PASS if content above is 221."
