#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${PROFILE:-initial}"
case "$PROFILE" in
  initial) exec "$HERE/serve_tp8_initial.sh" "$@" ;;
  ideal) exec "$HERE/serve_tp8_ideal.sh" "$@" ;;
  *) echo "PROFILE must be initial or ideal" >&2; exit 2 ;;
esac
