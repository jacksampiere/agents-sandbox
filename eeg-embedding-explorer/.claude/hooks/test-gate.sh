#!/usr/bin/env bash
# Test-gate Stop hook for the EEG Embedding Explorer.
# Runs the pytest suite when the agent tries to finish. If tests fail, it blocks
# the stop and feeds the failure back so the agent keeps working. It is a no-op
# before any tests exist (early gates) and is loop-safe.
#
# Stop-hook protocol (Claude Code):
#   - stdin: JSON with `stop_hook_active` (true if a prior Stop hook already blocked this turn)
#   - to block: print {"decision":"block","reason":"..."} on stdout, exit 0
#   - to allow: exit 0 with no block decision
# A built-in safety cap ends the session after 8 consecutive blocks, so a stuck
# suite can't loop forever.

set -uo pipefail

input="$(cat)"

# Loop guard: if a previous Stop hook in this turn already blocked, let it stop.
stop_active="$(printf '%s' "$input" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin).get("stop_hook_active",False))' \
  2>/dev/null || echo False)"
if [ "$stop_active" = "True" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# No tests yet (Gate 0 / early Gate 1) → nothing to gate on.
shopt -s nullglob
test_files=(tests/test_*.py tests/*_test.py)
if [ "${#test_files[@]}" -eq 0 ]; then
  exit 0
fi

output="$(uv run pytest -q 2>&1)"
status=$?

if [ "$status" -eq 0 ]; then
  exit 0
fi

# Tests failed → block the stop and hand the tail of the log back to the agent.
reason="$(printf 'Test gate FAILED — fix failing tests before finishing.\n\n%s' "$output" | tail -c 3000)"
REASON="$reason" python3 - <<'PY'
import json, os
print(json.dumps({"decision": "block", "reason": os.environ["REASON"]}))
PY
exit 0
