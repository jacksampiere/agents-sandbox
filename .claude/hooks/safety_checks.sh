#!/bin/bash
INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // ""')

if echo "$CMD" | grep -qE '(rm -rf /|DROP TABLE|format [A-Z]:)'; then
  echo '{"block": true, "message": "Potentially destructive command blocked. Please confirm manually."}' >&2
  exit 2
fi

exit 0