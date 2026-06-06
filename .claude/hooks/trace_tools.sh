#!/usr/bin/env bash
ts=$(date +%H:%M:%S)
jq -r --arg ts "$ts" '"\($ts) [\(.tool_name)] \(.tool_input | tostring | .[0:80])"' >> /tmp/agent-trace.log

# Local tool tracing notes:
#     While the agent is running:
#         Open a second terminal pane and `tail -f` the log:
#         - `tail -f /tmp/agent-trace.log`
#     After a run:
#         Just `cat` or `less` it:
#         - `cat /tmp/agent-trace.log`
# Note:
#     `/tmp/` gets wiped on reboot, and if you run multiple sessions the lines from all of them append together
#      If that gets noisy you can either clear it between runs (`> /tmp/agent-trace.log`) or change the path in the script
#      The path could be something like `$CLAUDE_PROJECT_DIR/.claude/traces/$(date +%Y%m%d-%H%M%S).log` so each session gets its own file