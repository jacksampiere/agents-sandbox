# agents-sandbox

A collection of standalone agentic engineering projects; each project lives in its own subdirectory.

## Claude Code configuration
`.claude/` contains repository-wide Claude Code configurations:
- `settings.json` + `hooks/`:
  - ruff format on Python file edits
  - bash safety checks (blocks destructive commands)
- `agents/`:
  - `code-reviewer.md`: sub-agent for code review
- `skills/`
  - `eda-report/`: skill for exploratory data analysis reports

## Projects

### ml-experiment-explorer
Analyzes local ML experiment artifacts across a set of runs and produces a written report with recommended next steps. A deterministic extractor computes per-experiment signals (config diffs, convergence states, overfitting gaps), then passes compact summaries to an LLM that reasons across the full trajectory.

### simple-tool-use
A lightweight implementation of the Anthropic tool use loop from scratch via the SDK. Demonstrates the full cycle of defining tools as JSON schemas, parsing tool calls from responses, executing them, and feeding results back to get a final answer.
