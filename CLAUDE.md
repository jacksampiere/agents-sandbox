# agents-sandbox

A collection of standalone agentic engineering projects. Each project lives in its own subdirectory, and those built extensively with Claude Code contain a project-specific `CLAUDE.md`.

## Conventions
- Each project is self-contained with its own dependencies, entrypoint, and `README.md`
- When starting a new project, create the subdirectory with a `README.md` and `pyproject.toml` before writing any code
- Prefer uv and `pyproject.toml` for Python environment management
  - Set Python version via `requires-python = ">=3.12, <3.13"`
  - Pin runtime dependency versions exactly (e.g. `anthropic==0.94.0`); dev tooling (e.g. `ruff`) may use loose ranges
  - Commit `uv.lock` for reproducibility
- Prefer simple scripts; don't over-engineer
  - Don't create classes where functions suffice
  - Avoid command line args; prefer config variables at the top of the script that the user edits directly before running
- Restrict error catching/handling to edge cases that arise in the core logic; avoid trivial checks such as syntax errors in the script call
- Take your time; clarify implementation details by asking questions rather than making assumptions
- Use ruff for linting and formatting; don't add custom ruff config unless the project requires it

## Task Observer (meta-skill)

Note: this requires `.claude/skills/task-observer/` and refers to a local skill installed via symlink (not committed). Source: https://github.com/rebelytics/one-skill-to-rule-them-all

At the start of any task-oriented session — any interaction where you will use tools and produce deliverables — invoke the task-observer skill before beginning work. This ensures skill improvement opportunities are captured throughout the session.

When loading any skill, check the observation log at `~/.claude/skill-commons/skill-observations/log.md` for OPEN observations tagged to that skill. Apply their insights to the current work, even if the skill file hasn't been updated yet.

When the user signals they are wrapping up or archiving a session (e.g. "done", "that's all", "wrapping up", "archive this"), proactively remind them to ask "Any observations logged?" before closing.
