# agents-sandbox

A collection of standalone agentic engineering projects. Each project lives in its own subdirectory with its own CLAUDE.md.

## Conventions
- Each project is self-contained with its own dependencies, entrypoint, and README
- Prefer uv and `pyproject.toml` for environment management
  - Set Python version via `requires-python = ">=3.12, <3.13"`
- Prefer simple scripts over frameworks unless complexity justifies it
- No external API dependencies unless explicitly scoped in the project CLAUDE.md
