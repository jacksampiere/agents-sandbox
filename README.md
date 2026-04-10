# agents-sandbox

A collection of standalone agentic engineering projects. Each project lives in its own subdirectory.

## Projects

### ml-experiment-explorer
Analyzes local ML experiment artifacts across a set of runs and produces a written recommendations report. A deterministic extractor computes per-experiment signals (config diffs, convergence state, overfit gap), then passes compact summaries to an LLM that reasons across the full trajectory and recommends what to try next.
