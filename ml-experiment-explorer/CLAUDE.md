# ml-experiment-explorer

An agent that reads local ML experiment artifacts, reasons across the full set of runs, and produces a written report with recommended next steps.

## Experiment schema
```
experiments/
  exp_NNN/
    config.json     # hyperparams, model arch, data config
    metrics.json    # train/val loss + eval metrics, per epoch where available
    notes.md        # hypothesis going in + any observations during the run
```

## Agent behavior
- Reads ALL experiments in experiments/
- Reasons about the trajectory across runs (not just the latest)
- Produces a markdown report to reports/report_YYYY-MM-DD.md

## Output format
The report should have:
1. A summary of the experiment trajectory (what changed, what happened)
2. Key patterns or inflection points observed
3. Concrete ranked recommendations for what to try next, each with a rationale

## Phase 1 scope
- Local files only, no external APIs
- Read-only agent (no config modification)
- Mock experiments are fine; see scripts/generate_mocks.py

## Stack
- Python, stdlib + json only for file reading
- Anthropic Python SDK for the agent
