# ml-experiment-explorer

An agent that reads local ML experiment artifacts, reasons across the full set of runs, and produces a written report with recommended next steps.

## How it works

1. Each experiment lives in `experiments/exp_NNN/` with three files:
   - `config.json` — hyperparams, model architecture, data config
   - `metrics.json` — per-epoch train/val loss and accuracy
   - `notes.md` — hypothesis going in and observations from the run

2. `scripts/analyze.py` reads all experiments and builds a structured summary for each one: config deltas from the previous run, key metrics, a trajectory signal indicating whether the model was still improving at the final epoch, and the researcher's notes.

3. All summaries are passed to Claude in a single API call. Claude reasons across the full experiment trajectory and writes a markdown report to `reports/report_YYYY-MM-DD.md` with three sections: experiment trajectory, key patterns, and ranked recommendations.

## Usage

```shell
# Install env; restores pinned deps from committed uv.lock
uv sync

# Generate mock experiments (first time)
python scripts/generate_mocks.py

# Run the agent
ANTHROPIC_API_KEY=<your-key> python scripts/analyze.py
```

The report is written to `reports/` and also printed to stdout.

## Project structure

```
experiments/         ML experiment artifacts (config, metrics, notes)
reports/             Generated reports (safe to gitignore)
scripts/
  analyze.py         Main entrypoint — builds summaries, calls Claude, writes report
  generate_mocks.py  Generates 5 mock experiments with a realistic narrative arc
```
