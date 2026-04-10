"""
analyze.py — ML experiment explorer.

Reads all experiments in experiments/, builds per-experiment summaries
(config deltas, key metrics, trajectory signal, notes), then makes a
single Anthropic API call to reason across the full set and writes a
markdown recommendations report to reports/.

Usage:
    python scripts/analyze.py [--experiments-dir PATH] [--model MODEL]
"""

import argparse
import json
import os
import sys
from datetime import date

import anthropic

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(_PROJECT_ROOT, "experiments")
REPORTS_DIR = os.path.join(_PROJECT_ROOT, "reports")
DEFAULT_MODEL = "claude-sonnet-4-6"


# ── Loading ────────────────────────────────────────────────────────────────────


def load_experiment(exp_dir: str) -> dict:
    name = os.path.basename(exp_dir)
    with open(os.path.join(exp_dir, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(exp_dir, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(exp_dir, "notes.md")) as f:
        notes = f.read()
    return {"name": name, "config": config, "metrics": metrics, "notes": notes}


def load_all_experiments(experiments_dir: str) -> list[dict]:
    if not os.path.isdir(experiments_dir):
        print(
            f"ERROR: experiments directory not found: {experiments_dir}",
            file=sys.stderr,
        )
        print("Run: python scripts/generate_mocks.py", file=sys.stderr)
        sys.exit(1)
    dirs = sorted(
        d
        for d in os.listdir(experiments_dir)
        if os.path.isdir(os.path.join(experiments_dir, d))
    )
    if not dirs:
        print("ERROR: no experiment subdirectories found.", file=sys.stderr)
        sys.exit(1)
    return [load_experiment(os.path.join(experiments_dir, d)) for d in dirs]


# ── Config delta ───────────────────────────────────────────────────────────────

SKIP_KEYS = {"experiment_id", "date", "val_samples"}


def flatten_config(cfg: dict, prefix: str = "") -> dict:
    """Flatten nested config into dot-separated key/value pairs, skipping metadata."""
    out = {}
    for k, v in cfg.items():
        if k in SKIP_KEYS:
            continue
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_config(v, full_key))
        else:
            out[full_key] = v
    return out


def config_deltas(curr: dict, prev: dict) -> list[str]:
    """Return human-readable lines describing what changed between two configs."""
    a = flatten_config(prev)
    b = flatten_config(curr)
    all_keys = sorted(set(a) | set(b))
    lines = []
    for k in all_keys:
        av, bv = a.get(k), b.get(k)
        if av != bv:
            if av is None:
                lines.append(f"  + {k}: {bv}  (new)")
            elif bv is None:
                lines.append(f"  - {k}  (removed)")
            else:
                lines.append(f"  ~ {k}: {av} → {bv}")
    return lines if lines else ["  (no changes)"]


# ── Trajectory signal ──────────────────────────────────────────────────────────


def trajectory_signal(metrics: dict) -> dict:
    """
    Compute whether the model was still improving at the final epoch.

    Returns a dict with:
      label       — "still_improving" | "plateau" | "degrading"
      val_acc_gain_last3   — val_accuracy delta over last 3 epochs
      val_loss_delta_last3 — val_loss delta over last 3 epochs (neg = improving)
      val_train_gap_final  — final (val_loss - train_loss); positive = generalization gap
    """
    epochs = metrics.get("per_epoch", [])
    if len(epochs) < 4:
        return {"label": "insufficient_data"}

    last3 = epochs[-3:]
    acc_gain = round(last3[-1]["val_accuracy"] - last3[0]["val_accuracy"], 4)
    loss_delta = round(last3[-1]["val_loss"] - last3[0]["val_loss"], 4)
    gap = round(metrics["final_val_loss"] - metrics["final_train_loss"], 4)

    # Characterise trajectory from first-half average delta
    mid = len(epochs) // 2
    first_half_deltas = [
        epochs[i + 1]["val_loss"] - epochs[i]["val_loss"]
        for i in range(min(mid, len(epochs) - 1))
    ]
    avg_early_delta = (
        sum(first_half_deltas) / len(first_half_deltas) if first_half_deltas else 0
    )

    if loss_delta > 0.005:
        label = "degrading"
    elif avg_early_delta != 0 and abs(loss_delta) < abs(avg_early_delta) * 0.08:
        label = "plateau"
    else:
        label = "still_improving"

    return {
        "label": label,
        "val_acc_gain_last3_epochs": acc_gain,
        "val_loss_delta_last3_epochs": loss_delta,
        "val_train_gap_final": gap,
    }


# ── Per-experiment summary ─────────────────────────────────────────────────────


def build_experiment_summary(exp: dict, prev: dict | None) -> str:
    name = exp["name"]
    cfg = exp["config"]
    m = exp["metrics"]

    lines = [f"## {name}  ({cfg.get('date', 'n/a')})", ""]

    # Config deltas
    if prev is None:
        lines.append("**Config:** baseline (no previous experiment)")
    else:
        lines.append(f"**Config changes from {prev['name']}:**")
        lines.extend(config_deltas(cfg, prev["config"]))
    lines.append("")

    # Key metrics
    lines.append("**Metrics:**")
    lines.append(f"  best_val_accuracy  : {m.get('best_val_accuracy')}")
    lines.append(f"  best_val_loss      : {m.get('best_val_loss')}")
    lines.append(f"  final_train_loss   : {m.get('final_train_loss')}")
    lines.append(f"  final_val_loss     : {m.get('final_val_loss')}")
    lines.append(f"  total_epochs       : {m.get('epochs')}")
    lines.append("")

    # Trajectory
    traj = trajectory_signal(m)
    lines.append("**Trajectory signal:**")
    for k, v in traj.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Notes
    lines.append("**Researcher notes:**")
    lines.append(exp["notes"].strip())
    lines.append("")

    return "\n".join(lines)


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert ML researcher reviewing a series of experiments.
You will receive structured summaries of each experiment in chronological order.
Each summary includes: config changes from the previous run, key metrics, a
trajectory signal (whether the model was still improving at the final epoch),
and the researcher's own notes.

Your task is to reason across the FULL set of experiments — not just the latest — \
and produce a written report. The report must have exactly these sections:

1. **Experiment Trajectory** — a concise narrative of what changed across runs and
   what happened to the metrics at each step. Highlight the inflection points.

2. **Key Patterns** — 3–5 bullet points identifying the most important signals:
   what worked, what didn't, what is ambiguous, and any signs of over/underfitting
   or training instability.

3. **Recommendations** — a ranked list of concrete next experiments to run, each with:
   - What to try
   - Why (grounded in the evidence from the experiments above)
   - What result would confirm or refute the hypothesis

Be specific and quantitative where the data supports it. Do not hedge excessively.
"""


def build_prompt(summaries: list[str]) -> str:
    body = "\n---\n\n".join(summaries)
    return (
        f"Here are {len(summaries)} experiments in chronological order:\n\n"
        f"---\n\n{body}"
    )


# ── API call ───────────────────────────────────────────────────────────────────


def generate_report(summaries: list[str], model: str) -> str:
    client = anthropic.Anthropic()
    prompt = build_prompt(summaries)

    print(f"Calling {model} with {len(summaries)} experiment summaries...", flush=True)

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


# ── Report writing ─────────────────────────────────────────────────────────────


def save_report(report_text: str, reports_dir: str) -> str:
    os.makedirs(reports_dir, exist_ok=True)
    filename = f"report_{date.today().isoformat()}.md"
    path = os.path.join(reports_dir, filename)
    with open(path, "w") as f:
        f.write(f"# ML Experiment Report — {date.today().isoformat()}\n\n")
        f.write(report_text)
        f.write("\n")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="ML experiment explorer")
    parser.add_argument("--experiments-dir", default=EXPERIMENTS_DIR)
    parser.add_argument("--reports-dir", default=REPORTS_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    experiments = load_all_experiments(args.experiments_dir)
    print(f"Loaded {len(experiments)} experiments.")

    summaries = []
    for i, exp in enumerate(experiments):
        prev = experiments[i - 1] if i > 0 else None
        summaries.append(build_experiment_summary(exp, prev))

    report_text = generate_report(summaries, args.model)
    path = save_report(report_text, args.reports_dir)

    print(f"Report written to: {path}")
    print()
    print(report_text)


if __name__ == "__main__":
    main()
