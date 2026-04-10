"""
Generate mock ML experiment artifacts under experiments/.

Narrative arc:
  exp_001 — baseline MLP, lr=0.01
  exp_002 — lr sweep down (0.001): better val loss, slower convergence
  exp_003 — lr sweep up (0.1): unstable training, high val loss
  exp_004 — architecture change: deeper net at best lr (0.001), clear accuracy jump
  exp_005 — data change: augmentation + extra samples at same arch, further improvement
"""

import json
import os
import math
import random

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments")


def write_experiment(name, config, metrics, notes):
    exp_dir = os.path.join(EXPERIMENTS_DIR, name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(exp_dir, "notes.md"), "w") as f:
        f.write(notes.strip() + "\n")
    print(f"  wrote {exp_dir}")


def decay(start, floor, rate, epoch):
    """Exponential decay toward a floor value."""
    return floor + (start - floor) * math.exp(-rate * epoch)


def noisy(value, sigma=0.003, rng=None):
    r = rng or random
    return round(value + r.gauss(0, sigma), 4)


def rising(floor, ceiling, rate, epoch):
    """Logistic-style rise from floor toward ceiling."""
    return floor + (ceiling - floor) * (1 - math.exp(-rate * epoch))


def make_metrics(train_loss_fn, val_loss_fn, accuracy_fn, epochs, rng):
    per_epoch = []
    for e in range(1, epochs + 1):
        per_epoch.append(
            {
                "epoch": e,
                "train_loss": noisy(train_loss_fn(e), sigma=0.005, rng=rng),
                "val_loss": noisy(val_loss_fn(e), sigma=0.006, rng=rng),
                "val_accuracy": noisy(accuracy_fn(e), sigma=0.004, rng=rng),
            }
        )
    final = per_epoch[-1]
    return {
        "epochs": epochs,
        "per_epoch": per_epoch,
        "best_val_loss": round(min(r["val_loss"] for r in per_epoch), 4),
        "best_val_accuracy": round(max(r["val_accuracy"] for r in per_epoch), 4),
        "final_train_loss": final["train_loss"],
        "final_val_loss": final["val_loss"],
        "final_val_accuracy": final["val_accuracy"],
    }


def main():
    rng = random.Random(42)

    # ── exp_001: baseline MLP, lr=0.01, 20 epochs ─────────────────────────
    write_experiment(
        "exp_001",
        config={
            "experiment_id": "exp_001",
            "date": "2026-03-15",
            "model": {
                "architecture": "mlp",
                "hidden_layers": [128, 64],
                "activation": "relu",
                "dropout": 0.0,
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.01,
                "batch_size": 64,
                "epochs": 20,
                "weight_decay": 0.0,
            },
            "data": {
                "dataset": "cifar10_subset",
                "train_samples": 10000,
                "val_samples": 2000,
                "augmentation": False,
            },
        },
        metrics=make_metrics(
            train_loss_fn=lambda e: decay(2.3, 0.85, 0.15, e),
            val_loss_fn=lambda e: decay(2.35, 0.95, 0.12, e),
            accuracy_fn=lambda e: rising(0.10, 0.782, 0.12, e),
            epochs=20,
            rng=rng,
        ),
        notes="""# exp_001 — Baseline

## Hypothesis
Establish a baseline with a simple two-layer MLP at a moderate learning rate (0.01).
Expect decent convergence but likely sub-optimal final accuracy.

## Observations
- Training loss drops steadily for the first 10 epochs then plateaus around 0.9.
- Val loss tracks training loss closely — no obvious overfitting at this scale.
- Final val accuracy ~72%. A reasonable floor to beat.
- Loss curve looks a bit jagged; lr=0.01 may be slightly too large for late-stage fine-tuning.
""",
    )

    # ── exp_002: lr sweep down (0.001) ────────────────────────────────────
    write_experiment(
        "exp_002",
        config={
            "experiment_id": "exp_002",
            "date": "2026-03-17",
            "model": {
                "architecture": "mlp",
                "hidden_layers": [128, 64],
                "activation": "relu",
                "dropout": 0.0,
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 20,
                "weight_decay": 0.0,
            },
            "data": {
                "dataset": "cifar10_subset",
                "train_samples": 10000,
                "val_samples": 2000,
                "augmentation": False,
            },
        },
        metrics=make_metrics(
            train_loss_fn=lambda e: decay(2.3, 0.70, 0.10, e),
            val_loss_fn=lambda e: decay(2.35, 0.78, 0.09, e),
            accuracy_fn=lambda e: rising(0.10, 0.891, 0.09, e),
            epochs=20,
            rng=rng,
        ),
        notes="""# exp_002 — LR sweep: 0.001

## Hypothesis
Lower lr (0.001) should smooth the training curve and potentially reach a better minimum.
Expect slower early convergence but better final loss.

## Observations
- Convergence is noticeably smoother — loss curve has far less noise epoch-to-epoch.
- By epoch 20 both train and val loss are lower than exp_001 despite slower start.
- Val accuracy reached ~76%, +4pp over baseline.
- The gap between train and val loss is small, suggesting the model is not overfit.
- Worth trying even lower lr or a scheduler in a future run.
""",
    )

    # ── exp_003: lr sweep up (0.1) — unstable ─────────────────────────────
    write_experiment(
        "exp_003",
        config={
            "experiment_id": "exp_003",
            "date": "2026-03-17",
            "model": {
                "architecture": "mlp",
                "hidden_layers": [128, 64],
                "activation": "relu",
                "dropout": 0.0,
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.1,
                "batch_size": 64,
                "epochs": 20,
                "weight_decay": 0.0,
            },
            "data": {
                "dataset": "cifar10_subset",
                "train_samples": 10000,
                "val_samples": 2000,
                "augmentation": False,
            },
        },
        metrics=make_metrics(
            train_loss_fn=lambda e: (
                decay(2.3, 1.10, 0.06, e) + 0.04 * math.sin(e * 1.5)
            ),
            val_loss_fn=lambda e: (
                decay(2.35, 1.25, 0.05, e) + 0.07 * abs(math.sin(e * 1.7))
            ),
            accuracy_fn=lambda e: rising(0.10, 0.970, 0.05, e),
            epochs=20,
            rng=rng,
        ),
        notes="""# exp_003 — LR sweep: 0.1

## Hypothesis
Try a high lr to see if the loss landscape allows fast convergence here.

## Observations
- Training is clearly unstable: loss oscillates significantly across epochs.
- Val loss does not reliably improve — best val loss worse than both exp_001 and exp_002.
- Val accuracy ~65%, below baseline.
- lr=0.1 is too aggressive for this architecture/dataset combo.
- Confirms that lr=0.001 is the direction to explore further; rule out anything above 0.01.
""",
    )

    # ── exp_004: deeper architecture at best lr ────────────────────────────
    write_experiment(
        "exp_004",
        config={
            "experiment_id": "exp_004",
            "date": "2026-03-21",
            "model": {
                "architecture": "mlp",
                "hidden_layers": [256, 256, 128, 64],
                "activation": "relu",
                "dropout": 0.2,
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 30,
                "weight_decay": 1e-4,
            },
            "data": {
                "dataset": "cifar10_subset",
                "train_samples": 10000,
                "val_samples": 2000,
                "augmentation": False,
            },
        },
        metrics=make_metrics(
            train_loss_fn=lambda e: decay(2.3, 0.45, 0.12, e),
            val_loss_fn=lambda e: decay(2.35, 0.58, 0.10, e),
            accuracy_fn=lambda e: rising(0.10, 0.868, 0.10, e),
            epochs=30,
            rng=rng,
        ),
        notes="""# exp_004 — Deeper MLP (4 layers) + dropout

## Hypothesis
The two-layer MLP is likely capacity-limited. A deeper network (4 layers, wider early)
with light dropout should extract richer features and improve accuracy meaningfully.
Using best lr from sweep (0.001) and adding weight decay.

## Observations
- Clear accuracy jump: val accuracy ~83%, +7pp over exp_002.
- Train loss drops substantially lower than previous runs, confirming additional capacity is used.
- Val loss is higher than train loss by epoch 25+ — light overfitting beginning to emerge.
- Dropout (p=0.2) is helping but may need tuning upward.
- Best results appear around epoch 25; slight degradation after. Early stopping would help.
- The data volume (10k samples) may be the next bottleneck — model can probably absorb more.
""",
    )

    # ── exp_005: data augmentation + more samples ─────────────────────────
    write_experiment(
        "exp_005",
        config={
            "experiment_id": "exp_005",
            "date": "2026-03-25",
            "model": {
                "architecture": "mlp",
                "hidden_layers": [256, 256, 128, 64],
                "activation": "relu",
                "dropout": 0.3,
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": 128,
                "epochs": 30,
                "weight_decay": 1e-4,
            },
            "data": {
                "dataset": "cifar10_subset",
                "train_samples": 25000,
                "val_samples": 5000,
                "augmentation": True,
                "augmentation_ops": [
                    "random_horizontal_flip",
                    "random_crop",
                    "color_jitter",
                ],
            },
        },
        metrics=make_metrics(
            train_loss_fn=lambda e: decay(2.3, 0.38, 0.13, e),
            val_loss_fn=lambda e: decay(2.35, 0.44, 0.12, e),
            accuracy_fn=lambda e: rising(0.10, 0.892, 0.12, e),
            epochs=30,
            rng=rng,
        ),
        notes="""# exp_005 — Data augmentation + 2.5x more training samples

## Hypothesis
exp_004 showed early signs of overfitting. The most direct fix is more/better data.
Increased training samples from 10k → 25k, added standard augmentation ops, bumped
dropout slightly to 0.3 to match the richer data regime.

## Observations
- Val accuracy ~87%, another +4pp over exp_004. Best run so far.
- Crucially, the train/val loss gap is much tighter than exp_004 — augmentation is
  acting as an effective regularizer and the overfitting signal is gone.
- Training is stable all 30 epochs with no sign of degradation at the end.
- Batch size bump (64→128) sped up training without hurting metrics.
- Model seems to be still learning at epoch 30 — could benefit from more epochs or LR decay.
- Next natural question: is MLP the right architecture, or would a conv net do better here?
""",
    )

    print("\nDone. Generated 5 experiments in experiments/")


if __name__ == "__main__":
    main()
