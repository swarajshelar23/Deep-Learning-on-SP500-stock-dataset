"""From-scratch Particle Swarm Optimization for ANN hyperparameters."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf

from models import build_optimized_ann, set_seed


@dataclass
class PSOConfig:
    swarm_size: int = 6
    iterations: int = 5
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    epoch_budget: int = 3
    seed: int = 42


# Position vector:
# [layers, n1, n2, n3, n4, log10_lr, dropout, batch_size_code]
BOUNDS_LOW = np.array([1, 16, 16, 16, 16, -4.0, 0.0, 0], dtype=float)
BOUNDS_HIGH = np.array([4, 256, 256, 256, 256, -2.0, 0.5, 2], dtype=float)
BATCH_OPTIONS = [16, 32, 64]


def decode_position(pos: np.ndarray) -> dict:
    hidden_layers = int(np.clip(np.rint(pos[0]), 1, 4))
    neurons = [int(np.clip(np.rint(x), 16, 256)) for x in pos[1:5]]
    # Round to practical powers/multiples to keep the search meaningful.
    neurons = [max(16, int(round(n / 8) * 8)) for n in neurons]
    lr = 10 ** float(pos[5])
    dropout = float(np.clip(pos[6], 0.0, 0.5))
    batch_code = int(np.clip(np.rint(pos[7]), 0, len(BATCH_OPTIONS) - 1))
    batch_size = BATCH_OPTIONS[batch_code]
    return {
        "hidden_layers": hidden_layers,
        "neurons": neurons,
        "learning_rate": lr,
        "dropout": dropout,
        "batch_size": batch_size,
    }


def _fitness(model, x_val, y_val) -> tuple[float, float]:
    proba = model.predict(x_val, verbose=0).reshape(-1)
    pred = (proba >= 0.5).astype(np.int8)
    f1 = f1_score(y_val, pred, zero_division=0)
    acc = accuracy_score(y_val, pred)
    return float(f1), float(acc)


def run_pso(
    x_train,
    y_train,
    x_val,
    y_val,
    n_features: int,
    config: PSOConfig,
    outputs: Path,
):
    rng = np.random.default_rng(config.seed)
    dim = len(BOUNDS_LOW)

    positions = rng.uniform(BOUNDS_LOW, BOUNDS_HIGH, size=(config.swarm_size, dim))
    velocities = rng.uniform(-0.15, 0.15, size=(config.swarm_size, dim))

    pbest_positions = positions.copy()
    pbest_scores = np.full(config.swarm_size, -np.inf)
    gbest_position = None
    gbest_score = -np.inf
    gbest_acc = -np.inf

    logs = []

    for iteration in range(config.iterations):
        print(f"\nPSO iteration {iteration + 1}/{config.iterations}")
        for particle in range(config.swarm_size):
            set_seed(config.seed + iteration * 1000 + particle)
            hp = decode_position(positions[particle])
            model = build_optimized_ann(
                n_features=n_features,
                hidden_layers=hp["hidden_layers"],
                neurons=hp["neurons"],
                dropout=hp["dropout"],
                lr=hp["learning_rate"],
            )
            callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=max(1, min(2, config.epoch_budget // 2)),
                restore_best_weights=True,
                verbose=0,
            )
            model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=config.epoch_budget,
                batch_size=hp["batch_size"],
                shuffle=False,
                callbacks=[callback],
                verbose=0,
            )
            f1, acc = _fitness(model, x_val, y_val)
            del model
            tf.keras.backend.clear_session()

            record = {
                "iteration": iteration + 1,
                "particle": particle + 1,
                "fitness_val_f1": f1,
                "val_accuracy": acc,
                **hp,
                "neurons": json.dumps(hp["neurons"]),
            }
            logs.append(record)
            print(
                f"  particle={particle + 1} F1={f1:.5f} "
                f"Acc={acc:.5f} layers={hp['hidden_layers']} "
                f"neurons={hp['neurons']} lr={hp['learning_rate']:.6g} "
                f"dropout={hp['dropout']:.3f} batch={hp['batch_size']}"
            )

            if f1 > pbest_scores[particle]:
                pbest_scores[particle] = f1
                pbest_positions[particle] = positions[particle].copy()

            if f1 > gbest_score:
                gbest_score = f1
                gbest_position = positions[particle].copy()
                gbest_acc = acc

        r1 = rng.random((config.swarm_size, dim))
        r2 = rng.random((config.swarm_size, dim))
        velocities = (
            config.w * velocities
            + config.c1 * r1 * (pbest_positions - positions)
            + config.c2 * r2 * (gbest_position - positions)
        )
        positions = np.clip(
            positions + velocities,
            BOUNDS_LOW,
            BOUNDS_HIGH,
        )

    log_df = pd.DataFrame(logs)
    log_df.to_csv(outputs / "pso_log.csv", index=False)

    best = decode_position(gbest_position)
    best_payload = {
        **best,
        "best_validation_f1": float(gbest_score),
        "best_validation_accuracy": float(gbest_acc),
        "swarm_size": config.swarm_size,
        "iterations": config.iterations,
        "w": config.w,
        "c1": config.c1,
        "c2": config.c2,
        "epoch_budget": config.epoch_budget,
        "seed": config.seed,
    }
    (outputs / "pso_best_config.json").write_text(
        json.dumps(best_payload, indent=2),
        encoding="utf-8",
    )
    return best_payload
