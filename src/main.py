"""End-to-end entry point for the stock direction deep-learning lab."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

# Allow direct execution with: python src/main.py
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import tensorflow as tf

from data_prep import (
    EXPECTED_DOWN,
    EXPECTED_LABELLED_ROWS,
    EXPECTED_SPLIT_SIZES,
    EXPECTED_TICKERS,
    EXPECTED_UP,
    PreparedData,
    WindowSequence,
    prepare_data,
)
from evaluate import (
    classification_metrics,
    majority_baseline,
    predict_labels,
    predict_labels_sequence,
)
from models import (
    TrainConfig,
    build_ann1,
    build_ann2,
    build_ann3,
    build_ann4,
    build_ann5,
    build_cnn,
    build_optimized_ann,
    set_seed,
    safe_name,
    train_model,
    train_sequence_model,
)
from plots import plot_history, plot_metric_bars
from pso import PSOConfig, run_pso


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    p = argparse.ArgumentParser(description="Next-day stock direction DL lab.")
    p.add_argument("--data", default="./data/all_stocks_5yr.csv")
    p.add_argument("--outputs", default="./outputs")
    p.add_argument("--tickers", type=int, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--skip-pso", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=5)

    # PSO CLI controls requested by the experiment specification.
    p.add_argument("--pso-swarm-size", type=int, default=6)
    p.add_argument("--pso-iterations", type=int, default=5)
    p.add_argument("--pso-w", type=float, default=0.7)
    p.add_argument("--pso-c1", type=float, default=1.5)
    p.add_argument("--pso-c2", type=float, default=1.5)
    p.add_argument("--pso-epoch-budget", type=int, default=3)
    return p.parse_args()


def hardware_info():
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "cpu_count": os.cpu_count(),
        "tensorflow_cpu_devices": [d.name for d in cpus],
        "tensorflow_gpu_devices": [d.name for d in gpus],
        "gpu_detected": bool(gpus),
    }


def runtime_estimate(tickers, epochs, pso):
    if tickers is None:
        base = "Full 505-ticker run: expect roughly tens of minutes to several hours on CPU, depending on hardware."
    elif tickers <= 20:
        base = f"{tickers}-ticker trial: typically minutes to under an hour, depending on hardware."
    else:
        base = f"{tickers}-ticker run: likely tens of minutes to hours, depending on hardware."
    if pso:
        base += " PSO can dominate runtime; reduce --pso-swarm-size/--pso-iterations/--pso-epoch-budget for a fast trial."
    print(f"Runtime estimate: {base}")
    print(f"Requested epochs per standard model: {epochs}")


def save_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def sequence_predictions(model, seq, threshold):
    y_true, y_pred, y_proba = predict_labels_sequence(model, seq, threshold)
    return y_true, y_pred, y_proba


def evaluate_dense_model(model, data: PreparedData, threshold: float):
    train_pred, train_prob = predict_labels(model, data.X_train, threshold)
    val_pred, val_prob = predict_labels(model, data.X_val, threshold)
    test_pred, test_prob = predict_labels(model, data.X_test, threshold)

    train_m = classification_metrics(data.y_train, train_pred, train_prob)
    val_m = classification_metrics(data.y_val, val_pred, val_prob)
    test_m = classification_metrics(data.y_test, test_pred, test_prob)
    return train_m, val_m, test_m


def evaluate_cnn_model(model, train_seq, val_seq, test_seq, threshold):
    train_y, train_p, train_prob = sequence_predictions(model, train_seq, threshold)
    val_y, val_p, val_prob = sequence_predictions(model, val_seq, threshold)
    test_y, test_p, test_prob = sequence_predictions(model, test_seq, threshold)
    return (
        classification_metrics(train_y, train_p, train_prob),
        classification_metrics(val_y, val_p, val_prob),
        classification_metrics(test_y, test_p, test_prob),
    )


def main():
    args = parse_args()
    outputs = Path(args.outputs)
    outputs.mkdir(parents=True, exist_ok=True)

    log_path = outputs / "run_log.txt"
    with log_path.open("w", encoding="utf-8") as log_file, redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
        run(args, outputs)


def run(args, outputs: Path):
    set_seed(args.seed)
    hw = hardware_info()
    print("=" * 80)
    print("NEXT-DAY STOCK DIRECTION DEEP LEARNING LAB")
    print("=" * 80)
    print(json.dumps(hw, indent=2))
    if not hw["gpu_detected"]:
        print("WARNING: TensorFlow sees no GPU; training will run on CPU.")

    runtime_estimate(args.tickers, args.epochs, not args.skip_pso)

    data = prepare_data(args.data, args.tickers, args.seed)
    print("\nRaw/labelled dataset statistics:")
    print(json.dumps(data.raw_stats, indent=2, default=str))
    print("\nFeature count:", len(data.feature_names))
    print("Full feature list:")
    for i, name in enumerate(data.feature_names, 1):
        print(f"  {i:02d}. {name}")

    if args.tickers is None:
        assert len(data.full_labelled) == EXPECTED_LABELLED_ROWS
        assert int(data.full_labelled["target"].sum()) == EXPECTED_UP
        assert int((data.full_labelled["target"] == 0).sum()) == EXPECTED_DOWN

    # Create split-specific CNN sequences. They are built separately, so no
    # window can cross a date boundary or ticker boundary.
    train_seq = WindowSequence(data.train, data.X_train, data.y_train, args.window, args.batch_size)
    val_seq = WindowSequence(data.val, data.X_val, data.y_val, args.window, args.batch_size)
    test_seq = WindowSequence(data.test, data.X_test, data.y_test, args.window, args.batch_size)
    print(
        f"\nCNN windows: train={train_seq.n_samples}, "
        f"val={val_seq.n_samples}, test={test_seq.n_samples}, "
        f"window={args.window}"
    )

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        decision_threshold=0.5,
    )
    print(
        "\nExact standard training protocol: "
        f"epochs={config.epochs}, batch_size={config.batch_size}, "
        f"learning_rate={config.lr}, patience={config.patience}, "
        f"seed={config.seed}, loss=binary_crossentropy, optimizer=Adam, "
        f"threshold={config.decision_threshold}, shuffle=False"
    )

    builders = [
        ("ANN_1_Basic", build_ann1),
        ("ANN_2_Deep", build_ann2),
        ("ANN_3_Wide", build_ann3),
        ("ANN_4_Dropout", build_ann4),
        ("ANN_5_BatchNorm", build_ann5),
    ]

    results = []
    histories = {}

    for name, builder in builders:
        print("\n" + "=" * 80)
        print(f"TRAINING {name}")
        print("=" * 80)
        set_seed(args.seed)
        model = builder(len(data.feature_names), args.lr)
        model.summary(print_fn=print)
        history = train_model(
            model, data.X_train, data.y_train, data.X_val, data.y_val, config
        )
        histories[name] = history
        plot_history(history, safe_name(name), outputs)

        # Test is evaluated here exactly once for this trained model, after all
        # training/selection is complete.
        train_m, val_m, test_m = evaluate_dense_model(
            model, data, config.decision_threshold
        )
        results.append({
            "Model": name,
            "Acc_Train": train_m["accuracy"],
            "Acc_Val": val_m["accuracy"],
            "Acc_Test": test_m["accuracy"],
            "Precision": test_m["precision"],
            "Recall": test_m["recall"],
            "F1": test_m["f1"],
            "ROC_AUC": test_m["roc_auc"],
            "MCC": test_m["mcc"],
            "FPR": test_m["fpr"],
            "FNR": test_m["fnr"],
        })
        print("Final metrics:", results[-1])
        del model
        tf.keras.backend.clear_session()

    # 1D-CNN uses exactly the same standard epochs/batch/lr/early-stopping policy.
    print("\n" + "=" * 80)
    print("TRAINING CNN_1D")
    print("=" * 80)
    set_seed(args.seed)
    cnn = build_cnn(len(data.feature_names), args.window, args.lr)
    cnn.summary(print_fn=print)
    cnn_history = train_sequence_model(cnn, train_seq, val_seq, config)
    histories["CNN_1D"] = cnn_history
    plot_history(cnn_history, safe_name("CNN_1D"), outputs)
    train_m, val_m, test_m = evaluate_cnn_model(
        cnn, train_seq, val_seq, test_seq, config.decision_threshold
    )
    results.append({
        "Model": "CNN_1D",
        "Acc_Train": train_m["accuracy"],
        "Acc_Val": val_m["accuracy"],
        "Acc_Test": test_m["accuracy"],
        "Precision": test_m["precision"],
        "Recall": test_m["recall"],
        "F1": test_m["f1"],
        "ROC_AUC": test_m["roc_auc"],
        "MCC": test_m["mcc"],
        "FPR": test_m["fpr"],
        "FNR": test_m["fnr"],
    })
    print("Final metrics:", results[-1])
    del cnn
    tf.keras.backend.clear_session()

    # PSO uses validation only. Test data is never passed to PSO.
    if args.skip_pso:
        print("\nPSO skipped by --skip-pso.")
        best_hp = {
            "hidden_layers": 3,
            "neurons": [128, 64, 32, 32],
            "learning_rate": args.lr,
            "dropout": 0.2,
            "batch_size": args.batch_size,
            "skipped": True,
        }
        save_json(outputs / "pso_best_config.json", best_hp)
        pd.DataFrame(
            columns=[
                "iteration", "particle", "fitness_val_f1", "val_accuracy",
                "hidden_layers", "neurons", "learning_rate", "dropout", "batch_size"
            ]
        ).to_csv(outputs / "pso_log.csv", index=False)
    else:
        print("\n" + "=" * 80)
        print("PARTICLE SWARM OPTIMIZATION — VALIDATION ONLY")
        print("=" * 80)
        pso_cfg = PSOConfig(
            swarm_size=args.pso_swarm_size,
            iterations=args.pso_iterations,
            w=args.pso_w,
            c1=args.pso_c1,
            c2=args.pso_c2,
            epoch_budget=args.pso_epoch_budget,
            seed=args.seed,
        )
        best_hp = run_pso(
            data.X_train,
            data.y_train,
            data.X_val,
            data.y_val,
            len(data.feature_names),
            pso_cfg,
            outputs,
        )
        best_hp["skipped"] = False

    print("\nPSO best configuration:")
    print(json.dumps(best_hp, indent=2))

    # Retrain the best PSO configuration fully. Only now is the test set used.
    print("\n" + "=" * 80)
    print("TRAINING OPTIMIZED_ANN WITH BEST PSO CONFIGURATION")
    print("=" * 80)
    set_seed(args.seed)
    opt_model = build_optimized_ann(
        len(data.feature_names),
        best_hp["hidden_layers"],
        best_hp["neurons"],
        best_hp["dropout"],
        best_hp["learning_rate"],
    )
    opt_config = TrainConfig(
        epochs=args.epochs,
        batch_size=best_hp["batch_size"],
        lr=best_hp["learning_rate"],
        patience=args.patience,
        seed=args.seed,
        decision_threshold=0.5,
    )
    opt_model.summary(print_fn=print)
    opt_history = train_model(
        opt_model,
        data.X_train,
        data.y_train,
        data.X_val,
        data.y_val,
        opt_config,
    )
    histories["Optimized_ANN"] = opt_history
    plot_history(opt_history, safe_name("Optimized_ANN"), outputs)
    train_m, val_m, test_m = evaluate_dense_model(
        opt_model, data, opt_config.decision_threshold
    )
    results.append({
        "Model": "Optimized_ANN",
        "Acc_Train": train_m["accuracy"],
        "Acc_Val": val_m["accuracy"],
        "Acc_Test": test_m["accuracy"],
        "Precision": test_m["precision"],
        "Recall": test_m["recall"],
        "F1": test_m["f1"],
        "ROC_AUC": test_m["roc_auc"],
        "MCC": test_m["mcc"],
        "FPR": test_m["fpr"],
        "FNR": test_m["fnr"],
    })
    print("Final metrics:", results[-1])

    # Majority baseline is calculated from training majority class and reported
    # on each split. It is not used for model selection.
    baseline = majority_baseline(data.y_train, data.y_val, data.y_test)
    print("\nMajority-class baseline:")
    print(json.dumps(baseline, indent=2))
    baseline_row = {
        "Model": "Majority Baseline",
        "Acc_Train": baseline["train"]["accuracy"],
        "Acc_Val": baseline["val"]["accuracy"],
        "Acc_Test": baseline["test"]["accuracy"],
        "Precision": baseline["test"]["precision"],
        "Recall": baseline["test"]["recall"],
        "F1": baseline["test"]["f1"],
        "ROC_AUC": np.nan,
        "MCC": baseline["test"]["mcc"],
        "FPR": baseline["test"]["fpr"],
        "FNR": baseline["test"]["fnr"],
    }
    results.append(baseline_row)

    # For the verified full dataset, the requested test baseline should be 53.09%.
    if args.tickers is None:
        assert abs(baseline_row["Acc_Test"] - 0.5309) < 0.002, (
            f"Unexpected test majority baseline: {baseline_row['Acc_Test']:.6f}"
        )

    table = pd.DataFrame(
        results,
        columns=[
            "Model", "Acc_Train", "Acc_Val", "Acc_Test",
            "Precision", "Recall", "F1", "ROC_AUC", "MCC", "FPR", "FNR",
        ],
    )
    table.to_csv(outputs / "results_table3.csv", index=False)
    table.to_markdown(outputs / "results_table3.md", index=False, floatfmt=".6f")

    plot_metric_bars(results, outputs, baseline_row["Acc_Test"])

    dataset_stats = dict(data.raw_stats)
    dataset_stats["feature_count"] = len(data.feature_names)
    dataset_stats["feature_names"] = data.feature_names
    dataset_stats["cnn_window"] = args.window
    dataset_stats["cnn_window_counts"] = {
        "train": train_seq.n_samples,
        "val": val_seq.n_samples,
        "test": test_seq.n_samples,
    }
    dataset_stats["majority_baseline"] = baseline
    save_json(outputs / "dataset_stats.json", dataset_stats)

    config_payload = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "loss": "binary_crossentropy",
        "window_length": args.window,
        "early_stopping_patience": args.patience,
        "decision_threshold": 0.5,
        "seed": args.seed,
        "shuffle": False,
        "tickers_argument": args.tickers,
        "data_path": str(Path(args.data).resolve()),
        "pso": {
            "skipped": args.skip_pso,
            "swarm_size": args.pso_swarm_size,
            "iterations": args.pso_iterations,
            "w": args.pso_w,
            "c1": args.pso_c1,
            "c2": args.pso_c2,
            "epoch_budget": args.pso_epoch_budget,
        },
        "framework_versions": hw,
        "hardware": hw,
        "feature_count": len(data.feature_names),
        "feature_names": data.feature_names,
    }
    save_json(outputs / "experiment_config.json", config_payload)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results: {outputs / 'results_table3.csv'}")
    print(f"Report table: {outputs / 'results_table3.md'}")
    print(f"Dataset stats: {outputs / 'dataset_stats.json'}")
    print(f"Config: {outputs / 'experiment_config.json'}")
    print(f"Run log: {outputs / 'run_log.txt'}")
    print("\nFinal results:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
