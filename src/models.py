"""Model builders and training utilities."""
from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import tensorflow as tf


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    patience: int = 5
    seed: int = 42
    decision_threshold: float = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def compile_model(model: tf.keras.Model, lr: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def build_ann1(n_features: int, lr: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="ANN_1_Basic",
    )
    return compile_model(model, lr)


def build_ann2(n_features: int, lr: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="ANN_2_Deep",
    )
    return compile_model(model, lr)


def build_ann3(n_features: int, lr: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="ANN_3_Wide",
    )
    return compile_model(model, lr)


def build_ann4(n_features: int, lr: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="ANN_4_Dropout",
    )
    return compile_model(model, lr)


def build_ann5(n_features: int, lr: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="ANN_5_BatchNorm",
    )
    return compile_model(model, lr)


def build_cnn(n_features: int, window: int, lr: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, n_features)),
            tf.keras.layers.Conv1D(64, 5, padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(128, 3, padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="CNN_1D",
    )
    return compile_model(model, lr)


def build_optimized_ann(
    n_features: int,
    hidden_layers: int,
    neurons: list[int],
    dropout: float,
    lr: float,
) -> tf.keras.Model:
    layers = [tf.keras.layers.Input(shape=(n_features,))]
    for i in range(hidden_layers):
        layers.append(tf.keras.layers.Dense(int(neurons[i]), activation="relu"))
        if dropout > 0:
            layers.append(tf.keras.layers.Dropout(float(dropout)))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers, name="Optimized_ANN")
    return compile_model(model, lr)


def train_model(
    model: tf.keras.Model,
    x_train,
    y_train,
    x_val,
    y_val,
    config: TrainConfig,
):
    set_seed(config.seed)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=False,
        callbacks=[callback],
        verbose=2,
    )
    return history


def train_sequence_model(
    model: tf.keras.Model,
    train_seq,
    val_seq,
    config: TrainConfig,
):
    set_seed(config.seed)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=config.epochs,
        callbacks=[callback],
        verbose=2,
    )
    return history


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
