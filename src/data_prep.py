"""Data loading, target creation, chronological splitting and leakage guards."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf   # add near the top

from features import add_features


EXPECTED_RAW_ROWS = 619_040
EXPECTED_TICKERS = 505
EXPECTED_DAYS = 1_259   # was 1_258 — that number is the post-labelling count, not the raw one
EXPECTED_START = "2013-02-08"
EXPECTED_END = "2018-02-07"
EXPECTED_MISSING_TOTAL = 27
EXPECTED_MISSING_ROWS = 11
EXPECTED_LABELLED_ROWS = 618_524
EXPECTED_UP = 322_446
EXPECTED_DOWN = 296_078
EXPECTED_SPLIT_SIZES = (429_052, 94_340, 95_132)
EXPECTED_TRAIN_END = "2016-08-08"
EXPECTED_VAL_END = "2017-05-09"


@dataclass
class PreparedData:
    full_labelled: pd.DataFrame
    featured: pd.DataFrame
    feature_names: list[str]
    scaler: StandardScaler
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    split_dates: dict
    raw_stats: dict


def _assert_raw_facts(df: pd.DataFrame) -> dict:
    required = {"date", "open", "high", "low", "close", "volume", "Name"}
    if set(df.columns) != required:
        raise AssertionError(
            f"Expected exactly columns {sorted(required)}, got {sorted(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    assert len(df) == EXPECTED_RAW_ROWS, f"Raw rows: {len(df)} != {EXPECTED_RAW_ROWS}"
    assert df["Name"].nunique() == EXPECTED_TICKERS
    assert df["date"].nunique() == EXPECTED_DAYS
    assert df["date"].min() == pd.Timestamp(EXPECTED_START)
    assert df["date"].max() == pd.Timestamp(EXPECTED_END)

    full_dups = int(df.duplicated().sum())
    pair_dups = int(df.duplicated(["date", "Name"]).sum())
    assert full_dups == 0, f"Full duplicates found: {full_dups}"
    assert pair_dups == 0, f"Duplicate (date, Name) pairs found: {pair_dups}"

    missing_by_col = df.isna().sum().to_dict()
    missing_total = int(df.isna().sum().sum())
    missing_rows = int(df.isna().any(axis=1).sum())
    assert missing_total == EXPECTED_MISSING_TOTAL
    assert missing_rows == EXPECTED_MISSING_ROWS
    assert int(missing_by_col["open"]) == 11
    assert int(missing_by_col["high"]) == 8
    assert int(missing_by_col["low"]) == 8
    assert int(missing_by_col["close"]) == 0
    assert int(missing_by_col["volume"]) == 0
    assert int(missing_by_col["date"]) == 0
    assert int(missing_by_col["Name"]) == 0

    return {
        "raw_rows": len(df),
        "tickers": int(df["Name"].nunique()),
        "trading_days": int(df["date"].nunique()),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "missing_total": missing_total,
        "missing_rows": missing_rows,
        "missing_by_column": {k: int(v) for k, v in missing_by_col.items()},
        "full_duplicates": full_dups,
        "duplicate_date_ticker_pairs": pair_dups,
    }


def load_and_label(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    raw_stats = _assert_raw_facts(df)

    # The verified labelled-row count implies that the 11 rows with missing
    # OHLC values are removed before the per-ticker target shift:
    # 619,040 - 11 missing rows - 505 per-ticker terminal rows = 618,524.
    clean = df.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    assert len(df) - len(clean) == EXPECTED_MISSING_ROWS

    clean = clean.sort_values(["Name", "date"]).reset_index(drop=True)

    clean["next_close"] = clean.groupby("Name", sort=False)["close"].shift(-1)
    clean["target"] = (clean["next_close"] > clean["close"]).astype("Int64")

    before_drop = len(clean)
    labelled = clean.dropna(subset=["next_close", "target"]).copy()
    terminal_drops = before_drop - len(labelled)
    assert terminal_drops == EXPECTED_TICKERS
    labelled["target"] = labelled["target"].astype(np.int8)
    labelled = labelled.drop(columns=["next_close"])

    assert len(labelled) == EXPECTED_LABELLED_ROWS
    up = int(labelled["target"].sum())
    down = int((labelled["target"] == 0).sum())
    assert up == EXPECTED_UP, f"UP count {up} != {EXPECTED_UP}"
    assert down == EXPECTED_DOWN, f"DOWN count {down} != {EXPECTED_DOWN}"

    labelled = labelled.sort_values(["date", "Name"]).reset_index(drop=True)
    raw_stats["rows_removed_missing_ohlcv"] = EXPECTED_MISSING_ROWS
    raw_stats["rows_removed_target_shift"] = EXPECTED_TICKERS
    raw_stats["labelled_rows"] = len(labelled)
    raw_stats["up_count"] = up
    raw_stats["down_count"] = down
    raw_stats["up_pct"] = up / len(labelled) * 100
    raw_stats["down_pct"] = down / len(labelled) * 100
    return labelled, raw_stats


def subset_tickers(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is None or n <= 0:
        return df.copy()
    if n > df["Name"].nunique():
        raise ValueError(f"--tickers {n} exceeds available tickers {df['Name'].nunique()}")
    rng = np.random.default_rng(seed)
    names = np.array(sorted(df["Name"].unique()))
    chosen = rng.choice(names, size=n, replace=False)
    return df[df["Name"].isin(chosen)].copy()


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    dates = np.array(sorted(df["date"].unique()))
    n = len(dates)
    train_end_idx = int(np.floor(n * 0.70)) - 1
    val_end_idx = int(np.floor(n * 0.85)) - 1

    train_end_date = pd.Timestamp(dates[train_end_idx])
    val_end_date = pd.Timestamp(dates[val_end_idx])
    min_test_date = pd.Timestamp(dates[val_end_idx + 1])

    train = df[df["date"] <= train_end_date].copy()
    val = df[(df["date"] > train_end_date) & (df["date"] <= val_end_date)].copy()
    test = df[df["date"] > val_end_date].copy()

    assert train["date"].max() < val["date"].min() < test["date"].min()
    assert len(train) + len(val) + len(test) == len(df)

    split_info = {
        "n_trading_days": int(n),
        "train_days": int(train_end_idx + 1),
        "val_days": int(val_end_idx - train_end_idx),
        "test_days": int(n - val_end_idx - 1),
        "train_start": str(train["date"].min().date()),
        "train_end": str(train["date"].max().date()),
        "val_start": str(val["date"].min().date()),
        "val_end": str(val["date"].max().date()),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "train_rows_labelled": int(len(train)),
        "val_rows_labelled": int(len(val)),
        "test_rows_labelled": int(len(test)),
    }
    return train, val, test, split_info


def prepare_data(
    path: str | Path,
    ticker_count: Optional[int],
    seed: int,
) -> PreparedData:
    labelled, raw_stats = load_and_label(path)

    if ticker_count is not None:
        labelled = subset_tickers(labelled, ticker_count, seed)
        raw_stats["selected_tickers"] = int(labelled["Name"].nunique())
        raw_stats["subset_seed"] = seed
    else:
        raw_stats["selected_tickers"] = EXPECTED_TICKERS

    train0, val0, test0, split_info = chronological_split(labelled)

    if ticker_count is None:
        assert tuple(split_info[k] for k in ("train_rows_labelled", "val_rows_labelled", "test_rows_labelled")) == EXPECTED_SPLIT_SIZES
        assert split_info["train_end"] == EXPECTED_TRAIN_END
        assert split_info["val_end"] == EXPECTED_VAL_END

    # Feature engineering happens after the split definition. This ensures the
    # 70/15/15 boundaries are defined from the labelled population requested
    # by the experiment. Features themselves remain past/current-only.
    featured_all, feature_names = add_features(labelled)

    # Warm-up rows generated by rolling/lagged features are dropped. Missing
    # OHLC rows were already removed before target creation.
    featured_all = featured_all.dropna(subset=feature_names + ["target"]).copy()
    featured_all["target"] = featured_all["target"].astype(np.int8)
    featured_all = featured_all.sort_values(["date", "Name"]).reset_index(drop=True)

    # Re-split featured data using the exact already-established date boundaries.
    train = featured_all[featured_all["date"] <= pd.Timestamp(split_info["train_end"])].copy()
    val = featured_all[
        (featured_all["date"] > pd.Timestamp(split_info["train_end"]))
        & (featured_all["date"] <= pd.Timestamp(split_info["val_end"]))
    ].copy()
    test = featured_all[featured_all["date"] > pd.Timestamp(split_info["val_end"])].copy()

    assert train["date"].max() < val["date"].min() < test["date"].min()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_names].astype(np.float64))
    X_val = scaler.transform(val[feature_names].astype(np.float64))
    X_test = scaler.transform(test[feature_names].astype(np.float64))

    y_train = train["target"].to_numpy(np.float32)
    y_val = val["target"].to_numpy(np.float32)
    y_test = test["target"].to_numpy(np.float32)

    # Strong scaler leakage guard: the fitted scaler statistics must equal the
    # training matrix statistics before transformation.
    train_mean = train[feature_names].astype(np.float64).mean().to_numpy()
    train_std = train[feature_names].astype(np.float64).std(ddof=0).to_numpy()
    assert np.allclose(scaler.mean_, train_mean, rtol=1e-8, atol=1e-8)
    assert np.allclose(scaler.scale_, train_std, rtol=1e-8, atol=1e-8)

    # Update stats with feature-stage counts.
    raw_stats["feature_rows_after_warmup"] = int(len(featured_all))
    raw_stats["feature_count"] = len(feature_names)
    raw_stats["feature_names"] = feature_names
    raw_stats["split"] = split_info
    raw_stats["split_feature_rows"] = {
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
    }

    return PreparedData(
        full_labelled=labelled,
        featured=featured_all,
        feature_names=feature_names,
        scaler=scaler,
        train=train,
        val=val,
        test=test,
        X_train=X_train.astype(np.float32),
        y_train=y_train,
        X_val=X_val.astype(np.float32),
        y_val=y_val,
        X_test=X_test.astype(np.float32),
        y_test=y_test,
        split_dates=split_info,
        raw_stats=raw_stats,
    )


class WindowSequence(tf.keras.utils.Sequence):
    """Memory-efficient 30-day per-ticker window sequence.

    Windows are generated only inside a single split, so no window can cross
    a train/validation/test boundary or a ticker boundary.
    """
    def __init__(self, frame: pd.DataFrame, X: np.ndarray, y: np.ndarray, window: int, batch_size: int):
        self.frame = frame.reset_index(drop=True)
        self.X = X
        self.y = y
        self.window = int(window)
        self.batch_size = int(batch_size)
        self.indices: list[tuple[np.ndarray, int]] = []

        if len(self.frame) != len(self.X) or len(self.frame) != len(self.y):
            raise ValueError("Frame/X/y lengths do not match.")

        # Map each ticker's local row positions to windows ending at a row.
        for ticker, idx in self.frame.groupby("Name", sort=False).groups.items():
            positions = np.asarray(list(idx), dtype=np.int64)
            positions = positions[np.argsort(self.frame.loc[positions, "date"].to_numpy())]
            if len(positions) < self.window:
                continue
            for j in range(self.window - 1, len(positions)):
                self.indices.append((positions, j))

        if not self.indices:
            raise ValueError("No valid CNN windows were created.")

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_idx: int):
        batch = self.indices[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]
        xs = np.stack([self.X[pos[j - self.window + 1 : j + 1]] for pos, j in batch])
        ys = np.asarray([self.y[pos[j]] for pos, j in batch], dtype=np.float32)
        return xs.astype(np.float32), ys

    @property
    def n_samples(self) -> int:
        return len(self.indices)
