# Deep Learning Lab — Next-Day Stock Direction Prediction

This project implements the complete experiment requested for the S&P 500
`all_stocks_5yr.csv` dataset:

- 5 ANN variants
- 1D-CNN with 30-day within-ticker windows
- From-scratch Particle Swarm Optimization (PSO) for an optimized ANN
- Chronological 70/15/15 split
- Train-only StandardScaler fitting
- Past-only rolling/lagged features
- Leakage guards
- Report-ready CSV/Markdown tables
- Accuracy/F1/ROC-AUC plots
- Training/validation curves
- JSON dataset/config metadata
- Full stdout capture in `outputs/run_log.txt`

## Expected project layout

```text
stock_dl_lab/
├── data/
│   └── all_stocks_5yr.csv
├── outputs/
├── src/
│   ├── data_prep.py
│   ├── features.py
│   ├── models.py
│   ├── pso.py
│   ├── evaluate.py
│   ├── plots.py
│   └── main.py
├── requirements.txt
└── README.md
```

## 1. Create the environment

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows CMD

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Put the dataset in the required location

```text
./data/all_stocks_5yr.csv
```

The implementation asserts the verified full-dataset facts supplied for this
experiment before proceeding:

- 619,040 raw rows
- 505 tickers
- 1,258 trading days
- 2013-02-08 through 2018-02-07
- 27 missing values across 11 rows
- 0 full duplicates
- 0 duplicate `(date, Name)` pairs
- 618,524 labelled rows after removing the 11 missing-OHLC rows and one
  terminal target row per ticker
- 322,446 UP
- 296,078 DOWN
- chronological labelled split of 429,052 / 94,340 / 95,132 rows
- train end 2016-08-08
- validation end 2017-05-09

If the file is different, the program fails loudly instead of silently using
different data.

## 3. Fast smoke/trial run

For a quick run on 10 random tickers:

```bash
python src/main.py --tickers 10 --epochs 5 --batch-size 32 --pso-swarm-size 2 --pso-iterations 2 --pso-epoch-budget 2
```

For a faster trial without PSO:

```bash
python src/main.py --tickers 10 --epochs 5 --skip-pso
```

## 4. Full experiment

The requested full experiment is:

```bash
python src/main.py
```

This uses:

- all 505 tickers
- 30 epochs
- batch size 32
- Adam
- learning rate 0.001
- binary cross-entropy
- early stopping patience 5
- seed 42
- 30-day CNN window
- PSO swarm 6
- PSO 5 iterations
- PSO reduced epoch budget 3

You can change the standard training settings:

```bash
python src/main.py --epochs 50 --batch-size 64 --lr 0.0005 --seed 123
```

Skip PSO:

```bash
python src/main.py --skip-pso
```

Change CNN window:

```bash
python src/main.py --window 30
```

## 5. PSO controls

All requested PSO controls are command-line configurable:

```bash
python src/main.py \
  --pso-swarm-size 8 \
  --pso-iterations 10 \
  --pso-w 0.7 \
  --pso-c1 1.5 \
  --pso-c2 1.5 \
  --pso-epoch-budget 3
```

On Windows CMD, put the command on one line if needed.

PSO fitness is **validation F1 only**. Test data is not passed to PSO.

## 6. Outputs

The program writes to `./outputs/`:

```text
results_table3.csv
results_table3.md

fig3_accuracy.png
fig3_f1.png
fig3_rocauc.png

curves_ANN_1_Basic_accuracy.png
curves_ANN_1_Basic_loss.png
curves_ANN_2_Deep_accuracy.png
curves_ANN_2_Deep_loss.png
curves_ANN_3_Wide_accuracy.png
curves_ANN_3_Wide_loss.png
curves_ANN_4_Dropout_accuracy.png
curves_ANN_4_Dropout_loss.png
curves_ANN_5_BatchNorm_accuracy.png
curves_ANN_5_BatchNorm_loss.png
curves_CNN_1D_accuracy.png
curves_CNN_1D_loss.png
curves_Optimized_ANN_accuracy.png
curves_Optimized_ANN_loss.png

dataset_stats.json
experiment_config.json
pso_log.csv
pso_best_config.json
run_log.txt
```

## 7. Leakage controls implemented

### Chronological split

No random split and no shuffled training batches:

```text
earliest 70% of dates → train
next 15%              → validation
last 15%              → test
```

The code asserts:

```text
max(train date) < min(validation date) < min(test date)
```

### StandardScaler

The scaler is fitted only on training rows:

```text
fit(train)
transform(validation)
transform(test)
```

### Feature engineering

Every rolling/lagged feature is calculated within `Name` using only past and
current observations.

No centered rolling windows are used.

### Target

The next-day target is generated with:

```python
df.groupby("Name")["close"].shift(-1)
```

so one company cannot inherit another company's future price.

### CNN

The 30-day windows are constructed separately for train, validation and test
and separately by ticker. Therefore:

- no window crosses a ticker boundary
- no window crosses a train/validation boundary
- no window crosses a validation/test boundary

### PSO

PSO receives only training and validation arrays.

The test set is not used for PSO fitness or model selection.

### Test evaluation

The test set is evaluated only after the model has been fully trained/selected.
No test metric is used to choose hyperparameters.

## 8. Model list

### ANN 1 — Basic

```text
Dense(32, relu)
Dense(16, relu)
Dense(1, sigmoid)
```

### ANN 2 — Deep

```text
Dense(128, relu)
Dense(64, relu)
Dense(32, relu)
Dense(16, relu)
Dense(1, sigmoid)
```

### ANN 3 — Wide

```text
Dense(256, relu)
Dense(128, relu)
Dense(64, relu)
Dense(1, sigmoid)
```

### ANN 4 — Dropout

```text
Dense(128, relu)
Dropout(0.30)
Dense(64, relu)
Dropout(0.20)
Dense(32, relu)
Dense(1, sigmoid)
```

### ANN 5 — Batch Normalization

```text
Dense(128)
BatchNormalization
ReLU
Dense(64)
BatchNormalization
ReLU
Dense(32, relu)
Dense(1, sigmoid)
```

### 1D-CNN

```text
Conv1D
ReLU
MaxPooling1D
Conv1D
ReLU
GlobalMaxPooling1D
Dense(64, relu)
Dropout(0.30)
Dense(1, sigmoid)
```

### Optimized ANN

PSO chooses:

- 1–4 hidden layers
- neurons for up to four layers
- learning rate on a log scale
- dropout rate
- batch size

The best configuration is retrained with the normal epoch budget and then
evaluated on the test set.

## 9. Report use

`outputs/results_table3.md` can be pasted into the report.

The three `fig3_*.png` files are the requested model comparison graphs.

The per-model `curves_*.png` files are training-vs-validation curves.

`dataset_stats.json` contains the feature count and complete feature list needed
for the dataset-description section.

`experiment_config.json` contains the exact experiment configuration and
detected hardware/framework versions.

`pso_log.csv` and `pso_best_config.json` provide the optimization evidence for
the methodology/results section.

## 10. CPU-only execution

If TensorFlow detects no GPU, the program continues on CPU and prints a warning.
For the full 505-ticker experiment, CPU execution can take a long time.

For a trial run use:

```bash
python src/main.py --tickers 10 --epochs 5 --skip-pso
```

Then increase the ticker count and epoch budget after verifying the pipeline.
