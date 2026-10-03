"""Simple assessment for the Sokoto MinMax wind dataset.

This script keeps only the parts needed to:
1. load the CSV,
2. build a valid datetime index,
3. summarize the wind-speed data,
4. evaluate a simple persistence baseline for a few forecast horizons.
5. optionally search a bounded set of LSTM parameters and save forecasts.

It intentionally avoids the larger two-tier BiLSTM/Weibull pipeline and is
better suited for quick assessment of the dataset you have in Sokoto minmax.csv.
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(42)


def load_sokoto_csv(input_path: str) -> pd.DataFrame:
    """Load the Sokoto MinMax CSV and build a proper datetime column."""
    path = Path(input_path)
    if not path.is_absolute() and not path.exists():
        candidates = [
            Path(__file__).resolve().parent / path,
            Path(__file__).resolve().parent / "thesis project" / path,
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), path)

    df = pd.read_csv(path)
    df = df.dropna(axis=1, how="all")

    required_date_cols = ["YEAR", "MO", "DY", "HR"]
    if not all(col in df.columns for col in required_date_cols):
        raise ValueError(
            f"The file must contain YEAR, MO, DY, and HR columns. Found: {list(df.columns[:10])}"
        )

    df["datetime"] = pd.to_datetime(
        df[["YEAR", "MO", "DY", "HR"]].rename(
            columns={"YEAR": "year", "MO": "month", "DY": "day", "HR": "hour"}
        ),
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def summarize_wind(df: pd.DataFrame, target: str) -> dict:
    values = pd.to_numeric(df[target], errors="coerce")
    summary = {
        "rows": int(len(df)),
        "valid_target_values": int(values.notna().sum()),
        "missing_target_pct": float((1 - values.notna().mean()) * 100),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "start_datetime": str(df["datetime"].min()),
        "end_datetime": str(df["datetime"].max()),
    }
    return summary


def evaluate_persistence(df: pd.DataFrame, target: str, horizon_hours: int) -> dict:
    series = pd.to_numeric(df[target], errors="coerce")
    series = series.set_axis(df["datetime"])
    series = series.sort_index()
    series = series.interpolate().ffill().bfill()

    y_true = series.iloc[horizon_hours:].to_numpy()
    y_pred = series.shift(horizon_hours).iloc[horizon_hours:].to_numpy()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"horizon_hours": horizon_hours, "RMSE": None, "MAE": None, "R2": None}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "horizon_hours": horizon_hours,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def make_sequences(values: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for end in range(lookback, len(values) - horizon + 1):
        X.append(values[end - lookback:end])
        y.append(values[end + horizon - 1])
    return np.asarray(X), np.asarray(y)


def build_tunable_lstm(lookback: int, units: int, dropout: float):
    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(max(8, units // 2), activation="relu"),
        layers.Dense(1),
    ])
    return model


def prepare_lstm_data(df: pd.DataFrame, target: str, horizon: int, lookback: int):
    series = pd.to_numeric(df[target], errors="coerce").interpolate().ffill().bfill().to_numpy()
    if len(series) <= lookback + horizon + 20:
        raise ValueError("Not enough valid data for the requested lookback and horizon")

    split_train = int(len(series) * 0.70)
    split_val = int(len(series) * 0.85)
    scaler = MinMaxScaler().fit(series[:split_train].reshape(-1, 1))
    scaled = scaler.transform(series.reshape(-1, 1)).ravel()
    X, y = make_sequences(scaled, lookback, horizon)
    train_end = max(1, split_train - lookback - horizon + 1)
    val_end = max(train_end + 1, split_val - lookback - horizon + 1)
    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:], scaler, series,
    )


def score_predictions(actual, predicted):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)),
    }


def train_candidate(data, config, max_train_samples: int, max_val_samples: int):
    X_train, y_train, X_val, y_val = data[:4]
    if max_train_samples and len(X_train) > max_train_samples:
        X_train, y_train = X_train[-max_train_samples:], y_train[-max_train_samples:]
    if max_val_samples and len(X_val) > max_val_samples:
        X_val, y_val = X_val[-max_val_samples:], y_val[-max_val_samples:]

    model = build_tunable_lstm(config["lookback"], config["units"], config["dropout"])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="mse",
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    model.fit(
        X_train[..., np.newaxis], y_train,
        validation_data=(X_val[..., np.newaxis], y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=0,
        callbacks=[early_stop],
    )
    validation_prediction = model.predict(X_val[..., np.newaxis], verbose=0).ravel()
    return model, score_predictions(y_val, validation_prediction)


def tune_lstm(df: pd.DataFrame, target: str, horizon: int, outdir: Path, args):
    search_space = {
        "lookback": [int(value) for value in args.tune_lookbacks.split(",")],
        "units": [int(value) for value in args.tune_units.split(",")],
        "dropout": [float(value) for value in args.tune_dropouts.split(",")],
        "learning_rate": [float(value) for value in args.tune_learning_rates.split(",")],
        "batch_size": [int(value) for value in args.tune_batch_sizes.split(",")],
        "epochs": [args.tune_epochs],
    }
    keys = list(search_space)
    combinations = [dict(zip(keys, values)) for values in itertools.product(*(search_space[key] for key in keys))]
    combinations = combinations[:args.tuning_combinations]
    results = []
    best = None

    for index, config in enumerate(combinations, start=1):
        data = prepare_lstm_data(df, target, horizon, config["lookback"])
        model, metrics = train_candidate(
            data, config, args.tune_max_train_samples, args.tune_max_val_samples
        )
        result = {**config, **metrics}
        results.append(result)
        if best is None or result["RMSE"] < best["RMSE"]:
            best = result
        print(f"Tuning {index}/{len(combinations)}: RMSE={result['RMSE']:.6f} {config}")
        del model
        keras.backend.clear_session()

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(outdir / "lstm_parameter_search.csv", index=False)
    with open(outdir / "best_lstm_parameters.json", "w") as file:
        json.dump({"horizon_hours": horizon, "best_parameters": best}, file, indent=2)

    data = prepare_lstm_data(df, target, horizon, best["lookback"])
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, series = data
    X_fit = np.concatenate([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    model = build_tunable_lstm(best["lookback"], best["units"], best["dropout"])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best["learning_rate"]),
        loss="mse",
    )
    model.fit(
        X_fit[..., np.newaxis], y_fit,
        epochs=best["epochs"], batch_size=best["batch_size"], verbose=0,
    )
    prediction_scaled = model.predict(X_test[..., np.newaxis], verbose=0).ravel()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()
    forecast = pd.DataFrame({"actual": actual, "forecast": prediction})
    forecast.to_csv(outdir / "best_lstm_forecasts.csv", index=False)
    best["test_metrics"] = score_predictions(actual, prediction)
    with open(outdir / "best_lstm_parameters.json", "w") as file:
        json.dump({"horizon_hours": horizon, "best_parameters": best}, file, indent=2)
    return best


def save_monthly_summary(df: pd.DataFrame, target: str, outdir: Path):
    values = pd.to_numeric(df[target], errors="coerce")
    monthly = values.set_axis(df["datetime"]).resample("MS").mean().to_frame("mean_wind_speed")
    monthly.to_csv(outdir / "monthly_mean_wind.csv")
    return monthly


def plot_time_series(df: pd.DataFrame, target: str, outdir: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["datetime"], pd.to_numeric(df[target], errors="coerce"), alpha=0.7)
    ax.set_title(f"{target} time series")
    ax.set_xlabel("datetime")
    ax.set_ylabel(target)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "wind_time_series.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Quick assessment of the Sokoto MinMax wind dataset")
    parser.add_argument("--input", default="Sokoto minmax.csv", help="Path to the dataset CSV")
    parser.add_argument("--target", default="WS10M", help="Wind speed column to assess")
    parser.add_argument("--output-dir", default="sokoto_assessment", help="Directory to save outputs")
    parser.add_argument(
        "--horizons",
        default="1,6,24,168",
        help="Persistence forecast horizons in hours, comma-separated",
    )
    parser.add_argument("--tune", action="store_true", help="Search LSTM parameter combinations")
    parser.add_argument("--tune-horizon", type=int, default=24, help="Horizon used during LSTM tuning")
    parser.add_argument("--tuning-combinations", type=int, default=24, help="Maximum combinations to test")
    parser.add_argument("--tune-lookbacks", default="24,48,72", help="Lookbacks to test")
    parser.add_argument("--tune-units", default="16,32", help="LSTM unit counts to test")
    parser.add_argument("--tune-dropouts", default="0.0,0.2", help="Dropout values to test")
    parser.add_argument("--tune-learning-rates", default="0.001,0.0005", help="Learning rates to test")
    parser.add_argument("--tune-batch-sizes", default="64", help="Batch sizes to test")
    parser.add_argument("--tune-epochs", type=int, default=8, help="Maximum epochs per candidate")
    parser.add_argument("--tune-max-train-samples", type=int, default=6000, help="Training samples per candidate")
    parser.add_argument("--tune-max-val-samples", type=int, default=1500, help="Validation samples per candidate")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(exist_ok=True, parents=True)

    df = load_sokoto_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Available columns: {list(df.columns[:20])}")

    report = summarize_wind(df, args.target)
    save_monthly_summary(df, args.target, outdir)
    plot_time_series(df, args.target, outdir)

    horizon_values = [int(v.strip()) for v in args.horizons.split(",") if v.strip()]
    baseline_results = {}
    for horizon in horizon_values:
        baseline_results[f"h{horizon}h"] = evaluate_persistence(df, args.target, horizon)

    output = {
        "dataset_summary": report,
        "persistence_baseline": baseline_results,
    }

    if args.tune:
        output["best_lstm_search"] = tune_lstm(
            df, args.target, args.tune_horizon, outdir, args
        )

    with open(outdir / "assessment_summary.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Sokoto wind assessment summary:")
    print(json.dumps(report, indent=2))
    print("\nPersistence baseline metrics:")
    for name, metrics in baseline_results.items():
        print(
            f"  {name}: RMSE={metrics['RMSE']:.4f} "
            f"MAE={metrics['MAE']:.4f} R2={metrics['R2']:.4f}"
        )
    if "best_lstm_search" in output:
        best = output["best_lstm_search"]
        print(f"\nBest LSTM parameters: {json.dumps(best, indent=2)}")
    print(f"\nSaved outputs to: {outdir.resolve()}")
    print(f"Monthly means saved to: {outdir / 'monthly_mean_wind.csv'}")


if __name__ == "__main__":
    main()
