"""Train baseline and production ETA models; write metrics and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from generate_data import FEATURE_COLUMNS, TARGET_COLUMN, generate_deliveries

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_min": float(mean_absolute_error(y_true, y_pred)),
        "rmse_min": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / "deliveries.csv"
    if not csv_path.exists():
        df = generate_deliveries()
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=120, max_depth=12, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        predictions[name] = pred
        results[name] = metrics_dict(y_test.to_numpy(), pred)
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        print(f"{name}: MAE={results[name]['mae_min']:.3f} RMSE={results[name]['rmse_min']:.3f}")

    best_name = min(results, key=lambda k: results[k]["mae_min"])
    best_model = models[best_name]
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": results[best_name],
        },
        MODEL_DIR / "production_eta.joblib",
    )

    summary = {
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "feature_columns": FEATURE_COLUMNS,
        "models": results,
        "selected_production_model": best_name,
        "note": "Metrics are from this synthetic demo run (measured), not the architecture design targets.",
    }
    metrics_path = OUTPUT_DIR / "training_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sample = x_test.head(8).copy()
    sample["actual_delivery_min"] = y_test.head(8).to_numpy()
    sample["pred_xgboost_min"] = predictions["xgboost"][:8]
    sample["abs_error_min"] = (
        sample["actual_delivery_min"] - sample["pred_xgboost_min"]
    ).abs()
    sample_path = OUTPUT_DIR / "sample_predictions.csv"
    sample.to_csv(sample_path, index=False)

    lines = [
        "# Training Results — Food Delivery ETA Predictor",
        "",
        f"- Train samples: **{len(x_train)}** | Test samples: **{len(x_test)}**",
        f"- Selected production model: **{best_name}**",
        "",
        "| Model | MAE (min) | RMSE (min) | R² |",
        "|-------|-----------|------------|----|",
    ]
    for name, m in results.items():
        mark = " ← production" if name == best_name else ""
        lines.append(
            f"| {name}{mark} | {m['mae_min']:.3f} | {m['rmse_min']:.3f} | {m['r2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Sample predictions (XGBoost)",
            "",
            "See `sample_predictions.csv` for full rows.",
            "",
            "```",
            sample.round(2).to_string(index=False),
            "```",
            "",
            f"Artifacts: `{MODEL_DIR}` | Metrics: `{metrics_path}`",
        ]
    )
    (OUTPUT_DIR / "training_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nProduction model: {best_name}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {sample_path}")
    print(f"Wrote {OUTPUT_DIR / 'training_report.md'}")


if __name__ == "__main__":
    main()
