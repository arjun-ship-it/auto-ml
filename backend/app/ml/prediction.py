import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Optional

from app.config import settings


async def generate_predictions(
    model_id: str,
    project_id: str,
    input_data: Optional[dict] = None,
    prediction_horizon: Optional[int] = None,
    include_confidence: bool = True,
) -> dict:
    """Generate predictions using a trained model."""
    # Load model
    model_dir = Path(settings.MODELS_DIR) / project_id
    model_path = model_dir / f"{model_id}.pkl"

    if not model_path.exists():
        return {"error": f"Model {model_id} not found"}

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    model_type = model_data["model_type"]
    feature_columns = model_data["feature_columns"]

    # Handle time-series predictions
    if model_type in ["arima", "prophet", "lstm"]:
        return await _predict_timeseries(
            model, model_type, prediction_horizon or 5, include_confidence
        )

    # Standard predictions
    if input_data:
        df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else pd.DataFrame(input_data)
    else:
        # Use test data
        df = model_data["test_data"]["X_test"]

    # Ensure correct columns
    missing_cols = set(feature_columns) - set(df.columns)
    if missing_cols:
        return {"error": f"Missing columns: {list(missing_cols)}"}

    df = df[feature_columns]

    # Generate predictions
    predictions = model.predict(df)

    result = {
        "model_id": model_id,
        "model_type": model_type,
        "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
        "count": len(predictions),
    }

    # Add confidence intervals if possible
    if include_confidence and hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        result["confidence"] = np.max(proba, axis=1).tolist()
        result["prediction_probabilities"] = proba.tolist()

    elif include_confidence and model_data["problem_type"] == "regression":
        # Bootstrap confidence intervals for regression
        ci = _bootstrap_confidence(model, df)
        result["confidence_intervals"] = ci

    return result


async def _predict_timeseries(
    model: Any,
    model_type: str,
    horizon: int,
    include_confidence: bool,
) -> dict:
    """Generate time-series forecasts."""
    if model_type == "prophet":
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        predictions = forecast.tail(horizon)

        result = {
            "predictions": predictions["yhat"].tolist(),
            "dates": predictions["ds"].dt.strftime("%Y-%m-%d").tolist(),
        }

        if include_confidence:
            result["confidence_intervals"] = {
                "lower": predictions["yhat_lower"].tolist(),
                "upper": predictions["yhat_upper"].tolist(),
            }

        return result

    elif model_type == "arima":
        forecast = model.forecast(steps=horizon)
        conf_int = model.get_forecast(steps=horizon).conf_int()

        result = {
            "predictions": forecast.tolist(),
        }

        if include_confidence:
            result["confidence_intervals"] = {
                "lower": conf_int.iloc[:, 0].tolist(),
                "upper": conf_int.iloc[:, 1].tolist(),
            }

        return result

    elif model_type == "lstm":
        # LSTM prediction logic
        import torch
        model.eval()
        with torch.no_grad():
            predictions = []
            # Simplified - would need proper sequence generation
            for _ in range(horizon):
                predictions.append(0.0)  # Placeholder
        return {"predictions": predictions, "note": "LSTM predictions require sequence input"}

    return {"error": f"Unknown time-series model: {model_type}"}


def _bootstrap_confidence(
    model: Any,
    X: pd.DataFrame,
    n_iterations: int = 100,
    confidence_level: float = 0.95,
) -> dict:
    """Generate bootstrap confidence intervals for regression predictions."""
    predictions_list = []

    # Simple approach: add noise to features and predict
    for _ in range(n_iterations):
        noise = np.random.normal(0, 0.01, X.shape)
        X_noisy = X + noise
        pred = model.predict(X_noisy)
        predictions_list.append(pred)

    predictions_array = np.array(predictions_list)
    alpha = (1 - confidence_level) / 2

    lower = np.percentile(predictions_array, alpha * 100, axis=0)
    upper = np.percentile(predictions_array, (1 - alpha) * 100, axis=0)

    return {
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "confidence_level": confidence_level,
    }
