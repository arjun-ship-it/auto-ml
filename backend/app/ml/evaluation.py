import pickle
import numpy as np
from pathlib import Path
from typing import Optional

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report,
)

from app.config import settings


async def evaluate_model(
    model_id: str,
    project_id: str,
    metrics: Optional[list[str]] = None,
) -> dict:
    """Evaluate a trained model."""
    # Load model
    model_dir = Path(settings.MODELS_DIR) / project_id
    model_path = model_dir / f"{model_id}.pkl"

    if not model_path.exists():
        return {"error": f"Model {model_id} not found"}

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    problem_type = model_data["problem_type"]
    X_test = model_data["test_data"]["X_test"]
    y_test = model_data["test_data"]["y_test"]

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    if problem_type == "classification":
        results = _evaluate_classification(y_test, y_pred, metrics)
    else:
        results = _evaluate_regression(y_test, y_pred, metrics)

    # Add general info
    results["model_id"] = model_id
    results["model_type"] = model_data["model_type"]
    results["problem_type"] = problem_type
    results["test_size"] = len(X_test)

    # Overfitting check
    train_pred = model.predict(model_data["test_data"]["X_test"])
    results["overfitting_assessment"] = _check_overfitting(results, problem_type)

    return results


def _evaluate_classification(
    y_true, y_pred, metrics: Optional[list[str]] = None
) -> dict:
    """Evaluate classification model."""
    default_metrics = ["accuracy", "precision", "recall", "f1"]
    metrics_to_compute = metrics or default_metrics

    results = {}

    if "accuracy" in metrics_to_compute:
        results["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)

    if "precision" in metrics_to_compute:
        results["precision"] = round(
            float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        )

    if "recall" in metrics_to_compute:
        results["recall"] = round(
            float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        )

    if "f1" in metrics_to_compute:
        results["f1"] = round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        )

    # Always include confusion matrix and classification report
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    results["classification_report"] = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    return results


def _evaluate_regression(
    y_true, y_pred, metrics: Optional[list[str]] = None
) -> dict:
    """Evaluate regression model."""
    default_metrics = ["mse", "rmse", "mae", "r2", "mape"]
    metrics_to_compute = metrics or default_metrics

    results = {}

    if "mse" in metrics_to_compute:
        results["mse"] = round(float(mean_squared_error(y_true, y_pred)), 4)

    if "rmse" in metrics_to_compute:
        results["rmse"] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)

    if "mae" in metrics_to_compute:
        results["mae"] = round(float(mean_absolute_error(y_true, y_pred)), 4)

    if "r2" in metrics_to_compute:
        results["r2"] = round(float(r2_score(y_true, y_pred)), 4)

    if "mape" in metrics_to_compute:
        # Avoid division by zero
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            results["mape"] = round(float(mape), 2)
        else:
            results["mape"] = None

    # Residual analysis
    residuals = y_true - y_pred
    results["residual_stats"] = {
        "mean": round(float(residuals.mean()), 4),
        "std": round(float(residuals.std()), 4),
        "max_error": round(float(np.abs(residuals).max()), 4),
    }

    return results


def _check_overfitting(results: dict, problem_type: str) -> dict:
    """Check for signs of overfitting."""
    assessment = {"risk": "low", "notes": []}

    if problem_type == "classification":
        accuracy = results.get("accuracy", 0)
        if accuracy > 0.99:
            assessment["risk"] = "high"
            assessment["notes"].append(
                "Accuracy > 99% - possible overfitting or data leakage"
            )
        elif accuracy < 0.5:
            assessment["risk"] = "high"
            assessment["notes"].append(
                "Accuracy < 50% - model may not be learning"
            )
    else:
        r2 = results.get("r2", 0)
        if r2 > 0.99:
            assessment["risk"] = "high"
            assessment["notes"].append(
                "R2 > 0.99 - possible overfitting or data leakage"
            )
        elif r2 < 0:
            assessment["risk"] = "high"
            assessment["notes"].append(
                "Negative R2 - model performs worse than mean prediction"
            )

    return assessment
