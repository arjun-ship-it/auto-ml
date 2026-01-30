import uuid
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from app.config import settings


async def train_model(
    dataset_id: str,
    project_id: str,
    model_type: str,
    target_column: str,
    hyperparameters: dict[str, Any] = {},
    test_size: float = 0.2,
) -> dict:
    """Train a machine learning model."""
    # Load data
    upload_dir = Path(settings.UPLOAD_DIR) / project_id
    df = _load_processed_data(upload_dir, dataset_id)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in dataset"}

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Determine problem type
    problem_type = _detect_problem_type(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Train model
    model, training_info = await _train(
        model_type, X_train, y_train, hyperparameters, problem_type
    )

    # Generate model ID and save
    model_id = str(uuid.uuid4())[:8]
    model_dir = Path(settings.MODELS_DIR) / project_id
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_id}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "model_type": model_type,
            "target_column": target_column,
            "feature_columns": list(X.columns),
            "problem_type": problem_type,
            "test_data": {"X_test": X_test, "y_test": y_test},
        }, f)

    return {
        "model_id": model_id,
        "model_type": model_type,
        "problem_type": problem_type,
        "training_info": training_info,
        "data_split": {
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "features_used": list(X.columns),
    }


def _load_processed_data(upload_dir: Path, dataset_id: str) -> pd.DataFrame:
    for ext in [".csv", ".xlsx", ".parquet"]:
        filepath = upload_dir / f"{dataset_id}{ext}"
        if filepath.exists():
            if ext == ".csv":
                return pd.read_csv(filepath)
            elif ext == ".xlsx":
                return pd.read_excel(filepath)
            elif ext == ".parquet":
                return pd.read_parquet(filepath)
    raise FileNotFoundError(f"Dataset {dataset_id} not found")


def _detect_problem_type(y: pd.Series) -> str:
    """Detect if this is classification or regression."""
    if y.dtype == "object" or y.nunique() < 20:
        return "classification"
    return "regression"


async def _train(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameters: dict,
    problem_type: str,
) -> tuple[Any, dict]:
    """Train the specified model."""
    model = _get_model(model_type, problem_type, hyperparameters)

    if model_type in ["arima", "prophet", "lstm"]:
        return await _train_timeseries(model_type, X_train, y_train, hyperparameters)

    if model_type == "neural_network":
        return await _train_neural_network(X_train, y_train, hyperparameters, problem_type)

    # Standard sklearn-compatible training
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)

    return model, {
        "train_score": round(float(train_score), 4),
        "hyperparameters": hyperparameters or "default",
    }


def _get_model(model_type: str, problem_type: str, params: dict) -> Any:
    """Get the model instance."""
    models = {
        "linear_regression": LinearRegression,
        "logistic_regression": LogisticRegression,
        "random_forest": (
            RandomForestClassifier if problem_type == "classification"
            else RandomForestRegressor
        ),
        "xgboost": _get_xgboost_model,
        "lightgbm": _get_lightgbm_model,
        "svm": SVC if problem_type == "classification" else SVR,
        "knn": (
            KNeighborsClassifier if problem_type == "classification"
            else KNeighborsRegressor
        ),
    }

    model_class = models.get(model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}")

    if callable(model_class) and not isinstance(model_class, type):
        return model_class(problem_type, params)

    return model_class(**params) if params else model_class()


def _get_xgboost_model(problem_type: str, params: dict):
    import xgboost as xgb
    if problem_type == "classification":
        return xgb.XGBClassifier(**params) if params else xgb.XGBClassifier()
    return xgb.XGBRegressor(**params) if params else xgb.XGBRegressor()


def _get_lightgbm_model(problem_type: str, params: dict):
    import lightgbm as lgb
    if problem_type == "classification":
        return lgb.LGBMClassifier(**params) if params else lgb.LGBMClassifier()
    return lgb.LGBMRegressor(**params) if params else lgb.LGBMRegressor()


async def _train_timeseries(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> tuple[Any, dict]:
    """Train time-series models."""
    if model_type == "prophet":
        from prophet import Prophet
        model = Prophet(**params) if params else Prophet()
        # Prophet expects 'ds' and 'y' columns
        prophet_df = pd.DataFrame({"ds": X_train.index, "y": y_train.values})
        model.fit(prophet_df)
        return model, {"model": "Prophet", "params": params}

    elif model_type == "arima":
        from statsmodels.tsa.arima.model import ARIMA
        order = params.get("order", (1, 1, 1))
        model = ARIMA(y_train, order=order)
        fitted = model.fit()
        return fitted, {"model": "ARIMA", "order": order, "aic": fitted.aic}

    elif model_type == "lstm":
        return await _train_lstm(X_train, y_train, params)

    raise ValueError(f"Unknown time-series model: {model_type}")


async def _train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    problem_type: str,
) -> tuple[Any, dict]:
    """Train a neural network using PyTorch."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    input_size = X_train.shape[1]
    hidden_size = params.get("hidden_size", 64)
    num_layers = params.get("num_layers", 2)
    epochs = params.get("epochs", 100)
    lr = params.get("learning_rate", 0.001)
    batch_size = params.get("batch_size", 32)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train.values)
    if problem_type == "classification":
        y_tensor = torch.LongTensor(y_train.values)
        output_size = int(y_train.nunique())
    else:
        y_tensor = torch.FloatTensor(y_train.values)
        output_size = 1

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build model
    layers = []
    prev_size = input_size
    for _ in range(num_layers):
        layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(0.2)])
        prev_size = hidden_size
    layers.append(nn.Linear(prev_size, output_size))

    model = nn.Sequential(*layers)

    # Loss and optimizer
    if problem_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            if problem_type == "regression":
                output = output.squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    return model, {
        "model": "NeuralNetwork",
        "architecture": str(model),
        "epochs": epochs,
        "final_loss": round(losses[-1], 4),
    }


async def _train_lstm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> tuple[Any, dict]:
    """Train an LSTM model for time-series."""
    import torch
    import torch.nn as nn

    hidden_size = params.get("hidden_size", 50)
    num_layers = params.get("num_layers", 2)
    epochs = params.get("epochs", 50)
    seq_length = params.get("seq_length", 10)

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    input_size = X_train.shape[1]
    model = LSTMModel(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create sequences
    X_tensor = torch.FloatTensor(X_train.values).unsqueeze(0)
    y_tensor = torch.FloatTensor(y_train.values[-1:])

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output.squeeze(), y_tensor)
        loss.backward()
        optimizer.step()

    return model, {"model": "LSTM", "hidden_size": hidden_size, "epochs": epochs}
