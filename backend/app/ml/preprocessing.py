import pandas as pd
import numpy as np
from typing import Any
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

from app.config import settings


async def preprocess_data(
    dataset_id: str,
    project_id: str,
    steps: list[str],
    config: dict[str, Any] = {},
) -> dict:
    """Apply preprocessing steps to a dataset."""
    # Load dataset
    upload_dir = Path(settings.UPLOAD_DIR) / project_id
    df = _load_df(upload_dir, dataset_id)

    original_shape = df.shape
    applied_steps = []

    for step in steps:
        if step == "handle_missing":
            df, info = _handle_missing(df, config.get("missing_strategy", "auto"))
            applied_steps.append({"step": step, "details": info})

        elif step == "encode_categorical":
            df, info = _encode_categorical(df, config.get("encoding", "auto"))
            applied_steps.append({"step": step, "details": info})

        elif step == "scale_features":
            df, info = _scale_features(df, config.get("scaler", "standard"))
            applied_steps.append({"step": step, "details": info})

        elif step == "remove_outliers":
            df, info = _remove_outliers(df, config.get("outlier_method", "iqr"))
            applied_steps.append({"step": step, "details": info})

        elif step == "feature_engineering":
            df, info = _feature_engineering(df, config.get("features", {}))
            applied_steps.append({"step": step, "details": info})

    # Save preprocessed dataset
    processed_id = f"{dataset_id}_processed"
    output_path = upload_dir / f"{processed_id}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return {
        "processed_dataset_id": processed_id,
        "original_shape": {"rows": original_shape[0], "columns": original_shape[1]},
        "new_shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "steps_applied": applied_steps,
        "columns": list(df.columns),
    }


def _load_df(upload_dir: Path, dataset_id: str) -> pd.DataFrame:
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


def _handle_missing(df: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, dict]:
    """Handle missing values."""
    missing_before = df.isnull().sum().sum()

    if strategy == "auto":
        # Numeric: median, Categorical: mode
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown", inplace=True)
    elif strategy == "drop_rows":
        df = df.dropna()
    elif strategy == "drop_columns":
        threshold = len(df) * 0.5
        df = df.dropna(axis=1, thresh=threshold)
    elif strategy == "knn":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    missing_after = df.isnull().sum().sum()
    return df, {
        "strategy": strategy,
        "missing_before": int(missing_before),
        "missing_after": int(missing_after),
    }


def _encode_categorical(df: pd.DataFrame, encoding: str) -> tuple[pd.DataFrame, dict]:
    """Encode categorical variables."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoded_cols = []

    if not cat_cols:
        return df, {"message": "No categorical columns found"}

    if encoding == "auto":
        for col in cat_cols:
            if df[col].nunique() <= 5:
                # One-hot for low cardinality
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                encoded_cols.append({"column": col, "method": "one_hot"})
            else:
                # Label encode for high cardinality
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoded_cols.append({"column": col, "method": "label"})
    elif encoding == "onehot":
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        encoded_cols = [{"column": col, "method": "one_hot"} for col in cat_cols]
    elif encoding == "label":
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoded_cols.append({"column": col, "method": "label"})

    return df, {"encoded_columns": encoded_cols}


def _scale_features(df: pd.DataFrame, scaler_type: str) -> tuple[pd.DataFrame, dict]:
    """Scale numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return df, {"message": "No numeric columns to scale"}

    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, {
        "scaler": scaler_type,
        "columns_scaled": numeric_cols,
    }


def _remove_outliers(df: pd.DataFrame, method: str) -> tuple[pd.DataFrame, dict]:
    """Remove outliers from numeric columns."""
    original_len = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if method == "iqr":
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    elif method == "zscore":
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[numeric_cols]))
        df = df[(z_scores < 3).all(axis=1)]

    return df, {
        "method": method,
        "rows_removed": original_len - len(df),
        "rows_remaining": len(df),
    }


def _feature_engineering(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Apply feature engineering."""
    new_features = []

    # Auto-detect datetime columns and extract features
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                dt_col = pd.to_datetime(df[col], errors="coerce")
                if dt_col.notna().sum() > len(df) * 0.8:
                    df[f"{col}_year"] = dt_col.dt.year
                    df[f"{col}_month"] = dt_col.dt.month
                    df[f"{col}_day"] = dt_col.dt.day
                    df[f"{col}_dayofweek"] = dt_col.dt.dayofweek
                    df = df.drop(col, axis=1)
                    new_features.extend([
                        f"{col}_year", f"{col}_month",
                        f"{col}_day", f"{col}_dayofweek",
                    ])
            except (ValueError, TypeError):
                pass

    # Interaction features if specified
    if config.get("interactions"):
        for feat_pair in config["interactions"]:
            if len(feat_pair) == 2 and all(f in df.columns for f in feat_pair):
                new_col = f"{feat_pair[0]}_x_{feat_pair[1]}"
                df[new_col] = df[feat_pair[0]] * df[feat_pair[1]]
                new_features.append(new_col)

    return df, {"new_features": new_features}
