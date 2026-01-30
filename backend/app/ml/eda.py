import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

from app.config import settings


async def run_eda(
    dataset_id: str,
    project_id: str,
    target_column: Optional[str] = None,
    analysis_type: str = "full",
) -> dict:
    """Perform exploratory data analysis on a dataset."""
    # Load dataset
    df = await _load_dataset(dataset_id, project_id)

    if analysis_type == "summary":
        return _get_summary(df)
    elif analysis_type == "missing_values":
        return _get_missing_values(df)
    elif analysis_type == "correlations":
        return _get_correlations(df, target_column)
    elif analysis_type == "distributions":
        return _get_distributions(df)
    else:
        # Full analysis
        return {
            "summary": _get_summary(df),
            "missing_values": _get_missing_values(df),
            "correlations": _get_correlations(df, target_column),
            "distributions": _get_distributions(df),
            "data_types": _get_data_types(df),
            "outliers": _detect_outliers(df),
            "recommendations": _get_recommendations(df, target_column),
        }


async def _load_dataset(dataset_id: str, project_id: str) -> pd.DataFrame:
    """Load a dataset by ID."""
    upload_dir = Path(settings.UPLOAD_DIR) / project_id
    # Try common extensions
    for ext in [".csv", ".xlsx", ".parquet"]:
        filepath = upload_dir / f"{dataset_id}{ext}"
        if filepath.exists():
            if ext == ".csv":
                return pd.read_csv(filepath)
            elif ext == ".xlsx":
                return pd.read_excel(filepath)
            elif ext == ".parquet":
                return pd.read_parquet(filepath)

    raise FileNotFoundError(f"Dataset {dataset_id} not found for project {project_id}")


def _get_summary(df: pd.DataFrame) -> dict:
    """Get basic summary statistics."""
    return {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_stats": df.describe().to_dict(),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "duplicates": int(df.duplicated().sum()),
    }


def _get_missing_values(df: pd.DataFrame) -> dict:
    """Analyze missing values."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    return {
        "total_missing": int(missing.sum()),
        "columns_with_missing": {
            col: {"count": int(missing[col]), "percentage": float(missing_pct[col])}
            for col in df.columns
            if missing[col] > 0
        },
        "complete_rows": int((~df.isnull().any(axis=1)).sum()),
        "complete_rows_pct": round((~df.isnull().any(axis=1)).mean() * 100, 2),
    }


def _get_correlations(df: pd.DataFrame, target_column: Optional[str] = None) -> dict:
    """Get correlation analysis."""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return {"message": "No numeric columns found for correlation analysis"}

    corr_matrix = numeric_df.corr().round(3)
    result = {"correlation_matrix": corr_matrix.to_dict()}

    if target_column and target_column in numeric_df.columns:
        target_corr = corr_matrix[target_column].drop(target_column).sort_values(
            key=abs, ascending=False
        )
        result["target_correlations"] = target_corr.to_dict()
        result["top_features"] = list(target_corr.head(10).index)

    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append({
                    "feature_1": corr_matrix.columns[i],
                    "feature_2": corr_matrix.columns[j],
                    "correlation": float(corr_matrix.iloc[i, j]),
                })
    result["high_correlation_pairs"] = high_corr_pairs

    return result


def _get_distributions(df: pd.DataFrame) -> dict:
    """Get distribution information for columns."""
    distributions = {}

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            distributions[col] = {
                "type": "numeric",
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "skewness": float(df[col].skew()),
                "kurtosis": float(df[col].kurtosis()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        else:
            value_counts = df[col].value_counts()
            distributions[col] = {
                "type": "categorical",
                "unique_values": int(df[col].nunique()),
                "top_values": value_counts.head(10).to_dict(),
                "mode": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
            }

    return distributions


def _get_data_types(df: pd.DataFrame) -> dict:
    """Categorize columns by data type."""
    return {
        "numeric": list(df.select_dtypes(include=[np.number]).columns),
        "categorical": list(df.select_dtypes(include=["object", "category"]).columns),
        "datetime": list(df.select_dtypes(include=["datetime64"]).columns),
        "boolean": list(df.select_dtypes(include=["bool"]).columns),
    }


def _detect_outliers(df: pd.DataFrame) -> dict:
    """Detect outliers using IQR method."""
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())

        if outlier_count > 0:
            outliers[col] = {
                "count": outlier_count,
                "percentage": round(outlier_count / len(df) * 100, 2),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }

    return outliers


def _get_recommendations(df: pd.DataFrame, target_column: Optional[str]) -> list[str]:
    """Generate data quality recommendations."""
    recommendations = []
    missing = df.isnull().sum()

    # Missing value recommendations
    high_missing = missing[missing / len(df) > 0.3]
    if not high_missing.empty:
        recommendations.append(
            f"Columns with >30% missing values ({list(high_missing.index)}): "
            "Consider dropping or using advanced imputation."
        )

    # Cardinality check
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() > 50:
            recommendations.append(
                f"Column '{col}' has {df[col].nunique()} unique values. "
                "Consider grouping or using target encoding."
            )

    # Class imbalance check
    if target_column and target_column in df.columns:
        if df[target_column].dtype == "object" or df[target_column].nunique() < 10:
            value_counts = df[target_column].value_counts(normalize=True)
            if value_counts.min() < 0.1:
                recommendations.append(
                    f"Target '{target_column}' is imbalanced. "
                    "Consider SMOTE or class weights."
                )

    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        recommendations.append(
            f"Constant columns detected ({constant_cols}): Remove these."
        )

    return recommendations
