"""
Intelligent Model Selection Module

This module does NOT rely on a fixed list of models. Instead, it:
1. Analyzes data characteristics deeply (shape, distributions, relationships)
2. Understands the problem nature (time-series, classification, regression, anomaly detection)
3. Uses Claude to reason about the best approach
4. Can generate custom model architectures when standard models won't suffice
5. Considers all factors: data size, feature types, target behavior, business constraints
"""

import json
import pandas as pd
import numpy as np
from typing import Any, Optional
from anthropic import AsyncAnthropic

from app.config import settings


class IntelligentModelSelector:
    """Uses LLM reasoning to select or design the best ML model for any problem."""

    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def analyze_and_recommend(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_description: str,
        constraints: Optional[dict] = None,
    ) -> dict:
        """
        Deeply analyze data and recommend the best ML approach.
        This can recommend standard models OR design custom architectures.
        """
        # Step 1: Deep data profiling
        data_profile = self._deep_profile(df, target_column)

        # Step 2: Detect problem characteristics
        problem_characteristics = self._analyze_problem(df, target_column)

        # Step 3: Use Claude to reason about the best approach
        recommendation = await self._llm_recommend(
            data_profile=data_profile,
            problem_characteristics=problem_characteristics,
            problem_description=problem_description,
            constraints=constraints,
        )

        return recommendation

    def _deep_profile(self, df: pd.DataFrame, target_column: str) -> dict:
        """Create a comprehensive data profile for LLM analysis."""
        target = df[target_column]
        features = df.drop(columns=[target_column])

        profile = {
            "data_size": {
                "rows": len(df),
                "features": len(features.columns),
                "size_category": self._size_category(len(df), len(features.columns)),
            },
            "target_analysis": {
                "dtype": str(target.dtype),
                "unique_values": int(target.nunique()),
                "distribution": self._analyze_distribution(target),
                "is_temporal": self._is_temporal_target(df, target_column),
                "has_seasonality": self._detect_seasonality(target),
                "stationarity": self._check_stationarity(target),
            },
            "feature_analysis": {
                "numeric_count": len(features.select_dtypes(include=[np.number]).columns),
                "categorical_count": len(features.select_dtypes(include=["object"]).columns),
                "datetime_count": len(features.select_dtypes(include=["datetime64"]).columns),
                "high_cardinality_features": self._high_cardinality_features(features),
                "feature_interactions": self._detect_interactions(features, target),
                "multicollinearity": self._check_multicollinearity(features),
                "non_linear_relationships": self._detect_nonlinearity(features, target),
            },
            "data_quality": {
                "missing_pattern": self._missing_pattern(df),
                "outlier_percentage": self._outlier_percentage(features),
                "noise_level": self._estimate_noise(features, target),
            },
            "complexity_indicators": {
                "feature_to_sample_ratio": len(features.columns) / max(len(df), 1),
                "class_separability": self._class_separability(features, target),
                "data_heterogeneity": self._data_heterogeneity(features),
            },
        }

        return profile

    def _analyze_problem(self, df: pd.DataFrame, target_column: str) -> dict:
        """Detect what kind of problem this is beyond simple classification/regression."""
        target = df[target_column]

        problem_type = "unknown"
        sub_type = None

        if target.dtype == "object" or target.nunique() < 20:
            problem_type = "classification"
            if target.nunique() == 2:
                sub_type = "binary"
            else:
                sub_type = "multiclass"
            if target.value_counts(normalize=True).min() < 0.05:
                sub_type = f"imbalanced_{sub_type}"
        else:
            problem_type = "regression"
            if self._is_temporal_target(df, target_column):
                problem_type = "time_series_forecasting"
            elif target.nunique() < 50:
                sub_type = "ordinal_regression"
            elif (target >= 0).all() and target.skew() > 2:
                sub_type = "count_regression"

        return {
            "problem_type": problem_type,
            "sub_type": sub_type,
            "target_nature": self._target_nature(target),
            "temporal_component": self._has_temporal_features(df),
            "spatial_component": self._has_spatial_features(df),
            "text_features": self._has_text_features(df),
            "recommended_loss": self._suggest_loss(problem_type, sub_type),
        }

    async def _llm_recommend(
        self,
        data_profile: dict,
        problem_characteristics: dict,
        problem_description: str,
        constraints: Optional[dict] = None,
    ) -> dict:
        """Use Claude to reason about the best ML approach."""
        prompt = f"""You are an expert ML engineer. Based on the following data analysis, recommend the BEST machine learning approach.

## Problem Description
{problem_description}

## Data Profile
{json.dumps(data_profile, indent=2, default=str)}

## Problem Characteristics
{json.dumps(problem_characteristics, indent=2, default=str)}

## Constraints
{json.dumps(constraints or {}, indent=2)}

## Your Task
Analyze ALL factors and recommend the best approach. You are NOT limited to standard models.
You can recommend:
1. A standard algorithm (with specific hyperparameter suggestions)
2. An ensemble of multiple models
3. A custom neural network architecture (provide the PyTorch code)
4. A multi-stage pipeline (e.g., clustering then classification)
5. Any creative approach that fits the data best

Consider:
- Data size vs model complexity
- Feature types and relationships
- Target distribution and nature
- Potential for overfitting
- Interpretability requirements
- Non-linear relationships in the data
- Whether the problem needs sequence modeling, attention mechanisms, etc.

Respond with a JSON object containing:
{{
    "recommended_approach": "description of the approach",
    "model_type": "standard|ensemble|custom_nn|multi_stage|hybrid",
    "models": [
        {{
            "name": "model name",
            "reason": "why this model",
            "hyperparameters": {{}},
            "expected_performance": "high/medium/low",
            "custom_code": "Python code if custom architecture (optional)"
        }}
    ],
    "preprocessing_recommendations": ["list of preprocessing steps"],
    "feature_engineering_suggestions": ["list of feature ideas"],
    "evaluation_strategy": "description of how to evaluate",
    "risks": ["potential issues to watch for"],
    "confidence": 0.0-1.0
}}
"""

        response = await self.client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse the response
        response_text = response.content[0].text

        # Extract JSON from response
        try:
            # Find JSON in response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                recommendation = json.loads(response_text[start:end])
            else:
                recommendation = {"raw_response": response_text}
        except json.JSONDecodeError:
            recommendation = {"raw_response": response_text}

        return recommendation

    async def generate_custom_model(
        self,
        data_profile: dict,
        problem_characteristics: dict,
        requirements: str,
    ) -> str:
        """Generate custom PyTorch model code based on data analysis."""
        prompt = f"""Design a custom PyTorch model architecture for this specific problem.

## Data Profile
{json.dumps(data_profile, indent=2, default=str)}

## Problem
{json.dumps(problem_characteristics, indent=2, default=str)}

## Requirements
{requirements}

Generate complete, runnable PyTorch code including:
1. Model class definition
2. Training loop
3. Evaluation function
4. Any custom layers or loss functions needed

The code should be production-ready and handle edge cases.
Return ONLY the Python code, no explanations.
"""

        response = await self.client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    # --- Helper methods for data analysis ---

    def _size_category(self, rows: int, cols: int) -> str:
        if rows < 100:
            return "very_small"
        elif rows < 1000:
            return "small"
        elif rows < 10000:
            return "medium"
        elif rows < 100000:
            return "large"
        return "very_large"

    def _analyze_distribution(self, series: pd.Series) -> dict:
        if series.dtype in [np.float64, np.int64]:
            return {
                "type": "continuous",
                "skewness": round(float(series.skew()), 3),
                "kurtosis": round(float(series.kurtosis()), 3),
                "is_normal": abs(series.skew()) < 0.5 and abs(series.kurtosis()) < 3,
            }
        return {
            "type": "categorical",
            "balance": round(float(series.value_counts(normalize=True).std()), 3),
        }

    def _is_temporal_target(self, df: pd.DataFrame, target_col: str) -> bool:
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        if len(datetime_cols) > 0:
            return True
        # Check if index is datetime-like
        for col in df.columns:
            if col != target_col and df[col].dtype == "object":
                try:
                    pd.to_datetime(df[col].head(10))
                    return True
                except (ValueError, TypeError):
                    pass
        return False

    def _detect_seasonality(self, series: pd.Series) -> Optional[str]:
        if series.dtype not in [np.float64, np.int64] or len(series) < 24:
            return None
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            result = seasonal_decompose(series.dropna(), period=min(12, len(series)//3), extrapolate_trend='freq')
            seasonal_strength = result.seasonal.std() / series.std()
            if seasonal_strength > 0.3:
                return "strong"
            elif seasonal_strength > 0.1:
                return "moderate"
        except Exception:
            pass
        return "none"

    def _check_stationarity(self, series: pd.Series) -> Optional[bool]:
        if series.dtype not in [np.float64, np.int64] or len(series) < 20:
            return None
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series.dropna())
            return result[1] < 0.05  # p-value < 0.05 means stationary
        except Exception:
            return None

    def _high_cardinality_features(self, df: pd.DataFrame) -> list[str]:
        return [
            col for col in df.select_dtypes(include=["object"]).columns
            if df[col].nunique() > 50
        ]

    def _detect_interactions(self, features: pd.DataFrame, target: pd.Series) -> dict:
        numeric = features.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2 or target.dtype not in [np.float64, np.int64]:
            return {"detected": False}

        # Simple interaction detection via correlation of products
        interactions = []
        cols = numeric.columns[:10]  # Limit to first 10 for speed
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                interaction = numeric[cols[i]] * numeric[cols[j]]
                corr = abs(interaction.corr(target))
                individual_max = max(
                    abs(numeric[cols[i]].corr(target)),
                    abs(numeric[cols[j]].corr(target))
                )
                if corr > individual_max + 0.1:
                    interactions.append(f"{cols[i]} x {cols[j]}")

        return {"detected": len(interactions) > 0, "pairs": interactions[:5]}

    def _check_multicollinearity(self, features: pd.DataFrame) -> dict:
        numeric = features.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return {"has_multicollinearity": False}

        corr = numeric.corr().abs()
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if corr.iloc[i, j] > 0.9:
                    high_corr.append((corr.columns[i], corr.columns[j]))

        return {
            "has_multicollinearity": len(high_corr) > 0,
            "pairs": high_corr[:5],
        }

    def _detect_nonlinearity(self, features: pd.DataFrame, target: pd.Series) -> dict:
        if target.dtype not in [np.float64, np.int64]:
            return {"detected": "unknown"}

        numeric = features.select_dtypes(include=[np.number])
        non_linear_features = []

        for col in numeric.columns[:10]:
            linear_corr = abs(numeric[col].corr(target))
            # Compare with rank correlation (Spearman)
            rank_corr = abs(numeric[col].rank().corr(target.rank()))
            if rank_corr - linear_corr > 0.1:
                non_linear_features.append(col)

        return {
            "detected": len(non_linear_features) > 0,
            "features": non_linear_features,
        }

    def _missing_pattern(self, df: pd.DataFrame) -> str:
        missing_cols = df.columns[df.isnull().any()].tolist()
        if not missing_cols:
            return "none"
        if len(missing_cols) > df.shape[1] * 0.5:
            return "widespread"
        return "localized"

    def _outlier_percentage(self, features: pd.DataFrame) -> float:
        numeric = features.select_dtypes(include=[np.number])
        if numeric.empty:
            return 0.0
        Q1 = numeric.quantile(0.25)
        Q3 = numeric.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric < (Q1 - 1.5 * IQR)) | (numeric > (Q3 + 1.5 * IQR))).any(axis=1)
        return round(float(outliers.mean() * 100), 2)

    def _estimate_noise(self, features: pd.DataFrame, target: pd.Series) -> str:
        if target.dtype not in [np.float64, np.int64]:
            return "unknown"
        numeric = features.select_dtypes(include=[np.number])
        if numeric.empty:
            return "unknown"
        max_corr = numeric.corrwith(target).abs().max()
        if max_corr > 0.7:
            return "low"
        elif max_corr > 0.3:
            return "medium"
        return "high"

    def _class_separability(self, features: pd.DataFrame, target: pd.Series) -> Optional[str]:
        if target.dtype not in ["object"] and target.nunique() >= 20:
            return None
        # Simple check: how well can a decision stump separate classes
        numeric = features.select_dtypes(include=[np.number])
        if numeric.empty:
            return "unknown"
        max_corr = numeric.corrwith(target.astype("category").cat.codes).abs().max()
        if max_corr > 0.5:
            return "easy"
        elif max_corr > 0.2:
            return "moderate"
        return "hard"

    def _data_heterogeneity(self, features: pd.DataFrame) -> str:
        dtypes = features.dtypes.value_counts()
        if len(dtypes) > 3:
            return "high"
        elif len(dtypes) > 1:
            return "medium"
        return "low"

    def _target_nature(self, target: pd.Series) -> str:
        if target.dtype == "object":
            return "categorical"
        if target.nunique() == 2:
            return "binary"
        if target.nunique() < 20:
            return "ordinal"
        if (target >= 0).all():
            if (target == target.astype(int)).all():
                return "count"
            return "positive_continuous"
        return "continuous"

    def _has_temporal_features(self, df: pd.DataFrame) -> bool:
        return len(df.select_dtypes(include=["datetime64"]).columns) > 0

    def _has_spatial_features(self, df: pd.DataFrame) -> bool:
        spatial_keywords = ["lat", "lon", "lng", "latitude", "longitude", "geo", "coord"]
        return any(
            any(kw in col.lower() for kw in spatial_keywords)
            for col in df.columns
        )

    def _has_text_features(self, df: pd.DataFrame) -> bool:
        for col in df.select_dtypes(include=["object"]).columns:
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > 50:
                return True
        return False

    def _suggest_loss(self, problem_type: str, sub_type: Optional[str]) -> str:
        if problem_type == "classification":
            if sub_type and "imbalanced" in sub_type:
                return "focal_loss or weighted_cross_entropy"
            return "cross_entropy"
        elif problem_type == "regression":
            if sub_type == "count_regression":
                return "poisson_loss"
            return "mse or huber_loss"
        elif problem_type == "time_series_forecasting":
            return "mse or quantile_loss"
        return "mse"
