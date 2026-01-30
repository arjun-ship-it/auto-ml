import pandas as pd
from typing import Optional


class DataValidator:
    """Validates data quality and compatibility for ML tasks."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues: list[dict] = []
        self.warnings: list[dict] = []

    def validate(self, target_column: Optional[str] = None) -> dict:
        """Run all validation checks."""
        self._check_empty()
        self._check_duplicates()
        self._check_missing_values()
        self._check_data_types()
        self._check_constant_columns()
        self._check_high_cardinality()

        if target_column:
            self._check_target(target_column)

        return {
            "is_valid": len(self.issues) == 0,
            "issues": self.issues,
            "warnings": self.warnings,
            "summary": {
                "rows": self.df.shape[0],
                "columns": self.df.shape[1],
                "issues_count": len(self.issues),
                "warnings_count": len(self.warnings),
            },
        }

    def _check_empty(self):
        if self.df.empty:
            self.issues.append({
                "type": "empty_dataset",
                "message": "Dataset is empty",
                "severity": "critical",
            })

        if self.df.shape[0] < 10:
            self.warnings.append({
                "type": "small_dataset",
                "message": f"Dataset has only {self.df.shape[0]} rows. ML models need more data.",
                "severity": "warning",
            })

    def _check_duplicates(self):
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            self.warnings.append({
                "type": "duplicates",
                "message": f"Found {dup_count} duplicate rows ({dup_count/len(self.df)*100:.1f}%)",
                "severity": "warning",
                "action": "Consider removing duplicates",
            })

    def _check_missing_values(self):
        for col in self.df.columns:
            missing_pct = self.df[col].isnull().mean() * 100
            if missing_pct > 50:
                self.issues.append({
                    "type": "high_missing",
                    "column": col,
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values",
                    "severity": "high",
                    "action": "Consider dropping this column",
                })
            elif missing_pct > 10:
                self.warnings.append({
                    "type": "missing_values",
                    "column": col,
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values",
                    "severity": "medium",
                    "action": "Imputation recommended",
                })

    def _check_data_types(self):
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                # Check if it's actually numeric
                try:
                    pd.to_numeric(self.df[col], errors="raise")
                    self.warnings.append({
                        "type": "mistyped_column",
                        "column": col,
                        "message": f"Column '{col}' contains numeric values stored as text",
                        "severity": "low",
                        "action": "Convert to numeric type",
                    })
                except (ValueError, TypeError):
                    pass

    def _check_constant_columns(self):
        for col in self.df.columns:
            if self.df[col].nunique() <= 1:
                self.issues.append({
                    "type": "constant_column",
                    "column": col,
                    "message": f"Column '{col}' has only one unique value",
                    "severity": "medium",
                    "action": "Remove this column - it provides no information",
                })

    def _check_high_cardinality(self):
        for col in self.df.select_dtypes(include=["object"]).columns:
            cardinality = self.df[col].nunique()
            if cardinality > len(self.df) * 0.5:
                self.warnings.append({
                    "type": "high_cardinality",
                    "column": col,
                    "message": f"Column '{col}' has {cardinality} unique values (possible ID column)",
                    "severity": "medium",
                    "action": "Consider removing or using target encoding",
                })

    def _check_target(self, target_column: str):
        if target_column not in self.df.columns:
            self.issues.append({
                "type": "missing_target",
                "message": f"Target column '{target_column}' not found in dataset",
                "severity": "critical",
            })
            return

        target = self.df[target_column]

        # Check for missing target values
        if target.isnull().any():
            self.issues.append({
                "type": "missing_target_values",
                "message": f"Target column has {target.isnull().sum()} missing values",
                "severity": "high",
                "action": "Remove rows with missing target values",
            })

        # Check class imbalance for classification
        if target.dtype == "object" or target.nunique() < 20:
            value_counts = target.value_counts(normalize=True)
            if value_counts.min() < 0.05:
                self.warnings.append({
                    "type": "class_imbalance",
                    "message": f"Severe class imbalance detected. Minority class: {value_counts.min()*100:.1f}%",
                    "severity": "high",
                    "action": "Use SMOTE, class weights, or undersampling",
                })
