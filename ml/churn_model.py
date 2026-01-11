import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

class ChurnModel:
    """
    This class encapsulates:
    - Feature engineering
    - Encoding
    - Model training
    - Prediction

    It is reusable for both training and FastAPI inference.
    """

    def __init__(self):
        # Label encoders for categorical columns
        self.le_geo = LabelEncoder()
        self.le_gender = LabelEncoder()

        # XGBoost classifier with tuned hyperparameters
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.7,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.5,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42
        )

        # To ensure feature order consistency
        self.features_ = None

    def _feature_engineering(self, x: pd.DataFrame, fit=False):
        """
        Applies ALL preprocessing and feature engineering.

        This same function is used during:
        - training (fit=True)
        - prediction (fit=False)

        This avoids training–serving mismatch.
        """
        x = x.copy()

        # Drop columns that do not help prediction
        x = x.drop(columns=["RowNumber", "CustomerId", "Surname"])

        # Encode categorical variables
        if fit:
            x["Geography"] = self.le_geo.fit_transform(x["Geography"])
            x["Gender"] = self.le_gender.fit_transform(x["Gender"])
        else:
            x["Geography"] = self.le_geo.transform(x["Geography"])
            x["Gender"] = self.le_gender.transform(x["Gender"])

        # Feature interactions based on domain knowledge
        x["old_inactive"] = ((x["IsActiveMember"] == 0) & (x["Age"] >= 40)).astype(int)
        x["young_active"] = ((x["IsActiveMember"] == 1) & (x["Age"] <= 40)).astype(int)

        # Log transformations (reduce skew)
        x["log_balance"] = np.log1p(x["Balance"])
        x["log_salary"] = np.log1p(x["EstimatedSalary"])
        x["log_credit"] = np.log1p(x["CreditScore"])
        x["log_products"] = np.log1p(x["NumOfProducts"])

        # Square-root transformations
        x["sqrt_balance"] = np.sqrt(x["Balance"])
        x["sqrt_salary"] = np.sqrt(x["EstimatedSalary"])
        x["sqrt_credit"] = np.sqrt(x["CreditScore"])
        x["sqrt_products"] = np.sqrt(x["NumOfProducts"])

        # Interaction terms
        x["balance_x_products"] = x["Balance"] * x["NumOfProducts"]
        x["balance_x_age"] = x["Balance"] * x["Age"]
        x["products_x_age"] = x["NumOfProducts"] * x["Age"]

        # Special business rule
        x["no_balance_no_active"] = (
            (x["Balance"] == 0) & (x["IsActiveMember"] == 0)
        ).astype(int)

        return x

    def fit(self, x: pd.DataFrame, y: pd.Series):
        """
        Trains the model and stores feature order.
        """
        x_transformed = self._feature_engineering(x, fit=True)
        self.features_ = x_transformed.columns
        self.model.fit(x_transformed, y)
        return self

    def predict_proba(self, x: pd.DataFrame):
        """
        Returns churn probability.
        """
        x_transformed = self._feature_engineering(x, fit=False)
        x_transformed = x_transformed[self.features_]
        return self.model.predict_proba(x_transformed)[:, 1]

    def predict(self, x: pd.DataFrame, threshold=0.3):
        """
        Converts probability to binary churn prediction.
        """
        prob = self.predict_proba(x)
        return (prob >= threshold).astype(int)
