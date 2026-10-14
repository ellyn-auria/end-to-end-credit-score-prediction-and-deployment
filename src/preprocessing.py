import os
import re
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler

class CreditScorePreprocessor:
    CONTAINER_BASE = "/opt/ml/processing"
    LOCAL_BASE = "/home/ec2-user/SageMaker"
    SCORE_MAP = {"Poor": 0, "Standard": 1, "Good": 2}

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

        self.cap_thresholds: Dict[str, float] = {}
        self.customer_mode_map: Dict[str, pd.Series] = {}

        self.cap_cols = [
            "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
            "Num_of_Delayed_Payment", "Num_Credit_Inquiries", "Total_EMI_per_month",
            "Annual_Income",
        ]
        self.noisy_numeric_cols = [
            "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
            "Changed_Credit_Limit", "Outstanding_Debt",
            "Amount_invested_monthly", "Monthly_Balance",
        ]
        self.features_to_drop = ["Annual_Income"]

    @staticmethod
    def _clean_numeric_string(series: pd.Series) -> pd.Series:
        s = series.astype(str)
        s = s.replace(["_", "nan", "NaN", ""], np.nan)
        s = s.str.replace("_", "", regex=False).str.strip()
        return pd.to_numeric(s, errors="coerce")

    @staticmethod
    def _parse_credit_history_age(val) -> float:
        if pd.isna(val):
            return np.nan
        m = re.match(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(val))
        return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan

    def _fit_cap_thresholds(self, X: pd.DataFrame):
        for col in self.cap_cols:
            if col in X.columns:
                self.cap_thresholds[col] = X[col].quantile(0.99)

    def _clean_data(self, X: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        X = X.copy()

        for col in self.noisy_numeric_cols:
            if col in X.columns:
                X[col] = self._clean_numeric_string(X[col])

        if is_train:
            self._fit_cap_thresholds(X)

        X["Occupation"] = X.get("Occupation", pd.Series(dtype=str)).replace("_______", np.nan)
        X["Credit_Mix"] = X.get("Credit_Mix", pd.Series(dtype=str)).replace("_", np.nan)
        X["Payment_Behaviour"] = X.get("Payment_Behaviour", pd.Series(dtype=str)).replace("!@9#%8", np.nan)
        X["Payment_of_Min_Amount"] = X.get("Payment_of_Min_Amount", pd.Series(dtype=str)).replace("NM", np.nan)
        if "SSN" in X.columns:
            X["SSN"] = X["SSN"].replace("#F%$D@*&8", np.nan)

        if "Credit_History_Age" in X.columns:
            X["Credit_History_Age_Months"] = X["Credit_History_Age"].apply(self._parse_credit_history_age)
            X = X.drop(columns=["Credit_History_Age"])

        if is_train and "Customer_ID" in X.columns:
            self.customer_mode_map = {
                "Age": X.groupby("Customer_ID")["Age"].apply(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
                ),
                "Occupation": X.groupby("Customer_ID")["Occupation"].apply(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
                ),
            }

        if self.customer_mode_map and "Customer_ID" in X.columns:
            X["Age"] = X["Customer_ID"].map(self.customer_mode_map.get("Age", pd.Series())).combine_first(X.get("Age", pd.Series()))
            X["Occupation"] = X["Customer_ID"].map(self.customer_mode_map.get("Occupation", pd.Series())).combine_first(X.get("Occupation", pd.Series()))

        X.loc[(X["Age"] < 0) | (X["Age"] > 100), "Age"] = np.nan
        X.loc[X.get("Num_Bank_Accounts", pd.Series()) < 0, "Num_Bank_Accounts"] = np.nan
        X.loc[X.get("Num_of_Loan", pd.Series()) < 0, "Num_of_Loan"] = np.nan
        if "Delay_from_due_date" in X.columns:
            X.loc[X["Delay_from_due_date"] < 0, "Delay_from_due_date"] = np.nan
        X.loc[X.get("Num_of_Delayed_Payment", pd.Series()) < 0, "Num_of_Delayed_Payment"] = np.nan
        X.loc[X.get("Amount_invested_monthly", pd.Series()) >= 10000, "Amount_invested_monthly"] = np.nan
        X.loc[X.get("Monthly_Balance", pd.Series()).abs() > 1e6, "Monthly_Balance"] = np.nan

        for col, cap in self.cap_thresholds.items():
            if col in X.columns:
                X[col] = np.where(X[col] > cap, cap, X[col])

        drop_cols = ["ID", "Customer_ID", "Name", "SSN", "Month", "Type_of_Loan"]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
        X = X.drop(columns=[c for c in self.features_to_drop if c in X.columns])

        return X

    def get_transformer(self, X_train: pd.DataFrame) -> ColumnTransformer:
        ord_feat = ["Credit_Mix", "Payment_Behaviour"]
        num_feat = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        ohe_feat = (
            X_train.select_dtypes(include=["object", "category"])
            .drop(columns=ord_feat, errors="ignore")
            .columns.tolist()
        )

        credit_mix_order = ["Bad", "Standard", "Good"]
        payment_behaviour_order = [
            "Low_spent_Small_value_payments", "Low_spent_Medium_value_payments",
            "Low_spent_Large_value_payments", "High_spent_Small_value_payments",
            "High_spent_Medium_value_payments", "High_spent_Large_value_payments",
        ]

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  RobustScaler()),
        ])
        ohe_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        ordinal_transformer = Pipeline([
            ("imputer",  SimpleImputer(strategy="most_frequent")),
            ("ordinal",  OrdinalEncoder(
                categories=[credit_mix_order, payment_behaviour_order],
                handle_unknown="use_encoded_value", unknown_value=-1,
            )),
        ])

        return ColumnTransformer(transformers=[
            ("num", numeric_transformer,  num_feat),
            ("ohe", ohe_transformer,      ohe_feat),
            ("ord", ordinal_transformer,  ord_feat),
        ], remainder="drop")

    def clean_and_split(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        df = pd.read_csv(data_path)

        X = df.drop(columns=["Credit_Score"])
        y = df["Credit_Score"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

        X_train_clean = self._clean_data(X_train, is_train=True)
        X_test_clean = self._clean_data(X_test, is_train=False)

        y_train_enc = y_train.map(self.SCORE_MAP)
        y_test_enc = y_test.map(self.SCORE_MAP)

        return X_train_clean, X_test_clean, y_train_enc, y_test_enc

    def run(self) -> Tuple[str, str]:
        base = self.CONTAINER_BASE if os.path.exists(self.CONTAINER_BASE) else self.LOCAL_BASE
        input_file = os.path.join(base, "ingested", "data_D.csv")
        train_dir = os.path.join(base, "train")
        test_dir = os.path.join(base, "test")

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File tidak ditemukan: {input_file}")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir,  exist_ok=True)

        X_train, X_test, y_train, y_test = self.clean_and_split(input_file)

        train_df = X_train.copy(); train_df["Credit_Score"] = y_train.values
        test_df = X_test.copy();  test_df["Credit_Score"] = y_test.values

        train_path = os.path.join(train_dir, "train.csv")
        test_path = os.path.join(test_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print("Selesai")
        return train_path, test_path


if __name__ == "__main__":
    CreditScorePreprocessor().run()
