import argparse
import os
from typing import Dict, List, Optional
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from preprocessing import CreditScorePreprocessor

class CreditScoreModelTrainer:
    TARGET_COL = "Credit_Score"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = CreditScorePreprocessor(random_state=random_state)

    def _get_classifier(self, model_name: str):
        if model_name == "XGBoost":
            return XGBClassifier(
                n_estimators=200, max_depth=7, learning_rate=0.1,
                subsample=0.8, eval_metric="mlogloss",
                random_state=self.random_state,
            )
        if model_name == "RandomForest":
            return RandomForestClassifier(
                n_estimators=200, max_depth=None,
                min_samples_split=2, min_samples_leaf=1,
                random_state=self.random_state,
            )
        if model_name == "DecisionTree":
            return DecisionTreeClassifier(
                criterion="gini", max_depth=10,
                min_samples_split=5, min_samples_leaf=2,
                random_state=self.random_state,
            )
        raise ValueError(f"Model '{model_name}' tidak dikenal.")

    def _train_single(self, model_name: str, transformer, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,) -> Dict:
        classifier = self._get_classifier(model_name)

        pipeline = ImbPipeline([
            ("preprocessor", transformer),
            ("smote", SMOTE(random_state=self.random_state)),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        print(f"{model_name} Accuracy={acc:.4f} | F1={f1:.4f}")
        return {"model_name": model_name, "accuracy": acc, "f1_score": f1, "pipeline": pipeline}

    def run(self, train_dir: str, model_dir: str, model_names: Optional[List[str]] = None,) -> str:
        if model_names is None:
            model_names = ["XGBoost", "RandomForest", "DecisionTree"]

        train_file = os.path.join(train_dir, "train.csv")

        df = pd.read_csv(train_file)
        X_train = df.drop(columns=[self.TARGET_COL])
        y_train = df[self.TARGET_COL]

        transformer = self.preprocessor.get_transformer(X_train)

        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train)

        print(f"Training {len(model_names)} model...")
        results = [
            self._train_single(name, transformer, X_tr, y_tr, X_val, y_val)
            for name in model_names
        ]

        best = max(results, key=lambda r: r["f1_score"])
        print(f"\nBest model: {best['model_name']} (F1={best['f1_score']:.4f})")
        
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "best_model.pkl")
        joblib.dump(best["pipeline"], model_path)
        print(f"Model disimpan ke: {model_path}")
        return model_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-names", type=str, default="XGBoost,RandomForest,DecisionTree",
        help="Comma-separated model names to train"
    )
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/home/ec2-user/SageMaker/train"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/home/ec2-user/SageMaker/model"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_names = [m.strip() for m in args.model_names.split(",")]

    CreditScoreModelTrainer().run(train_dir = args.train, model_dir = args.model_dir, model_names = model_names,)
