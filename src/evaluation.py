import json
import os
import tarfile
import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score,precision_score, recall_score,)

class CreditScoreModelEvaluator:
    CONTAINER_BASE = "/opt/ml/processing"
    LOCAL_BASE = "/home/ec2-user/SageMaker"
    MODEL_FILENAME = "best_model.pkl"
    TARGET_COL = "Credit_Score"
    CLASS_NAMES = ["Poor", "Standard", "Good"]

    def __init__(self):
        base = self.CONTAINER_BASE if os.path.exists(self.CONTAINER_BASE) else self.LOCAL_BASE
        self.model_dir = os.path.join(base, "model")
        self.test_dir = os.path.join(base, "test")
        self.output_dir = os.path.join(base, "evaluation")
        os.makedirs(self.output_dir, exist_ok=True)

    def _resolve_model(self):
        direct_path = os.path.join(self.model_dir, self.MODEL_FILENAME)
        if os.path.exists(direct_path):
            print(f"Loading model")
            return joblib.load(direct_path)

        tar_path = os.path.join(self.model_dir, "model.tar.gz")
        if os.path.exists(tar_path):
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=self.model_dir)

        if os.path.exists(direct_path):
            return joblib.load(direct_path)

    def run(self) -> dict:
        model = self._resolve_model()
        test_file = os.path.join(self.test_dir, "test.csv")

        df = pd.read_csv(test_file)
        X_test = df.drop(columns=[self.TARGET_COL])
        y_test = df[self.TARGET_COL]

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted") 
        
        report = {
            "multiclass_classification_metrics": {
                "accuracy": {"value": round(acc, 4)},
                "f1": {"value": round(f1, 4)},
                "precision": {"value": round(prec, 4)},
                "recall": {"value": round(rec, 4)},
            }
        }

        output_path = os.path.join(self.output_dir, "evaluation.json")
        with open(output_path, "w") as fp:
            json.dump(report, fp, indent=4)

        print(f"Accuracy  : {acc:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, target_names=self.CLASS_NAMES))

        return report

if __name__ == "__main__":
    CreditScoreModelEvaluator().run()