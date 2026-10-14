import json
import os
import traceback
import joblib
import pandas as pd

JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"
CLASS_NAMES = ["POOR", "STANDARD", "GOOD"]
FEATURE_NAMES = [
    "Unnamed: 0",
    "Age",
    "Occupation",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Credit_Mix",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Payment_Behaviour",
    "Monthly_Balance",
    "Credit_History_Age_Months",
]

NUMERIC_COLS = [
    "Unnamed: 0", "Age", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance", "Credit_History_Age_Months",
]

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "best_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model.pkl tidak ditemukan di {model_dir}")

    try:
        model = joblib.load(model_path)
        print(f"Model loaded OK: {type(model)}")
        return model
    except Exception as exc:
        print(f"ERROR saat joblib.load: {exc}")
        traceback.print_exc()
        raise

def input_fn(request_body, request_content_type):
    try:
        if request_content_type == JSON_CONTENT_TYPE:
            if isinstance(request_body, (bytes, bytearray)):
                request_body = request_body.decode("utf-8")
            payload = json.loads(request_body)
            instances = payload["instances"]
            df = pd.DataFrame(instances, columns=FEATURE_NAMES)

        elif request_content_type == CSV_CONTENT_TYPE:
            if isinstance(request_body, (bytes, bytearray)):
                request_body = request_body.decode("utf-8")
            rows = [
                line.split(",")
                for line in request_body.strip().splitlines()
                if line.strip()
            ]
            df = pd.DataFrame(rows, columns=FEATURE_NAMES)

        else:
            raise ValueError(
                f"Content-Type tidak didukung: '{request_content_type}'")

        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
        return df

    except Exception as exc:
        traceback.print_exc()
        raise

def predict_fn(input_data, model):
    try:
        preds = model.predict(input_data)
        labels = [CLASS_NAMES[int(p)] for p in preds]

        result = {
            "predictions": [int(p) for p in preds],
            "labels": labels,
        }
        print(f"Result: {result}")
        return result

    except Exception as exc:
        traceback.print_exc()
        raise

def output_fn(prediction, accept):
    if accept in (JSON_CONTENT_TYPE, "*/*", ""):
        return json.dumps(prediction)

    raise ValueError(f"Accept type tidak didukung: '{accept}'")
