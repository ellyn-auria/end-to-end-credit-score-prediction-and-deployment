import json
import os

import boto3
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Score Predictor", layout="wide")
st.title("Credit Score Prediction App")

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


runtime = get_runtime_client()

st.subheader("Input Customer Data")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    occupations_list = [
        "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer",
        "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager",
        "Accountant", "Musician", "Mechanic", "Writer", "Architect",
    ]
    occupation = st.selectbox("Occupation", occupations_list, index=0)

    monthly_inhand_salary  = st.number_input("Monthly Inhand Salary ($)", min_value=0.0, value=3000.0)
    num_bank_accounts = st.number_input("Num of Bank Accounts", min_value=0, max_value=20, value=3)
    num_credit_card = st.number_input("Num of Credit Card", min_value=0, max_value=20, value=3)
    interest_rate = st.number_input("Interest Rate (%)", min_value=1, max_value=50, value=10)
    num_of_loan = st.number_input("Num of Loan", min_value=0, max_value=20, value=2)

with col2:
    delay_from_due_date = st.number_input("Delay from due date (Days)",  min_value=0, max_value=100, value=10)
    num_of_delayed_payment = st.number_input("Num of Delayed Payment", min_value=0, max_value=50, value=5)
    changed_credit_limit = st.number_input("Changed Credit Limit", min_value=0.0, value=5.0)
    num_credit_inquiries = st.number_input("Num of Credit Inquiries", min_value=0, max_value=30, value=3)

    credit_mix_list = ["Bad", "Standard", "Good"]
    credit_mix = st.selectbox("Credit Mix", credit_mix_list, index=1)

    outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, value=1000.0)

with col3:
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)

    payment_min_list = ["Yes", "No", "NM"]
    payment_of_min_amount = st.selectbox("Payment of Min Amount", payment_min_list, index=0)

    total_emi_per_month = st.number_input("Total EMI per month ($)", min_value=0.0, value=50.0)
    amount_invested_monthly = st.number_input("Amount invested monthly ($)", min_value=0.0, value=50.0)
    monthly_balance = st.number_input("Monthly Balance ($)", min_value=0.0, value=300.0)
    credit_history_months = st.number_input("Credit History Age (in Months)", min_value=0, value=120)

payment_behaviour_list = [
    "Low_spent_Small_value_payments", "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments", "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments", "High_spent_Large_value_payments",
]
payment_behaviour = st.selectbox("Payment Behaviour", payment_behaviour_list, index=0)

with st.sidebar:
    st.caption("Endpoint")
    st.code(ENDPOINT_NAME)
    st.caption("Region")
    st.code(REGION)

st.markdown("---")

if st.button("Analyze Credit Score", use_container_width=True):
    instance = [
        0,
        age,
        occupation,
        monthly_inhand_salary,
        num_bank_accounts,
        num_credit_card,
        interest_rate,
        num_of_loan,
        delay_from_due_date,
        num_of_delayed_payment,
        changed_credit_limit,
        num_credit_inquiries,
        credit_mix,
        outstanding_debt,
        credit_utilization_ratio,
        payment_of_min_amount,
        total_emi_per_month,
        amount_invested_monthly,
        payment_behaviour,
        monthly_balance,
        credit_history_months,
    ]

    payload = json.dumps({"instances": [instance]})

    with st.spinner("Analyzing..."):
        try:
            response = runtime.invoke_endpoint(
                EndpointName = ENDPOINT_NAME,
                ContentType  = "application/json",
                Accept       = "application/json",
                Body         = payload,
            )
            result       = json.loads(response["Body"].read().decode("utf-8"))
            prediction   = result["predictions"][0]
            result_label = result["labels"][0]

            st.subheader("Analysis Result:")
            if prediction == 0:
                st.error(f"This customer's credit score is classified as: **{result_label}**")
                st.write(
                    "The customer has a risky debt ratio and a history of late payments. "
                    "It is highly discouraged to approve new credit or increase their credit limit at this time."
                )
            elif prediction == 1:
                st.warning(f"This customer's credit score is classified as: **{result_label}**")
                st.write(
                    "The customer is at an average threshold. Credit products can be offered, "
                    "but with moderate limits and standard monitoring."
                )
            else:
                st.success(f"This customer's credit score is classified as: **{result_label}**")
                st.write(
                    "The customer has a very healthy credit profile! They are highly eligible "
                    "for credit approvals, priority limits, or promotional offers."
                )

        except Exception as e:
            st.error(
                f"Error {e}")