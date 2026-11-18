from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent

# load the saved model bundle once, when the server starts
bundle = joblib.load(BASE_DIR / "model" / "loan_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]  # the 10 features, in training order

app = FastAPI(title="LoanOrNot")


class LoanApplication(BaseModel):
    Applicant_Income: float = Field(..., ge=0)
    Coapplicant_Income: float = Field(..., ge=0)
    Age: float = Field(..., ge=18, le=100)
    Existing_Loans: float = Field(..., ge=0)
    Credit_Score: float = Field(..., ge=0)
    DTI_Ratio: float = Field(..., ge=0)
    Savings: float = Field(..., ge=0)
    Collateral_Value: float = Field(..., ge=0)
    Loan_Amount: float = Field(..., ge=0)
    Loan_Term: float = Field(..., gt=0)


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "app" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(application: LoanApplication):
    # same feature engineering as the notebook: squares of Credit_Score and DTI_Ratio
    row = {
        "Credit_score_sq": application.Credit_Score ** 2,
        "DTI_Ratio_sq": application.DTI_Ratio ** 2,
        "Applicant_Income": application.Applicant_Income,
        "Loan_Amount": application.Loan_Amount,
        "Collateral_Value": application.Collateral_Value,
        "Savings": application.Savings,
        "Age": application.Age,
        "Coapplicant_Income": application.Coapplicant_Income,
        "Loan_Term": application.Loan_Term,
        "Existing_Loans": application.Existing_Loans,
    }

    # keep the exact column order used during training
    df = pd.DataFrame([row])[feature_columns]
    scaled = scaler.transform(df)

    prediction = int(model.predict(scaled)[0])  # 0 = No, 1 = Yes
    probability = float(model.predict_proba(scaled)[0][1])

    return {
        "approved": prediction == 1,
        "decision": "Loan Approved" if prediction == 1 else "Loan Rejected",
        "approval_probability": round(probability, 4),
    }
