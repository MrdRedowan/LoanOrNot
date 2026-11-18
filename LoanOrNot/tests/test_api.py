from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

good_applicant = {
    "Applicant_Income": 10000,
    "Coapplicant_Income": 5000,
    "Age": 35,
    "Existing_Loans": 1,
    "Credit_Score": 800,
    "DTI_Ratio": 0.2,
    "Savings": 5000,
    "Collateral_Value": 30000,
    "Loan_Amount": 25000,
    "Loan_Term": 36,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_page_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "LoanOrNot" in response.text


def test_predict_returns_expected_fields():
    response = client.post("/predict", json=good_applicant)
    assert response.status_code == 200
    result = response.json()
    assert "approved" in result
    assert "decision" in result
    assert 0 <= result["approval_probability"] <= 1


def test_strong_applicant_is_approved():
    response = client.post("/predict", json=good_applicant)
    assert response.json()["approved"] is True


def test_weak_applicant_is_rejected():
    weak_applicant = dict(good_applicant, Credit_Score=450, DTI_Ratio=0.7)
    response = client.post("/predict", json=weak_applicant)
    assert response.json()["approved"] is False


def test_missing_field_gives_422():
    incomplete = dict(good_applicant)
    del incomplete["Age"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
