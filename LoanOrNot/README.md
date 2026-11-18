# LoanOrNot

Predicts whether a loan gets approved. A stacking classifier (top 10 features) is served with FastAPI and a small HTML page.
Pipeline: GitHub -> GitHub Actions (test) -> Render (deploy).

## Run locally
```
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000 (API docs at /docs).

## Run tests
```
pytest
```

## Deploy (Render + GitHub Actions)
1. Push this folder to a GitHub repo (branch `main`).
2. Render: New > Web Service > connect the repo > Runtime: Docker > Instance: Free.
   In the service settings set **Auto-Deploy = Off** (so GitHub Actions controls deployment).
3. Render service > Settings > **Deploy Hook** > copy the URL.
4. GitHub repo > Settings > Secrets and variables > Actions > New secret:
   name `RENDER_DEPLOY_HOOK`, value = the hook URL.
5. Push to `main`: tests run, and only if they pass the deploy hook fires.

## Important
`loan_model.pkl` was saved with scikit-learn 1.6.1, so `requirements.txt` pins that exact version.
If you retrain with another version, update the pin.
