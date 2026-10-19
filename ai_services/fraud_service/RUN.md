# DeFiLens Fraud Detection Module (refactored)

## What changed
- `model_service.py` now prepares the wallet's feature matrix once per wallet assessment and reuses it for XGBoost prediction, boosting progression, and local SHAP.
- The dataset is loaded from `data/transaction_dataset.csv`, with a fallback to the legacy root-level CSV.
- `train_model.py` uses the same dataset path convention.
- Risk formulas, model artifacts, API routes, response keys, and frontend behavior are preserved.

## Backend (PowerShell)
```powershell
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
# Optional: add GEMINI_API_KEY to .env for GenAI explanations
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```
Swagger: http://localhost:8001/docs
Health: http://localhost:8001/api/v1/health

## Frontend (new terminal)
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

The frontend calls `POST /api/v1/assess-risk`.

## Notes
- The actual `.env`, virtual environment, `node_modules`, and temporary Module 1/2 payload JSON files are not included.
- `saved_model/` contains the existing trained model and artifacts. Do not run `train_model.py` unless you intentionally want to retrain/replace those artifacts.
