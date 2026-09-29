# Complete DeFiLens Fraud Detection Module

## Backend
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# edit .env and add GEMINI_API_KEY
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

Swagger: http://localhost:8001/docs
Health: http://localhost:8001/api/v1/health

## Frontend
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

The frontend calls `POST /api/v1/assess-risk` and understands the current API response wrapper (`assessment`).

## Current model assets
`transaction_dataset.csv` and `saved_model/` are included.

The actual `.env` is intentionally not included in this package. Copy `.env.example` to `.env`.
