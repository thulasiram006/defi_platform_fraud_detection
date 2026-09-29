import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_service import get_model_service

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

BASE_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")

GEMINI_MODEL = "gemini-2.5-flash"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="DeFiLens Fraud Detection API",
    description=(
        "Live wallet fraud detection and DeFi transaction "
        "risk assessment API powered by XGBoost."
    ),
    version="3.0.0",
)


# ============================================================
# CORS
# ============================================================
# React/Vite frontend runs on port 5173 while FastAPI runs on
# port 8001. This allows the browser to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_service = get_model_service()


# ============================================================
# SERVICE URLS
# ============================================================

# Optional old gateway endpoint.
NODE_GATEWAY_URL = "http://localhost:5000/api/fraud-evaluation"

# Module 3 = Blockchain Module
#
# IMPORTANT:
# Change this URL if your Blockchain Module uses a
# different port or endpoint.
BLOCKCHAIN_MODULE_URL = (
    "http://localhost:5000/api/blockchain/process-transaction"
)


# ============================================================
# GEMINI EXPLANATION
# ============================================================

def _simple_feature_name(feature):
    mapping = {
        "sent tnx": "transactions sent",
        "received tnx": "transactions received",
        "total ether sent": "ETH sent",
        "total ether received": "ETH received",
        "total ether balance": "ETH balance",
        "number of created contracts": "contracts created",
        "unique received from addresses": "different sending addresses",
        "unique sent to addresses": "different receiving addresses",
        "avg min between sent tnx": "average time between sent transactions",
        "avg min between received tnx": "average time between received transactions",
        "time diff between first and last (mins)": "wallet activity period",
        "min value received": "smallest received amount",
        "max value received": "largest received amount",
        "avg val received": "average received amount",
        "min val sent": "smallest sent amount",
        "max val sent": "largest sent amount",
        "avg val sent": "average sent amount",
    }
    key = str(feature or "").strip().lower()
    return mapping.get(key, str(feature or "transaction behaviour").strip())


def _fallback_genai(assessment):
    probability = float(
        assessment.get("base_xgboost_probability",
                       assessment.get("on_chain_risk", 0.0)) or 0.0
    )
    prediction = bool(assessment.get("model_prediction", 0))
    title = (
        "Why is this transaction suspicious?"
        if prediction
        else "Why is this transaction considered safe?"
    )

    rows = []
    for row in assessment.get("local_shap", []) or []:
        if isinstance(row, dict):
            try:
                value = float(row.get("shap_value", 0) or 0)
            except (TypeError, ValueError):
                value = 0.0
            if (prediction and value > 0) or (not prediction and value < 0):
                rows.append((abs(value), value, _simple_feature_name(row.get("feature"))))

    rows.sort(reverse=True)
    points = []
    for _, value, name in rows[:3]:
        if prediction:
            points.append(f"The wallet's {name} increased the suspiciousness of the transaction.")
        else:
            points.append(f"The wallet's {name} helped keep the transaction risk lower.")

    defaults = (
        [
            "The wallet shows transaction behaviour that is less typical of legitimate activity.",
            "Several transaction signals together increased the fraud risk.",
            f"The fraud detector estimated a {probability * 100:.2f}% fraud probability.",
        ]
        if prediction else
        [
            "The wallet shows transaction behaviour closer to legitimate activity.",
            "The fraud detector did not find strong signals pointing toward fraud.",
            f"The fraud detector estimated a {probability * 100:.2f}% fraud probability.",
        ]
    )
    for point in defaults:
        if len(points) >= 3:
            break
        points.append(point)

    return {"title": title, "points": points[:3], "source": "Fallback explanation"}


def generate_genai_explanation(assessment):
    fallback = _fallback_genai(assessment)

    if genai is None or types is None:
        return fallback

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI ERROR: GEMINI_API_KEY is not set.")
        return fallback

    probability = float(
        assessment.get("base_xgboost_probability",
                       assessment.get("on_chain_risk", 0.0)) or 0.0
    )
    prediction = bool(assessment.get("model_prediction", 0))
    title = fallback["title"]

    signals = []
    for row in assessment.get("local_shap", []) or []:
        if not isinstance(row, dict):
            continue
        try:
            value = float(row.get("shap_value", 0) or 0)
        except (TypeError, ValueError):
            continue
        if (prediction and value <= 0) or (not prediction and value >= 0):
            continue
        signals.append({
            "behaviour": _simple_feature_name(row.get("feature")),
            "direction": "higher fraud risk" if value > 0 else "lower fraud risk",
            "strength": round(abs(value), 6),
        })

    signals.sort(key=lambda x: x["strength"], reverse=True)
    signals = signals[:4]

    decision = "suspicious" if prediction else "likely legitimate"
    prompt = f"""
You are the explanation component of a DeFi fraud detection dashboard.

The fraud detector has already classified this transaction as: {decision}.
Fraud probability: {probability * 100:.2f}%.

Strongest supplied signals:
{json.dumps(signals, indent=2)}

Write exactly 3 short points explaining the result to an ordinary DeFi user.
Rules:
- Support the existing classification; do not change it.
- Use only the supplied signals.
- Do not invent facts.
- Do not mention SHAP, XGBoost, machine learning, algorithms, weights, or model internals.
- Do not say the transaction is definitely fraudulent.
- One sentence per point.
- Return only the three points, one per line.
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=180,
            ),
        )
        text = (getattr(response, "text", None) or "").strip()
        points = []
        for line in text.splitlines():
            line = re.sub(r"^[-•*]\s*", "", line.strip())
            line = re.sub(r"^\d+[.)]\s*", "", line).strip()
            if len(line) >= 15:
                points.append(line)

        if len(points) < 3:
            raise RuntimeError(f"Gemini returned only {len(points)} usable points.")

        return {"title": title, "points": points[:3], "source": "Gemini GenAI"}

    except Exception as exc:
        print("========== GEMINI ACTUAL ERROR ==========")
        print(type(exc).__name__)
        print(repr(exc))
        print("==========================================")
        return fallback


# ============================================================
# LOCAL CACHE
# ============================================================

LATEST_MODULE2_PATH = (
    Path(__file__).resolve().parent
    / "latest_module2_payload.json"
)


# ============================================================
# REQUEST MODELS
# ============================================================

class RiskAssessmentRequest(BaseModel):

    # --------------------------------------------------------
    # Wallet supplied by Module 1
    # --------------------------------------------------------

    borrower_wallet_address: Optional[str] = None
    borrower_address: Optional[str] = None
    wallet_address: Optional[str] = None

    # --------------------------------------------------------
    # Financial information supplied by Module 1
    # --------------------------------------------------------

    verified_monthly_income: float = 0.0
    total_liabilities: float = 0.0
    requested_loan_amount: float = 0.0
    discrepancy_ratio: float = 0.0
    dr_risk_penalty: float = 0.0

    # --------------------------------------------------------
    # Optional direct transaction data
    # --------------------------------------------------------

    transaction: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional direct transaction feature dictionary."
    )

    transaction_hash: Optional[str] = None

    # --------------------------------------------------------
    # Model options
    # --------------------------------------------------------

    include_shap: bool = True
    include_boosting: bool = True

    # --------------------------------------------------------
    # Forwarding options
    # --------------------------------------------------------

    # Optional old gateway forwarding.
    forward_to_gateway: bool = False

    # Blockchain forwarding.
    #
    # IMPORTANT:
    # Even if this is True, the Blockchain Module will
    # ONLY be called when the fraud decision is APPROVED.
    forward_to_blockchain: bool = True

    # Compatibility with Module 2 Streamlit app.
    forward_to_module3: bool = True

    # DeFi transaction information supplied by Module 1.
    action: Optional[str] = "borrow"
    collateral_amount: float = 0.0
    amount: Optional[float] = None


class GatewayRequest(BaseModel):

    assessment: Dict[str, Any]

    wallet_address: Optional[str] = None
    borrower_address: Optional[str] = None
    transaction_hash: Optional[str] = None


# ============================================================
# HELPERS
# ============================================================

def current_timestamp():
    """
    Return the current UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def send_to_service(
    url: str,
    payload: Dict[str, Any],
    timeout: int = 30
):
    """
    Send JSON data to another backend service using HTTP POST.
    """

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=timeout
        )

        try:
            data = response.json()

        except Exception:

            data = {
                "raw_response": response.text
            }

        return {
            "success": response.ok,
            "status_code": response.status_code,
            "response": data
        }

    except requests.exceptions.RequestException as e:

        return {
            "success": False,
            "status_code": None,
            "response": {
                "error": str(e)
            }
        }


def extract_borrower_address(
    request: RiskAssessmentRequest
):
    """
    Extract the wallet address supplied by Module 1.
    """

    return (
        request.borrower_wallet_address
        or request.borrower_address
        or request.wallet_address
    )


# ============================================================
# ROOT / QUICK CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "service": "DeFiLens Fraud Detection API",
        "status": "running",
        "assessment_endpoint": "/api/v1/assess-risk",
        "health_endpoint": "/api/v1/health"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/v1/health")
def health():

    return {
        "status": "healthy",
        "service": "DeFiLens Fraud Detection Engine",
        "timestamp": current_timestamp(),
        "model": model_service.get_model_info()
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/api/v1/model")
def model_information():

    return {
        "success": True,
        "model": model_service.get_model_info()
    }


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

@app.get("/api/v1/explainability")
def explainability():

    return {
        "success": True,
        "features": model_service.get_shap_data()
    }


# ============================================================
# LIVE WALLET FRAUD ASSESSMENT
# ============================================================

@app.post("/api/v1/assess-risk")
@app.post("/api/v1/module1-input")
def assess_risk(
    request: RiskAssessmentRequest
):

    # ========================================================
    # STEP 1
    # Receive wallet from Module 1
    # ========================================================

    borrower_address = extract_borrower_address(request)

    # --------------------------------------------------------
    # Store Module 1 input locally for dashboard/debugging
    # --------------------------------------------------------

    try:

        module1_received_path = (
            Path(__file__).resolve().parent
            / "latest_module1_payload.json"
        )

        with open(
            module1_received_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                request.dict(),
                file,
                indent=2
            )

    except Exception as cache_error:

        print(
            "Warning: could not save Module 1 payload: "
            f"{cache_error}"
        )

    # ========================================================
    # STEP 2
    # Run Fraud Detection
    # ========================================================

    # --------------------------------------------------------
    # LIVE WALLET MODE
    # Module 1 supplies borrower wallet address.
    # --------------------------------------------------------

    if borrower_address:

        try:

            assessment = model_service.assess_wallet(

                borrower_address=borrower_address,

                verified_monthly_income=(
                    request.verified_monthly_income
                ),

                total_liabilities=(
                    request.total_liabilities
                ),

                requested_loan_amount=(
                    request.requested_loan_amount
                ),

                discrepancy_ratio=(
                    request.discrepancy_ratio
                ),

                dr_penalty=(
                    request.dr_risk_penalty
                ),

                include_shap=(
                    request.include_shap
                ),

                include_boosting=(
                    request.include_boosting
                )
            )

        except Exception as e:

            import traceback

            print("\n" + "=" * 70)
            print("DEFILENS WALLET FRAUD ASSESSMENT ERROR")
            print("=" * 70)

            traceback.print_exc()

            print("=" * 70 + "\n")

            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

        # ----------------------------------------------------
        # Wallet not found
        # ----------------------------------------------------

        if not assessment.get(
            "wallet_found",
            False
        ):

            raise HTTPException(
                status_code=404,
                detail=(
                    "Borrower wallet was not found in "
                    "transaction_dataset.csv."
                )
            )

    # ========================================================
    # BACKWARD COMPATIBILITY
    # Direct transaction mode
    # ========================================================

    elif request.transaction:

        try:

            assessment = model_service.assess(

                transaction=request.transaction,

                dr_penalty=(
                    request.dr_risk_penalty
                )
            )

        except Exception as e:

            import traceback

            print("\n" + "=" * 70)
            print("DEFILENS TRANSACTION ASSESSMENT ERROR")
            print("=" * 70)

            traceback.print_exc()

            print("=" * 70 + "\n")

            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

    else:

        raise HTTPException(
            status_code=400,
            detail=(
                "Provide borrower_address from Module 1 "
                "or direct transaction data."
            )
        )

    # ========================================================
    # STEP 3
    # Add metadata to fraud assessment
    # ========================================================

    assessment["timestamp"] = current_timestamp()

    assessment["borrower_address"] = (
        borrower_address
    )

    assessment["wallet_address"] = (
        request.wallet_address
        or borrower_address
    )

    assessment["transaction_hash"] = (
        request.transaction_hash
    )

    # ========================================================
    # STEP 3.5
    # Expose explainability results to the frontend
    # ========================================================

    # model_service already computes these when the request flags are True.
    assessment["shap"] = assessment.get("local_shap", [])
    assessment["boosting"] = assessment.get("boosting_progression", [])

    # Gemini explains the model output; it never changes the prediction.
    assessment["genai_explanation"] = generate_genai_explanation(assessment)
    assessment["genai_points"] = assessment["genai_explanation"].get("points", [])

# ========================================================
    # STEP 4
    # Extract Fraud Decision
    # ========================================================

    decision = assessment.get(
        "decision"
    )

    risk_level = assessment.get(
        "risk_level"
    )

    fraud_probability = assessment.get(
        "on_chain_risk"
    )

    # --------------------------------------------------------
    # Normalize decision for reliable comparison
    # --------------------------------------------------------

    if isinstance(decision, str):

        decision_normalized = (
            decision.strip()
            .upper()
        )

    else:

        decision_normalized = ""


    # ========================================================
    # STEP 5
    # Create Blockchain Module Payload
    # ========================================================

    blockchain_payload = {

        # ----------------------------------------------------
        # Borrower information
        # ----------------------------------------------------

        "borrower_wallet_address": (
            borrower_address
        ),

        "wallet_address": (
            request.wallet_address
            or borrower_address
        ),

        # ----------------------------------------------------
        # Financial information from Module 1
        # ----------------------------------------------------

        "verified_monthly_income": (
            request.verified_monthly_income
        ),

        "total_liabilities": (
            request.total_liabilities
        ),

        "requested_loan_amount": (
            request.requested_loan_amount
        ),

        "discrepancy_ratio": (
            request.discrepancy_ratio
        ),

        "dr_risk_penalty": (
            assessment.get(
                "dr_penalty",
                request.dr_risk_penalty
            )
        ),

        # ----------------------------------------------------
        # DeFi transaction information from Module 1
        # ----------------------------------------------------

        "action": (
            request.action
            or "borrow"
        ),

        "amount": (
            request.amount
            if request.amount is not None
            else request.requested_loan_amount
        ),

        "collateral_amount": (
            request.collateral_amount
        ),

        "collateral": (
            request.collateral_amount
        ),

        # ----------------------------------------------------
        # Fraud detection results from Module 2
        # ----------------------------------------------------

        "fraud_probability": (
            fraud_probability
        ),

        "fraud_status": (
            risk_level
        ),

        "decision": (
            decision
        ),

        "risk_level": (
            risk_level
        ),

        "final_risk": (
            assessment.get("final_risk", 0.0)
        ),

        "on_chain_risk": (
            assessment.get("on_chain_risk", 0.0)
        ),

        "off_chain_risk": (
            assessment.get("off_chain_risk", 0.0)
        ),

        "authorized_for_module3": (
            decision_normalized == "APPROVED"
        ),

        # ----------------------------------------------------
        # Transaction information
        # ----------------------------------------------------

        "transaction_hash": (
            request.transaction_hash
        ),

        "timestamp": (
            current_timestamp()
        )
    }


    # ========================================================
    # STEP 6
    # Save Module 2 result locally
    # ========================================================

    try:

        with open(
            LATEST_MODULE2_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                blockchain_payload,
                file,
                indent=2
            )

    except Exception as cache_error:

        print(
            "Warning: could not save Module 2 payload: "
            f"{cache_error}"
        )


    # ========================================================
    # STEP 7
    # SEND TO BLOCKCHAIN ONLY IF APPROVED
    # ========================================================

    blockchain_result = None

    if (
        (
            request.forward_to_blockchain
            or request.forward_to_module3
        )
        and decision_normalized == "APPROVED"
    ):

        print("\n" + "=" * 70)
        print("DEFILENS → BLOCKCHAIN MODULE")
        print("=" * 70)

        print(
            f"Wallet: {borrower_address}"
        )

        print(
            f"Fraud Probability: {fraud_probability}"
        )

        print(
            f"Risk Level: {risk_level}"
        )

        print(
            f"Decision: {decision}"
        )

        print(
            f"Sending to: {BLOCKCHAIN_MODULE_URL}"
        )

        print("=" * 70 + "\n")


        blockchain_result = send_to_service(

            BLOCKCHAIN_MODULE_URL,

            blockchain_payload,

            timeout=30
        )


    # ========================================================
    # STEP 8
    # REJECTED TRANSACTION
    # ========================================================

    else:

        if decision_normalized != "APPROVED":

            print("\n" + "=" * 70)
            print("DEFILENS TRANSACTION REJECTED")
            print("=" * 70)

            print(
                f"Wallet: {borrower_address}"
            )

            print(
                f"Fraud Probability: "
                f"{fraud_probability}"
            )

            print(
                f"Risk Level: {risk_level}"
            )

            print(
                f"Decision: {decision}"
            )

            print(
                "Blockchain Module was NOT called."
            )

            print(
                "No on-chain write will be performed."
            )

            print("=" * 70 + "\n")


    # ========================================================
    # OPTIONAL OLD GATEWAY
    # ========================================================

    gateway_result = None

    if request.forward_to_gateway:

        gateway_result = send_to_service(

            NODE_GATEWAY_URL,

            {
                "wallet_address": (
                    request.wallet_address
                    or borrower_address
                ),

                "transaction_hash": (
                    request.transaction_hash
                ),

                "assessment": assessment
            }
        )


    # Make the exact Module 3 payload available to the
    # Streamlit dashboard through the existing assessment object.
    assessment["module3_payload"] = blockchain_payload

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "success": True,

        "service": (
            "DeFiLens Hybrid Fraud Detection Engine"
        ),

        # ----------------------------------------------------
        # Fraud assessment
        # ----------------------------------------------------

        "assessment": assessment,

        # ----------------------------------------------------
        # Blockchain payload
        # ----------------------------------------------------

        "blockchain_payload": (
            blockchain_payload
        ),

        # ----------------------------------------------------
        # Blockchain response
        # ----------------------------------------------------

        "blockchain": (
            blockchain_result
        ),

        # ----------------------------------------------------
        # Status information
        # ----------------------------------------------------

        "blockchain_forwarded": (
            blockchain_result is not None
        ),

        "blockchain_forward_reason": (

            "Transaction approved by fraud detection"
            if blockchain_result is not None
            else
            "Transaction was not approved by fraud detection"
        ),

        # ----------------------------------------------------
        # Optional gateway
        # ----------------------------------------------------

        "gateway": gateway_result
    }


# ============================================================
# SIMPLE PREDICTION
# ============================================================

@app.post("/api/v1/predict")
def predict(
    request: RiskAssessmentRequest
):

    borrower_address = extract_borrower_address(
        request
    )

    # --------------------------------------------------------
    # Wallet prediction
    # --------------------------------------------------------

    if borrower_address:

        try:

            result = model_service.assess_wallet(

                borrower_address=borrower_address,

                verified_monthly_income=(
                    request.verified_monthly_income
                ),

                total_liabilities=(
                    request.total_liabilities
                ),

                requested_loan_amount=(
                    request.requested_loan_amount
                ),

                discrepancy_ratio=(
                    request.discrepancy_ratio
                ),

                dr_penalty=(
                    request.dr_risk_penalty
                ),

                include_shap=False,

                include_boosting=False
            )

            if not result.get(
                "wallet_found"
            ):

                raise HTTPException(
                    status_code=404,
                    detail=(
                        "Borrower wallet not found "
                        "in dataset."
                    )
                )

            return {

                "success": True,

                "prediction": result
            }

        except HTTPException:

            raise

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=str(e)
            )


    # --------------------------------------------------------
    # Direct transaction prediction
    # --------------------------------------------------------

    if not request.transaction:

        raise HTTPException(
            status_code=400,
            detail=(
                "Transaction data cannot be empty."
            )
        )

    try:

        result = model_service.predict(
            request.transaction
        )

        return {

            "success": True,

            "prediction": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# GATEWAY-FRIENDLY ENDPOINT
# ============================================================

@app.post("/api/fraud-evaluation")
def fraud_evaluation(
    request: GatewayRequest
):

    assessment = request.assessment

    return {

        "success": True,

        "wallet_address": (
            request.wallet_address
            or request.borrower_address
        ),

        "transaction_hash": (
            request.transaction_hash
        ),

        "risk": (
            assessment.get("final_risk")
        ),

        "risk_level": (
            assessment.get("risk_level")
        ),

        "decision": (
            assessment.get("decision")
        ),

        "action": (
            assessment.get("action")
        ),

        "timestamp": current_timestamp()
    }