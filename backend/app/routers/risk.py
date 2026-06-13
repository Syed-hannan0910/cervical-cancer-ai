"""
API Router: Clinical Risk Assessment (XGBoost + SHAP)
POST /api/risk/assess
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional

from app.models.xgboost_model import predict_risk, FEATURE_NAMES

logger = logging.getLogger("CervicalAI.Router.Risk")

router = APIRouter()


class RiskAssessmentRequest(BaseModel):
    # Demographics
    age: float = Field(..., ge=13, le=90, description="Patient age in years")
    num_sexual_partners: float = Field(1.0, ge=0, le=50, description="Number of sexual partners")
    first_sexual_intercourse: float = Field(18.0, ge=10, le=40, description="Age at first sexual intercourse")
    num_pregnancies: float = Field(0.0, ge=0, le=20, description="Number of pregnancies")

    # Smoking
    smokes: float = Field(0.0, ge=0, le=1, description="Smoker (0=No, 1=Yes)")
    smokes_years: float = Field(0.0, ge=0, le=50, description="Years of smoking")
    smokes_packs_year: float = Field(0.0, ge=0, le=60, description="Packs smoked per year")

    # Contraceptives
    hormonal_contraceptives: float = Field(0.0, ge=0, le=1, description="Using hormonal contraceptives (0/1)")
    hormonal_contraceptives_years: float = Field(0.0, ge=0, le=40, description="Years on hormonal contraceptives")

    # IUD
    iud: float = Field(0.0, ge=0, le=1, description="IUD use (0=No, 1=Yes)")
    iud_years: float = Field(0.0, ge=0, le=30, description="Years with IUD")

    # STDs
    stds: float = Field(0.0, ge=0, le=1, description="Has any STD (0/1)")
    stds_number: float = Field(0.0, ge=0, le=10, description="Total number of STDs")
    stds_condylomatosis: float = Field(0.0, ge=0, le=1, description="Condylomatosis (0/1)")
    stds_hpv: float = Field(0.0, ge=0, le=1, description="HPV infection (0/1)")
    stds_hiv: float = Field(0.0, ge=0, le=1, description="HIV positive (0/1)")
    stds_syphilis: float = Field(0.0, ge=0, le=1, description="Syphilis (0/1)")

    # Prior diagnoses
    dx_cancer: float = Field(0.0, ge=0, le=1, description="Prior cancer diagnosis (0/1)")
    dx_cin: float = Field(0.0, ge=0, le=1, description="CIN diagnosis (0/1)")
    dx_hpv: float = Field(0.0, ge=0, le=1, description="HPV diagnosis (0/1)")

    # Optional patient info for report
    patient_id: Optional[str] = Field(None, description="Patient identifier for report")
    patient_name: Optional[str] = Field(None, description="Patient name for report")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "num_sexual_partners": 3,
                "first_sexual_intercourse": 17,
                "num_pregnancies": 2,
                "smokes": 1,
                "smokes_years": 5,
                "smokes_packs_year": 10,
                "hormonal_contraceptives": 1,
                "hormonal_contraceptives_years": 8,
                "iud": 0,
                "iud_years": 0,
                "stds": 1,
                "stds_number": 1,
                "stds_condylomatosis": 0,
                "stds_hpv": 1,
                "stds_hiv": 0,
                "stds_syphilis": 0,
                "dx_cancer": 0,
                "dx_cin": 0,
                "dx_hpv": 1,
                "patient_id": "PT-20481",
                "patient_name": "Anonymous"
            }
        }


@router.post("/assess", summary="Clinical risk assessment with SHAP explainability")
async def assess_risk(request: RiskAssessmentRequest):
    """
    ## Stage 1: XGBoost Clinical Risk Assessment

    Analyzes patient clinical and demographic features to predict cervical cancer risk.

    Returns:
    - **Risk score** (0.0–1.0)
    - **Risk tier** (Low / Moderate / High / Critical)
    - **Clinical recommendation** with urgency level
    - **SHAP values** for each feature (explainability)
    - **SHAP waterfall chart** (base64 PNG)
    - **Top risk factors** ranked by impact

    This implements Stage 1 of the two-stage detection framework.
    """
    # Map request fields to model feature names
    features = {
        "Age": request.age,
        "Number of sexual partners": request.num_sexual_partners,
        "First sexual intercourse": request.first_sexual_intercourse,
        "Num of pregnancies": request.num_pregnancies,
        "Smokes": request.smokes,
        "Smokes (years)": request.smokes_years,
        "Smokes (packs/year)": request.smokes_packs_year,
        "Hormonal Contraceptives": request.hormonal_contraceptives,
        "Hormonal Contraceptives (years)": request.hormonal_contraceptives_years,
        "IUD": request.iud,
        "IUD (years)": request.iud_years,
        "STDs": request.stds,
        "STDs (number)": request.stds_number,
        "STDs:condylomatosis": request.stds_condylomatosis,
        "STDs:HPV": request.stds_hpv,
        "STDs:HIV": request.stds_hiv,
        "STDs:syphilis": request.stds_syphilis,
        "Dx:Cancer": request.dx_cancer,
        "Dx:CIN": request.dx_cin,
        "Dx:HPV": request.dx_hpv,
    }

    logger.info(f"Risk assessment request - Age: {request.age}, STDs:HPV: {request.stds_hpv}, Dx:HPV: {request.dx_hpv}")

    try:
        result = predict_risk(features)
        result["patient_id"] = request.patient_id
        result["patient_name"] = request.patient_name
        result["input_features"] = features
        logger.info(f"Risk score: {result['risk_score']:.3f} ({result['risk_tier']})")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Risk assessment error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")


@router.get("/features", summary="Get list of clinical features used in risk model")
async def get_features():
    """Returns the list of clinical features accepted by the XGBoost risk model."""
    return {"features": FEATURE_NAMES, "count": len(FEATURE_NAMES)}
