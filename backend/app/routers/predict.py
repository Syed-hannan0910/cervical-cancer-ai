"""
API Router: Image Classification (FastViT + GradCAM)
POST /api/predict/image
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from app.models.fastvit_model import predict_image

logger = logging.getLogger("CervicalAI.Router.Predict")

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/tiff", "image/bmp"}
MAX_SIZE_MB = 20


class PredictionResponse(BaseModel):
    predicted_class: int
    class_name: str
    confidence: float
    risk_level: str
    risk_score: float
    description: str
    class_probabilities: list
    gradcam_image: str
    model: str
    dataset: str


@router.post("/image", response_model=PredictionResponse, summary="Classify pap smear image")
async def classify_image(file: UploadFile = File(..., description="Pap smear cytological image (JPG/PNG/TIFF)")):
    """
    ## Pap Smear Image Classification

    Classifies a cervical cytological image into one of 5 SipakMed categories using FastViT-T8.

    Returns:
    - **Predicted cell type** with confidence score
    - **Risk level** (low/medium/high)
    - **GradCAM heatmap** overlaid on original image (base64 PNG)
    - **Per-class probabilities** for all 5 SipakMed categories

    ### SipakMed Categories:
    | Class | Name | Risk |
    |-------|------|------|
    | 0 | Dyskeratotic | High |
    | 1 | Koilocytotic | High |
    | 2 | Metaplastic | Medium |
    | 3 | Parabasal | Medium |
    | 4 | Superficial-Intermediate | Low |
    """
    # Validate file type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES and not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp")):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Please upload a JPG, PNG, or TIFF image."
        )

    # Read and validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Maximum allowed: {MAX_SIZE_MB}MB."
        )

    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupted.")

    logger.info(f"Processing image: {file.filename} ({size_mb:.2f}MB)")

    try:
        result = predict_image(contents)
        logger.info(f"Prediction: {result['class_name']} (confidence={result['confidence']:.3f})")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image classification failed: {str(e)}")
