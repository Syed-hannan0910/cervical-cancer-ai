"""
CervicalAI - Explainable Two-Stage Cervical Cancer Detection Framework
FastAPI Backend - Main Application Entry Point
"""

import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CervicalAI")

# Import routers
from app.routers import predict, risk, report, health

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup & shutdown events."""
    logger.info("🚀 CervicalAI Backend starting up...")
    # Pre-load models at startup
    try:
        from app.models.fastvit_model import get_fastvit_model
        from app.models.xgboost_model import get_xgboost_model
        get_fastvit_model()
        get_xgboost_model()
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Model pre-load warning (will load on first request): {e}")
    yield
    logger.info("🛑 CervicalAI Backend shutting down...")


app = FastAPI(
    title="CervicalAI - Cervical Cancer Detection API",
    description="""
## Explainable Two-Stage Cervical Cancer Detection Framework

### Architecture
1. **Stage 1 - XGBoost Risk Stratification**: Clinical risk factor analysis with SHAP explainability
2. **Stage 2 - FastViT Image Classification**: Pap smear cytological image analysis with GradCAM

### Features
- Pap smear image classification (SipakMed categories)
- Clinical risk factor prediction with SHAP values
- GradCAM saliency maps for visual explainability  
- SHAP waterfall/beeswarm plots for feature importance
- Automated PDF report generation
- Combined risk scoring

### Datasets
- [Cervical Cancer Risk Factors](https://www.kaggle.com/datasets/ranzeet013/cervical-cancer-dataset) 
- [SipakMed Cytological Images](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed)
    """,
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration for Vercel frontend
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS]

# Add production Vercel URL if set
vercel_url = os.getenv("VERCEL_URL")
if vercel_url:
    ALLOWED_ORIGINS.append(f"https://{vercel_url}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS + ["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for model artifacts / reports
os.makedirs("static/reports", exist_ok=True)
os.makedirs("static/gradcam", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Register routers
app.include_router(health.router, prefix="/api", tags=["Health Check"])
app.include_router(predict.router, prefix="/api/predict", tags=["Image Classification"])
app.include_router(risk.router, prefix="/api/risk", tags=["Risk Assessment"])
app.include_router(report.router, prefix="/api/report", tags=["PDF Report"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "CervicalAI Backend",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "stages": {
            "stage1": "XGBoost Clinical Risk Prediction + SHAP",
            "stage2": "FastViT Pap Smear Classification + GradCAM"
        }
    }
