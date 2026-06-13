"""Health check router."""
from fastapi import APIRouter
from datetime import datetime
import sys, platform

router = APIRouter()

@router.get("/health", summary="Health check")
async def health_check():
    return {
        "status": "healthy",
        "service": "CervicalAI Backend",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "python": sys.version,
        "platform": platform.system(),
    }

@router.get("/ping")
async def ping():
    return {"pong": True}
