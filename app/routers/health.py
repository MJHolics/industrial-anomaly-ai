from fastapi import APIRouter
from app.models.anomaly_detector import detector
from app.schemas.sensor import ModelInfo

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "model_fitted": detector.is_fitted}


@router.get("/model/info", response_model=ModelInfo)
async def model_info():
    return ModelInfo(
        model_type="IsolationForest",
        contamination=detector.contamination,
        n_estimators=detector.n_estimators,
        trained_at=detector.trained_at,
        is_fitted=detector.is_fitted,
    )
