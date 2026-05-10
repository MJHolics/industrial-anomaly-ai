from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class SensorReading(BaseModel):
    sensor_id: str = Field(..., example="sensor_01")
    temperature: float = Field(..., ge=-50, le=200, example=72.3)
    vibration: float = Field(..., ge=0, le=100, example=0.42)
    current: float = Field(..., ge=0, le=50, example=12.1)
    timestamp: Optional[datetime] = None


class DetectionResult(BaseModel):
    sensor_id: str
    timestamp: datetime
    temperature: float
    vibration: float
    current: float
    anomaly_score: float
    is_anomaly: bool
    severity: str  # normal / warning / critical


class ModelInfo(BaseModel):
    model_type: str
    contamination: float
    n_estimators: int
    trained_at: Optional[str]
    is_fitted: bool
