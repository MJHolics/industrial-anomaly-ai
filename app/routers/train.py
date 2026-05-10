import os
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.metrics import precision_score, recall_score, f1_score
from app.models.anomaly_detector import detector

router = APIRouter()

DATA_PATH = "data/raw/sensor_data.csv"
FEATURES = ["temperature", "vibration", "current"]


class TrainResult(BaseModel):
    status: str
    trained_at: str
    n_samples: int
    contamination: float
    precision: float
    recall: float
    f1: float
    message: str


@router.post("/model/train", response_model=TrainResult)
async def train_model(contamination: float = 0.05):
    """
    data/raw/sensor_data.csv 로 모델 재학습.
    contamination: 전체 데이터 중 이상 비율 추정값 (기본 0.05 = 5%)
    """
    if not os.path.exists(DATA_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"학습 데이터 없음: {DATA_PATH} — python scripts/generate_data.py 를 먼저 실행하세요."
        )

    df = pd.read_csv(DATA_PATH)
    if not all(f in df.columns for f in FEATURES):
        raise HTTPException(status_code=400, detail=f"필수 컬럼 없음: {FEATURES}")

    X = df[FEATURES].values
    y = df["label"].values if "label" in df.columns else None

    # contamination 업데이트 후 학습
    detector.contamination = contamination
    detector.model.set_params(contamination=contamination)
    detector.train(X[y == 0] if y is not None else X)

    # 성능 평가 (레이블 있을 때만)
    precision = recall = f1 = -1.0
    if y is not None:
        from sklearn.preprocessing import StandardScaler
        X_scaled = detector.scaler.transform(X)
        raw_preds = detector.model.predict(X_scaled)
        y_pred = (raw_preds == -1).astype(int)
        precision = round(float(precision_score(y, y_pred, zero_division=0)), 3)
        recall    = round(float(recall_score(y, y_pred, zero_division=0)), 3)
        f1        = round(float(f1_score(y, y_pred, zero_division=0)), 3)

    return TrainResult(
        status="success",
        trained_at=detector.trained_at or datetime.now().isoformat(),
        n_samples=len(X),
        contamination=contamination,
        precision=precision,
        recall=recall,
        f1=f1,
        message=f"모델 학습 완료. F1={f1:.3f}",
    )
