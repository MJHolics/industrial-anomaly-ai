import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "models/isolation_forest.joblib"
SCALER_PATH = "models/scaler.joblib"


class AnomalyDetector:
    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.trained_at: str | None = None
        self._try_load()

    def _try_load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.is_fitted = True

    def _to_array(self, temperature: float, vibration: float, current: float) -> np.ndarray:
        return np.array([[temperature, vibration, current]])

    def train(self, X: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        self.trained_at = datetime.now().isoformat()
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

    def predict(self, temperature: float, vibration: float, current: float) -> dict:
        if not self.is_fitted:
            self._quick_fit()

        X = self._to_array(temperature, vibration, current)
        X_scaled = self.scaler.transform(X)

        score = float(self.model.score_samples(X_scaled)[0])
        label = int(self.model.predict(X_scaled)[0])
        is_anomaly = label == -1

        if not is_anomaly:
            severity = "normal"
        elif score > -0.15:
            severity = "warning"
        else:
            severity = "critical"

        return {
            "anomaly_score": round(score, 4),
            "is_anomaly": is_anomaly,
            "severity": severity,
        }

    def _quick_fit(self):
        """서버 시작 시 저장된 모델이 없으면 기본 데이터로 즉시 학습."""
        rng = np.random.RandomState(42)
        n = 1000
        normal = np.column_stack([
            rng.normal(70, 5, n),   # temperature
            rng.normal(0.5, 0.1, n),  # vibration
            rng.normal(12, 1, n),   # current
        ])
        self.train(normal)


detector = AnomalyDetector()
