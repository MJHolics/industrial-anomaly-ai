"""
학습 데이터 생성 스크립트.
정상 패턴 95% + 이상 패턴 5% 혼합 CSV 생성.

실행: python scripts/generate_data.py
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

SEED = 42
N_NORMAL = 10000
N_ANOMALY = 500
OUTPUT_PATH = "data/raw/sensor_data.csv"

rng = np.random.RandomState(SEED)


def generate_normal(n: int) -> pd.DataFrame:
    """정상 운전 조건 — 좁은 분포."""
    return pd.DataFrame({
        "temperature": rng.normal(70, 3, n),    # °C
        "vibration":   rng.normal(0.5, 0.08, n), # mm/s
        "current":     rng.normal(12, 0.8, n),   # A
        "label": 0,
    })


def generate_anomaly(n: int) -> pd.DataFrame:
    """이상 패턴 — 과열 / 과진동 / 전류 스파이크 혼합."""
    n3 = n // 3
    remainder = n - 2 * n3

    overheating = pd.DataFrame({
        "temperature": rng.normal(95, 5, n3),
        "vibration":   rng.normal(0.5, 0.08, n3),
        "current":     rng.normal(12, 0.8, n3),
        "label": 1,
    })
    over_vibration = pd.DataFrame({
        "temperature": rng.normal(70, 3, n3),
        "vibration":   rng.normal(2.5, 0.3, n3),
        "current":     rng.normal(12, 0.8, n3),
        "label": 1,
    })
    current_spike = pd.DataFrame({
        "temperature": rng.normal(70, 3, remainder),
        "vibration":   rng.normal(0.5, 0.08, remainder),
        "current":     rng.normal(30, 3, remainder),
        "label": 1,
    })
    return pd.concat([overheating, over_vibration, current_spike], ignore_index=True)


def add_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    start = datetime(2024, 1, 1)
    df["timestamp"] = [
        (start + timedelta(seconds=i * 10)).isoformat()
        for i in range(len(df))
    ]
    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    normal = generate_normal(N_NORMAL)
    anomaly = generate_anomaly(N_ANOMALY)
    df = pd.concat([normal, anomaly], ignore_index=True).sample(frac=1, random_state=SEED)
    df = add_timestamps(df.reset_index(drop=True))
    df = df[["timestamp", "temperature", "vibration", "current", "label"]]

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"  정상: {(df.label == 0).sum():,}개")
    print(f"  이상: {(df.label == 1).sum():,}개")
    print(df.describe().round(2))
