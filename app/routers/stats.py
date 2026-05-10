import sqlite3
from datetime import datetime, date
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

DB_PATH = "data/anomaly_log.db"


class DailyStats(BaseModel):
    date: str
    total_anomalies: int
    critical_count: int
    warning_count: int
    avg_anomaly_score: float


class StatsResponse(BaseModel):
    total_anomalies: int
    today_anomalies: int
    critical_count: int
    warning_count: int
    avg_anomaly_score: float
    anomaly_rate_pct: float
    top_sensors: list[dict]
    hourly_trend: list[dict]


def _query(sql: str, params: tuple = ()):
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(sql, params).fetchall()
        return rows
    finally:
        conn.close()


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    today = date.today().isoformat()

    total_rows   = _query("SELECT COUNT(*) FROM anomaly_log")
    total        = total_rows[0][0] if total_rows else 0

    today_rows   = _query("SELECT COUNT(*) FROM anomaly_log WHERE timestamp LIKE ?", (f"{today}%",))
    today_count  = today_rows[0][0] if today_rows else 0

    critical_rows = _query("SELECT COUNT(*) FROM anomaly_log WHERE severity='critical'")
    critical      = critical_rows[0][0] if critical_rows else 0

    warning_rows  = _query("SELECT COUNT(*) FROM anomaly_log WHERE severity='warning'")
    warning       = warning_rows[0][0] if warning_rows else 0

    score_rows    = _query("SELECT AVG(anomaly_score) FROM anomaly_log")
    avg_score     = round(float(score_rows[0][0] or 0), 4)

    # 요청 총 수 추정 (anomaly_log는 이상만 저장 — 이상률은 DB에서 직접 계산 불가)
    # 간단하게 탐지된 이상 건수 기준으로 표시
    anomaly_rate = 100.0  # 로그는 이상만 저장됨

    # 센서별 이상 건수 TOP 5
    sensor_rows = _query("""
        SELECT sensor_id, COUNT(*) as cnt
        FROM anomaly_log
        GROUP BY sensor_id
        ORDER BY cnt DESC
        LIMIT 5
    """)
    top_sensors = [{"sensor_id": r[0], "anomaly_count": r[1]} for r in sensor_rows]

    # 시간대별 이상 건수 (최근 24시간)
    hourly_rows = _query("""
        SELECT substr(timestamp, 12, 2) as hour, COUNT(*) as cnt
        FROM anomaly_log
        WHERE timestamp LIKE ?
        GROUP BY hour
        ORDER BY hour
    """, (f"{today}%",))
    hourly_trend = [{"hour": f"{r[0]}:00", "count": r[1]} for r in hourly_rows]

    return StatsResponse(
        total_anomalies=total,
        today_anomalies=today_count,
        critical_count=critical,
        warning_count=warning,
        avg_anomaly_score=avg_score,
        anomaly_rate_pct=anomaly_rate,
        top_sensors=top_sensors,
        hourly_trend=hourly_trend,
    )
