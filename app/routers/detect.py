import json
import asyncio
import sqlite3
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.models.anomaly_detector import detector
from app.schemas.sensor import SensorReading, DetectionResult

router = APIRouter()

DB_PATH = "data/anomaly_log.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id TEXT,
            timestamp TEXT,
            temperature REAL,
            vibration REAL,
            current REAL,
            anomaly_score REAL,
            is_anomaly INTEGER,
            severity TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_log(result: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO anomaly_log
        (sensor_id, timestamp, temperature, vibration, current, anomaly_score, is_anomaly, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result["sensor_id"], result["timestamp"], result["temperature"],
        result["vibration"], result["current"],
        result["anomaly_score"], int(result["is_anomaly"]), result["severity"],
    ))
    conn.commit()
    conn.close()


@router.on_event("startup")
async def startup():
    import os
    os.makedirs("data", exist_ok=True)
    init_db()


@router.post("/detect", response_model=DetectionResult)
async def detect(reading: SensorReading):
    ts = reading.timestamp or datetime.now()
    result = detector.predict(reading.temperature, reading.vibration, reading.current)

    output = {
        "sensor_id": reading.sensor_id,
        "timestamp": ts,
        "temperature": reading.temperature,
        "vibration": reading.vibration,
        "current": reading.current,
        **result,
    }
    if result["is_anomaly"]:
        save_log({**output, "timestamp": ts.isoformat()})

    return output


@router.get("/logs")
async def get_logs(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM anomaly_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    cols = ["id", "sensor_id", "timestamp", "temperature", "vibration",
            "current", "anomaly_score", "is_anomaly", "severity"]
    return [dict(zip(cols, r)) for r in rows]


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                pass


manager = ConnectionManager()


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """클라이언트가 센서 데이터를 보내면 실시간 탐지 결과를 반환."""
    await manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            reading = SensorReading(**data)
            ts = datetime.now()
            result = detector.predict(reading.temperature, reading.vibration, reading.current)
            output = {
                "sensor_id": reading.sensor_id,
                "timestamp": ts.isoformat(),
                "temperature": reading.temperature,
                "vibration": reading.vibration,
                "current": reading.current,
                **result,
            }
            if result["is_anomaly"]:
                save_log(output)
            await websocket.send_json(output)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
