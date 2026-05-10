"""
실시간 이상탐지 대시보드.
FastAPI WebSocket에서 데이터를 받아 실시간 차트로 표시.

실행: streamlit run dashboard/streamlit_app.py
"""

import streamlit as st
import requests
import json
import time
import random
import numpy as np
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_BASE = "http://localhost:8000"
WINDOW = 60  # 차트에 표시할 최근 N개

st.set_page_config(
    page_title="산업 이상탐지 대시보드",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 산업 IoT 실시간 이상탐지 대시보드")
st.caption("FastAPI + IsolationForest 기반 실시간 센서 모니터링")

# ── 누적 통계 KPI 카드 (stats API) ────────────────
st.subheader("누적 통계")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
try:
    s = requests.get(f"{API_BASE}/stats", timeout=2).json()
    kpi1.metric("전체 이상 감지", f"{s['total_anomalies']}건")
    kpi2.metric("오늘 이상 감지", f"{s['today_anomalies']}건")
    kpi3.metric("Critical", f"{s['critical_count']}건")
    kpi4.metric("Warning", f"{s['warning_count']}건")
    kpi5.metric("평균 이상 점수", f"{s['avg_anomaly_score']:.3f}")
except Exception:
    kpi1.metric("누적 통계", "서버 연결 필요")

st.divider()

# ── 사이드바 ──────────────────────────────────────
with st.sidebar:
    st.header("설정")
    sensor_id = st.text_input("센서 ID", value="sensor_01")
    update_interval = st.slider("갱신 주기 (초)", 0.5, 5.0, 1.0, 0.5)
    inject_anomaly = st.toggle("이상 패턴 주입", value=False)

    st.divider()
    st.subheader("API 상태")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            st.success("API 연결됨")
        else:
            st.error("API 응답 오류")
    except Exception:
        st.error("API 연결 실패\n`uvicorn app.main:app` 먼저 실행하세요")

# ── 세션 상태 초기화 ───────────────────────────────
for key, default in {
    "timestamps": deque(maxlen=WINDOW),
    "temperatures": deque(maxlen=WINDOW),
    "vibrations": deque(maxlen=WINDOW),
    "currents": deque(maxlen=WINDOW),
    "scores": deque(maxlen=WINDOW),
    "severities": deque(maxlen=WINDOW),
    "anomaly_count": 0,
    "total_count": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def generate_reading(anomaly: bool) -> dict:
    rng = np.random
    if anomaly:
        atype = random.choice(["overheating", "vibration", "current"])
        t = rng.normal(95, 5) if atype == "overheating" else rng.normal(70, 3)
        v = rng.normal(2.5, 0.3) if atype == "vibration" else rng.normal(0.5, 0.08)
        c = rng.normal(30, 3) if atype == "current" else rng.normal(12, 0.8)
    else:
        t, v, c = rng.normal(70, 3), rng.normal(0.5, 0.08), rng.normal(12, 0.8)
    return {
        "sensor_id": sensor_id,
        "temperature": round(float(t), 2),
        "vibration": round(float(v), 3),
        "current": round(float(c), 2),
    }


def call_api(reading: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/detect", json=reading, timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def severity_color(s: str) -> str:
    return {"normal": "🟢", "warning": "🟡", "critical": "🔴"}.get(s, "⚪")


# ── 지표 카드 ─────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
metric_temp  = col1.empty()
metric_vib   = col2.empty()
metric_curr  = col3.empty()
metric_anom  = col4.empty()

# ── 차트 ─────────────────────────────────────────
chart_placeholder = st.empty()
log_placeholder   = st.empty()

# ── 최근 이상 로그 ────────────────────────────────
st.subheader("최근 이상 감지 로그")
log_table = st.empty()

# ── 실시간 루프 ──────────────────────────────────
run_btn, stop_btn = st.columns(2)
running = st.session_state.get("running", False)

if run_btn.button("▶ 시작", use_container_width=True):
    st.session_state["running"] = True
    running = True
if stop_btn.button("⏹ 중지", use_container_width=True):
    st.session_state["running"] = False
    running = False

if running:
    reading = generate_reading(inject_anomaly)
    result = call_api(reading)

    if result:
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state.timestamps.append(ts)
        st.session_state.temperatures.append(result["temperature"])
        st.session_state.vibrations.append(result["vibration"])
        st.session_state.currents.append(result["current"])
        st.session_state.scores.append(result["anomaly_score"])
        st.session_state.severities.append(result["severity"])
        st.session_state.total_count += 1
        if result["is_anomaly"]:
            st.session_state.anomaly_count += 1

        # 지표 업데이트
        metric_temp.metric("온도 (°C)", f"{result['temperature']:.1f}")
        metric_vib.metric("진동 (mm/s)", f"{result['vibration']:.3f}")
        metric_curr.metric("전류 (A)", f"{result['current']:.1f}")
        anom_rate = st.session_state.anomaly_count / max(st.session_state.total_count, 1) * 100
        metric_anom.metric(
            f"{severity_color(result['severity'])} 이상 감지율",
            f"{anom_rate:.1f}%",
            delta=f"총 {st.session_state.anomaly_count}건",
        )

        # 차트 업데이트
        xs = list(st.session_state.timestamps)
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=["온도 (°C)", "진동 (mm/s)", "전류 (A)", "이상 점수"],
            vertical_spacing=0.08,
        )
        colors = ["#EF4444" if s != "normal" else "#3B82F6" for s in st.session_state.severities]

        fig.add_trace(go.Scatter(x=xs, y=list(st.session_state.temperatures),
                                  mode="lines+markers", marker=dict(color=colors),
                                  name="온도"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=list(st.session_state.vibrations),
                                  mode="lines+markers", marker=dict(color=colors),
                                  name="진동"), row=2, col=1)
        fig.add_trace(go.Scatter(x=xs, y=list(st.session_state.currents),
                                  mode="lines+markers", marker=dict(color=colors),
                                  name="전류"), row=3, col=1)
        fig.add_trace(go.Scatter(x=xs, y=list(st.session_state.scores),
                                  mode="lines+markers", marker=dict(color=colors),
                                  name="이상 점수"), row=4, col=1)
        fig.add_hline(y=-0.05, line_dash="dash", line_color="orange",
                      annotation_text="이상 임계값", row=4, col=1)

        fig.update_layout(height=700, showlegend=False, template="plotly_dark")
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    # 이상 로그 테이블
    try:
        logs = requests.get(f"{API_BASE}/logs?limit=10", timeout=2).json()
        if logs:
            import pandas as pd
            df = pd.DataFrame(logs)[["timestamp", "sensor_id", "temperature",
                                      "vibration", "current", "severity"]]
            log_table.dataframe(df, use_container_width=True)
    except Exception:
        pass

    time.sleep(update_interval)
    st.rerun()
