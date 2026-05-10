# Industrial Anomaly AI — 프로젝트 컨텍스트

## 목표
IoT 센서 데이터 기반 실시간 이상탐지 AI 시스템.
센서 데이터(온도/진동/전류) → IsolationForest 모델 → FastAPI → 실시간 대시보드.

## 사용자 배경
- LLM/RAG/PEFT, CV(YOLO/SAM), 자율주행 파이프라인 경험
- FastAPI + Docker 경험
- 목표: 캠스텍 등 산업 AI 스타트업 취업 포트폴리오

## 기술 스택
- **IsolationForest (Scikit-learn)** — 비지도 이상탐지 모델
- **FastAPI** — REST API + WebSocket 실시간 스트리밍
- **MQTT (mosquitto)** — 센서 데이터 pub/sub 시뮬레이션
- **Streamlit** — 실시간 대시보드
- **Docker + Docker Compose** — 전체 시스템 배포

## 시스템 구조
```
센서 시뮬레이터 (MQTT publish)
        ↓
FastAPI 서버 (MQTT subscribe)
        ↓
IsolationForest → 이상 여부 판단
        ↓
WebSocket → 실시간 대시보드 (Streamlit)
        ↓
이상 감지 시 알림 로그 저장 (SQLite)
```

## 4단계 노트북 로드맵

### Notebook 01 — 센서 데이터 시뮬레이션
- 정상 / 이상 패턴 데이터 생성
- 온도/진동/전류 3채널
- 시각화 및 EDA

### Notebook 02 — 이상탐지 모델 학습
- IsolationForest 학습 및 평가
- 오염률(contamination) 튜닝
- 모델 저장 (joblib)

### Notebook 03 — MQTT + FastAPI 실시간 파이프라인
- mosquitto 브로커 연결
- FastAPI WebSocket 구현
- 실시간 예측 파이프라인

### Notebook 04 — 풀 통합 테스트
- Docker Compose로 전체 실행
- Streamlit 대시보드 연동
- 부하 테스트

## 파일 구조
```
industrial_anomaly_ai/
├── app/
│   ├── main.py               # FastAPI 엔트리포인트
│   ├── models/
│   │   └── anomaly_detector.py
│   ├── routers/
│   │   ├── detect.py         # POST /detect, WebSocket /ws
│   │   └── health.py
│   └── schemas/
│       └── sensor.py
├── data/raw/                 # 학습 데이터
├── models/                   # 저장된 모델 (joblib)
├── scripts/
│   ├── generate_data.py      # 학습 데이터 생성
│   └── mqtt_simulator.py     # 센서 MQTT 시뮬레이터
├── dashboard/
│   └── streamlit_app.py      # 실시간 대시보드
├── notebooks/                # 4단계 노트북
├── docker-compose.yml
└── Dockerfile
```

## 환경
- Python: Anaconda (agent_env 또는 신규 anomaly_env 커널)
- API 키: 불필요 (완전 로컬)
- MQTT: Docker mosquitto 브로커

## 진행 상황
- [x] 프로젝트 구조 생성
- [ ] Notebook 01 — 데이터 시뮬레이션
- [ ] Notebook 02 — 이상탐지 모델
- [ ] Notebook 03 — MQTT + FastAPI
- [ ] Notebook 04 — 풀 통합
- [ ] FastAPI 서빙
- [ ] Streamlit 대시보드
- [ ] Docker 배포
