from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import detect, health, train, stats

app = FastAPI(
    title="Industrial Anomaly Detection API",
    description="IoT 센서 기반 실시간 이상탐지 시스템",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(detect.router, tags=["Detection"])
app.include_router(train.router,  tags=["Training"])
app.include_router(stats.router,  tags=["Statistics"])


@app.on_event("startup")
async def startup():
    import os
    os.makedirs("data", exist_ok=True)
    detect.init_db()
